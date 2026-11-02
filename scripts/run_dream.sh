#!/usr/bin/env bash
# The dream pipeline's planner experiments: one process per (person, line), at most MAX_JOBS
# at once, then the averaged and per-person figures.
#
#   scripts/run_dream.sh warmstart [N=20] [D=4]
#       6 repeated goals: plain, the deployed MLP and the new model (net:...:hist), with 0
#       and D dreams per episode, slowest first -> data/dream/warmstart_compare
#       SEED_MODE=step ... 20 0          # model seeds every MPC step; learned lines only
#                                        # (plain does not depend on the seed mode), tagged `_step`
#       ZERO=0 SEED_MODE=step ... 20 4   # dreaming lines only
#       CACHE=1 LEARNED=0 ...            # the goal-keyed cache instead of the learned models
#       MODELS="net" ...                 # which learned models (net, mlp), when LEARNED=1
#   scripts/run_dream.sh fresh [N=20] [D=4]
#       Novel tasks: every real episode a goal of its own (--fresh-goals), dreams from a
#       separate pool. Plain, the cache and the new model (D dreams, step mode), all at
#       once, same load -> data/dream/fresh_goals
#   MODELS="gen:flow:hist gen:cvae:hist" scripts/run_dream.sh generative [N=20] [D=4]
#       Generative warm starts: fresh goals, varied geometry, D dreams, step mode, K=8 seed
#       candidates per MPC step (the model's most likely plan and 7 draws; for regression,
#       its plan and 7 noisy copies) -> data/dream/generative_models. No figures. Its plain,
#       cache and 1-candidate regression lines come from
#       EXTRA=--vary-geometry OUT=data/dream/generative_models \
#           FIG=figures/dream/generative_models scripts/run_dream.sh fresh 20 4
#   scripts/run_dream.sh planners [N=20] [D=4]
#       PDDLStream vs POMDP (PLANNERS="pddlstream pouct"): plain, and each MODELS model with
#       D and 0 dreams; fresh goals, varied geometry, step mode, K=8 (K="3 8" runs both)
#       -> data/dream/planners. Longest first, interleaved; MAX_JOBS default 12. Compare
#       `plan_cpu_time` (not inflated by core contention) as well as `plan_time`.
#       FAIR=1: runs rotate one pinned core every ROTATE s (default 60), and filler runs
#       ($OUT/filler) keep MAX_JOBS busy until the last real run ends.
#   scripts/run_dream.sh prior [N=20] [D=4]
#       Prior-only baseline: dream HOMES (default 100) hypothetical people from the prior, D
#       dreams each, sharded over MAX_JOBS cores (able.tasks.limb.prior_factory); fit one flow
#       model per dream budget in BUDGETS (default "400 80"), at 4000 and 12000 steps, and
#       ship the one with the lower held-out loss; then deploy it frozen on PEOPLE with
#       pddlstream_full's settings (FAIR=1 by default) -> data/dream/prior_only
#
# PEOPLE, OUT, FIG, VARIANT (scene and limb, default isolated-left-arm) and EXTRA (more
# dream_pipeline flags for every run, e.g. --random-goals) apply to every mode.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:?usage: run_dream.sh warmstart|fresh|generative|planners [N=20] [D=4]}"
N="${2:-20}"
D="${3:-4}"
case "$MODE" in
    warmstart) NAME=warmstart_compare ;;
    fresh) NAME=fresh_goals ;;
    generative) NAME=generative_models ;;
    planners) NAME=planners; MAX_JOBS="${MAX_JOBS:-12}" ;;
    prior) NAME=prior_only; FAIR="${FAIR:-1}"; EXTRA="${EXTRA:---no-feedback}" ;;
    *) echo "unknown mode: $MODE" >&2; exit 1 ;;
esac
PEOPLE="${PEOPLE:-23 24 25 26 29}"
DEPLOYED="${DEPLOYED:-mlp:128x128:a0.001}"
NEW="${NEW:-net:256x256:d0.3:hist}"
MAX_JOBS="${MAX_JOBS:-16}"
OUT="${OUT:-data/dream/$NAME}"
FIG="${FIG:-figures/dream/$NAME}"
EXTRA="${EXTRA:-}"
VARIANT="${VARIANT:-isolated-left-arm}"
LOGS="$OUT/logs"
mkdir -p "$LOGS"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [ -z "${CONDA_PREFIX:-}" ] && [ -d "$HOME/anaconda3/envs/kinder/bin" ]; then
    export PATH="$HOME/anaconda3/envs/kinder/bin:$PATH"
fi

# One physical core per run, alternating L3 domains (the 9950X's two CCDs). PIN=0 disables.
PIN="${PIN:-1}"
CORES=()
if [ "$PIN" = 1 ] && command -v taskset > /dev/null; then
    mapfile -t by_cache < <(for c in /sys/devices/system/cpu/cpu[0-9]*; do
        echo "$(cat "$c/cache/index3/shared_cpu_list" 2>/dev/null || echo 0) $(cat "$c/topology/thread_siblings_list")"
    done | sort -u | sort -t' ' -k2,2n)
    mapfile -t domains < <(printf '%s\n' "${by_cache[@]}" | cut -d' ' -f1 | sort -u)
    for ((i = 0; ; i++)); do
        added=0
        for d in "${domains[@]}"; do
            mapfile -t mine < <(printf '%s\n' "${by_cache[@]}" | awk -v d="$d" '$1 == d {print $2}')
            if [ "$i" -lt "${#mine[@]}" ]; then CORES+=("${mine[$i]}"); added=1; fi
        done
        [ "$added" = 1 ] || break
    done
    [ "$MAX_JOBS" -le "${#CORES[@]}" ] || MAX_JOBS="${#CORES[@]}"
fi
SLOT_PIDS=()
REAL_PIDS=()
FAIR="${FAIR:-0}"
ROTATE="${ROTATE:-60}"

run() { # person condition n_dreams tag seed_mode model [more dream_pipeline flags]
    local person="$1" condition="$2" dreams="$3" tag="$4" seed_mode="$5" model="$6"
    shift 6
    local slot=-1 i pin=()
    while :; do
        for ((i = 0; i < MAX_JOBS; i++)); do
            if [ -z "${SLOT_PIDS[$i]:-}" ] || ! kill -0 "${SLOT_PIDS[$i]}" 2>/dev/null; then
                slot=$i; break
            fi
        done
        [ "$slot" -ge 0 ] && break
        wait -n || true
    done
    [ "${#CORES[@]}" -gt 0 ] && pin=(taskset -c "${CORES[$slot]}")
    "${pin[@]}" python -m able.tasks.limb.dream_pipeline --condition "$condition" --n-dreams "$dreams" \
        --num-episodes "$N" --variation 0.3 --variant "$VARIANT" \
        --person "$person" --model "$model" --particles 32 --confidence 0.5 \
        --seed-mode "$seed_mode" --out "$OUT/${tag}_p$person.csv" --tag "$tag" "$@" $EXTRA \
        > "$LOGS/${tag}_p$person.log" 2>&1 &
    SLOT_PIDS[$slot]=$!
    REAL_PIDS+=("$!")
}

rotate() { # shift the runs under $OUT one core every ROTATE seconds
    local offset=0 n="${#CORES[@]}" pids i pid
    [ "$MAX_JOBS" -lt "$n" ] && n="$MAX_JOBS"
    while :; do
        mapfile -t pids < <(ps -eo pid=,args= | awk -v out="--out $OUT/" \
            '$2 ~ /python/ && index($0, "able.tasks.limb.dream_pipeline") && index($0, out) {print $1}' | sort -n)
        for ((i = 0; i < ${#pids[@]} && i < n; i++)); do
            for pid in "${pids[$i]}" $(pgrep -P "${pids[$i]}" || true); do
                taskset -a -cp "${CORES[$(((i + offset) % n))]}" "$pid" > /dev/null 2>&1 || true
            done
        done
        offset=$((offset + 1))
        sleep "$ROTATE"
    done
}

finish() { # wait for the real runs; with FAIR=1, fill freed cores with filler runs
    if [ "$FAIR" != 1 ]; then wait; return; fi
    local fillers=() alive busy pid k=0 planner
    mkdir -p "$OUT/filler"
    while :; do
        alive=0
        for pid in "${REAL_PIDS[@]}"; do kill -0 "$pid" 2> /dev/null && alive=$((alive + 1)); done
        [ "$alive" -eq 0 ] && break
        busy="$alive"
        for pid in "${fillers[@]}"; do kill -0 "$pid" 2> /dev/null && busy=$((busy + 1)); done
        while [ "$busy" -lt "$MAX_JOBS" ]; do
            planner=pddlstream; [ $((k % 2)) = 1 ] && planner=pouct
            python -m able.tasks.limb.dream_pipeline --condition plain --n-dreams 0 \
                --num-episodes 100 --variation 0.3 --variant "$VARIANT" --person $((100 + k)) \
                --model ridge --particles 32 --confidence 0.5 --seed-mode plan \
                --planner "$planner" --fresh-goals --vary-geometry --tag filler $EXTRA \
                --out "$OUT/filler/filler_$k.csv" > "$LOGS/filler_$k.log" 2>&1 &
            fillers+=("$!")
            k=$((k + 1)); busy=$((busy + 1))
        done
        sleep 20
    done
    [ "${#fillers[@]}" -gt 0 ] && kill "${fillers[@]}" 2> /dev/null || true
    [ -n "${ROTATOR:-}" ] && kill "$ROTATOR" 2> /dev/null || true
    wait || true
}

warmstart() {
    local mode="${SEED_MODE:-plan}" zero="${ZERO:-1}" cache="${CACHE:-0}" learned="${LEARNED:-1}"
    local models="${MODELS:-net mlp}" suffix="" p
    [ "$mode" = step ] && suffix="_step"
    line() { run "$1" "$2" "$3" "$4" "$mode" "${5:-ridge}" --num-goals 6; }
    if [ "$D" -gt 0 ]; then
        for p in $PEOPLE; do
            [ "$cache" = 1 ] && line "$p" cache "$D" "cache_$D$suffix"
            [ "$learned" = 1 ] || continue
            [[ " $models " == *" net "* ]] && line "$p" learned "$D" "net_$D$suffix" "$NEW"
            [[ " $models " == *" mlp "* ]] && line "$p" learned "$D" "mlp_$D$suffix" "$DEPLOYED"
        done
    fi
    for p in $PEOPLE; do
        [ "$zero" = 1 ] || break
        [ "$mode" = plan ] && line "$p" plain 0 plain_0
        [ "$cache" = 1 ] && line "$p" cache 0 "cache_0$suffix"
        [ "$learned" = 1 ] || continue
        [[ " $models " == *" net "* ]] && line "$p" learned 0 "net_0$suffix" "$NEW"
        [[ " $models " == *" mlp "* ]] && line "$p" learned 0 "mlp_0$suffix" "$DEPLOYED"
    done
    wait
    python scripts/dream_figures.py summary --csv "$OUT"/*_p*.csv --out "$FIG/summary.png" --per-person
}

fresh() {
    local p
    for p in $PEOPLE; do
        run "$p" learned "$D" "net_${D}_step" step "$NEW" --fresh-goals
        run "$p" cache "$D" "cache_${D}_step" step ridge --fresh-goals
        run "$p" plain 0 plain_0 plan ridge --fresh-goals
    done
    wait
    python scripts/dream_figures.py summary --csv "$OUT"/*_p??.csv --out "$FIG/summary.png" --per-person
}

generative() {
    local models="${MODELS:?set MODELS, e.g. \"gen:flow:hist gen:cvae:hist\"}" k="${K:-8}" p m tag
    for p in $PEOPLE; do
        for m in $models; do
            tag="$(echo "$m" | cut -d: -f1-2 | tr ':' '_')_${D}_step_k$k"
            [ "${m%%:*}" = net ] && tag="net_${D}_step_k$k"
            run "$p" learned "$D" "$tag" step "$m" --fresh-goals --vary-geometry --seed-rollouts "$k"
        done
    done
    wait
}

planners() {
    # Longest first, interleaved, so every line sees roughly the same load.
    local planners="${PLANNERS:-pddlstream pouct}" ks="${K:-8}"
    local models="${MODELS:-gen:flow:hist gen:ddpm:hist}" p planner m name k dreams
    local flags=(--fresh-goals --vary-geometry)
    if [ "$FAIR" = 1 ] && [ "${#CORES[@]}" -gt 0 ]; then rotate & ROTATOR=$!; fi
    for dreams in "$D" 0; do
        for p in $PEOPLE; do
            for planner in $planners; do
                for m in $models; do
                    for k in $ks; do
                        name="$(echo "$m" | cut -d: -f1-2 | tr ':' '_')"
                        run "$p" learned "$dreams" "${planner}_${name}_${dreams}_step_k$k" \
                            step "$m" --planner "$planner" "${flags[@]}" --seed-rollouts "$k"
                    done
                done
            done
        done
    done
    for p in $PEOPLE; do
        for planner in $planners; do
            run "$p" plain 0 "${planner}_plain_0" plan ridge --planner "$planner" "${flags[@]}"
        done
    done
    finish
    python scripts/dream_figures.py summary --csv "$OUT"/*_p*[0-9].csv --out "$FIG/summary.png" --per-person
}

prior() {
    local homes="${HOMES:-100}" budgets="${BUDGETS:-400 80}" factory="$OUT/factory"
    local shard b steps best pin pids=()
    mkdir -p "$factory"
    # 1. Factory dreams, one shard per core; a rerun resumes each shard's finished homes.
    for ((shard = 0; shard < MAX_JOBS; shard++)); do
        pin=()
        [ "${#CORES[@]}" -gt 0 ] && pin=(taskset -c "${CORES[$shard]}")
        "${pin[@]}" python -m able.tasks.limb.prior_factory dream --homes "$homes" \
            --dreams-per-home "$D" --shard "$shard" --num-shards "$MAX_JOBS" \
            --variant "$VARIANT" --out-dir "$factory" > "$LOGS/factory_$shard.log" 2>&1 &
        pids+=("$!")
    done
    wait "${pids[@]}"
    # 2. One model per budget; the step count with the lower held-out loss is shipped.
    for b in $budgets; do
        for steps in 4000 12000; do
            python -m able.tasks.limb.prior_factory fit --factory "$factory" --dreams "$b" \
                --model "gen:flow:s$steps:hist" --out "$factory/g_${b}_s$steps.pkl" \
                > "$LOGS/fit_${b}_s$steps.log" 2>&1 &
        done
    done
    wait
    for b in $budgets; do
        best=$(python - "$factory" "$b" <<'PY'
import json, sys
loss = {s: json.loads(open(f"{sys.argv[1]}/g_{sys.argv[2]}_s{s}.jsonl").readline())["val_loss"]
        for s in (4000, 12000)}
loss = {s: v if v == v else float("inf") for s, v in loss.items()}  # no held-out rows: 4000
print(min(loss, key=loss.get))
PY
)
        echo "budget $b: shipping s$best"
        ln -sf "g_${b}_s$best.pkl" "$factory/g_$b.pkl"
    done
    # 3. Deploy each budget's model, frozen, on every person.
    if [ "$FAIR" = 1 ] && [ "${#CORES[@]}" -gt 0 ]; then rotate & ROTATOR=$!; fi
    for p in $PEOPLE; do
        for b in $budgets; do
            run "$p" prior 0 "pddlstream_prior_${b}_step_k8" step gen:flow:hist \
                --prior-model "$factory/g_$b.pkl" --fresh-goals --vary-geometry --seed-rollouts 8
        done
    done
    finish
    python scripts/dream_figures.py summary --csv data/dream/pddlstream_full/*_p*[0-9].csv \
        "$OUT"/*_p*[0-9].csv --out "$FIG/summary.png" --per-person
}

"$MODE"
