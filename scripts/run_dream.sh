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
#
# PEOPLE, OUT, FIG and EXTRA (more dream_pipeline flags for every run, e.g. --random-goals)
# apply to every mode.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:?usage: run_dream.sh warmstart|fresh|generative [N=20] [D=4]}"
N="${2:-20}"
D="${3:-4}"
case "$MODE" in
    warmstart) NAME=warmstart_compare ;;
    fresh) NAME=fresh_goals ;;
    generative) NAME=generative_models ;;
    *) echo "unknown mode: $MODE" >&2; exit 1 ;;
esac
PEOPLE="${PEOPLE:-23 24 25 26 29}"
DEPLOYED="${DEPLOYED:-mlp:128x128:a0.001}"
NEW="${NEW:-net:256x256:d0.3:hist}"
MAX_JOBS="${MAX_JOBS:-16}"
OUT="${OUT:-data/dream/$NAME}"
FIG="${FIG:-figures/dream/$NAME}"
EXTRA="${EXTRA:-}"
LOGS="$OUT/logs"
mkdir -p "$LOGS"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [ -z "${CONDA_PREFIX:-}" ] && [ -d "$HOME/anaconda3/envs/kinder/bin" ]; then
    export PATH="$HOME/anaconda3/envs/kinder/bin:$PATH"
fi

run() { # person condition n_dreams tag seed_mode model [more dream_pipeline flags]
    local person="$1" condition="$2" dreams="$3" tag="$4" seed_mode="$5" model="$6"
    shift 6
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do wait -n || true; done
    python -m limb_baseline.dream_pipeline --condition "$condition" --n-dreams "$dreams" \
        --num-episodes "$N" --variation 0.3 --variant isolated-left-arm \
        --person "$person" --model "$model" --particles 32 --confidence 0.5 \
        --seed-mode "$seed_mode" --out "$OUT/${tag}_p$person.csv" --tag "$tag" "$@" $EXTRA \
        > "$LOGS/${tag}_p$person.log" 2>&1 &
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

"$MODE"
