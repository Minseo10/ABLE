#!/usr/bin/env bash
# Step 2 of the warm-start work: planner outcomes for plain, the deployed MLP and the new
# model (net:...:hist), with 0 and D dreams per episode, one process per (person, line),
# at most MAX_JOBS at once, slowest first. Then the averaged and per-person figures.
#
#   scripts/run_warmstart_comparison.sh [N=20] [D=4]
#   SEED_MODE=step scripts/run_warmstart_comparison.sh 20 0   # model seeds every MPC step;
#       learned lines only (plain does not depend on the seed mode), tagged `_step`
#   ZERO=0 SEED_MODE=step scripts/run_warmstart_comparison.sh 20 4   # dreaming lines only
#   CACHE=1 LEARNED=0 ... # the goal-keyed cache instead of the learned models
set -euo pipefail
cd "$(dirname "$0")/.."

N="${1:-20}"
D="${2:-4}"
PEOPLE="${PEOPLE:-23 24 25 26 29}"
DEPLOYED="${DEPLOYED:-mlp:128x128:a0.001}"
NEW="${NEW:-net:256x256:d0.3:hist}"
MAX_JOBS="${MAX_JOBS:-16}"
SEED_MODE="${SEED_MODE:-plan}"
ZERO="${ZERO:-1}"
CACHE="${CACHE:-0}"
LEARNED="${LEARNED:-1}"
MODELS="${MODELS:-net mlp}"   # which learned models, when LEARNED=1
SUFFIX=""
[ "$SEED_MODE" = step ] && SUFFIX="_step"
OUT="${OUT:-data/dream/warmstart_compare}"
FIG="${FIG:-figures/dream/warmstart_compare}"
LOGS="$OUT/logs"
mkdir -p "$LOGS"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [ -z "${CONDA_PREFIX:-}" ] && [ -d "$HOME/anaconda3/envs/kinder/bin" ]; then
    export PATH="$HOME/anaconda3/envs/kinder/bin:$PATH"
fi

run() { # person condition n_dreams name [model]
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do wait -n || true; done
    python -m limb_baseline.dream_pipeline --condition "$2" --n-dreams "$3" \
        --num-episodes "$N" --num-goals 6 --variation 0.3 --variant isolated-left-arm \
        --person "$1" --model "${5:-ridge}" --particles 32 --confidence 0.5 \
        --seed-mode "$SEED_MODE" \
        --out "$OUT/$4_p$1.csv" --tag "$4" > "$LOGS/$4_p$1.log" 2>&1 &
}

if [ "$D" -gt 0 ]; then
    for p in $PEOPLE; do
        [ "$CACHE" = 1 ] && run "$p" cache "$D" "cache_$D$SUFFIX"
        [ "$LEARNED" = 1 ] || continue
        [[ " $MODELS " == *" net "* ]] && run "$p" learned "$D" "net_$D$SUFFIX" "$NEW"
        [[ " $MODELS " == *" mlp "* ]] && run "$p" learned "$D" "mlp_$D$SUFFIX" "$DEPLOYED"
    done
fi
for p in $PEOPLE; do
    [ "$ZERO" = 1 ] || break
    [ "$SEED_MODE" = plan ] && run "$p" plain 0 plain_0
    [ "$CACHE" = 1 ] && run "$p" cache 0 "cache_0$SUFFIX"
    [ "$LEARNED" = 1 ] || continue
    [[ " $MODELS " == *" net "* ]] && run "$p" learned 0 "net_0$SUFFIX" "$NEW"
    [[ " $MODELS " == *" mlp "* ]] && run "$p" learned 0 "mlp_0$SUFFIX" "$DEPLOYED"
done
wait

python scripts/make_dream_figures.py --csv "$OUT"/*_p*.csv --out "$FIG/summary.png" --per-person
