#!/usr/bin/env bash
# Step 2 of the warm-start work: planner outcomes for plain, the deployed MLP and the new
# model (net:...:hist), with 0 and D dreams per episode, one process per (person, line),
# at most MAX_JOBS at once, slowest first. Then the averaged and per-person figures.
#
#   scripts/run_warmstart_comparison.sh [N=20] [D=4]
#   SEED_MODE=step scripts/run_warmstart_comparison.sh 20 0   # model seeds every MPC step;
#       learned lines only (plain does not depend on the seed mode), tagged `_step`
set -euo pipefail
cd "$(dirname "$0")/.."

N="${1:-20}"
D="${2:-4}"
PEOPLE="${PEOPLE:-23 24 25 26 29}"
DEPLOYED="${DEPLOYED:-mlp:128x128:a0.001}"
NEW="${NEW:-net:256x256:d0.3:hist}"
MAX_JOBS="${MAX_JOBS:-16}"
SEED_MODE="${SEED_MODE:-plan}"
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
        run "$p" learned "$D" "net_$D$SUFFIX" "$NEW"
        run "$p" learned "$D" "mlp_$D$SUFFIX" "$DEPLOYED"
    done
fi
for p in $PEOPLE; do
    [ "$SEED_MODE" = plan ] && run "$p" plain 0 plain_0
    run "$p" learned 0 "net_0$SUFFIX" "$NEW"
    run "$p" learned 0 "mlp_0$SUFFIX" "$DEPLOYED"
done
wait

python scripts/make_dream_figures.py --csv "$OUT"/*_p*.csv --out "$FIG/summary.png" --per-person
