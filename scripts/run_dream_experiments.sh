#!/usr/bin/env bash
# The dream pipeline (limb_baseline.dream_pipeline), notebook section 5 on the arm: for
# every person, plain, and the cache and both learned models with 0 and D dreams per
# episode, one goal per dream. One process per (person, line), so the memory persists
# across episodes; at most MAX_JOBS at once (one per physical core: 35 runs on 16 cores
# made every plan 3-4x slower and a third of them hit the time limit), the slow
# D-dream runs first. Then the averaged and per-person figures.
#
#   scripts/run_dream_experiments.sh [N=20] [D=4] [V=0.3] [VARIANT=isolated-left-arm]
#   PEOPLE="23 24 25 26 29" RIDGE=ridge MLP=mlp:128x128:a0.001 scripts/run_dream_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/.."

N="${1:-20}"        # episodes per person
D="${2:-4}"         # dreams per episode
V="${3:-0.3}"       # goal variation, rad
VARIANT="${4:-isolated-left-arm}"
PEOPLE="${PEOPLE:-23 24 25 26 29}"
RIDGE="${RIDGE:-ridge}"
MLP="${MLP:-mlp:128x128:a0.001}"
GOALS="${GOALS:-6}"
PARTICLES="${PARTICLES:-32}"     # 0 plans under the belief's mean
CONFIDENCE="${CONFIDENCE:-0.5}"
MAX_JOBS="${MAX_JOBS:-16}"
OUT="${OUT:-data/dream/main}"
FIG="${FIG:-figures/dream/main}"
LOGS="$OUT/logs"
mkdir -p "$LOGS"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# The planner compiles its IKFast module on first use and needs the env's g++ on PATH.
if [ -z "${CONDA_PREFIX:-}" ] && [ -d "$HOME/anaconda3/envs/kinder/bin" ]; then
    export PATH="$HOME/anaconda3/envs/kinder/bin:$PATH"
fi

run() { # person condition n_dreams name [model]
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do wait -n || true; done
    python -m limb_baseline.dream_pipeline --condition "$2" --n-dreams "$3" \
        --num-episodes "$N" --num-goals "$GOALS" --variation "$V" --variant "$VARIANT" \
        --person "$1" --model "${5:-ridge}" --particles "$PARTICLES" \
        --confidence "$CONFIDENCE" --out "$OUT/$4_p$1.csv" --tag "$4" \
        > "$LOGS/$4_p$1.log" 2>&1 &
}

# Slowest first, so the short runs fill in around them.
for p in $PEOPLE; do
    run "$p" learned "$D" "ridge_$D" "$RIDGE"
    run "$p" learned "$D" "mlp_$D" "$MLP"
done
for p in $PEOPLE; do run "$p" cache "$D" "cache_$D"; done
for p in $PEOPLE; do
    run "$p" plain 0 plain_0
    run "$p" cache 0 cache_0
    run "$p" learned 0 ridge_0 "$RIDGE"
    run "$p" learned 0 mlp_0 "$MLP"
done
wait

python scripts/make_dream_figures.py --csv "$OUT"/*_p*.csv \
    --out "$FIG/summary.png" --per-person
