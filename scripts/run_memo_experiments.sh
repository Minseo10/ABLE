#!/usr/bin/env bash
# Plain PDDLStream against the memorized plan-limb-motion stream, two experiments:
#   1. one task repeated N times            -> data/memo/same_task.csv
#   2. N tasks whose goals differ by <= V rad -> data/memo/varied_task.csv
# The four sequences run in parallel, one process each, so the memo can persist across
# iterations and the two conditions see the same machine load.
#
#   scripts/run_memo_experiments.sh [N=20] [V=0.3] [VARIANT=isolated-left-arm]
set -euo pipefail
cd "$(dirname "$0")/.."

N="${1:-20}"
V="${2:-0.3}"
VARIANT="${3:-isolated-left-arm}"
OUT=data/memo
LOGS="$OUT/logs"
mkdir -p "$LOGS"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# The planner compiles its IKFast module on first use and needs the env's g++ on PATH.
if [ -z "${CONDA_PREFIX:-}" ] && [ -d "$HOME/anaconda3/envs/kinder/bin" ]; then
    export PATH="$HOME/anaconda3/envs/kinder/bin:$PATH"
fi

run() { # condition variation csv
    python -m limb_baseline.memo --condition "$1" --variation "$2" --num-tasks "$N" \
        --variant "$VARIANT" --out "$OUT/$3.csv" > "$LOGS/$3_$1.log" 2>&1 &
}

run plain 0    same_task
run warm  0    same_task
run plain "$V" varied_task
run warm  "$V" varied_task
wait

python scripts/make_memo_figures.py --csv "$OUT/same_task.csv" --csv "$OUT/varied_task.csv" \
    --out figures/memo/plan_time.png
python scripts/make_memo_figures.py --csv "$OUT/same_task.csv" --csv "$OUT/varied_task.csv" \
    --out figures/memo/plan_time_log.png --log
