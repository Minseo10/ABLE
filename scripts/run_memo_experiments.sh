#!/usr/bin/env bash
# The MPC seeded from a memorized plan (warm) or a model refit per iteration (learned),
# against plain PDDLStream, on one task repeated N times and on N tasks whose goals
# differ by <= V rad. Four sequences run in parallel, one process each, so the memo and
# the model persist across iterations; plain rows are reused from data/memo/replay.
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

run() { # condition variation csv [model]
    python -m limb_baseline.memo --condition "$1" --variation "$2" --num-tasks "$N" \
        --variant "$VARIANT" --model "${4:-mlp_light}" --out "$OUT/$3.csv" \
        > "$LOGS/$3_$1${4:+_$4}.log" 2>&1 &
}

# Plain rows come from data/memo/replay, which planned the same tasks with the same seeds.
for v in 0 "$V"; do
    csv=same_task; [ "$v" != 0 ] && csv=varied_task
    run warm    "$v" $csv
    run learned "$v" $csv ridge
    run learned "$v" $csv mlp_light
done
wait

for pair in "plain,warm memo" "plain,learned:ridge,learned:mlp_light learned"; do
    set -- $pair
    python scripts/make_memo_figures.py --csv "$OUT/same_task.csv" --csv "$OUT/varied_task.csv" \
        --plain-csv "$OUT/replay/same_task.csv" --plain-csv "$OUT/replay/varied_task.csv" \
        --conditions "$1" --out "figures/memo/${2}_plan_time.png"
done
