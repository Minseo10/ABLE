#!/usr/bin/env bash
# Novel tasks: every real episode a goal of its own (--fresh-goals), dreams from a separate
# pool. Plain, the cache and the new model (4 dreams, step mode), all at once, same load.
#
#   scripts/run_fresh_goals.sh [N=20] [D=4]
set -euo pipefail
cd "$(dirname "$0")/.."

N="${1:-20}"
D="${2:-4}"
PEOPLE="${PEOPLE:-23 24 25 26 29}"
NEW="${NEW:-net:256x256:d0.3:hist}"
OUT="${OUT:-data/dream/fresh_goals}"
FIG="${FIG:-figures/dream/fresh_goals}"
LOGS="$OUT/logs"
mkdir -p "$LOGS"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [ -z "${CONDA_PREFIX:-}" ] && [ -d "$HOME/anaconda3/envs/kinder/bin" ]; then
    export PATH="$HOME/anaconda3/envs/kinder/bin:$PATH"
fi

run() { # person condition n_dreams name seed_mode [model]
    python -m limb_baseline.dream_pipeline --condition "$2" --n-dreams "$3" --fresh-goals \
        --num-episodes "$N" --variation 0.3 --variant isolated-left-arm \
        --person "$1" --model "${6:-ridge}" --particles 32 --confidence 0.5 \
        --seed-mode "$5" --out "$OUT/$4_p$1.csv" --tag "$4" > "$LOGS/$4_p$1.log" 2>&1 &
}

for p in $PEOPLE; do
    run "$p" learned "$D" "net_${D}_step" step "$NEW"
    run "$p" cache "$D" "cache_${D}_step" step
    run "$p" plain 0 plain_0 plan
done
wait

python scripts/make_dream_figures.py --csv "$OUT"/*_p??.csv --out "$FIG/summary.png" --per-person
