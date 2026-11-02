#!/bin/bash
# Runs the complexity sweep: 2 envs x 4 levels x 3 budgets x 10 seeds, 14 at a time.
# Usage: scripts/friction/run_sweep.sh   (from the repo root)
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-python}
OUT=$ROOT/data/friction/complexity
mkdir -p $OUT/logs
one() {
  env=$1; level=$2; sims=$3; seed=$4
  f=$OUT/${env}_${level}_s${sims}_seed${seed}.json
  [ -f "$f" ] && return 0
  $PYTHON $ROOT/scripts/friction/run_one.py --env $env --level $level \
    --sims $sims --seed $seed --out $f > $OUT/logs/${env}_${level}_s${sims}_seed${seed}.log 2>&1
}
export -f one; export OUT ROOT PYTHON
# Slowest jobs first.
for sims in 3000 1000 300; do for env in 2d 3d; do for level in L1 L2 L3 L4; do
  for seed in 0 1 2 3 4 5 6 7 8 9; do echo "$env $level $sims $seed"; done; done; done; done |
  xargs -P 14 -L 1 bash -c 'one "$@"' _
