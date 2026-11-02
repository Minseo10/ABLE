#!/bin/bash
# One experiment: every person x condition for a config, at most $JOBS jobs at once.
#
#   scripts/run.sh CONFIG NAME [PERSONS] [CONDITIONS]
#   scripts/run.sh configs/friction/full.yaml friction_full "0 1 2 3 4" "plain learned prior_only"
#
# Writes data/NAME/<condition>_p<person>.csv (+ .pkl checkpoint; reruns resume).
# prior_only first trains the factory model: FACTORY_SHARDS x FACTORY_DREAMS dreams at the prior.
# Env: JOBS (16), PYTHON (python), FACTORY_SHARDS (=JOBS), FACTORY_DREAMS (8, per shard).
set -eu
CONFIG=$(realpath "$1"); NAME=$2
PERSONS=${3:-"0 1 2 3 4"}; CONDITIONS=${4:-"plain learned prior_only"}
JOBS=${JOBS:-16}; PYTHON=${PYTHON:-python}
FACTORY_SHARDS=${FACTORY_SHARDS:-$JOBS}; FACTORY_DREAMS=${FACTORY_DREAMS:-8}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/data/$NAME
mkdir -p "$OUT/logs" "$OUT/work"
export PYTHONPATH=$ROOT/src${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CONFIG OUT PYTHON

# Each job in its own working dir: PDDLStream writes temp files into the cwd.
job() {
  kind=$1; arg=$2; cond=${3:-}
  work=$OUT/work/${kind}_${cond}_$arg; mkdir -p "$work"; cd "$work"
  if [ "$kind" = factory ]; then
    $PYTHON -m able.run factory --config "$CONFIG" --dreams "$FACTORY_DREAMS" --seed "$arg" \
      --out "$OUT/factory/rows_$arg.pkl" > "$OUT/logs/factory_$arg.log" 2>&1
  else
    extra=(); [ "$cond" = prior_only ] && extra=(--prior-model "$OUT/factory.pkl")
    $PYTHON -m able.run person --config "$CONFIG" --condition "$cond" --person "$arg" --resume \
      --out "$OUT/${cond}_p$arg" "${extra[@]}" > "$OUT/logs/${cond}_p$arg.log" 2>&1
  fi || echo "failed: $kind $cond $arg (see $OUT/logs)"
}
export -f job

if [[ " $CONDITIONS " == *" prior_only "* ]] && [ ! -f "$OUT/factory.pkl" ]; then
  seq 0 $((FACTORY_SHARDS - 1)) | xargs -P "$JOBS" -I{} bash -c 'job factory {}'
  $PYTHON -m able.run fit --config "$CONFIG" --rows "$OUT"/factory/rows_*.pkl --out "$OUT/factory.pkl"
fi
for cond in $CONDITIONS; do for p in $PERSONS; do echo "$cond $p"; done; done |
  xargs -P "$JOBS" -L 1 bash -c 'job person "$2" "$1"' _
echo "done: $OUT"
