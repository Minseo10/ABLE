#!/bin/bash
# Post-merge check: Stage A (golden fingerprints) and Stage B (end-to-end runs), in parallel.
#
#   scripts/check.sh [--tasks "limb friction bedside"] [--update-golden] [--keep]
#
# Prints one table and the log dir; exits non-zero on any FAIL.
# Env: PYTHON (python), JOBS (12 python processes at once; a limb one takes ~1 GB).
# Baselines: tests/golden/<hostname>/<task>.json, written by --update-golden.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=$ROOT/src${PYTHONPATH:+:$PYTHONPATH}
export WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${PYTHON:-python}" "$ROOT/scripts/check/run.py" "$@"
