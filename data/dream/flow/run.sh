#!/usr/bin/env bash
cd "$(dirname "$0")/../../.."
export PATH="$HOME/anaconda3/envs/kinder/bin:$PATH" OMP_NUM_THREADS=4
python scripts/eval_flow_policy.py search > data/dream/flow/search.log 2>&1 &&
python scripts/eval_flow_policy.py scaling --seeds 2 > data/dream/flow/scaling.log 2>&1
echo "EXIT $?" >> data/dream/flow/scaling.log
