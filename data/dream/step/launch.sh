#!/usr/bin/env bash
# Option 2: the learned lines with the model's prediction as batch 0 of every MPC step
# (--seed-mode step) instead of a whole policy rollout. Same people and settings as rerun.
cd "$(dirname "$0")/../../.."
export PATH="$HOME/anaconda3/envs/kinder/bin:$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MAX_JOBS=16; OUT=data/dream/step
RIDGE=ridge; MLP=mlp:128x128:a0.001
busy() { ps -eo args | grep "limb_baseline.dream_pipeline" | grep -v grep | wc -l; }
run() { # person n_dreams name model
    while [ "$(busy)" -ge "$MAX_JOBS" ]; do sleep 15; done
    python -m limb_baseline.dream_pipeline --condition learned --n-dreams "$2" \
        --num-episodes 10 --num-goals 6 --variation 0.3 --person "$1" \
        --model "$4" --particles 32 --confidence 0.5 --seed-mode step \
        --out "$OUT/$3_p$1.csv" --tag "$3" > "$OUT/logs/$3_p$1.log" 2>&1 &
    sleep 2
}
for p in 23 24 25 26 29; do run $p 4 ridge_4_step $RIDGE; run $p 4 mlp_4_step $MLP; done
for p in 23 24 25 26 29; do run $p 0 ridge_0_step $RIDGE; run $p 0 mlp_0_step $MLP; done
wait
echo "$(date +%H:%M) all step runs finished"
