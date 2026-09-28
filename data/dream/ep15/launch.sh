#!/usr/bin/env bash
# Every rerun and step line again, 15 episodes, with per-episode checkpoints: rerunning
# this script (or raising EPISODES) resumes each run where it stopped. Slowest first,
# at most MAX_JOBS pipeline processes on the machine at once.
cd "$(dirname "$0")/../../.."
export PATH="$HOME/anaconda3/envs/kinder/bin:$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
EPISODES=${EPISODES:-15}; MAX_JOBS=16; OUT=data/dream/ep15
RIDGE=ridge; MLP=mlp:128x128:a0.001
PEOPLE="23 24 25 26 29"
busy() { ps -eo args | grep "limb_baseline.dream_pipeline" | grep -v grep | wc -l; }
run() { # person condition n_dreams name seed_mode [model]
    while [ "$(busy)" -ge "$MAX_JOBS" ]; do sleep 15; done
    python -m limb_baseline.dream_pipeline --condition "$2" --n-dreams "$3" \
        --num-episodes "$EPISODES" --num-goals 6 --variation 0.3 --person "$1" \
        --model "${6:-ridge}" --particles 32 --confidence 0.5 --seed-mode "$5" --resume \
        --out "$OUT/$4_p$1.csv" --tag "$4" >> "$OUT/logs/$4_p$1.log" 2>&1 &
    sleep 2
}
for p in $PEOPLE; do
    run $p learned 4 ridge_4 plan $RIDGE; run $p learned 4 mlp_4 plan $MLP
    run $p learned 4 ridge_4_step step $RIDGE; run $p learned 4 mlp_4_step step $MLP
done
for p in $PEOPLE; do run $p cache 4 cache_4 plan; done
for p in $PEOPLE; do
    run $p plain 0 plain_0 plan; run $p cache 0 cache_0 plan
    run $p learned 0 ridge_0 plan $RIDGE; run $p learned 0 mlp_0 plan $MLP
    run $p learned 0 ridge_0_step step $RIDGE; run $p learned 0 mlp_0_step step $MLP
done
wait
echo "$(date '+%F %H:%M') all ep15 runs finished"
