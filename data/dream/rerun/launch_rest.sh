#!/usr/bin/env bash
# The rest of the rerun, 10 episodes: the no-dream lines of persons 25, 26, 29 and all
# seven lines of persons $1 $2 again (their first runs were stopped), slowest first. At most MAX_JOBS
# pipeline processes on the machine at once, counting the runs already going.
cd "$(dirname "$0")/../../.."
export PATH="$HOME/anaconda3/envs/kinder/bin:$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
NEW="$1 $2"; MAX_JOBS=16; OUT=data/dream/rerun
RIDGE=ridge; MLP=mlp:128x128:a0.001
busy() { ps -eo args | grep "limb_baseline.dream_pipeline" | grep -v grep | wc -l; }
run() { # person condition n_dreams name [model]
    while [ "$(busy)" -ge "$MAX_JOBS" ]; do sleep 15; done
    python -m limb_baseline.dream_pipeline --condition "$2" --n-dreams "$3" \
        --num-episodes 10 --num-goals 6 --variation 0.3 --person "$1" \
        --model "${5:-ridge}" --particles 32 --confidence 0.5 \
        --out "$OUT/$4_p$1.csv" --tag "$4" > "$OUT/logs/$4_p$1.log" 2>&1 &
    sleep 2
}
for p in $NEW; do run $p learned 4 ridge_4 $RIDGE; run $p learned 4 mlp_4 $MLP; done
for p in $NEW; do run $p cache 4 cache_4; done
for p in $NEW 25 26 29; do
    # Person 23's plain run already has 12 episodes from before; reuse it.
    [ "$p" != 23 ] && run $p plain 0 plain_0
    run $p cache 0 cache_0
    run $p learned 0 ridge_0 $RIDGE; run $p learned 0 mlp_0 $MLP
done
wait
