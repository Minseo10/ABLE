#!/bin/bash
# Usage: run_one.sh METHOD(baseline|object-first) GOAL_SCALE SEED PATH_STEPS
source ~/anaconda3/etc/profile.d/conda.sh && conda activate kinder
dir="$(dirname "$0")/$1/isolated-right-arm-g$2-s$3"; mkdir -p "$dir" && cd "$dir"
flags="--muscle-tone none --max-time 600 --goal-scale $2 --seed $3 --log-path log.json"
[[ $1 == object-first ]] && flags="$flags --object-first --limb-path-steps $4"
start=$(date +%s)
timeout -s INT 1800 python -m kinder_pddlstream_planning.limbrepositioning3d.run --variant isolated-right-arm $flags > out.txt 2>&1
echo "exit=$? wall=$(( $(date +%s) - start ))" >> out.txt
