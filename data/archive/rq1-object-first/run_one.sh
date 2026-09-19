#!/bin/bash
# Usage: run_one.sh METHOD(baseline|object-first) VARIANT GOAL_SCALE SEED
source ~/anaconda3/etc/profile.d/conda.sh && conda activate kinder
dir="$(dirname "$0")/$1/$2-g$3-s$4"; mkdir -p "$dir" && cd "$dir"
flags="--muscle-tone none --max-time 600 --goal-scale $3 --seed $4 --log-path log.json"
[[ $1 == object-first ]] && flags="$flags --object-first"
start=$(date +%s)
timeout -s INT 1800 python -m kinder_pddlstream_planning.limbrepositioning3d.run --variant "$2" $flags > out.txt 2>&1
echo "exit=$? wall=$(( $(date +%s) - start ))" >> out.txt
