#!/bin/bash
# Usage: run_one.sh GOAL_SCALE SEED PATH_STEPS
source ~/anaconda3/etc/profile.d/conda.sh && conda activate kinder
dir="$(dirname "$0")/object-first/bed-right-arm-g$1-s$2"; mkdir -p "$dir" && cd "$dir"
start=$(date +%s)
timeout -s INT 1800 python -m kinder_pddlstream_planning.limbrepositioning3d.run --variant bed-right-arm \
  --muscle-tone none --max-time 600 --goal-scale $1 --seed $2 --no-check-base-furniture-collisions \
  --object-first --limb-path-steps $3 --log-path log.json > out.txt 2>&1
echo "exit=$? wall=$(( $(date +%s) - start ))" >> out.txt
