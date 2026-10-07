#!/usr/bin/env bash
# Stop persons 23 and 24's 4-dream runs (cache, ridge, MLP) at 6 real episodes; their
# dreams keep hitting the time limit and would run until ~18:00 otherwise.
cd "$(dirname "$0")"
while true; do
    left=0
    for f in cache_4_p23.csv ridge_4_p23.csv mlp_4_p23.csv cache_4_p24.csv ridge_4_p24.csv mlp_4_p24.csv; do
        pid=$(ps -eo pid,args | grep "limb_baseline.dream_pipeline" | grep -v grep \
              | grep -- "--out data/dream/rerun/$f " | awk '{print $1}')
        [ -z "$pid" ] && continue
        n=$(( $(wc -l < "$f") - 1 ))
        if [ "$n" -ge 6 ]; then kill "$pid" && echo "$(date +%H:%M) stopped ${f%.csv} at $n episodes"
        else left=1; fi
    done
    [ "$left" -eq 0 ] && { echo "$(date +%H:%M) all six capped or finished"; break; }
    sleep 30
done
