#!/usr/bin/env bash
# Stop each learned 4-dream run (ridge_4, mlp_4) once it has 10 real episodes; the
# other lines run their full 15. Detached from the editor session.
cd "$(dirname "$0")"
CAP=10
while true; do
    left=0
    for f in ridge_4_p*.csv mlp_4_p*.csv; do
        case $f in *_dreams.csv) continue;; esac
        pid=$(ps -eo pid,args | grep "limb_baseline.dream_pipeline" | grep -v grep \
              | grep -- "--out data/dream/rerun/$f " | awk '{print $1}')
        [ -z "$pid" ] && continue
        n=$(( $(wc -l < "$f") - 1 ))
        if [ "$n" -ge "$CAP" ]; then
            kill "$pid" && echo "$(date +%H:%M) stopped ${f%.csv} at $n episodes"
        else
            left=1
        fi
    done
    [ "$left" -eq 0 ] && { echo "$(date +%H:%M) all learned 4-dream runs capped or finished"; break; }
    sleep 30
done
