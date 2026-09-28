#!/usr/bin/env bash
# Stop each rerun process once its CSV has 10 real episodes, and everything at 14:50;
# then draw the final figures (first 10 episodes of every run).
cd "$(dirname "$0")"
DEADLINE=$(date -d "tomorrow 23:59" +%s)  # no deadline: every run completes 10 episodes
while true; do
    for line in $(ps -eo pid,args | grep "limb_baseline.dream_pipeline" | grep -v grep \
                  | grep -o "^ *[0-9]* .*--out data/dream/rerun/[a-z_0-9]*\.csv" \
                  | awk '{print $1":"$NF}'); do
        pid=${line%%:*}; f=$(basename "${line#*:}")
        [ -f "$f" ] || continue
        n=$(( $(wc -l < "$f") - 1 ))
        [ "$n" -ge 10 ] && kill "$pid" && echo "$(date +%H:%M) stopped ${f%.csv} at $n episodes"
    done
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        ps -eo pid,args | grep "[l]aunch_rest.sh" | awk '{print $1}' | xargs -r kill
        ps -eo pid,args | grep "limb_baseline.dream_pipeline" | grep -v grep \
            | grep "data/dream/rerun" | awk '{print $1}' | xargs -r kill
        echo "$(date +%H:%M) deadline: stopped everything still running"
        break
    fi
    if ! ps -eo args | grep -q "[l]aunch_rest.sh" \
       && ! ps -eo args | grep "limb_baseline.dream_pipeline" | grep -v grep | grep -q "data/dream/rerun"; then
        echo "$(date +%H:%M) all runs finished"; break
    fi
    sleep 30
done
sleep 5
cd ../../..
export PATH="$HOME/anaconda3/envs/kinder/bin:$PATH"
python scripts/make_dream_figures.py --csv $(ls data/dream/rerun/*_p*.csv | grep -v _dreams) \
    --out figures/dream/rerun/summary.png --per-person --max-episodes 10 \
    > figures/dream/rerun/summary.txt 2>&1
echo "$(date +%H:%M) final figures written: figures/dream/rerun/summary*.png"
