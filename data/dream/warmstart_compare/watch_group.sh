#!/usr/bin/env bash
# Exit once the 5 runs of group $1 (cache, net, mlp) have >= $2 real episodes, or on a traceback.
cd "$(dirname "$0")"
group="$1"; need="${2:-20}"
while true; do
  if grep -l "Traceback" logs/${group}_4_step_*.log 2>/dev/null | grep -q .; then echo "CRASH in $group"; grep -l Traceback logs/${group}_4_step_*.log; exit 0; fi
  n=0
  for p in 23 24 25 26 29; do
    f=${group}_4_step_p$p.csv
    [ -f "$f" ] && [ "$(python3 -c "import csv;print(len({r['episode'] for r in csv.DictReader(open('$f'))}))")" -ge "$need" ] && n=$((n+1))
  done
  [ "$n" -ge 5 ] && { echo "$group: all 5 runs >= $need episodes"; exit 0; }
  sleep 120
done
