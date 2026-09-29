#!/usr/bin/env bash
# Exit once every 4-dream step run (net, mlp, cache: 15) has >= $1 real episodes, or on a traceback.
cd "$(dirname "$0")"
need="${1:-20}"
while true; do
  if grep -l "Traceback" logs/*_4_step_*.log 2>/dev/null | grep -q .; then echo CRASH; grep -l Traceback logs/*_4_step_*.log; exit 0; fi
  n=0
  for m in net mlp cache; do for p in 23 24 25 26 29; do
    f=${m}_4_step_p$p.csv
    [ -f "$f" ] && [ "$(python3 -c "import csv;print(len({r['episode'] for r in csv.DictReader(open('$f'))}))")" -ge "$need" ] && n=$((n+1))
  done; done
  [ "$n" -ge 15 ] && { echo "ALL 15 RUNS >= $need EPISODES"; exit 0; }
  sleep 120
done
