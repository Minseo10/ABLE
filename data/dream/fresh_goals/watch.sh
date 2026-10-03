#!/usr/bin/env bash
# Exit once all 15 runs have >= $1 real episodes, or on a traceback.
cd "$(dirname "$0")"
need="${1:-20}"
while true; do
  if grep -l "Traceback" logs/*.log 2>/dev/null | grep -q .; then echo CRASH; grep -l Traceback logs/*.log; exit 0; fi
  n=0
  for m in net_4_step cache_4_step plain_0; do for p in 23 24 25 26 29; do
    f=${m}_p$p.csv
    [ -f "$f" ] && [ "$(python3 -c "import csv;print(len({r['episode'] for r in csv.DictReader(open('$f'))}))")" -ge "$need" ] && n=$((n+1))
  done; done
  [ "$n" -ge 15 ] && { echo "all 15 runs >= $need episodes"; exit 0; }
  sleep 120
done
