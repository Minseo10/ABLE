#!/usr/bin/env bash
# Exit once every *_0_step run has >= $1 episodes, or on a traceback.
cd "$(dirname "$0")"
need="${1:-2}"
while true; do
  if grep -l "Traceback" logs/*_step_*.log 2>/dev/null | grep -q .; then echo CRASH; grep -l Traceback logs/*_step_*.log; exit 0; fi
  n=0
  for f in net_0_step_p*.csv mlp_0_step_p*.csv; do
    [ -f "$f" ] && [ "$(python3 -c "import csv;print(len({r['episode'] for r in csv.DictReader(open('$f'))}))")" -ge "$need" ] && n=$((n+1))
  done
  [ "$n" -ge 10 ] && { echo "ALL >= $need EPISODES"; exit 0; }
  sleep 60
done
