#!/usr/bin/env bash
# Exit when every 0-dream run has 20 episodes, or a log shows a traceback.
cd "$(dirname "$0")"
while true; do
  if grep -l "Traceback" logs/*.log 2>/dev/null | head -1 | grep -q .; then echo "CRASH:"; grep -l Traceback logs/*.log; exit 0; fi
  done_n=0
  for p in 23 24 25 26 29; do for c in plain_0 net_0 mlp_0; do
    f=${c}_p$p.csv
    [ -f "$f" ] && [ "$(python3 -c "import csv;r=list(csv.DictReader(open('$f')));print(len({x['episode'] for x in r}))")" -ge 20 ] && done_n=$((done_n+1))
  done; done
  [ "$done_n" -ge 15 ] && { echo "ZERO-DREAM DONE"; exit 0; }
  sleep 120
done
