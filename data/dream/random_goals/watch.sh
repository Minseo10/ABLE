#!/usr/bin/env bash
# watch.sh <episodes> <tag>...: exit once every <tag>_p* run has that many real episodes,
# or a log of those tags shows a traceback.
cd "$(dirname "$0")"
need="$1"; shift
while true; do
  for t in "$@"; do
    if grep -l "Traceback" logs/${t}_p*.log 2>/dev/null | grep -q .; then echo "CRASH in $t"; grep -l Traceback logs/${t}_p*.log; exit 0; fi
  done
  n=0; total=0
  for t in "$@"; do for p in 23 24 25 26 29; do
    total=$((total+1)); f=${t}_p$p.csv
    [ -f "$f" ] && [ "$(python3 -c "import csv;print(len({r['episode'] for r in csv.DictReader(open('$f'))}))")" -ge "$need" ] && n=$((n+1))
  done; done
  [ "$n" -ge "$total" ] && { echo "all $total runs >= $need episodes"; exit 0; }
  sleep 120
done
