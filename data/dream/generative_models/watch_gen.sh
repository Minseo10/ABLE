#!/usr/bin/env bash
# Exit once the 15 k8 runs have >= $1 real episodes, or on a traceback in their logs.
cd "$(dirname "$0")"
need="${1:-20}"
while true; do
  if grep -l "Traceback" logs/*_k8_*.log 2>/dev/null | grep -q .; then echo CRASH; grep -l Traceback logs/*_k8_*.log; exit 0; fi
  n=0
  for t in gen_flow_4_step_k8 gen_ddpm_4_step_k8 net_4_step_k8; do for p in 23 24 25 26 29; do
    f=${t}_p$p.csv
    [ -f "$f" ] && [ "$(python3 -c "import csv;print(len({r['episode'] for r in csv.DictReader(open('$f'))}))")" -ge "$need" ] && n=$((n+1))
  done; done
  [ "$n" -ge 15 ] && { echo "all 15 runs >= $need episodes"; exit 0; }
  sleep 120
done
