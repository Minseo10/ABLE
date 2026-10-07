#!/usr/bin/env bash
# Draw interim figures every 30 minutes while any rerun process is alive, detached from
# the editor session. Log: data/dream/rerun/interim_loop.log
cd "$(dirname "$0")"
while ps -eo args | grep -q "[r]un_dream_experiments.sh"; do
    sleep 1800
    ./interim.sh
done
echo "$(date +%H:%M) rerun finished; interim loop stopped"
