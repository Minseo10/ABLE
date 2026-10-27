#!/usr/bin/env bash
# Workstation shortcuts. No hardware activation or real execution.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$repo/real_world/scripts/sprout_python.sh"
cd "$repo"
run() { bash "$repo/real_world/scripts/sprout_rehearsal.sh" "$@"; }
action=${1:-preview}
if [[ $# -gt 0 ]]; then shift; fi
case "$action" in
  package) run setup; "${SPROUT_PYTHON:-python3}" real_world/scripts/sprout_package.py ;;
  setup|check) run "$action" "$@" ;;
  capture)
    run setup
    run capture --host 127.0.0.1 --port 16003 --output .local/sprout/current.jpg "$@"
    ;;
  init) run initialize --selection .local/sprout/reference-selection.json ;;
  preview)
    run check
    run detect --config .local/sprout/detection.json --image .local/sprout/current.jpg \
      --output .local/sprout/detection
    run preview --config .local/sprout/scene.json --robot-state .local/sprout/robot-state.json \
      --observation .local/sprout/detection/observation.json --live-config .local/sprout/live.json \
      --output .local/sprout/camera-preview "$@"
    ;;
  demo)
    run setup
    model=$("$SPROUT_PYTHON" -c 'import json; print(json.load(open(".local/sprout/scene.json"))["tidybot_model"])')
    run demo --tidybot-model "$model" \
      --sprout-model private_sprout/v0.1.4/sprout.xml --lift-height 0.03 --output .local/sprout/rehearsal "$@"
    ;;
  *) echo 'Usage: bash real_world/scripts/sprout_sim.sh [setup|check|capture|init|preview|demo] [extra CLI args]' >&2; exit 2 ;;
esac
