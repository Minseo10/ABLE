#!/usr/bin/env bash
# Workstation preview; never deploys code or starts robot servers.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$repo/real_world/scripts/sprout_python.sh"
tidy_repo=${SPROUT_TIDYBOT_REPO:-"$repo/../prpl-tidybot"}
python_bin=${SPROUT_PYTHON:-python3}
export PYTHONPATH="$repo/real_world/src:$repo/src:$tidy_repo/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
exec "$python_bin" -u -m sprout_experiment.rehearsal "$@"
