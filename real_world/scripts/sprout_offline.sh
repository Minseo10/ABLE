#!/usr/bin/env bash
# Offline only. Does not run launch.sh, run_remote.sh, git sync, or any robot server.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$repo/real_world/scripts/sprout_python.sh"
tidy_repo=${SPROUT_TIDYBOT_REPO:-"$repo/../prpl-tidybot"}
python_bin=${SPROUT_PYTHON:-python3}
auth_file="$repo/.local/sprout/authkey"
export PYTHONPATH="$repo/real_world/src:$repo/src:$tidy_repo/src${PYTHONPATH:+:$PYTHONPATH}"
command=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
case "$command" in
  init)
    "$python_bin" - "$auth_file" <<'PY'
import os
import secrets
import sys
from pathlib import Path
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
try:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
except FileExistsError:
    print("Existing offline authkey retained.")
else:
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(secrets.token_bytes(32))
    print("Created private offline authkey; its contents are not printed.")
PY
    ;;
  server)
    exec "$python_bin" -m prpl_tidybot.sprout.offline_server --authkey-file "$auth_file" "$@"
    ;;
  model-info)
    exec "$python_bin" -m sprout_experiment.model "$@"
    ;;
  cycle|status|stop|watch-stop)
    exec "$python_bin" -m sprout_experiment.client "$command" --authkey-file "$auth_file" "$@"
    ;;
  *)
    echo 'Usage: bash real_world/scripts/sprout_offline.sh {init|server|cycle|status|stop|watch-stop|model-info}'
    echo 'Synthetic offline rehearsal only. No hardware execution command exists.'
    exit 2
    ;;
esac
