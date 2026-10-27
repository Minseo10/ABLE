#!/usr/bin/env bash
# Source after setting repo. Prefer this account's environment, never another user's.
if [[ -z ${SPROUT_PYTHON:-} ]]; then
  if [[ -x "$repo/.venv/bin/python" ]]; then
    export SPROUT_PYTHON="$repo/.venv/bin/python"
  elif [[ -x "${SPROUT_TIDYBOT_REPO:-$repo/../prpl-tidybot}/.venv/bin/python" ]]; then
    export SPROUT_PYTHON="${SPROUT_TIDYBOT_REPO:-$repo/../prpl-tidybot}/.venv/bin/python"
  else
    export SPROUT_PYTHON=python3
  fi
fi
