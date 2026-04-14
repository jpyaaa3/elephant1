#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

UI_CMD=(python3 ex/robot_ui.py)
TRACKER_CMD=(python3 tracker.py --width 1920 --height 1080 --show low)

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  jobs -pr | xargs -r kill
  wait || true
  exit "$code"
}

trap cleanup EXIT INT TERM

"${UI_CMD[@]}" &
"${TRACKER_CMD[@]}" &

echo "Launched:"
printf '  %q ' "${UI_CMD[@]}"; echo
printf '  %q ' "${TRACKER_CMD[@]}"; echo

wait
