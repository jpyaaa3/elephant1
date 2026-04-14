#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BRIDGE_CMD=(python3 ex/bridge.py)
SIM_CMD=(python3 ex/sim.py)
TRACKER_CMD=(python3 tracker.py --width 1920 --height 1080 --show low)
RECORDER_CMD=(python3 recorder.py)

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  jobs -pr | xargs -r kill
  wait || true
  exit "$code"
}

trap cleanup EXIT INT TERM

"${BRIDGE_CMD[@]}" &
"${SIM_CMD[@]}" &
"${TRACKER_CMD[@]}" &
"${RECORDER_CMD[@]}" &

echo "Launched:"
printf '  %q ' "${BRIDGE_CMD[@]}"; echo
printf '  %q ' "${SIM_CMD[@]}"; echo
printf '  %q ' "${TRACKER_CMD[@]}"; echo
printf '  %q ' "${RECORDER_CMD[@]}"; echo

wait
