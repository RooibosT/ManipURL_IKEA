#!/usr/bin/env bash
# Run the whole loop against the mocks, with the cameras this submission
# actually declares.
#
# `conformance.py` deliberately runs `mock_orin --no-wrists`, which publishes
# only the mono `ego_view` -- enough to validate the action contract, but none
# of our three declared keys. This script fills that gap: --stereo-ego makes the
# mock publish ego_view_left/right as well, so the server sees the same keys it
# will see on the robot.
#
#   scripts/dev_stack.sh                        # hold-still policy
#   scripts/dev_stack.sh --checkpoint /weights/... --device cuda
#
# Anything passed through goes to the server.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
LANE=decoupled
EXPECT="${EXPECT:-40}"
pids=()
cleanup() { for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

echo "--- mock orin (stereo ego + both wrists) ---"
$PYTHON -u mocks/mock_orin.py --stereo-ego --fps 30 &
pids+=($!)
sleep 2

echo "--- policy server ---"
$PYTHON -u components/server.py --lane "$LANE" --port 8765 "$@" &
pids+=($!)
sleep 3

echo "--- policy client ---"
$PYTHON -u components/client.py --lane "$LANE" --thor 127.0.0.1 --orin 127.0.0.1 &
pids+=($!)
sleep 2

echo "--- mock whole-body controller ---"
$PYTHON -u mocks/mock_wbc.py --lane "$LANE" --expect "$EXPECT" --timeout-s 20
