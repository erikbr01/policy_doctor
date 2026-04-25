#!/usr/bin/env bash
# Start the DAgger visualization server (separate process, own cv2 window).
# Run this BEFORE starting the DAgger runner.
#
# Usage:
#   ./scripts/experiments/run_viz_server.sh [--port 5002] [--fps 30]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

PORT=5002
FPS=30
DEVICE="auto"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)   PORT="$2";   shift 2 ;;
        --fps)    FPS="$2";    shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== DAgger Viz Server ==="
echo "  url : http://127.0.0.1:$PORT"
echo "  fps : $FPS"
echo ""
echo "Then in another terminal:"
echo "  ./scripts/experiments/run_dagger_square_feb5.sh --viz-url http://127.0.0.1:$PORT ..."
echo ""

CONDA_PYTHON="$(conda run -n cupid which python 2>/dev/null || echo python)"
PYTHONPATH="$POLICY_DOCTOR_ROOT:$CUPID_CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
exec "$CONDA_PYTHON" \
    -m policy_doctor.envs.viz_server \
    --port "$PORT" \
    --fps "$FPS" \
    --device "$DEVICE"
