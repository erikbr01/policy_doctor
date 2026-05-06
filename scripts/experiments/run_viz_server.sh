#!/usr/bin/env bash
# Start the DAgger visualization server (separate process, own cv2 window).
# Run this BEFORE starting the DAgger runner.
#
# Usage:
#   ./scripts/experiments/run_viz_server.sh [--device spacemouse] [--spacemouse-vid 0x256f --spacemouse-pid 0xc62f] ...

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

PORT=5002
FPS=30
DEVICE="spacemouse"
DAGGER_CONFIG="${DAGGER_CONFIG:-spacemouse_default}"
TASK=""
SM_VID=""
SM_PID=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)   PORT="$2";   shift 2 ;;
        --fps)    FPS="$2";    shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --dagger-config) DAGGER_CONFIG="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --spacemouse-vid) SM_VID="$2"; shift 2 ;;
        --spacemouse-pid) SM_PID="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== DAgger Viz Server ==="
echo "  url : http://127.0.0.1:$PORT"
echo "  fps : $FPS"
echo "  input: $DEVICE"
echo "  dagger_config: $DAGGER_CONFIG (device bindings / SpaceMouse spatial_mapping when applicable)"
if [[ -n "$TASK" ]]; then echo "  task overlay   : $TASK (merge pygame/spacemouse from data_collection/tasks)"; fi
if [[ -n "$SM_VID$SM_PID" ]]; then
    if [[ -z "$SM_VID" || -z "$SM_PID" ]]; then echo "Need both --spacemouse-vid and --spacemouse-pid"; exit 1; fi
    echo "  spacemouse usb : vid=$SM_VID pid=$SM_PID"
fi
echo ""
echo "Then in another terminal:"
echo "  conda run -n cupid python scripts/run_dagger.py"
echo ""

CONDA_PYTHON="$(conda run -n cupid which python 2>/dev/null || echo python)"
PYTHONPATH="$POLICY_DOCTOR_ROOT:$CUPID_CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
VIZ_TASK_ARGS=()
if [[ -n "$TASK" ]]; then VIZ_TASK_ARGS+=(--task "$TASK"); fi
if [[ -n "$SM_VID" ]]; then VIZ_TASK_ARGS+=(--spacemouse-vid "$SM_VID" --spacemouse-pid "$SM_PID"); fi
exec "$CONDA_PYTHON" \
    -m policy_doctor.envs.viz_server \
    --port "$PORT" \
    --fps "$FPS" \
    --device "$DEVICE" \
    --dagger-config "$DAGGER_CONFIG" \
    "${VIZ_TASK_ARGS[@]}"
