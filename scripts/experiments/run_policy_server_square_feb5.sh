#!/usr/bin/env bash
# Start the HTTP policy server for square_mh_feb5.
# Run this in a separate terminal BEFORE starting the DAgger runner.
#
# Usage:
#   ./scripts/experiments/run_policy_server_square_feb5.sh [--device mps|cpu] [--port 5001]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

DEVICE="mps"
PORT=5001

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --port)   PORT="$2";   shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

EXPERIMENT="feb5"
SEED=0
MODEL_NAME="feb5_train_diffusion_unet_lowdim_square_mh_${SEED}"
CKPT_DIR="$CUPID_DATA_ROOT/data/outputs/train/${EXPERIMENT}/${MODEL_NAME}/checkpoints"

# Find best checkpoint
CKPT=$(ls "$CKPT_DIR"/epoch=*test_mean_score=*.ckpt 2>/dev/null \
    | sort -t= -k3 -rn | head -1)
[[ -z "$CKPT" ]] && CKPT="$CKPT_DIR/latest.ckpt"

echo "=== Policy Server (square_mh_feb5) ==="
echo "  checkpoint : $CKPT"
echo "  device     : $DEVICE"
echo "  url        : http://127.0.0.1:$PORT"
echo ""
echo "Leave this running. In another terminal:"
echo "  ./scripts/experiments/run_dagger_square_feb5.sh --no-monitor --server-url http://127.0.0.1:$PORT"
echo ""

cd "$CUPID_DATA_ROOT"
CONDA_PYTHON="$(conda run -n cupid which python 2>/dev/null || echo python)"
PYTHONPATH="$POLICY_DOCTOR_ROOT:$CUPID_CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
exec "$CONDA_PYTHON" \
    -m policy_doctor.envs.policy_server \
    --checkpoint "$CKPT" \
    --device "$DEVICE" \
    --port "$PORT"
