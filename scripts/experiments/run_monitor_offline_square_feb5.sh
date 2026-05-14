#!/usr/bin/env bash
# Classify saved rollout episodes from square_mh_feb5 offline (no sim required).
# Use this to verify the monitoring/viz pipeline without needing a working sim env.
#
# Usage:
#   ./scripts/experiments/run_monitor_offline_square_feb5.sh [options]
#
# Options:
#   --episode   PKL    Specific episode pkl (default: ep0000_succ.pkl)
#   --device    DEV    PyTorch device [default: mps]
#   --output    CSV    Save assignments CSV [default: /tmp/offline_monitor.csv]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/experiments/_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

DEVICE="mps"
OUTPUT="/tmp/offline_monitor.csv"
EPISODE=""  # defaults to first succ episode below

while [[ $# -gt 0 ]]; do
    case "$1" in
        --episode) EPISODE="$2";  shift 2 ;;
        --device)  DEVICE="$2";   shift 2 ;;
        --output)  OUTPUT="$2";   shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

EXPERIMENT="feb5"
SEED=0
MODEL_NAME="feb5_train_diffusion_unet_lowdim_square_mh_${SEED}"
CKPT_DIR="$CUPID_DATA_ROOT/data/outputs/train/${EXPERIMENT}/${MODEL_NAME}/checkpoints"
TRAK_DIR="$CUPID_DATA_ROOT/data/outputs/eval_save_episodes/${EXPERIMENT}/${MODEL_NAME}/latest/default_trak_results-proj_dim=4096-lambda_reg=0.0-num_ckpts=1-seed=0-loss_fn=square-num_timesteps=64"
EPISODES_DIR="$CUPID_DATA_ROOT/data/outputs/eval_save_episodes/${EXPERIMENT}/${MODEL_NAME}/latest/episodes"
CLUSTERING_DIR="$CUPID_DATA_ROOT/influence_visualizer/configs/square_mh_feb5/clustering/sliding_window_rollout_kmeans_k15_2026_03_11"

if [[ -z "$EPISODE" ]]; then
    EPISODE="$(ls "$EPISODES_DIR"/ep*_succ.pkl 2>/dev/null | head -1)"
    if [[ -z "$EPISODE" ]]; then
        EPISODE="$(ls "$EPISODES_DIR"/ep*.pkl | head -1)"
    fi
fi

# Resolve best checkpoint
CKPT=$(ls "$CKPT_DIR"/epoch=*test_mean_score=*.ckpt 2>/dev/null \
    | sort -t= -k3 -rn | head -1)
[[ -z "$CKPT" ]] && CKPT="$CKPT_DIR/latest.ckpt"

echo "=== offline monitor (no sim) ==="
echo "  episode   : $EPISODE"
echo "  checkpoint: $CKPT"
echo "  device    : $DEVICE"
echo "  output    : $OUTPUT"
echo ""

cd "$CUPID_DATA_ROOT"
PYTHONPATH="$POLICY_DOCTOR_ROOT:$CUPID_CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
exec conda run -n cupid --no-capture-output \
    python "$POLICY_DOCTOR_ROOT/scripts/monitor_offline.py" \
        --episode "$EPISODE" \
        --checkpoint "$CKPT" \
        --infembed_fit "$TRAK_DIR/infembed_fit.pt" \
        --infembed_npz "$TRAK_DIR/infembed_embeddings.npz" \
        --clustering_dir "$CLUSTERING_DIR" \
        --episodes_dir "$EPISODES_DIR" \
        --device "$DEVICE" \
        --output "$OUTPUT"
