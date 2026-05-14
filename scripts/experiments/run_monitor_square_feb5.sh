#!/usr/bin/env bash
# Run monitor_online.py on square_mh_feb5 (no interventions, viz pipeline test).
# All paths resolve from CUPID_DATA_ROOT (default: sibling cupid/ next to policy_doctor).
#
# Usage:
#   ./scripts/experiments/run_monitor_square_feb5.sh [options]
#
# Options:
#   --ckpt       CKPT       Checkpoint: 'best', 'latest', or epoch int  [default: best]
#   --clustering SLUG       Clustering subdirectory slug                 [default: kmeans_k15]
#   --episodes   N          Number of test episodes                      [default: 5]
#   --output     DIR        Output directory                             [default: /tmp/monitor_square_feb5]
#   --device     DEVICE     PyTorch device                               [default: cpu]
#   --overwrite             Delete and recreate output_dir if it exists
#
# Environment override:
#   CUPID_DATA_ROOT=/your/path/to/cupid  # if your layout is not the default sibling

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/experiments/_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

# ── Defaults ──────────────────────────────────────────────────────────────────
CKPT="best"
CLUSTERING_SLUG="sliding_window_rollout_kmeans_k15_2026_03_11"
NUM_EPISODES=5
OUTPUT_DIR="/tmp/monitor_square_feb5"
DEVICE="mps"
OVERWRITE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)        CKPT="$2";             shift 2 ;;
        --clustering)  CLUSTERING_SLUG="$2";  shift 2 ;;
        --episodes)    NUM_EPISODES="$2";     shift 2 ;;
        --output)      OUTPUT_DIR="$2";       shift 2 ;;
        --device)      DEVICE="$2";           shift 2 ;;
        --overwrite)   OVERWRITE=true;        shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
EXPERIMENT="feb5"
SEED=0
MODEL_NAME="feb5_train_diffusion_unet_lowdim_square_mh_${SEED}"

TRAIN_DIR="$CUPID_DATA_ROOT/data/outputs/train/${EXPERIMENT}/${MODEL_NAME}"

TRAK_DIR="$CUPID_DATA_ROOT/data/outputs/eval_save_episodes/${EXPERIMENT}/${MODEL_NAME}/latest/default_trak_results-proj_dim=4096-lambda_reg=0.0-num_ckpts=1-seed=0-loss_fn=square-num_timesteps=64"

CLUSTERING_DIR="$CUPID_DATA_ROOT/influence_visualizer/configs/square_mh_feb5/clustering/${CLUSTERING_SLUG}"

EPISODES_DIR="$CUPID_DATA_ROOT/data/outputs/eval_save_episodes/${EXPERIMENT}/${MODEL_NAME}/latest/episodes"

if "$OVERWRITE" && [[ -d "$OUTPUT_DIR" ]]; then
    rm -rf "$OUTPUT_DIR"
fi

echo "=== monitor_online (no interventions) ==="
echo "  cupid data root : $CUPID_DATA_ROOT"
echo "  train dir       : $TRAIN_DIR"
echo "  checkpoint      : $CKPT"
echo "  clustering      : $CLUSTERING_SLUG"
echo "  episodes        : $NUM_EPISODES"
echo "  output          : $OUTPUT_DIR"
echo "  device          : $DEVICE"
echo ""

run_monitor \
    "$OUTPUT_DIR" \
    "$TRAIN_DIR" \
    "$CKPT" \
    "$TRAK_DIR/infembed_fit.pt" \
    "$TRAK_DIR/infembed_embeddings.npz" \
    "$CLUSTERING_DIR" \
    "$NUM_EPISODES" \
    "$DEVICE" \
    "$EPISODES_DIR"
