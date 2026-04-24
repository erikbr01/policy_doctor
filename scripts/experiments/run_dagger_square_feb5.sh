#!/usr/bin/env bash
# Run DAgger interactive rollouts on square_mh_feb5.
# All paths resolve from CUPID_DATA_ROOT (default: sibling cupid/ next to policy_doctor).
#
# Usage:
#   ./scripts/experiments/run_dagger_square_feb5.sh [options]
#
# Options:
#   --ckpt         CKPT    Checkpoint: 'best', 'latest', or epoch int   [default: best]
#   --clustering   SLUG    Clustering subdirectory slug                  [default: kmeans_k15]
#   --episodes     N       Number of DAgger episodes                     [default: 5]
#   --output       DIR     Output directory for episode pkls             [default: /tmp/dagger_square_feb5]
#   --device       DEVICE  PyTorch device                                [default: cpu]
#   --dagger-config NAME   DAgger config preset (keyboard_default, spacemouse_default)
#                                                                        [default: keyboard_default]
#   --no-viz               Disable live visualization
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
OUTPUT_DIR="/tmp/dagger_square_feb5"
DEVICE="mps"
DAGGER_CONFIG="keyboard_default"
NO_VIZ=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)           CKPT="$2";            shift 2 ;;
        --clustering)     CLUSTERING_SLUG="$2"; shift 2 ;;
        --episodes)       NUM_EPISODES="$2";    shift 2 ;;
        --output)         OUTPUT_DIR="$2";      shift 2 ;;
        --device)         DEVICE="$2";          shift 2 ;;
        --dagger-config)  DAGGER_CONFIG="$2";   shift 2 ;;
        --no-viz)         NO_VIZ=true;          shift   ;;
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

mkdir -p "$OUTPUT_DIR"

echo "=== DAgger interactive rollout ==="
echo "  cupid data root : $CUPID_DATA_ROOT"
echo "  train dir       : $TRAIN_DIR"
echo "  checkpoint      : $CKPT"
echo "  clustering      : $CLUSTERING_SLUG"
echo "  episodes        : $NUM_EPISODES"
echo "  output          : $OUTPUT_DIR"
echo "  device          : $DEVICE"
echo "  dagger config   : $DAGGER_CONFIG"
echo ""

EXTRA_ARGS=()
"$NO_VIZ" && EXTRA_ARGS+=(--no_visualization)

# Use direct conda activation for DAgger visualization (matplotlib needs macOS GUI thread).
# conda run spawns a subprocess that breaks NSApp on macOS.
cd "$CUPID_DATA_ROOT"
export PYTHONPATH="$POLICY_DOCTOR_ROOT:$CUPID_CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
CONDA_PYTHON="$(conda run -n cupid which python 2>/dev/null || echo python)"
exec "$CONDA_PYTHON" "$POLICY_DOCTOR_ROOT/scripts/run_dagger.py" \
      --task square_mh \
      --output_dir "$OUTPUT_DIR" \
      --train_dir "$TRAIN_DIR" \
      --train_ckpt "$CKPT" \
      --infembed_fit "$TRAK_DIR/infembed_fit.pt" \
      --infembed_npz "$TRAK_DIR/infembed_embeddings.npz" \
      --clustering_dir "$CLUSTERING_DIR" \
      --num_episodes "$NUM_EPISODES" \
      --device "$DEVICE" \
      --dagger_config "$DAGGER_CONFIG" \
      --episodes_dir "$CUPID_DATA_ROOT/data/outputs/eval_save_episodes/feb5/feb5_train_diffusion_unet_lowdim_square_mh_0/latest/episodes" \
      "${EXTRA_ARGS[@]}"
