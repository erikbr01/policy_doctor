#!/usr/bin/env bash
# Train diffusion policy (CNN, low-dim) on MimicGen Square D1.
# Conda env: cupid  |  Robosuite: 1.2.0 (pinned by mimicgen)  |  Runner: MimicgenLowdimRunner
#
# MimicGen uses a different action layout (7-DoF) from the cupid/robomimic square MH (10-DoF),
# so it uses a separate config and dataset adapter.
#
# Usage:
#   ./scripts/experiments/train_mimicgen_square.sh [extra hydra overrides...]
#
# Examples:
#   # default full run
#   ./scripts/experiments/train_mimicgen_square.sh
#
#   # custom HDF5 (e.g. a larger generated set)
#   ./scripts/experiments/train_mimicgen_square.sh \
#     "task.dataset.dataset_path=/path/to/my_square.hdf5"
#
# Key overridable Hydra keys:
#   task.dataset.dataset_path=<path>     MimicGen HDF5 dataset
#   task.env_runner.dataset_path=<path>  same HDF5 for eval rollouts
#   training.device=cuda:0
#   training.num_epochs=800
#   multi_run.run_dir=<output_dir>
set -euo pipefail

# ---------------------------------------------------------------------------
# Optional acceleration flags (translated to Hydra overrides):
#   --compile    enable torch.compile  (+training.compile=true)
#   --tf32       enable TF32 matmul    (+training.tf32=true)
#   --no-compile disable torch.compile (+training.compile=false)
#   --no-tf32    disable TF32          (+training.tf32=false)
# Any remaining arguments are forwarded verbatim to train.py as Hydra overrides.
# ---------------------------------------------------------------------------
EXTRA_OVERRIDES=()
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --compile)    EXTRA_OVERRIDES+=("+training.compile=true");  shift ;;
    --no-compile) EXTRA_OVERRIDES+=("+training.compile=false"); shift ;;
    --tf32)       EXTRA_OVERRIDES+=("+training.tf32=true");     shift ;;
    --no-tf32)    EXTRA_OVERRIDES+=("+training.tf32=false");    shift ;;
    *)            PASSTHROUGH+=("$1"); shift ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CUPID_ROOT="${REPO_ROOT}/third_party/cupid"

# Config path is relative to train.py (in CUPID_ROOT)
CONFIG_PATH="configs/low_dim/square_mimicgen_lowdim/diffusion_policy_cnn"

# Absolute path to the source HDF5 (MimicGen Square D1 generated demos)
HDF5="${REPO_ROOT}/data/source/mimicgen/core_datasets/square/demo_src_square_task_D1/demo.hdf5"

if [[ ! -f "$HDF5" ]]; then
  echo "ERROR: dataset not found at $HDF5" >&2
  exit 1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_MODE="${WANDB_MODE:-online}"

echo "=== train_mimicgen_square: cupid env, MimicGen Square D1 low-dim ==="
echo "    HDF5:   $HDF5"
echo "    config: ${CUPID_ROOT}/${CONFIG_PATH}/config.yaml"
echo ""

cd "$CUPID_ROOT"
exec conda run -n cupid_torch2 --no-capture-output \
  python train.py \
    --config-path "$CONFIG_PATH" \
    --config-name config \
    "task.dataset.dataset_path=${HDF5}" \
    "task.dataset_path=${HDF5}" \
    "task.env_runner.dataset_path=${HDF5}" \
    "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}" \
    "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"
