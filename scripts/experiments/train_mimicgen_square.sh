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
# Optional flags:
#   --compile      enable torch.compile  (+training.compile=true)
#   --no-compile   disable torch.compile (+training.compile=false)
#   --tf32         enable TF32 matmul    (+training.tf32=true)
#   --no-tf32      disable TF32          (+training.tf32=false)
#   --num-gpus N   use N GPUs via torchrun (default: 1, uses plain python)
# Any remaining arguments are forwarded verbatim to train.py as Hydra overrides.
# ---------------------------------------------------------------------------
USE_COMPILE=true
USE_TF32=true
PASSTHROUGH=()
NUM_GPUS=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --compile)    USE_COMPILE=true;  shift ;;
    --no-compile) USE_COMPILE=false; shift ;;
    --tf32)       USE_TF32=true;     shift ;;
    --no-tf32)    USE_TF32=false;    shift ;;
    --num-gpus)   NUM_GPUS="$2"; shift 2 ;;
    *)            PASSTHROUGH+=("$1"); shift ;;
  esac
done

EXTRA_OVERRIDES=(
  "+training.compile=${USE_COMPILE}"
  "+training.tf32=${USE_TF32}"
)

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

TRAIN_ARGS=(
  --config-path "$CONFIG_PATH"
  --config-name config
  "task.dataset.dataset_path=${HDF5}"
  "task.dataset_path=${HDF5}"
  "task.env_runner.dataset_path=${HDF5}"
  "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"
  "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"
)

if [[ "$NUM_GPUS" -gt 1 ]]; then
  echo "    gpus:   ${NUM_GPUS} (torchrun)"
  exec conda run -n mimicgen_torch2 --no-capture-output \
    torchrun --nproc_per_node="$NUM_GPUS" train.py "${TRAIN_ARGS[@]}"
else
  exec conda run -n mimicgen_torch2 --no-capture-output \
    python train.py "${TRAIN_ARGS[@]}"
fi
