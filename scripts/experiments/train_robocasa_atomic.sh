#!/usr/bin/env bash
# Train diffusion policy (transformer, image/hybrid) on RoboCasa atomic tasks.
# Conda env: robocasa  |  Robosuite: 1.5.2  |  Dataset: LeRobot v2 (no HDF5)
#            Runner: RobocasaImageRunner (eval via live robosuite env, no eval HDF5 needed)
#
# Usage:
#   ./scripts/experiments/train_robocasa_atomic.sh [TASK] [extra hydra overrides...]
#
# TASK is the robosuite/robocasa env name (default: PickPlaceCounterToCabinet).
# The lerobot dataset path is derived from TASK automatically if it follows the
# standard layout: data/source/robocasa/v1.0/target/atomic/<TASK>/<DATE>/lerobot
# You can override it explicitly via task.dataset.dataset_path=<path>.
#
# Examples:
#   # PickPlaceCounterToCabinet (default)
#   ./scripts/experiments/train_robocasa_atomic.sh
#
#   # Different task
#   ./scripts/experiments/train_robocasa_atomic.sh OpenCabinet
#
#   # Explicit dataset path + custom output dir
#   ./scripts/experiments/train_robocasa_atomic.sh PickPlaceCounterToCabinet \
#     "task.dataset.dataset_path=/path/to/lerobot" \
#     multi_run.run_dir=/data/outputs/my_run
#
# Key overridable Hydra keys:
#   task.dataset.dataset_path=<lerobot_root>  LeRobot dataset directory
#   task.env_runner.env_name=<env>            robocasa env name for eval rollouts
#   training.device=cuda:0
#   training.num_epochs=800
#   multi_run.run_dir=<output_dir>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CUPID_ROOT="${REPO_ROOT}/third_party/cupid"

# Config path is relative to train.py (in CUPID_ROOT)
CONFIG_PATH="configs/image/robocasa_lerobot_atomic/diffusion_policy_transformer"

TASK="${1:-PickPlaceCounterToCabinet}"
if [[ $# -ge 1 ]]; then shift; fi

ROBOCASA_DATA_ROOT="${REPO_ROOT}/data/source/robocasa/v1.0/target/atomic"

# Auto-discover the latest date-stamped lerobot directory for this task
TASK_DIR="${ROBOCASA_DATA_ROOT}/${TASK}"
if [[ ! -d "$TASK_DIR" ]]; then
  echo "ERROR: task directory not found: $TASK_DIR" >&2
  echo "       Available tasks:" >&2
  ls "$ROBOCASA_DATA_ROOT" 2>/dev/null | sed 's/^/         /' >&2
  exit 1
fi

# Pick the most recent date-stamped subdirectory
LEROBOT_ROOT="$(ls -d "${TASK_DIR}"/*/lerobot 2>/dev/null | sort | tail -1)"
if [[ -z "$LEROBOT_ROOT" ]]; then
  echo "ERROR: no lerobot dataset found under $TASK_DIR/<date>/lerobot" >&2
  exit 1
fi

if [[ ! -d "${LEROBOT_ROOT}/meta" ]]; then
  echo "ERROR: lerobot directory missing 'meta/' subfolder: $LEROBOT_ROOT" >&2
  exit 1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_MODE="${WANDB_MODE:-online}"

echo "=== train_robocasa_atomic: robocasa env, ${TASK} image/hybrid ==="
echo "    LeRobot: $LEROBOT_ROOT"
echo "    env:     $TASK"
echo "    config:  ${CUPID_ROOT}/${CONFIG_PATH}/config.yaml"
echo ""

cd "$CUPID_ROOT"
exec conda run -n robocasa --no-capture-output \
  python train.py \
    --config-path "$CONFIG_PATH" \
    --config-name config \
    "task.dataset.dataset_path=${LEROBOT_ROOT}" \
    "task.dataset_path=${LEROBOT_ROOT}" \
    "task.env_runner.env_name=${TASK}" \
    "$@"
