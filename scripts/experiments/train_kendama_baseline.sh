#!/usr/bin/env bash
# Baseline training run for kendama real-robot task (may13 dataset).
#
# Prerequisites:
#   Run scripts/convert_droid_hdf5_debug.py to produce data/source/droid/kendama_may13.hdf5.
#
# Usage:
#   ./scripts/train_kendama_baseline.sh [extra hydra overrides...]
#   ./scripts/train_kendama_baseline.sh training.device=cuda:1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CUPID_DIR="${REPO_ROOT}/third_party/cupid"
CONFIG_PATH="${CUPID_DIR}/configs/image/droid/diffusion_policy_cnn"
HDF5="/mnt/ssdB/erik/droid_data/kendama_may13.hdf5"

if [[ ! -f "$HDF5" ]]; then
  echo "ERROR: dataset not found at $HDF5" >&2
  echo "  Run scripts/convert_droid_hdf5_debug.py first." >&2
  exit 1
fi

echo "=== train_kendama_baseline: cupid_torch2 env, DROID image ==="
echo "    HDF5:   $HDF5"
echo "    config: ${CONFIG_PATH}/config.yaml"
echo ""

cd "${CUPID_DIR}"
exec conda run -n cupid_torch2 --no-capture-output \
  python train.py \
    --config-path "${CONFIG_PATH}" \
    --config-name config \
    "++task.dataset.dataset_path=${HDF5}" \
    "++task.env_runner.dataset_path=${HDF5}" \
    "+training.tf32=true" \
    "$@"
