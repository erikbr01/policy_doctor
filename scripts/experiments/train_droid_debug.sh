#!/usr/bin/env bash
# Debug training run for DROID real-robot data with image observations.
#
# Prerequisites:
#   1. Run convert_droid_hdf5_debug.py to produce the HDF5 dataset
#   2. ZED SDK extracted at /mnt/ssdB/erik/zed_sdk_extracted/
#      (only needed if re-running the converter; training itself does not use pyzed)
#
# Training does NOT need LD_LIBRARY_PATH — pyzed is only used by the converter.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CUPID_DIR="${REPO_ROOT}/third_party/cupid"
CONFIG_PATH="${CUPID_DIR}/configs/image/droid_debug/diffusion_policy_cnn"

cd "${CUPID_DIR}"
conda run -n cupid_torch2 python train.py \
    --config-path="${CONFIG_PATH}" \
    --config-name=config \
    "$@"
