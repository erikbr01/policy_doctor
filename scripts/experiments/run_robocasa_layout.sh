#!/usr/bin/env bash
# RoboCasa demos in robomimic-layout HDF5 (data_source=robocasa_layout).
# Requires submodule: git submodule update --init third_party/robocasa
# Place merged HDF5 at data/robocasa/datasets/kitchen_lowdim_merged.hdf5 (or override baseline.diffusion_dataset_path).
# Example:
#   ./scripts/experiments/run_robocasa_layout.sh cupid steps=[train_baseline] dry_run=true
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/experiments/_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

ENV_NAME="${1:-cupid}"
if [[ $# -ge 1 ]]; then shift; fi
run_pipeline "$ENV_NAME" data_source=robocasa_layout "$@"
