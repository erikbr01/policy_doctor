#!/usr/bin/env bash
# MimicGen-merged square HDF5 profile (data_source=mimicgen_square).
# Train/attribution: conda env cupid. Datagen: conda env mimicgen (run MimicGen outside this script).
# Example dry clustering:
#   ./scripts/experiments/run_mimicgen_square.sh cupid steps=[run_clustering] dry_run=true run_name=mg_square_smoke
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/experiments/_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

ENV_NAME="${1:-cupid}"
if [[ $# -ge 1 ]]; then shift; fi
run_pipeline "$ENV_NAME" data_source=mimicgen_square "$@"
