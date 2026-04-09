#!/usr/bin/env bash
# Full pipeline (or subset) with default data_source=cupid_robomimic (transport MH + cupid).
# Usage:
#   ./scripts/experiments/run_cupid_robomimic_transport.sh cupid 'steps=[run_clustering]' dry_run=true
# Env (positional): cupid | policy_doctor — use cupid for train/eval/TRAK; policy_doctor for clustering-only.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/experiments/_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

ENV_NAME="${1:-cupid}"
if [[ $# -ge 1 ]]; then shift; fi
run_pipeline "$ENV_NAME" data_source=cupid_robomimic "$@"
