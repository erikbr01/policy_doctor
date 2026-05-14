#!/usr/bin/env bash
# E2 sim runner: pulls DemonstrationRequests from the proposal server and
# executes them via the existing DAgger runner. Use the cupid env (sim stack).
#
# Usage:
#   ./scripts/experiments/run_e2_sim.sh \
#       task=square_mh \
#       proposal_server=http://127.0.0.1:5003 \
#       viz_url=http://127.0.0.1:5002 \
#       output_dir=/tmp/e2_demos \
#       train_dir=/path/to/train_dir

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

cd "$CUPID_CODE_DIR"
export PYTHONPATH="$POLICY_DOCTOR_ROOT:$CUPID_CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec conda run -n cupid --no-capture-output \
    python "$POLICY_DOCTOR_ROOT/scripts/run_e2_sim.py" "$@"
