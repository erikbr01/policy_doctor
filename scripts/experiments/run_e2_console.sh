#!/usr/bin/env bash
# Streamlit operator console for E2. Talks to the proposal_server via HTTP.
#
# Usage:
#   ./scripts/experiments/run_e2_console.sh [--port 8501]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

PORT=8501
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== E2 Operator Console (Streamlit) ==="
echo "  url : http://localhost:$PORT"

cd "$POLICY_DOCTOR_ROOT"
export PYTHONPATH="$POLICY_DOCTOR_ROOT:$CUPID_CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec conda run -n policy_doctor --no-capture-output \
    streamlit run policy_doctor/streamlit_app/app.py \
        --server.port "$PORT"
