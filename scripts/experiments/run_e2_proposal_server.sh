#!/usr/bin/env bash
# Start the E2 proposal HTTP server (separate process; owns the Qwen3-VL model
# and the request queue). Run this BEFORE the sim runner / Streamlit console.
#
# Usage:
#   ./scripts/experiments/run_e2_proposal_server.sh \
#       --config policy_doctor/configs/e2/example_run.yaml \
#       --port 5003

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

PORT=5003
HOST=127.0.0.1
CONFIG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)   PORT="$2";   shift 2 ;;
        --host)   HOST="$2";   shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required (path to e2 config yaml)"
    exit 1
fi

echo "=== E2 Proposal Server ==="
echo "  url    : http://$HOST:$PORT"
echo "  config : $CONFIG"
echo ""
echo "Streamlit console:"
echo "  conda activate policy_doctor && streamlit run policy_doctor/streamlit_app/app.py"
echo ""

cd "$POLICY_DOCTOR_ROOT"
export PYTHONPATH="$POLICY_DOCTOR_ROOT:$CUPID_CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec conda run -n policy_doctor --no-capture-output \
    python -m policy_doctor.vlm.proposals.server \
        --config "$CONFIG" \
        --host "$HOST" \
        --port "$PORT"
