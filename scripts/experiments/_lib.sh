#!/usr/bin/env bash
# Shared helpers for experiment runners (sourced by other scripts).
set -euo pipefail

experiments_lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DOCTOR_ROOT="$(cd "${experiments_lib_dir}/../.." && pwd)"

run_pipeline() {
  local env_name="${1:?conda env}"
  shift
  cd "$POLICY_DOCTOR_ROOT"
  export PYTHONPATH=.
  exec conda run -n "$env_name" --no-capture-output \
    python -m policy_doctor.scripts.run_pipeline "$@"
}
