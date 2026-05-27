#!/usr/bin/env bash
# Orchestration / package tests (no cupid or mimicgen sim required beyond normal deps).
# Refresh deps: ``conda env update -f environment_policy_doctor.yaml`` then
# ``./scripts/install_policy_doctor_env.sh`` (installs ``requirements_policy_doctor.txt``).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH=.
exec conda run -n policy_doctor --no-capture-output \
  python run_tests.py --suite policy_doctor "$@"
