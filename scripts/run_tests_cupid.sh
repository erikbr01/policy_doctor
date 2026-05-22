#!/usr/bin/env bash
# Cupid stack integration: transport pipeline, fingerprint / episode_ends checks.
# Runs under the unified ``policy_doctor`` env (formerly ``cupid_torch2``).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH=.
exec conda run -n policy_doctor --no-capture-output \
  python run_tests.py --suite cupid "$@"
