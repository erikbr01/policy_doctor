#!/usr/bin/env bash
# Cupid stack integration: mar27 transport pipeline, fingerprint / episode_ends checks.
# Needs ``pandas`` / ``six``; fingerprint tests need ``pytorch3d`` (see cupid ``conda_environment.yaml``).
# Use ``conda env update`` on ``third_party/cupid/conda_environment.yaml`` if imports fail.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH=.
exec conda run -n cupid --no-capture-output \
  python run_tests.py --suite cupid "$@"
