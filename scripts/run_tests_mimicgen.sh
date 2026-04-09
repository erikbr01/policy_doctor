#!/usr/bin/env bash
# MimicGen seed abstractions + optional E2E (set MIMICGEN_E2E=1 for full sim + HF).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH=.
exec conda run -n mimicgen --no-capture-output \
  python run_tests.py --suite mimicgen "$@"
