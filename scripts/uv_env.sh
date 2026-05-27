#!/usr/bin/env bash
# Run a command inside one of the uv-managed envs (the conda-run replacement).
#
# Usage:
#   ./scripts/uv_env.sh <env-name> <command> [args...]
#
# Example:
#   ./scripts/uv_env.sh analysis pytest tests/golden/
#   ./scripts/uv_env.sh mimicgen python -m policy_doctor.scripts.run_pipeline ...
#
# Each env-name maps to:
#   - a uv extra in pyproject.toml (the [project.optional-dependencies] keys)
#   - a venv at .venvs/<env-name>/
#
# Run `./scripts/uv_env.sh <env-name> --setup` to (re)create the venv.

set -euo pipefail

usage() {
    cat >&2 <<EOF
Usage: $0 <env-name> <command> [args...]

env-name: analysis | cupid | mimicgen | robocasa
EOF
    exit 1
}

if [[ $# -lt 2 ]]; then
    usage
fi

ENV_NAME="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_PROJECT_ENVIRONMENT="${REPO_ROOT}/.venvs/${ENV_NAME}"

case "$ENV_NAME" in
    analysis|cupid|mimicgen|robocasa) ;;
    *) echo "Unknown env-name: $ENV_NAME" >&2; usage ;;
esac

# Ensure the env exists; recreate if missing or if --setup was passed.
if [[ "${1:-}" == "--setup" ]] || [[ ! -d "$UV_PROJECT_ENVIRONMENT" ]]; then
    shift || true
    # Dev extras (pytest, ruff) always come along — they're tiny.
    (cd "$REPO_ROOT" && uv sync --extra "$ENV_NAME" --extra dev) >&2
fi

exec uv run --no-sync "$@"
