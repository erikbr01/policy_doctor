#!/usr/bin/env bash
# Run a command inside one of the uv-managed envs (the conda-run replacement).
#
# Usage:
#   ./scripts/uv_env.sh <env-name> <command> [args...]            # via symlink
#   ./scripts/setup/uv_env.sh <env-name> <command> [args...]       # direct
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

# Resolve REPO_ROOT robustly whether invoked directly (scripts/setup/uv_env.sh)
# or via the back-compat symlink at scripts/uv_env.sh. Resolve symlinks on the
# script path so we always climb from the real location (scripts/setup/).
SOURCE="${BASH_SOURCE[0]}"
while [[ -h "$SOURCE" ]]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ "$SOURCE" != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export UV_PROJECT_ENVIRONMENT="${REPO_ROOT}/.venvs/${ENV_NAME}"

case "$ENV_NAME" in
    analysis|cupid|mimicgen|robocasa) ;;
    *) echo "Unknown env-name: $ENV_NAME" >&2; usage ;;
esac

# --setup: just (re)create the env and exit.
if [[ "${1:-}" == "--setup" ]]; then
    shift
    (cd "$REPO_ROOT" && uv sync --extra "$ENV_NAME" --extra dev)
    if [[ $# -eq 0 ]]; then
        exit 0
    fi
elif [[ ! -d "$UV_PROJECT_ENVIRONMENT" ]]; then
    # First use: sync silently to stderr, then run the command.
    (cd "$REPO_ROOT" && uv sync --extra "$ENV_NAME" --extra dev) >&2
fi

if [[ $# -eq 0 ]]; then
    echo "No command given; pass --setup to (re)create the env, or supply a command to run." >&2
    exit 2
fi

exec uv run --no-sync "$@"
