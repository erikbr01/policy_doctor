#!/usr/bin/env bash
# Sync worktree source into deploy/ for Docker builds (--no-collect fast path).
#
# Copies policy_doctor/ only. Does not touch deploy/.env, cloudflared/*.json,
# or survey_responses/ — those stay local and are excluded from the Docker build
# context via deploy/.dockerignore.
#
# Usage:
#   source deploy/sync_repo.sh && sync_repo
#   ./deploy/sync_repo.sh

set -euo pipefail

_SYNC_REPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SYNC_REPO_ROOT="$(cd "$_SYNC_REPO_SCRIPT_DIR/.." && pwd)"

sync_repo() {
    echo "→ Syncing policy_doctor → deploy/policy_doctor"
    mkdir -p "$_SYNC_REPO_SCRIPT_DIR/policy_doctor"
    rsync -a --delete \
        --exclude='.git/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='.env' \
        --exclude='.env.local' \
        --exclude='.env.production' \
        --exclude='.app_password' \
        --exclude='survey_responses/' \
        --exclude='*.pem' \
        --exclude='*.key' \
        --exclude='cloudflared/' \
        --exclude='**/credentials*.json' \
        "$_SYNC_REPO_ROOT/policy_doctor/" "$_SYNC_REPO_SCRIPT_DIR/policy_doctor/"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    sync_repo "$@"
fi
