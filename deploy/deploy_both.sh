#!/usr/bin/env bash
# Build and run BOTH the survey app (port 8501) and graph-demo (port 8502)
# on the local machine using Docker Compose.
#
# Usage:
#   ./deploy/deploy_both.sh               # collect, build, start both
#   ./deploy/deploy_both.sh --no-collect  # skip artifact collection
#   ./deploy/deploy_both.sh --no-build    # reuse existing image
#
# Env vars (optional — see docker-compose.yml):
#   SURVEY_PASSWORD_SHA256   password hash for the survey app
#   APP_PASSWORD_SHA256      password hash for the graph-demo
#   SURVEY_GCS_BUCKET        GCS bucket for response storage
#
# After startup:
#   Survey app (send to participants):  http://localhost:8501
#   Graph demo  (researchers):          http://localhost:8502

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="policy-doctor-demo"

COLLECT=true
BUILD=true
NO_CACHE=""

for arg in "$@"; do
    case "$arg" in
        --no-collect) COLLECT=false ;;
        --no-build)   BUILD=false ;;
        --no-cache)   NO_CACHE="--no-cache" ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

if $COLLECT; then
    echo "→ Collecting artifacts"
    "$SCRIPT_DIR/collect_artifacts.sh"
fi

if $BUILD; then
    echo "→ Building image $IMAGE_NAME"
    docker build $NO_CACHE -t "$IMAGE_NAME" "$SCRIPT_DIR"
fi

echo "→ Starting both services (survey on :8501, demo on :8502)"
docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d

echo
echo "================================================================"
echo "  Survey app (participants):  http://localhost:8501"
echo "  Graph demo  (researchers):  http://localhost:8502"
echo
echo "  Logs:  docker compose -f $SCRIPT_DIR/docker-compose.yml logs -f"
echo "  Stop:  docker compose -f $SCRIPT_DIR/docker-compose.yml down"
echo "================================================================"
