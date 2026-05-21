#!/usr/bin/env bash
# Build and run the policy-doctor demo locally via Docker.
#
# Usage:
#   ./deploy/deploy_local.sh           # collect artifacts, build, run
#   ./deploy/deploy_local.sh --no-collect  # skip collect_artifacts.sh (faster rebuild)
#   ./deploy/deploy_local.sh --no-build    # skip docker build (reuse existing image)
#
# Opens http://localhost:8503 when the container is ready.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER_NAME="policy-doctor-demo"
IMAGE_NAME="policy-doctor-demo"
PORT=8503

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

echo "→ Stopping existing container (if any)"
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "→ Starting $CONTAINER_NAME on port $PORT"
docker run -d \
    --name "$CONTAINER_NAME" \
    -p "${PORT}:8501" \
    "$IMAGE_NAME"

echo
echo "================================================================"
echo "  URL:  http://localhost:${PORT}"
echo "  Logs: docker logs -f $CONTAINER_NAME"
echo "  Stop: docker rm -f $CONTAINER_NAME"
echo "================================================================"
