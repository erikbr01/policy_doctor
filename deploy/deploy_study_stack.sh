#!/usr/bin/env bash
# Build and run the user-study stack: nginx reverse proxy + survey + demo.
#
#   Browser  ──►  proxy:80  ──►  survey:8501   (study.<domain>)
#                              \─►  demo:8501   (demo.<domain>)
#
# Usage:
#   ./deploy/deploy_study_stack.sh               # collect + build + start
#   ./deploy/deploy_study_stack.sh --no-collect  # skip artifact bundling
#   ./deploy/deploy_study_stack.sh --no-build    # reuse existing image
#   ./deploy/deploy_study_stack.sh --no-cache    # docker build --no-cache
#   ./deploy/deploy_study_stack.sh --tls         # include TLS override
#                                                # (needs init_letsencrypt.sh
#                                                #  to have run first)
#
# Reads .env (next to this script) for STUDY_DOMAIN / DEMO_DOMAIN / passwords /
# SURVEY_GCS_BUCKET. See .env.example.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="policy-doctor-demo"

COLLECT=true
BUILD=true
TLS=false
NO_CACHE=""

for arg in "$@"; do
    case "$arg" in
        --no-collect) COLLECT=false ;;
        --no-build)   BUILD=false ;;
        --no-cache)   NO_CACHE="--no-cache" ;;
        --tls)        TLS=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# Load .env so STUDY_DOMAIN / DEMO_DOMAIN are visible to the final banner.
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$SCRIPT_DIR/.env"
    set +a
fi

STUDY_DOMAIN="${STUDY_DOMAIN:-study.localhost}"
DEMO_DOMAIN="${DEMO_DOMAIN:-demo.localhost}"
HTTP_PORT="${HTTP_PORT:-80}"
HTTPS_PORT="${HTTPS_PORT:-443}"

COMPOSE_FILES=(-f "$SCRIPT_DIR/docker-compose.yml")
if $TLS; then
    COMPOSE_FILES+=(-f "$SCRIPT_DIR/docker-compose.tls.yml")
fi

if $COLLECT; then
    echo "→ Collecting artifacts"
    "$SCRIPT_DIR/collect_artifacts.sh"
fi

if $BUILD; then
    echo "→ Building image $IMAGE_NAME"
    docker build $NO_CACHE -t "$IMAGE_NAME" "$SCRIPT_DIR"
fi

echo "→ Starting stack (proxy + survey + demo)"
docker compose "${COMPOSE_FILES[@]}" up -d

# URL scheme depends on whether we're running with the TLS override.
if $TLS; then
    SCHEME=https
    PORT_SUFFIX=""
    [ "$HTTPS_PORT" = "443" ] || PORT_SUFFIX=":$HTTPS_PORT"
else
    SCHEME=http
    PORT_SUFFIX=""
    [ "$HTTP_PORT" = "80" ] || PORT_SUFFIX=":$HTTP_PORT"
fi

echo
echo "================================================================"
echo "  Survey app (participants):  $SCHEME://$STUDY_DOMAIN$PORT_SUFFIX"
echo "  Demo app   (researchers):   $SCHEME://$DEMO_DOMAIN$PORT_SUFFIX"
echo
if ! $TLS; then
    case "$STUDY_DOMAIN" in
        *.localhost|localhost)
            echo "  Local-only DNS: most systems resolve *.localhost to 127.0.0.1"
            echo "  automatically. If your browser shows DNS_PROBE_FINISHED_NXDOMAIN,"
            echo "  add this line to /etc/hosts:"
            echo "      127.0.0.1  $STUDY_DOMAIN $DEMO_DOMAIN"
            echo "  Or set STUDY_DOMAIN=study.lvh.me / DEMO_DOMAIN=demo.lvh.me"
            echo "  in .env (no /etc/hosts edit needed)."
            echo
            ;;
    esac
fi
echo "  Logs:  docker compose ${COMPOSE_FILES[*]} logs -f"
echo "  Stop:  docker compose ${COMPOSE_FILES[*]} down"
echo "================================================================"
