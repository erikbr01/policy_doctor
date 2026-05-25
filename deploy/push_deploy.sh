#!/usr/bin/env bash
# Build the image, push it to Artifact Registry, sync the deploy/ folder to
# the VM, and restart the Compose stack.
#
# Run this for every release (and for the initial deployment after create_vm.sh
# and setup_data_disk.sh have completed).
#
# Usage:
#   ./deploy/push_deploy.sh                          # collect + build + push + deploy
#   ./deploy/push_deploy.sh --no-collect             # skip artifact bundling
#   ./deploy/push_deploy.sh --no-build               # skip docker build (re-push existing image)
#   ./deploy/push_deploy.sh --no-push                # skip push (re-deploy already-pushed image)
#   ./deploy/push_deploy.sh --no-cache               # docker build --no-cache
#   ./deploy/push_deploy.sh --set-password=<pw>      # update APP_PASSWORD_SHA256 on VM and restart
#
# Override defaults:
#   VM_NAME=user-study-test ./deploy/push_deploy.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT="${PROJECT:-gcp-driven-data}"
REGION="${REGION:-us-west1}"
ZONE="${ZONE:-us-west1-a}"
VM_NAME="${VM_NAME:-user-study-test}"
REPO="${REPO:-policy-doctor}"
IMAGE_NAME="${IMAGE_NAME:-demo}"
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}"
REMOTE_DEPLOY_DIR="~/deploy"
TUNNEL_FLAG="--tunnel"

command -v gcloud >/dev/null || { echo "gcloud not on PATH"; exit 1; }
command -v docker  >/dev/null || { echo "docker not on PATH"; exit 1; }

# ── Flags ─────────────────────────────────────────────────────────────────────
COLLECT=true
BUILD=true
PUSH=true
NO_CACHE=""
SET_PASSWORD=""

for arg in "$@"; do
    case "$arg" in
        --no-collect)       COLLECT=false ;;
        --no-build)         BUILD=false ;;
        --no-push)          PUSH=false ;;
        --no-cache)         NO_CACHE="--no-cache" ;;
        --set-password=*)   SET_PASSWORD="${arg#--set-password=}" ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# ── Password-only shortcut ────────────────────────────────────────────────────
# --set-password updates just APP_PASSWORD_SHA256 in the VM's .env and
# restarts the stack. Skips build/push entirely.
if [ -n "$SET_PASSWORD" ]; then
    HASH="$(printf '%s' "$SET_PASSWORD" | sha256sum | awk '{print $1}')"
    echo "→ Updating APP_PASSWORD_SHA256 on $VM_NAME"
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --quiet \
        --command="
            set -e
            ENV_FILE=${REMOTE_DEPLOY_DIR}/.env
            if grep -q '^APP_PASSWORD_SHA256=' \"\$ENV_FILE\" 2>/dev/null; then
                sed -i 's|^APP_PASSWORD_SHA256=.*|APP_PASSWORD_SHA256=${HASH}|' \"\$ENV_FILE\"
            else
                echo 'APP_PASSWORD_SHA256=${HASH}' >> \"\$ENV_FILE\"
            fi
            echo '  .env updated'
            cd ${REMOTE_DEPLOY_DIR}
            bash deploy_study_stack.sh ${TUNNEL_FLAG} --no-collect --no-build
        "
    echo "Password updated and stack restarted."
    exit 0
fi

GIT_SHA="$(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo dev)"
IMAGE_LATEST="${REGISTRY}/${IMAGE_NAME}:latest"
IMAGE_SHA="${REGISTRY}/${IMAGE_NAME}:${GIT_SHA}"

echo "================================================================"
echo "  project:  $PROJECT  ($REGION / $ZONE)"
echo "  vm:       $VM_NAME"
echo "  image:    $IMAGE_SHA"
echo "================================================================"

gcloud config set project "$PROJECT" --quiet

# ── 1. Collect artifacts ──────────────────────────────────────────────────────
if $COLLECT; then
    echo "→ Collecting artifacts"
    "$SCRIPT_DIR/collect_artifacts.sh"
fi

# ── 2. Build (linux/amd64 — GCE is always x86-64) ────────────────────────────
if $BUILD; then
    echo "→ Building image for linux/amd64"
    docker build $NO_CACHE --platform linux/amd64 \
        -t "$IMAGE_LATEST" -t "$IMAGE_SHA" \
        "$SCRIPT_DIR"
fi

# ── 3. Push ───────────────────────────────────────────────────────────────────
if $PUSH; then
    # Ensure the Artifact Registry repo exists (idempotent).
    if ! gcloud artifacts repositories describe "$REPO" \
            --location="$REGION" --project="$PROJECT" \
            --format="value(name)" >/dev/null 2>&1; then
        echo "→ Creating Artifact Registry repo $REPO"
        gcloud artifacts repositories create "$REPO" \
            --repository-format=docker \
            --location="$REGION" \
            --project="$PROJECT" \
            --quiet
    fi

    echo "→ Authenticating Docker with Artifact Registry"
    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

    echo "→ Pushing $IMAGE_LATEST"
    docker push "$IMAGE_LATEST"
    docker push "$IMAGE_SHA"
fi

# ── 4. Sync deploy/ folder to VM ─────────────────────────────────────────────
# Pipe a tarball over SSH — avoids gcloud compute scp's lack of --exclude and
# the rsync-over-gcloud SSH transport issues.
# The data bundles (policy_doctor/, third_party/, data/) are baked into the
# Docker image; only scripts, configs, and nginx templates need to be on the VM.
echo "→ Syncing deploy/ to $VM_NAME:$REMOTE_DEPLOY_DIR"

gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" \
    --quiet --command="mkdir -p $REMOTE_DEPLOY_DIR/cloudflared"

tar -czf - -C "$SCRIPT_DIR" \
    --exclude='.git' \
    --exclude='policy_doctor' \
    --exclude='third_party' \
    --exclude='data' \
    --exclude='survey_responses' \
    --exclude='cloudflared/*.json' \
    --exclude='.env' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    . \
| gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" \
    --quiet --command="tar -xzf - -C $REMOTE_DEPLOY_DIR"

# Copy the tunnel credentials JSON separately (excluded from tar above).
CRED_JSON=$(ls "$SCRIPT_DIR/cloudflared/"*.json 2>/dev/null | head -1)
if [ -n "$CRED_JSON" ]; then
    echo "→ Copying tunnel credentials JSON"
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --quiet \
        --command="cat > ${REMOTE_DEPLOY_DIR}/cloudflared/$(basename "$CRED_JSON")" \
        < "$CRED_JSON"
else
    echo "WARNING: No cloudflared credentials JSON found in $SCRIPT_DIR/cloudflared/"
    echo "  The tunnel will not start until you copy it manually."
fi

# ── 5. Pull image + restart Compose stack on VM ───────────────────────────────
echo "→ Pulling image and restarting stack on $VM_NAME"
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --quiet \
    --command="
        set -e

        # Ensure the user-level Docker config has the AR credential helper.
        # The startup script wrote it system-wide (/etc/docker/config.json)
        # but Docker prefers ~/.docker/config.json, so configure it for this
        # user too.  Idempotent.
        mkdir -p ~/.docker
        docker-credential-gcr configure-docker \
            --registries="${REGION}-docker.pkg.dev" \
            --include-artifact-registry >/dev/null

        # Pull the new image and tag it with the local name Compose expects.
        docker pull '${IMAGE_LATEST}'
        docker tag  '${IMAGE_LATEST}' policy-doctor-demo

        # Restart the full stack (--no-build skips explicit docker build;
        # Compose finds the image via the tag we just set).
        cd ${REMOTE_DEPLOY_DIR}
        bash deploy_study_stack.sh ${TUNNEL_FLAG} --no-collect --no-build
    "

echo
echo "================================================================"
echo "  Deployed to $VM_NAME"
echo "  https://study.behavior-graphs.com"
echo "  https://demo.behavior-graphs.com"
echo "================================================================"
