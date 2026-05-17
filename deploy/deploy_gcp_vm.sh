#!/usr/bin/env bash
# Deploy the policy-doctor demo to a single GCE VM (Container-Optimized OS).
#
# Why this instead of Cloud Run? Cloud Run's domain-restricted access needs
# `run.services.setIamPolicy`, which `roles/editor` does not include. A VM
# costs ~$13/mo (e2-small) and we gate access at the app layer with a
# SHA-256-hashed shared password.
#
# Workflow (every release):
#   1. collect_artifacts.sh re-bundles policy_doctor + clusterings + MP4s.
#   2. docker build --platform linux/amd64 (Cloud Run / GCE both want amd64).
#   3. Push :latest and :<git-sha> to Artifact Registry.
#   4. Either create the VM (first run) or `docker pull`+restart on the VM.
#   5. Ensure firewall rule allowing tcp:8501.
#   6. Print external IP, login URL, and the plaintext password (once).
#
# Idempotency:
#   - VM exists  → SSH in, pull new image, recreate container with the new
#                  image and the same APP_PASSWORD_SHA256 env from the
#                  instance metadata (so the password is stable across
#                  redeploys without storing the plaintext anywhere).
#   - VM missing → create from cos-stable, attach startup script, set the
#                  app-password-sha256 metadata.
#   - Firewall rule already present → skip.
#
# Where the plaintext password lives:
#   - Only in `deploy/.app_password` on the deployer's machine (gitignored).
#   - On the VM, only the SHA-256 hex digest exists, as instance metadata
#     `app-password-sha256` and as the container env var
#     APP_PASSWORD_SHA256. The plaintext is never sent to GCP.
#
# Override defaults via env vars:
#   APP_PASSWORD=... ./deploy/deploy_gcp_vm.sh   # pin a specific password
#   MACHINE_TYPE=e2-medium ./deploy/deploy_gcp_vm.sh

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT="${PROJECT:-gcp-driven-data}"
REGION="${REGION:-us-west1}"
ZONE="${ZONE:-us-west1-a}"
REPO="${REPO:-policy-doctor}"                # Artifact Registry repo
IMAGE_NAME="${IMAGE_NAME:-demo}"             # image inside the AR repo
VM_NAME="${VM_NAME:-policy-doctor-demo}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-small}"
DISK_SIZE="${DISK_SIZE:-20GB}"
FIREWALL_NAME="${FIREWALL_NAME:-policy-doctor-allow-8501}"
NETWORK_TAG="${NETWORK_TAG:-policy-doctor-http}"
PORT=8501

# ── Sanity ───────────────────────────────────────────────────────────────────
command -v gcloud >/dev/null || { echo "gcloud not on PATH"; exit 1; }
command -v docker >/dev/null || { echo "docker not on PATH"; exit 1; }
command -v shasum >/dev/null || command -v sha256sum >/dev/null \
    || { echo "need shasum or sha256sum"; exit 1; }

if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "Not logged in to gcloud — run 'gcloud auth login' first." ; exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GIT_SHA="$(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo dev)"
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}"
IMAGE_LATEST="${REGISTRY}/${IMAGE_NAME}:latest"
IMAGE_SHA="${REGISTRY}/${IMAGE_NAME}:${GIT_SHA}"

# ── Password handling ────────────────────────────────────────────────────────
# Persist the plaintext locally so re-running the script doesn't change the
# password (and lock out study participants who already have it).
PW_FILE="$SCRIPT_DIR/.app_password"
if [ -n "${APP_PASSWORD:-}" ]; then
    PASSWORD="$APP_PASSWORD"
    printf '%s\n' "$PASSWORD" > "$PW_FILE"
    chmod 600 "$PW_FILE"
elif [ -f "$PW_FILE" ]; then
    PASSWORD="$(cat "$PW_FILE")"
else
    # 16 hex chars from openssl (avoids the tr | head SIGPIPE-vs-pipefail
    # trap; openssl emits exactly the bytes we ask for and exits 0).
    PASSWORD="$(openssl rand -hex 8)"
    printf '%s\n' "$PASSWORD" > "$PW_FILE"
    chmod 600 "$PW_FILE"
fi

if command -v sha256sum >/dev/null; then
    PASSWORD_HASH="$(printf '%s' "$PASSWORD" | sha256sum | awk '{print $1}')"
else
    PASSWORD_HASH="$(printf '%s' "$PASSWORD" | shasum -a 256 | awk '{print $1}')"
fi

# ── Banner ───────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  project:      $PROJECT"
echo "  region/zone:  $REGION / $ZONE"
echo "  vm:           $VM_NAME ($MACHINE_TYPE)"
echo "  image tag:    $GIT_SHA"
echo "  password:     stored in $PW_FILE (chmod 600)"
echo "================================================================"

gcloud config set project "$PROJECT" >/dev/null

# ── 1. Bundle, build, push ──────────────────────────────────────────────────
echo "→ Bundling artifacts"
"$SCRIPT_DIR/collect_artifacts.sh"

echo "→ Building image for linux/amd64"
docker build --platform linux/amd64 -t policy-doctor-demo "$SCRIPT_DIR"

echo "→ Tagging and pushing $IMAGE_LATEST / $IMAGE_SHA"
docker tag policy-doctor-demo "$IMAGE_LATEST"
docker tag policy-doctor-demo "$IMAGE_SHA"
docker push "$IMAGE_LATEST"
docker push "$IMAGE_SHA"

# ── 2. Firewall (idempotent) ─────────────────────────────────────────────────
if ! gcloud compute firewall-rules describe "$FIREWALL_NAME" --project="$PROJECT" \
        --format="value(name)" >/dev/null 2>&1; then
    echo "→ Creating firewall rule $FIREWALL_NAME (tcp:$PORT from 0.0.0.0/0)"
    gcloud compute firewall-rules create "$FIREWALL_NAME" \
        --project="$PROJECT" \
        --network=default \
        --direction=INGRESS \
        --action=ALLOW \
        --rules="tcp:$PORT" \
        --source-ranges=0.0.0.0/0 \
        --target-tags="$NETWORK_TAG" \
        --quiet
else
    echo "→ Firewall $FIREWALL_NAME already exists, skipping"
fi

# ── 3. VM create-or-update ───────────────────────────────────────────────────
# Startup script: pull image and run with APP_PASSWORD_SHA256 from metadata.
# We use COS so docker + the AR credential helper are preconfigured.
STARTUP_SCRIPT=$(cat <<'EOF'
#!/bin/bash
set -e
exec >>/var/log/policy-doctor-startup.log 2>&1
echo "[startup] $(date) — fetching metadata"
MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
IMAGE=$(curl -fsS -H 'Metadata-Flavor: Google' "$MD/container-image")
HASH=$(curl -fsS -H 'Metadata-Flavor: Google' "$MD/app-password-sha256")
echo "[startup] image=$IMAGE"
# COS's docker only auto-auths against gcr.io. For Artifact Registry
# (*.pkg.dev) we have to register the host with docker-credential-gcr so
# the VM service account's creds get used on pull.
docker-credential-gcr configure-docker --registries=us-west1-docker.pkg.dev
docker pull "$IMAGE"
docker rm -f policy-doctor 2>/dev/null || true
docker run -d --restart=always --name policy-doctor \
    -p 8501:8501 \
    -e APP_PASSWORD_SHA256="$HASH" \
    "$IMAGE"
echo "[startup] container started"
EOF
)

if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --project="$PROJECT" \
        --format="value(name)" >/dev/null 2>&1; then
    echo "→ VM $VM_NAME exists — updating container-image metadata and refreshing"
    gcloud compute instances add-metadata "$VM_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --metadata="container-image=$IMAGE_SHA,app-password-sha256=$PASSWORD_HASH" \
        --quiet >/dev/null

    # Re-run the same pull/restart logic over SSH so the running container
    # picks up the new image without rebooting the VM. configure-docker is
    # idempotent and required because COS's default docker auth covers
    # gcr.io but not Artifact Registry.
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --quiet \
        --command="
            set -e
            docker-credential-gcr configure-docker --registries=us-west1-docker.pkg.dev
            docker pull '$IMAGE_SHA'
            docker rm -f policy-doctor 2>/dev/null || true
            docker run -d --restart=always --name policy-doctor \
                -p 8501:8501 \
                -e APP_PASSWORD_SHA256='$PASSWORD_HASH' \
                '$IMAGE_SHA'
        "
else
    echo "→ Creating VM $VM_NAME ($MACHINE_TYPE) in $ZONE"
    TMP_STARTUP=$(mktemp)
    printf '%s\n' "$STARTUP_SCRIPT" > "$TMP_STARTUP"
    trap 'rm -f "$TMP_STARTUP"' EXIT

    gcloud compute instances create "$VM_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --image-family=cos-stable \
        --image-project=cos-cloud \
        --boot-disk-size="$DISK_SIZE" \
        --boot-disk-type=pd-standard \
        --tags="$NETWORK_TAG" \
        --metadata-from-file="startup-script=$TMP_STARTUP" \
        --metadata="container-image=$IMAGE_SHA,app-password-sha256=$PASSWORD_HASH" \
        --quiet
fi

# ── 4. Print access info ─────────────────────────────────────────────────────
IP=$(gcloud compute instances describe "$VM_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --format='value(networkInterfaces[0].accessConfigs[0].natIP)')

echo
echo "================================================================"
echo "  VM:        $VM_NAME ($MACHINE_TYPE, $ZONE)"
echo "  URL:       http://${IP}:${PORT}"
echo "  Password:  $PASSWORD"
echo "  Image:     $IMAGE_SHA"
echo
echo "  On first deploy, allow 1-2 min for the startup script to pull"
echo "  the image and start the container. Tail the startup log:"
echo "    gcloud compute ssh $VM_NAME --zone $ZONE --project $PROJECT \\"
echo "       --command='sudo tail -f /var/log/policy-doctor-startup.log'"
echo "================================================================"
