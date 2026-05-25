#!/usr/bin/env bash
# Create a new GCE VM for the Compose-based policy-doctor stack.
#
# This is a ONE-TIME setup script.  After it completes:
#   1. Run setup_data_disk.sh   — create + mount the persistent data disk
#   2. Create cloudflared/config.yml from config.yml.example
#   3. Copy <TUNNEL_ID>.json into cloudflared/
#   4. Run push_deploy.sh       — build + push image + sync + start stack
#
# Usage:
#   ./deploy/create_vm.sh
#
# Override defaults:
#   VM_NAME=user-study-test MACHINE_TYPE=e2-medium ./deploy/create_vm.sh

set -euo pipefail

PROJECT="${PROJECT:-gcp-driven-data}"
ZONE="${ZONE:-us-west1-a}"
VM_NAME="${VM_NAME:-user-study-test}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-small}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-20GB}"
REGION="${REGION:-us-west1}"
AR_REGISTRY="${REGION}-docker.pkg.dev"

command -v gcloud >/dev/null || { echo "gcloud not on PATH"; exit 1; }

echo "================================================================"
echo "  project:   $PROJECT"
echo "  zone:      $ZONE"
echo "  vm:        $VM_NAME ($MACHINE_TYPE)"
echo "  boot disk: $BOOT_DISK_SIZE"
echo "  No firewall ports needed — traffic via Cloudflare Tunnel only."
echo "================================================================"

# Bail early if the VM already exists.
if gcloud compute instances describe "$VM_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --format="value(name)" >/dev/null 2>&1; then
    echo "VM $VM_NAME already exists — nothing to do."
    echo "If you want to recreate it, delete it first:"
    echo "  gcloud compute instances delete $VM_NAME --zone=$ZONE --project=$PROJECT"
    exit 0
fi

# ── Startup script: install Docker CE + Compose plugin + AR auth ─────────────
# This runs as root on first boot.  Docker installation follows the official
# Docker docs for Debian 12 (Bookworm).  AR auth uses docker-credential-gcr
# so the VM's ADC (default Compute SA) is used to pull from Artifact Registry
# without any extra credentials.
STARTUP_SCRIPT=$(cat <<'STARTUP_EOF'
#!/bin/bash
set -euo pipefail
exec >>/var/log/vm-setup.log 2>&1
echo "[setup] $(date) — starting"

# ── Docker CE ────────────────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y --no-install-recommends ca-certificates curl gnupg

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/debian bookworm stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update -qq
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

systemctl enable --now docker

# Add every human user to the docker group so they don't need sudo.
for u in $(getent passwd | awk -F: '$3 >= 1000 && $3 < 65534 {print $1}'); do
    usermod -aG docker "$u" 2>/dev/null || true
done

# ── Artifact Registry auth ───────────────────────────────────────────────────
# Install docker-credential-gcr so Docker uses the VM's ADC to pull from AR.
GCR_VERSION=$(curl -fsS \
    "https://api.github.com/repos/GoogleCloudPlatform/docker-credential-gcr/releases/latest" \
    | grep '"tag_name"' | sed 's/.*"v\([^"]*\)".*/\1/')
ARCH=$(dpkg --print-architecture)
curl -fsSL \
    "https://github.com/GoogleCloudPlatform/docker-credential-gcr/releases/download/v${GCR_VERSION}/docker-credential-gcr_linux_${ARCH}-${GCR_VERSION}.tar.gz" \
    | tar xz -C /usr/local/bin docker-credential-gcr
chmod +x /usr/local/bin/docker-credential-gcr

# Configure for all users (write to /etc/docker/config.json).
docker-credential-gcr configure-docker \
    --registries=us-west1-docker.pkg.dev \
    --include-artifact-registry >/dev/null

echo "[setup] $(date) — complete"
STARTUP_EOF
)

gcloud config set project "$PROJECT" --quiet

echo "→ Creating VM $VM_NAME in $ZONE"
TMP=$(mktemp)
printf '%s\n' "$STARTUP_SCRIPT" > "$TMP"
trap 'rm -f "$TMP"' EXIT

gcloud compute instances create "$VM_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --boot-disk-type=pd-standard \
    --metadata-from-file="startup-script=$TMP" \
    --scopes=cloud-platform \
    --quiet

echo
echo "→ VM created. Waiting for startup script to finish installing Docker…"
echo "  (This takes ~2 min. Tailing the setup log:)"
echo

# Poll until Docker is available.
for i in $(seq 1 30); do
    sleep 10
    if gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" \
            --quiet --command="docker --version" 2>/dev/null; then
        echo
        echo "================================================================"
        echo "  VM $VM_NAME is ready."
        echo
        echo "  Next steps:"
        echo "    1. VM_NAME=$VM_NAME ./deploy/setup_data_disk.sh"
        echo "    2. Copy cloudflared credentials to the VM (see README)"
        echo "    3. VM_NAME=$VM_NAME ./deploy/push_deploy.sh"
        echo "================================================================"
        exit 0
    fi
    echo "  … waiting (${i}/30)"
done

echo "WARNING: Docker not yet ready after 5 min — startup script may still be running."
echo "  Check: gcloud compute ssh $VM_NAME --zone=$ZONE --command='tail /var/log/vm-setup.log'"
