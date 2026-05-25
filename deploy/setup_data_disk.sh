#!/usr/bin/env bash
# One-time setup: create a persistent data disk, attach it to the VM, format
# it, and configure it to auto-mount on reboot.
#
# Run this ONCE from your local machine before the first deployment.
# The disk is separate from the VM's boot disk and is NOT auto-deleted when
# the VM is deleted or recreated — your response data survives.
#
# After this script runs, set in deploy/.env on the VM:
#   SURVEY_RESPONSES_DIR=/mnt/data/survey_responses
#
# Usage:
#   ./deploy/setup_data_disk.sh
#
# Override defaults via env vars:
#   DISK_SIZE=20GB ./deploy/setup_data_disk.sh

set -euo pipefail

PROJECT="${PROJECT:-gcp-driven-data}"
ZONE="${ZONE:-us-west1-a}"
VM_NAME="${VM_NAME:-policy-doctor-demo}"
DATA_DISK_NAME="${DATA_DISK_NAME:-policy-doctor-data}"
DATA_DISK_SIZE="${DATA_DISK_SIZE:-10GB}"
MOUNT_POINT="/mnt/data"
DEVICE_NAME="data"   # logical device name used in the attach call

command -v gcloud >/dev/null || { echo "gcloud not on PATH"; exit 1; }

echo "================================================================"
echo "  project:    $PROJECT"
echo "  zone:       $ZONE"
echo "  vm:         $VM_NAME"
echo "  disk:       $DATA_DISK_NAME ($DATA_DISK_SIZE)"
echo "  mount:      $MOUNT_POINT"
echo "================================================================"

# ── 1. Create the disk (idempotent) ──────────────────────────────────────────
if gcloud compute disks describe "$DATA_DISK_NAME" \
        --zone="$ZONE" --project="$PROJECT" \
        --format="value(name)" >/dev/null 2>&1; then
    echo "→ Disk $DATA_DISK_NAME already exists, skipping creation"
else
    echo "→ Creating persistent disk $DATA_DISK_NAME ($DATA_DISK_SIZE)"
    gcloud compute disks create "$DATA_DISK_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --size="$DATA_DISK_SIZE" \
        --type=pd-standard \
        --quiet
fi

# ── 2. Attach the disk to the VM (idempotent) ─────────────────────────────────
ATTACHED=$(gcloud compute instances describe "$VM_NAME" \
    --zone="$ZONE" --project="$PROJECT" \
    --format="value(disks[].source)" \
    | tr ';' '\n' \
    | grep -c "$DATA_DISK_NAME" || true)

if [ "$ATTACHED" -gt 0 ]; then
    echo "→ Disk already attached to $VM_NAME, skipping"
else
    echo "→ Attaching $DATA_DISK_NAME to $VM_NAME (auto-delete=false)"
    gcloud compute instances attach-disk "$VM_NAME" \
        --disk="$DATA_DISK_NAME" \
        --device-name="$DEVICE_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --quiet
    # Explicitly disable auto-delete so the disk survives VM deletion.
    gcloud compute instances set-disk-auto-delete "$VM_NAME" \
        --no-auto-delete \
        --disk="$DATA_DISK_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --quiet
fi

# ── 3. Format, mount, and configure fstab on the VM ─────────────────────────
echo "→ Formatting and mounting disk on $VM_NAME"
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --quiet \
    --command="
        set -e
        DEVICE=/dev/disk/by-id/google-${DEVICE_NAME}

        # Format only if no filesystem exists yet.
        if ! sudo blkid \"\$DEVICE\" >/dev/null 2>&1; then
            echo '[disk] Formatting as ext4...'
            sudo mkfs.ext4 -F \"\$DEVICE\"
        else
            echo '[disk] Filesystem already present, skipping format'
        fi

        # Mount point
        sudo mkdir -p ${MOUNT_POINT}

        # Add to fstab if not already there (persists across reboots).
        FSTAB_LINE=\"\$DEVICE  ${MOUNT_POINT}  ext4  defaults,nofail  0  2\"
        if ! grep -qF \"${DEVICE_NAME}\" /etc/fstab; then
            echo '[disk] Adding fstab entry'
            echo \"\$FSTAB_LINE\" | sudo tee -a /etc/fstab
        else
            echo '[disk] fstab entry already present'
        fi

        sudo mount -a

        # Create the responses directory with open permissions so the
        # Docker container (running as a non-root user) can write to it.
        sudo mkdir -p ${MOUNT_POINT}/survey_responses
        sudo chmod 777 ${MOUNT_POINT}/survey_responses

        echo '[disk] Mount complete:'
        df -h ${MOUNT_POINT}
    "

echo
echo "================================================================"
echo "  Done. Disk is mounted at ${MOUNT_POINT} on $VM_NAME."
echo
echo "  Add this to deploy/.env on the VM:"
echo "    SURVEY_RESPONSES_DIR=${MOUNT_POINT}/survey_responses"
echo "================================================================"
