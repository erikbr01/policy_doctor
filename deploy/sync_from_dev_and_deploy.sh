#!/usr/bin/env bash
# Pull pi05-libero data from the dev VM, re-bundle, and deploy to GCP.
#
# Run this on your local machine (the one with robomimic clusterings + Docker):
#
#   ./deploy/sync_from_dev_and_deploy.sh
#
# Prerequisites:
#   - gcloud auth login done
#   - Docker running
#   - Robomimic clusterings already in third_party/influence_visualizer/configs/
#   - Robomimic MP4s already in /tmp/study_mp4s/
#
# What it does:
#   1. git pull (picks up the merged feat/pi05-libero code)
#   2. rsync pi05 clustering configs from the dev VM
#   3. rsync pi05 episode media (MP4s + metadata) from the dev VM
#   4. collect_artifacts.sh — bundles everything into deploy/
#   5. deploy_gcp_vm.sh — builds image, pushes to AR, updates VM

set -euo pipefail

DEV_VM="dev"
DEV_ZONE="us-west1-a"
DEV_PROJECT="gcp-driven-data"
DEV_ROOT="/home/erbauer/policy_doctor"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Use gcloud compute ssh as the rsync remote shell so IAP/project auth is
# handled automatically without needing a separate SSH key config.
RSYNC_RSH="gcloud compute ssh $DEV_VM --zone $DEV_ZONE --project $DEV_PROJECT --"

_rsync() {
    rsync -avz --progress -e "$RSYNC_RSH" "$@"
}

echo "================================================================"
echo "  dev VM:   $DEV_VM ($DEV_ZONE, $DEV_PROJECT)"
echo "  local:    $ROOT"
echo "================================================================"

# ── 1. Pull latest code ──────────────────────────────────────────────────────
echo
echo "→ git pull origin main"
git pull origin main

# ── 2. Pi05 clustering configs ───────────────────────────────────────────────
echo
echo "→ Syncing pi05 clustering configs (~613 MB)"
mkdir -p third_party/influence_visualizer/configs
for suite in libero_spatial libero_object libero_goal; do
    task="pi05_$suite"
    echo "   $task"
    _rsync \
        --exclude='clustering_models.pkl' \
        --exclude='embedding_models.pkl' \
        ":$DEV_ROOT/third_party/influence_visualizer/configs/$task" \
        third_party/influence_visualizer/configs/
done

# ── 3. Pi05 media (MP4s + episode metadata) ──────────────────────────────────
echo
echo "→ Syncing pi05 media (~166 MB)"
for suite in libero_spatial libero_object libero_goal; do
    echo "   $suite"
    mkdir -p "data/pi05_libero/$suite/media" "data/pi05_libero/$suite/episodes"
    _rsync \
        --include='*.mp4' --exclude='*' \
        ":$DEV_ROOT/data/pi05_libero/$suite/media/" \
        "data/pi05_libero/$suite/media/"
    _rsync \
        ":$DEV_ROOT/data/pi05_libero/$suite/episodes/metadata.yaml" \
        "data/pi05_libero/$suite/episodes/"
done

# ── 4. Bundle ────────────────────────────────────────────────────────────────
echo
echo "→ collect_artifacts.sh"
CONDA_PYTHON="${CONDA_PYTHON:-$(command -v python3)}"
CONDA_PYTHON="$CONDA_PYTHON" bash deploy/collect_artifacts.sh

# ── 5. Deploy ────────────────────────────────────────────────────────────────
echo
echo "→ deploy_gcp_vm.sh"
bash deploy/deploy_gcp_vm.sh
