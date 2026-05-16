#!/usr/bin/env bash
# Collect everything the docker build needs from the worktree into ./deploy/.
# Re-runnable: deletes the destination subtrees before copying.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== Bundling deploy/ from $ROOT =="

mkdir -p deploy/policy_doctor deploy/third_party deploy/data/study_mp4s

# 1. The policy_doctor source package (only the parts the demo uses).
rsync -a --delete \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    policy_doctor/ deploy/policy_doctor/

# 2. The clusterings needed by the user-study + graph-demo pages.
#    Just transport_mh_jan28 for now (all 47+ variants we generated).
mkdir -p deploy/third_party/influence_visualizer/configs
rsync -a --delete \
    --exclude='clustering_models.pkl' \
    --exclude='embedding_models.pkl' \
    third_party/influence_visualizer/configs/transport_mh_jan28 \
    deploy/third_party/influence_visualizer/configs/

# 3. MP4s for the user-study pages (and the graph-demo's click-to-explore).
if [ -d /tmp/study_mp4s/transport_mh_jan28 ]; then
    rsync -a --delete \
        /tmp/study_mp4s/transport_mh_jan28 \
        deploy/data/study_mp4s/
else
    echo "WARNING: /tmp/study_mp4s/transport_mh_jan28 not found —"
    echo "         user-study pages and graph-demo video panels will be empty."
fi

# 4. Sanity check.
echo "== Bundle contents =="
du -sh deploy/policy_doctor deploy/third_party deploy/data 2>&1

echo
echo "== Ready to build =="
echo "  cd deploy && docker build -t policy-doctor-demo ."
