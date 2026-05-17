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

# 2. Clusterings + MP4s for each task we want in the bundle.
TASKS=(
    transport_mh_jan28
    square_mh_feb5
    lift_mh_jan26
)
mkdir -p deploy/third_party/influence_visualizer/configs
for task in "${TASKS[@]}"; do
    src_clu="third_party/influence_visualizer/configs/$task"
    if [ -d "$src_clu/clustering" ]; then
        rsync -a --delete \
            --exclude='clustering_models.pkl' \
            --exclude='embedding_models.pkl' \
            "$src_clu" deploy/third_party/influence_visualizer/configs/
    else
        echo "WARNING: $src_clu/clustering not found — task '$task' will be missing from the bundle."
    fi
    src_mp4="/tmp/study_mp4s/$task"
    if [ -d "$src_mp4" ]; then
        rsync -a --delete "$src_mp4" deploy/data/study_mp4s/
    else
        echo "WARNING: $src_mp4 not found — task '$task' video panels will be empty."
    fi
done

# 4. Sanity check.
echo "== Bundle contents =="
du -sh deploy/policy_doctor deploy/third_party deploy/data 2>&1

echo
echo "== Ready to build =="
echo "  cd deploy && docker build -t policy-doctor-demo ."
