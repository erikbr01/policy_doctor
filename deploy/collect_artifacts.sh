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

# 3. pi0.5-LIBERO tasks.
#    Clusterings come from the iv-configs tree (same as above tasks).
#    Media lives permanently under data/pi05_libero/<suite>/media/ — we copy
#    it into deploy/data/study_mp4s/ and generate index.json with paths
#    matching /tmp/study_mp4s/<task>/ (the Docker landing dir). We also create
#    a local symlink so the demo works without Docker after running this script.
PI05_SUITES=(libero_spatial libero_object libero_goal)
for suite in "${PI05_SUITES[@]}"; do
    task="pi05_$suite"
    src_clu="third_party/influence_visualizer/configs/$task"
    src_media="$ROOT/data/pi05_libero/$suite/media"
    dst_mp4="$ROOT/deploy/data/study_mp4s/$task"

    # Clusterings
    if [ -d "$src_clu/clustering" ]; then
        rsync -a --delete \
            --exclude='clustering_models.pkl' \
            --exclude='embedding_models.pkl' \
            "$src_clu" deploy/third_party/influence_visualizer/configs/
    else
        echo "WARNING: $src_clu/clustering not found — pi05 task '$task' missing from bundle."
    fi

    # Media
    if [ -d "$src_media" ]; then
        mkdir -p "$dst_mp4"
        rsync -a --delete --include='*.mp4' --exclude='*' "$src_media/" "$dst_mp4/"

        # Generate index.json with paths pointing to /tmp/study_mp4s/<task>/ (Docker standard)
        META="$ROOT/data/pi05_libero/$suite/episodes/metadata.yaml"
        PYTHON="${CONDA_PYTHON:-$(command -v python3)}"
        if [ -f "$META" ]; then
            "$PYTHON" - "$META" "$dst_mp4" "$task" <<'PYEOF'
import sys, json, pathlib, yaml

meta_path, dst_dir, task = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
with open(meta_path) as f:
    meta = yaml.safe_load(f)

episodes = []
for ep_idx, (ep_len, success) in enumerate(
    zip(meta["episode_lengths"], meta["episode_successes"])
):
    suffix = "succ" if success else "fail"
    fname = f"ep{ep_idx:04d}_{suffix}.mp4"
    mp4_path = dst_dir / fname
    if mp4_path.exists():
        episodes.append({
            "index": ep_idx,
            "path": f"/tmp/study_mp4s/{task}/{fname}",
            "frame_count": ep_len,
            "success": bool(success),
        })

with open(dst_dir / "index.json", "w") as f:
    json.dump({"episodes": episodes}, f, indent=2)
print(f"  index.json: {len(episodes)} episodes")
PYEOF
        fi

        # Local symlink so the demo works immediately without Docker
        mkdir -p /tmp/study_mp4s
        ln -sfn "$dst_mp4" "/tmp/study_mp4s/$task"
        echo "  $task: $(ls "$dst_mp4"/*.mp4 2>/dev/null | wc -l) MP4s bundled, symlinked to /tmp/study_mp4s/$task"
    else
        echo "WARNING: $src_media not found — pi05 task '$task' video panels will be empty."
    fi
done

# 4. Sanity check.
echo "== Bundle contents =="
du -sh deploy/policy_doctor deploy/third_party deploy/data 2>&1

echo
echo "== Ready to build =="
echo "  cd deploy && docker build -t policy-doctor-demo ."
