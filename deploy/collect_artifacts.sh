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
        rsync -aL --delete \
            --exclude='clustering_models.pkl' \
            --exclude='embedding_models.pkl' \
            "$src_clu" deploy/third_party/influence_visualizer/configs/
    else
        echo "ERROR: $src_clu/clustering not found — task '$task' missing from bundle." >&2; exit 1
    fi
    src_mp4="$ROOT/data/study_mp4s/$task"
    if [ -d "$src_mp4" ] && [ -f "$src_mp4/index.json" ]; then
        mkdir -p "deploy/data/study_mp4s/$task"
        rsync -a --delete "$src_mp4/" "deploy/data/study_mp4s/$task/"
        mkdir -p /tmp/study_mp4s
        ln -sfn "$src_mp4" "/tmp/study_mp4s/$task"
        echo "  $task: $(ls "$src_mp4"/*.mp4 2>/dev/null | wc -l) MP4s bundled"
    else
        echo "ERROR: $src_mp4 not found or missing index.json — task '$task' video panels would be empty." >&2; exit 1
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
        rsync -aL --delete \
            --exclude='clustering_models.pkl' \
            --exclude='embedding_models.pkl' \
            "$src_clu" deploy/third_party/influence_visualizer/configs/
    else
        echo "ERROR: $src_clu/clustering not found — pi05 task '$task' missing from bundle." >&2; exit 1
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
        else
            echo "ERROR: $META not found — cannot generate index.json for '$task'." >&2; exit 1
        fi

        # Local symlink so the demo works immediately without Docker
        mkdir -p /tmp/study_mp4s
        ln -sfn "$dst_mp4" "/tmp/study_mp4s/$task"
        echo "  $task: $(ls "$dst_mp4"/*.mp4 2>/dev/null | wc -l) MP4s bundled, symlinked to /tmp/study_mp4s/$task"
    else
        echo "ERROR: $src_media not found — pi05 task '$task' video panels would be empty." >&2; exit 1
    fi
done

# 4. Data-support clusterings: graph_demo discovers tasks from
#    data/clusterings/<task>/<prep_mode>/<cid>/. Symlinks point at SSD —
#    rsync -L dereferences so the data is baked into the image. Exclude
#    bulky artifacts the demo doesn't load.
if [ -d "data/clusterings" ]; then
    mkdir -p deploy/data/clusterings
    rsync -aL --delete \
        --exclude='clustering_models.pkl' \
        --exclude='embedding_models.pkl' \
        --exclude='joint_umap.joblib' \
        data/clusterings/ deploy/data/clusterings/
    echo "  data/clusterings: $(find deploy/data/clusterings -name 'cluster_labels.npy' 2>/dev/null | wc -l) clusterings bundled"
else
    echo "WARNING: data/clusterings/ not found — data-support feature will be empty in the demo."
fi

# 5. Demo sweep clustering results (data/demo_sweep/<task>/run_clustering/clustering/).
DEMO_SWEEP_TASKS=(
    transport_mh_jan28
    square_mh_feb5
    lift_mh_jan26
    pi05_libero_spatial
    pi05_libero_object
    pi05_libero_goal
    kendama_may22
)
mkdir -p deploy/data/demo_sweep
for task in "${DEMO_SWEEP_TASKS[@]}"; do
    task_dir="$ROOT/data/demo_sweep/$task"
    clu_dir="$task_dir/run_clustering/clustering"
    if [ -d "$clu_dir" ] && [ -n "$(find "$clu_dir" -name "cluster_labels.npy" -print -quit 2>/dev/null)" ]; then
        rsync -aL --delete \
            --exclude='clustering_models.pkl' \
            --exclude='embedding_models.pkl' \
            --exclude='joint_umap.joblib' \
            --exclude='_trunks' \
            "$task_dir" deploy/data/demo_sweep/
        echo "  $task: $(find "$clu_dir" -name 'cluster_labels.npy' | wc -l) clusterings bundled"
    else
        echo "ERROR: data/demo_sweep/$task has no clusterings — run the sweep first." >&2; exit 1
    fi
done

# 6. Kendama rollout MP4s.
#    Source: /mnt/ssdB/erik/rollouts/rollouts_kendama_latest_may19/*/exterior.mp4
#    The cluster_kendama_rollouts.py script already created symlinks in
#    /tmp/study_mp4s/kendama_may22/ — here we copy the actual MP4 bytes into
#    deploy/data/study_mp4s/kendama_may22/ so they're self-contained in the image.
KENDAMA_ROLLOUTS_SRC="/mnt/ssdB/erik/rollouts/rollouts_kendama_latest_may19"
KENDAMA_MP4_SRC="/tmp/study_mp4s/kendama_may22"   # populated by cluster_kendama_rollouts.py
KENDAMA_MP4_DST="$ROOT/deploy/data/study_mp4s/kendama_may22"

if [ ! -d "$KENDAMA_MP4_SRC" ] || [ ! -f "$KENDAMA_MP4_SRC/index.json" ]; then
    echo "→ Kendama MP4s not found in $KENDAMA_MP4_SRC — running cluster_kendama_rollouts.py"
    conda run -n policy_doctor python "$ROOT/scripts/cluster_kendama_rollouts.py" \
        --rollouts "$KENDAMA_ROLLOUTS_SRC" \
        --out_dir  "data/demo_sweep/kendama_may22/run_clustering/clustering/state_w20_s10_k8" \
        --mp4_out  "$KENDAMA_MP4_SRC"
fi

if [ -f "$KENDAMA_MP4_SRC/index.json" ]; then
    mkdir -p "$KENDAMA_MP4_DST"
    # rsync -L dereferences symlinks so actual video bytes land in deploy/
    rsync -aL --delete \
        --include='*.mp4' --include='index.json' --exclude='*' \
        "$KENDAMA_MP4_SRC/" "$KENDAMA_MP4_DST/"
    echo "  kendama_may22: $(ls "$KENDAMA_MP4_DST"/*.mp4 2>/dev/null | wc -l) MP4s bundled"
else
    echo "WARNING: kendama_may22 MP4s not found — study page will be video-less."
fi

# 7. Sanity check.
echo "== Bundle contents =="
du -sh deploy/policy_doctor deploy/third_party deploy/data 2>&1

echo
echo "== Ready to build =="
echo "  cd deploy && docker build -t policy-doctor-demo ."
