#!/usr/bin/env bash
# Generate a K × W × S sweep of clusterings for the demo's Graph page.
#
# For each task × representation, builds the UMAP "trunk" once (slow),
# then runs the windowing+kmeans "branch" for every (W, S, K) combo
# (seconds each). Output goes directly into the
# third_party/influence_visualizer/configs/<task>/clustering/ tree so
# collect_artifacts.sh picks it up automatically.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TASKS=(
    "square_mh_feb5:/Users/erik/stanford/asl_rotation/cupid/data/outputs/eval_save_episodes/feb5/feb5_train_diffusion_unet_lowdim_square_mh_0/latest"
    "lift_mh_jan26:/Users/erik/stanford/asl_rotation/cupid/data/outputs/eval_save_episodes/jan26/jan26_train_diffusion_unet_lowdim_lift_mh_0/latest"
    "transport_mh_jan28:/Users/erik/stanford/asl_rotation/cupid/data/outputs/eval_save_episodes/jan28/jan28_train_diffusion_unet_lowdim_transport_mh_0/latest"
)

LAYER="bottleneck_plan_t0"
REP="policy_emb"
K_VALUES=(5 10 15 20)
W_VALUES=(3 5 8)
S_VALUES=(1 2)
AGG="mean"

TRUNK_ROOT="/tmp/sweep_trunks"
mkdir -p "$TRUNK_ROOT"

for entry in "${TASKS[@]}"; do
    task="${entry%%:*}"
    eval_dir="${entry##*:}"
    echo "================================================================"
    echo "TASK: $task"
    echo "================================================================"

    trunk_dir="$TRUNK_ROOT/${task}_${LAYER}"
    if [ ! -f "$trunk_dir/timestep_embeddings.npy" ]; then
        echo "Building trunk for $task/$LAYER..."
        python scripts/build_alt_clustering.py \
            --representation "$REP" \
            --layer "$LAYER" \
            --eval_dir "$eval_dir" \
            --out_dir "$trunk_dir" \
            --timestep_embed_only \
            --normalize none --prescale standard --reducer umap \
            --umap_n_components 50 --umap_n_jobs -1
    else
        echo "Trunk already exists: $trunk_dir"
    fi

    iv_dir="$ROOT/third_party/influence_visualizer/configs/$task/clustering"
    mkdir -p "$iv_dir"

    for w in "${W_VALUES[@]}"; do
        for s in "${S_VALUES[@]}"; do
            for k in "${K_VALUES[@]}"; do
                # Preserve existing naming (no w/s suffix) for the W=5,S=2
                # default that the user-study session yamls reference.
                if [ "$w" = "5" ] && [ "$s" = "2" ]; then
                    slug="policy_emb_${LAYER}_seed0_kmeans_k${k}"
                else
                    slug="policy_emb_${LAYER}_w${w}_s${s}_seed0_kmeans_k${k}"
                fi
                out_dir="$iv_dir/$slug"
                if [ -f "$out_dir/cluster_labels.npy" ] \
                   && grep -q "window_width: $w" "$out_dir/manifest.yaml" 2>/dev/null \
                   && grep -q "stride: $s" "$out_dir/manifest.yaml" 2>/dev/null \
                   && grep -q "n_clusters: $k" "$out_dir/manifest.yaml" 2>/dev/null; then
                    echo "  [skip] $slug (already up-to-date)"
                    continue
                fi
                echo "  [build] $slug"
                python scripts/build_alt_clustering.py \
                    --representation "$REP" \
                    --layer "$LAYER" \
                    --eval_dir "$eval_dir" \
                    --out_dir "$out_dir" \
                    --timestep_embed_dir "$trunk_dir" \
                    --window_width "$w" --stride "$s" --aggregation "$AGG" \
                    --n_clusters "$k" \
                    --task_config "$task" 2>&1 | tail -4
            done
        done
    done
done

echo "== sweep complete =="
