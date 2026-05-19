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

# Task entries: "task_name:eval_dir[:rep_override_1:rep_override_2:...]"
# If rep_overrides are given, only those reps are swept for that task.
# eval_dir can be absolute or relative to ROOT. Tasks with missing eval_dirs are skipped.
TASKS=(
    # Diffusion-policy tasks (eval dirs are on the original dev machine; skip gracefully if absent)
    "square_mh_feb5:third_party/cupid/data/outputs/eval_save_episodes/feb5/feb5_train_diffusion_unet_lowdim_square_mh_0/latest"
    "lift_mh_jan26:third_party/cupid/data/outputs/eval_save_episodes/jan26/jan26_train_diffusion_unet_lowdim_lift_mh_0/latest"
    "transport_mh_jan28:third_party/cupid/data/outputs/eval_save_episodes/jan28/jan28_train_diffusion_unet_lowdim_transport_mh_0/latest"
    # pi0.5-LIBERO tasks — uses policy_emb only (no infembed/trak/state embeddings available)
    "pi05_libero_spatial:data/pi05_libero/libero_spatial:policy_emb_bottleneck_plan_t0"
    "pi05_libero_object:data/pi05_libero/libero_object:policy_emb_bottleneck_plan_t0"
    "pi05_libero_goal:data/pi05_libero/libero_goal:policy_emb_bottleneck_plan_t0"
)

K_VALUES=(5 10 15 20)
W_VALUES=(3 5 8)
S_VALUES=(1 2)
AGG="mean"

# Representations to sweep. Format: "rep_id:flag1=val1:flag2=val2:..."
# rep_id is what appears in the slug; the flags are passed to
# build_alt_clustering.py.
REPS=(
    "infembed:--representation=infembed"
    "trak:--representation=trak"
    "policy_emb_bottleneck_plan_t0:--representation=policy_emb:--layer=bottleneck_plan_t0"
    "state_full_history:--representation=state:--obs_strategy=full_history"
    "state_action_full_history_full_plan:--representation=state_action:--obs_strategy=full_history:--action_strategy=full_plan"
)

# Per-task rep override: map task name to space-separated list of rep_ids to run.
# Used for pi05 tasks that only have policy_emb embeddings.
declare -A TASK_REP_OVERRIDE
TASK_REP_OVERRIDE["pi05_libero_spatial"]="policy_emb_bottleneck_plan_t0"
TASK_REP_OVERRIDE["pi05_libero_object"]="policy_emb_bottleneck_plan_t0"
TASK_REP_OVERRIDE["pi05_libero_goal"]="policy_emb_bottleneck_plan_t0"

# pi05 uses --layer=pi05 (loads policy_embeddings/pi05.npz) instead of --layer=bottleneck_plan_t0.
declare -A TASK_LAYER_OVERRIDE
TASK_LAYER_OVERRIDE["pi05_libero_spatial"]="pi05"
TASK_LAYER_OVERRIDE["pi05_libero_object"]="pi05"
TASK_LAYER_OVERRIDE["pi05_libero_goal"]="pi05"

TRUNK_ROOT="/tmp/sweep_trunks"
mkdir -p "$TRUNK_ROOT"

for entry in "${TASKS[@]}"; do
    task="${entry%%:*}"
    rest="${entry#*:}"
    eval_dir="${rest%%:*}"

    # Resolve relative eval_dir against ROOT
    if [[ "$eval_dir" != /* ]]; then
        eval_dir="$ROOT/$eval_dir"
    fi

    echo "================================================================"
    echo "TASK: $task  (eval: $eval_dir)"
    echo "================================================================"

    # Skip tasks whose eval dir doesn't exist
    if [ ! -d "$eval_dir" ]; then
        echo "  [skip] eval_dir not found: $eval_dir"
        continue
    fi

    # Determine which reps to run for this task
    rep_filter="${TASK_REP_OVERRIDE[$task]:-}"  # empty = run all reps
    layer_override="${TASK_LAYER_OVERRIDE[$task]:-}"

    for rep_entry in "${REPS[@]}"; do
        IFS=':' read -r -a parts <<< "$rep_entry"
        rep_id="${parts[0]}"

        # Apply per-task rep filter
        if [ -n "$rep_filter" ] && [[ " $rep_filter " != *" $rep_id "* ]]; then
            continue
        fi

        rep_flags=()
        for ((i=1; i<${#parts[@]}; i++)); do
            flag="${parts[i]}"
            # Apply per-task layer override (e.g. --layer=pi05 instead of --layer=bottleneck_plan_t0)
            if [ -n "$layer_override" ] && [[ "$flag" == --layer=* ]]; then
                flag="--layer=$layer_override"
            fi
            rep_flags+=("$flag")
        done
        slug_base="$rep_id"
        echo "--- rep: $rep_id ---"

        trunk_dir="$TRUNK_ROOT/${task}_${rep_id}"
        if [ ! -f "$trunk_dir/timestep_embeddings.npy" ]; then
            echo "Building trunk for $task/$rep_id..."
            python scripts/build_alt_clustering.py \
                "${rep_flags[@]}" \
                --eval_dir "$eval_dir" \
                --out_dir "$trunk_dir" \
                --timestep_embed_only \
                --normalize none --prescale standard --reducer umap \
                --umap_n_components 50 --umap_n_jobs -1 || {
                    echo "  [warn] trunk build failed for $task/$rep_id; skipping rep"
                    continue   # skip only this rep, not the whole task
                }
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
                        slug="${slug_base}_seed0_kmeans_k${k}"
                    else
                        slug="${slug_base}_w${w}_s${s}_seed0_kmeans_k${k}"
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
                        "${rep_flags[@]}" \
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
done

echo "== sweep complete =="
