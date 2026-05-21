#!/usr/bin/env bash
# Generate a K × W × S × ordering sweep of clusterings for the demo's Graph page.
#
# Two orderings are swept for each robomimic task × representation:
#
#   umap_first   (default / original):  UMAP per-timestep trunk → windowing → K-means.
#                Trunk is built once per (task, rep) and shared across all W/S/K.
#                Fast branch runs (seconds each).
#
#   agg_first    (aggregate-first):     windowing → UMAP → K-means (full pipeline).
#                Each (W, S) combo gets its own UMAP run.  Slower per combo.
#
# Output goes to:
#   data/clusterings/<task>/umap_first/<slug>/
#   data/clusterings/<task>/agg_first/<slug>/
#
# Pi05 tasks are NOT swept here — they use cluster_pi05_libero.py (agg_first only).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── Robomimic tasks only (pi05 uses a separate script) ───────────────────────
TASKS=(
    "square_mh_feb5:third_party/cupid/data/outputs/eval_save_episodes/feb5/feb5_train_diffusion_unet_lowdim_square_mh_0/latest"
    "lift_mh_jan26:third_party/cupid/data/outputs/eval_save_episodes/jan26/jan26_train_diffusion_unet_lowdim_lift_mh_0/latest"
    "transport_mh_jan28:third_party/cupid/data/outputs/eval_save_episodes/jan28/jan28_train_diffusion_unet_lowdim_transport_mh_0/latest"
)

K_VALUES=(5 10 15 20)
W_VALUES=(3 5 8)
S_VALUES=(1 2)
AGG="mean"
UMAP_N_JOBS=4

REPS=(
    "infembed:--representation=infembed"
    "trak:--representation=trak"
    "policy_emb_bottleneck_plan_t0:--representation=policy_emb:--layer=bottleneck_plan_t0"
    "state_full_history:--representation=state:--obs_strategy=full_history"
    "state_action_full_history_full_plan:--representation=state_action:--obs_strategy=full_history:--action_strategy=full_plan"
)

# Trunk cache lives in the project tree (not /tmp) so it survives reboots.
TRUNK_ROOT="$ROOT/data/sweep_trunks"
mkdir -p "$TRUNK_ROOT"

# ── Pass 1: umap_first (trunk → branch) ─────────────────────────────────────
echo "###################################################################"
echo "# PASS 1: umap_first"
echo "###################################################################"

for entry in "${TASKS[@]}"; do
    task="${entry%%:*}"
    eval_dir="${entry#*:}"
    [[ "$eval_dir" != /* ]] && eval_dir="$ROOT/$eval_dir"

    echo "================================================================"
    echo "TASK: $task  [umap_first]  (eval: $eval_dir)"
    echo "================================================================"

    if [ ! -d "$eval_dir" ]; then
        echo "  [skip] eval_dir not found: $eval_dir"
        continue
    fi

    for rep_entry in "${REPS[@]}"; do
        IFS=':' read -r -a parts <<< "$rep_entry"
        rep_id="${parts[0]}"
        rep_flags=("${parts[@]:1}")
        echo "--- rep: $rep_id ---"

        trunk_dir="$TRUNK_ROOT/${task}_${rep_id}"
        if [ ! -f "$trunk_dir/timestep_embeddings.npy" ]; then
            echo "Building trunk for $task/$rep_id ..."
            python scripts/build_alt_clustering.py \
                "${rep_flags[@]}" \
                --eval_dir "$eval_dir" \
                --out_dir "$trunk_dir" \
                --timestep_embed_only \
                --normalize none --prescale standard --reducer umap \
                --umap_n_components 50 --umap_n_jobs "$UMAP_N_JOBS" \
                --umap_init spectral || {
                    echo "  [warn] trunk build failed for $task/$rep_id; skipping rep"
                    continue
                }
        else
            echo "Trunk already exists: $trunk_dir"
        fi

        out_base="$ROOT/data/clusterings/$task/umap_first"
        mkdir -p "$out_base"

        for w in "${W_VALUES[@]}"; do
            for s in "${S_VALUES[@]}"; do
                for k in "${K_VALUES[@]}"; do
                    if [ "$w" = "5" ] && [ "$s" = "2" ]; then
                        slug="${rep_id}_seed0_kmeans_k${k}"
                    else
                        slug="${rep_id}_w${w}_s${s}_seed0_kmeans_k${k}"
                    fi
                    out_dir="$out_base/$slug"
                    if [ -f "$out_dir/cluster_labels.npy" ] \
                       && grep -q "window_width: $w" "$out_dir/manifest.yaml" 2>/dev/null \
                       && grep -q "stride: $s" "$out_dir/manifest.yaml" 2>/dev/null \
                       && grep -q "n_clusters: $k" "$out_dir/manifest.yaml" 2>/dev/null \
                       && grep -q "step: umap" "$out_dir/manifest.yaml" 2>/dev/null \
                       && python3 -c "
import yaml, sys
p = yaml.safe_load(open('$out_dir/manifest.yaml')).get('pipeline', [])
steps = [s['step'] for s in p]
sys.exit(0 if steps.index('umap') < steps.index('window') else 1)
" 2>/dev/null; then
                        echo "  [skip] $slug"
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

# ── Pass 2: agg_first (window → UMAP → K-means, full pipeline) ──────────────
echo "###################################################################"
echo "# PASS 2: agg_first"
echo "###################################################################"

for entry in "${TASKS[@]}"; do
    task="${entry%%:*}"
    eval_dir="${entry#*:}"
    [[ "$eval_dir" != /* ]] && eval_dir="$ROOT/$eval_dir"

    echo "================================================================"
    echo "TASK: $task  [agg_first]  (eval: $eval_dir)"
    echo "================================================================"

    if [ ! -d "$eval_dir" ]; then
        echo "  [skip] eval_dir not found: $eval_dir"
        continue
    fi

    for rep_entry in "${REPS[@]}"; do
        IFS=':' read -r -a parts <<< "$rep_entry"
        rep_id="${parts[0]}"
        rep_flags=("${parts[@]:1}")
        echo "--- rep: $rep_id ---"

        out_base="$ROOT/data/clusterings/$task/agg_first"
        mkdir -p "$out_base"

        for w in "${W_VALUES[@]}"; do
            for s in "${S_VALUES[@]}"; do
                for k in "${K_VALUES[@]}"; do
                    if [ "$w" = "5" ] && [ "$s" = "2" ]; then
                        slug="${rep_id}_seed0_kmeans_k${k}"
                    else
                        slug="${rep_id}_w${w}_s${s}_seed0_kmeans_k${k}"
                    fi
                    out_dir="$out_base/$slug"
                    if [ -f "$out_dir/cluster_labels.npy" ] \
                       && grep -q "window_width: $w" "$out_dir/manifest.yaml" 2>/dev/null \
                       && grep -q "stride: $s" "$out_dir/manifest.yaml" 2>/dev/null \
                       && grep -q "n_clusters: $k" "$out_dir/manifest.yaml" 2>/dev/null \
                       && grep -q "step: window" "$out_dir/manifest.yaml" 2>/dev/null \
                       && python3 -c "
import yaml, sys
p = yaml.safe_load(open('$out_dir/manifest.yaml')).get('pipeline', [])
steps = [s['step'] for s in p]
sys.exit(0 if steps.index('window') < steps.index('umap') else 1)
" 2>/dev/null; then
                        echo "  [skip] $slug"
                        continue
                    fi
                    echo "  [build] $slug"
                    python scripts/build_alt_clustering.py \
                        "${rep_flags[@]}" \
                        --eval_dir "$eval_dir" \
                        --out_dir "$out_dir" \
                        --normalize none --prescale standard --reducer umap \
                        --umap_n_components 50 --umap_n_jobs "$UMAP_N_JOBS" \
                        --umap_init spectral \
                        --window_width "$w" --stride "$s" --aggregation "$AGG" \
                        --n_clusters "$k" \
                        --task_config "$task" 2>&1 | tail -4 || {
                            echo "  [warn] agg_first build failed for $slug; skipping"
                        }
                done
            done
        done
    done
done

echo "== sweep complete =="
