#!/usr/bin/env bash
# Batch-compute data_support.json for every policy_emb clustering across
# the three robomimic demo_sweep tasks. Extracts demo embeddings for any
# task that doesn't have them yet (mimicgen_torch2 env, GPU), then runs
# scripts/batch_data_support.py for each task (policy_doctor env).
#
# Logs land under logs/data_support/ (which symlinks to /mnt/ssdB).
#
# Usage:
#   ./scripts/run_batch_data_support_robomimic.sh           # all three tasks
#   ./scripts/run_batch_data_support_robomimic.sh transport_mh_jan28
#
# Env knobs:
#   GPU=1                       # which CUDA device to use for demo extraction
#   RADIUS=1.0
#   KNN_K=10
#   UMAP_N_COMPONENTS=10
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/logs/data_support"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_DIR/batch_robomimic_${TS}.log"
echo "Writing master log to: $RUN_LOG"

GPU="${GPU:-1}"
RADIUS="${RADIUS:-1.0}"
KNN_K="${KNN_K:-10}"
UMAP_N_COMPONENTS="${UMAP_N_COMPONENTS:-10}"

# Resolve task list.
ALL_TASKS=(transport_mh_jan28 lift_mh_jan26 square_mh_feb5)
if [ $# -gt 0 ]; then
    TASKS=("$@")
else
    TASKS=("${ALL_TASKS[@]}")
fi

# Map task → train_subdir (must match TASK_REGISTRY in batch_data_support.py).
declare -A TRAIN_SUBDIR=(
    [transport_mh_jan28]="jan28/jan28_train_diffusion_unet_lowdim_transport_mh_0"
    [lift_mh_jan26]="jan26/jan26_train_diffusion_unet_lowdim_lift_mh_0"
    [square_mh_feb5]="feb5/feb5_train_diffusion_unet_lowdim_square_mh_0"
)

CUPID_ROOT="$ROOT/third_party/cupid"
LAYER="bottleneck_plan_t0"

# ── Step 1 — Demo embedding extraction (mimicgen_torch2 env, GPU) ──────────────
for task in "${TASKS[@]}"; do
    train_subdir="${TRAIN_SUBDIR[$task]:-}"
    if [ -z "$train_subdir" ]; then
        echo "Unknown task: $task" | tee -a "$RUN_LOG"
        exit 1
    fi
    train_dir="$CUPID_ROOT/data/outputs/train/$train_subdir"
    emb_path="$train_dir/policy_embeddings_demos/$LAYER.npz"

    if [ -f "$emb_path" ]; then
        echo "[$(date +%T)] $task — demo embeddings already exist, skipping extraction" | tee -a "$RUN_LOG"
        continue
    fi

    sub_log="$LOG_DIR/demo_emb_${task}_${TS}.log"
    echo "[$(date +%T)] $task — extracting demo embeddings (GPU $GPU) → $sub_log" | tee -a "$RUN_LOG"

    (
        cd "$CUPID_ROOT"
        conda run -n mimicgen_torch2 --no-capture-output python compute_policy_embeddings_demos.py \
            --train_dir "$train_dir" \
            --train_ckpt latest \
            --layer "$LAYER" \
            --batch_size 128 \
            --device "cuda:$GPU"
    ) >"$sub_log" 2>&1 && echo "  done." | tee -a "$RUN_LOG" \
        || { echo "  FAILED — see $sub_log" | tee -a "$RUN_LOG"; exit 2; }
done

# ── Step 2 — Batch data_support iteration (policy_doctor env) ─────────────────
for task in "${TASKS[@]}"; do
    sub_log="$LOG_DIR/batch_data_support_${task}_${TS}.log"
    echo "[$(date +%T)] $task — batch data_support → $sub_log" | tee -a "$RUN_LOG"

    conda run -n policy_doctor --no-capture-output python scripts/batch_data_support.py \
        --task "$task" \
        --radius "$RADIUS" \
        --knn_k "$KNN_K" \
        --umap_n_components "$UMAP_N_COMPONENTS" \
        >"$sub_log" 2>&1 && echo "  done." | tee -a "$RUN_LOG" \
        || { echo "  FAILED — see $sub_log" | tee -a "$RUN_LOG"; exit 3; }
done

echo "[$(date +%T)] All done." | tee -a "$RUN_LOG"
