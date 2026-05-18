#!/usr/bin/env bash
# Post-500-rollout-eval pipeline. Run this once the 500-rollout eval
# (launch_500_rollout_eval.sh) has finished for all three baseline tasks.
#
# Steps per task:
#   1. Extract policy_emb_bottleneck_plan_t0 on the new 500-rollout eval dir.
#   2. Build a UMAP-50 trunk over the new per-timestep features.
#   3. Run the rollout-budget sweep at N ∈ {20, 50, 100, 200, 300, 400, 500}
#      and K ∈ {5, 10, 15, 20} with 5 subsample seeds per cell.
#   4. Re-summarise (overwrites docs/rollout_budget_results/).
#
# All artifacts on /mnt/ssdB; logs co-located.
#
# Usage:
#   ./scripts/post_eval_500_pipeline.sh

set -euo pipefail
WORKTREE="$(cd "$(dirname "$0")/.." && pwd)"
SSD_ROOT=/mnt/ssdB/erik/cupid_data/graph_simplification
TRUNKS_DIR=$SSD_ROOT/trunks_500
LOG_DIR=$SSD_ROOT/logs/post_eval_500
mkdir -p "$TRUNKS_DIR" "$LOG_DIR"

EVAL_ROOT=/mnt/ssdB/erik/cupid_data/outputs/eval_save_episodes/graph_simplification

declare -a TASKS=(
  "transport_mh_jan28:cuda:0:/mnt/ssdB/erik/cupid_data/outputs/train/jan28/jan28_train_diffusion_unet_lowdim_transport_mh_0"
  "square_mh_feb5:cuda:1:/mnt/ssdB/erik/cupid_data/outputs/train/feb5/feb5_train_diffusion_unet_lowdim_square_mh_0"
  "lift_mh_jan26:cuda:0:/mnt/ssdB/erik/cupid_data/outputs/train/jan26/jan26_train_diffusion_unet_lowdim_lift_mh_0"
)

merge_chunks() {
  echo "[merge chunks]"
  cd "$WORKTREE"
  PYTHONPATH=. conda run -n policy_doctor --no-capture-output python scripts/merge_eval_chunks.py \
    > "$LOG_DIR/merge.log" 2>&1
}

extract_policy_emb() {
  local task="$1" device="$2" train_dir="$3"
  local eval_dir="$EVAL_ROOT/${task}_n500"
  if [[ ! -d "$eval_dir/episodes" ]] || [[ "$(ls -1 $eval_dir/episodes | wc -l)" -lt 500 ]]; then
    echo "[skip extract] $task: eval not complete (need 500 episodes)"
    return 1
  fi
  if [[ -f "$eval_dir/policy_embeddings/bottleneck_plan_t0.npz" ]]; then
    echo "[skip extract] $task: policy_emb already extracted"
    return 0
  fi
  echo "[extract] $task -> $eval_dir/policy_embeddings/"
  cd "$WORKTREE/third_party/cupid"
  conda run -n cupid_torch2 --no-capture-output python compute_policy_embeddings.py \
    --train_dir "$train_dir" --eval_dir "$eval_dir" \
    --layer bottleneck_plan_t0 --batch_size 128 --device "$device" \
    > "$LOG_DIR/${task}_extract.log" 2>&1
}

build_trunk() {
  local task="$1"
  local eval_dir="$EVAL_ROOT/${task}_n500"
  local out_dir="$TRUNKS_DIR/${task}__policy_emb_n500"
  if [[ -f "$out_dir/embed_manifest.yaml" ]]; then
    echo "[skip trunk] $task: already built"
    return 0
  fi
  echo "[trunk] $task -> $out_dir"
  cd "$WORKTREE"
  conda run -n policy_doctor --no-capture-output python scripts/build_alt_clustering.py \
    --representation policy_emb \
    --eval_dir "$eval_dir" \
    --layer bottleneck_plan_t0 \
    --prescale standard \
    --umap_n_components 50 \
    --seed 42 \
    --timestep_embed_only \
    --out_dir "$out_dir" \
    > "$LOG_DIR/${task}_trunk.log" 2>&1
}

run_budget_sweep() {
  local task="$1"
  local trunk="$TRUNKS_DIR/${task}__policy_emb_n500"
  if [[ ! -f "$trunk/embed_manifest.yaml" ]]; then
    echo "[skip budget] $task: trunk missing"
    return 1
  fi
  echo "[budget sweep] $task"
  cd "$WORKTREE"
  PYTHONPATH=. conda run -n policy_doctor --no-capture-output python scripts/run_rollout_budget_sweep.py \
    --task_label "${task}__policy_emb_n500" \
    --trunk "$trunk" \
    --budgets 20 50 100 200 300 400 500 \
    --ks 5 10 15 20 \
    --n_subsample_seeds 5 \
    --n_jobs 8 \
    > "$LOG_DIR/${task}_budget.log" 2>&1
}

# 0. Merge parallel chunks into the main <task>_n500/episodes/ dir.
merge_chunks

# 1. Extract policy_emb for each task (parallel where possible).
for entry in "${TASKS[@]}"; do
  IFS=':' read -r task device train_dir <<< "$entry"
  extract_policy_emb "$task" "$device" "$train_dir" || true
done

# 2. Build trunks (serial — UMAP is RAM-heavy).
for entry in "${TASKS[@]}"; do
  IFS=':' read -r task device train_dir <<< "$entry"
  build_trunk "$task" || true
done

# 3. Run budget sweep per task (sequential, each is CPU-bound and fast).
for entry in "${TASKS[@]}"; do
  IFS=':' read -r task device train_dir <<< "$entry"
  run_budget_sweep "$task" || true
done

# 4. Re-summarise.
echo "[summarise]"
cd "$WORKTREE"
PYTHONPATH=. conda run -n policy_doctor --no-capture-output python scripts/summarize_rollout_budget.py

echo "[DONE]"
