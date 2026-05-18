#!/usr/bin/env bash
# Launch 500-rollout eval for each baseline task in the background, distributing
# across the 2 GPUs. Output goes to a new eval dir with `graph_simplification`
# in the name.
#
# All logs go to /mnt/ssdB/erik/cupid_data/graph_simplification/logs/eval_500/
# (boot drive is space-limited).
#
# Usage:
#   ./scripts/launch_500_rollout_eval.sh

set -euo pipefail
WORKTREE="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_ROOT=/mnt/ssdB/erik/cupid_data/outputs/eval_save_episodes/graph_simplification
LOG_DIR=/mnt/ssdB/erik/cupid_data/graph_simplification/logs/eval_500
mkdir -p "$EVAL_ROOT" "$LOG_DIR"

N=500
CKPT=best
START_SEED=1000000  # disjoint from any prior eval-seed range (default 100000)

declare -a JOBS=(
  "transport_mh_jan28|cuda:0|/mnt/ssdB/erik/cupid_data/outputs/train/jan28/jan28_train_diffusion_unet_lowdim_transport_mh_0"
  "square_mh_feb5|cuda:1|/mnt/ssdB/erik/cupid_data/outputs/train/feb5/feb5_train_diffusion_unet_lowdim_square_mh_0"
  "lift_mh_jan26|cuda:0|/mnt/ssdB/erik/cupid_data/outputs/train/jan26/jan26_train_diffusion_unet_lowdim_lift_mh_0"
)

# Run transport and square in parallel on cuda:0/cuda:1; lift starts after the
# first one frees a GPU. We use plain wait+sequential to keep this simple — the
# user can interrupt without orphaning processes.

run_one() {
  local task="$1" device="$2" train_dir="$3"
  local out_dir="$EVAL_ROOT/${task}_n${N}"
  local log="$LOG_DIR/${task}_n${N}.log"
  if [[ -d "$out_dir" && -f "$out_dir/eval_log.json" ]]; then
    echo "[skip] $task already done -> $out_dir"
    return
  fi
  echo "[start] $task on $device -> $out_dir"
  cd "$WORKTREE/third_party/cupid"
  conda run -n mimicgen_torch2 --no-capture-output python eval_save_episodes.py \
    --output_dir "$out_dir" \
    --train_dir "$train_dir" \
    --train_ckpt "$CKPT" \
    --num_episodes "$N" \
    --test_start_seed "$START_SEED" \
    --device "$device" \
    --n_test_vis 0 \
    --overwrite false \
    > "$log" 2>&1
  echo "[done]  $task ($?) -> $log"
}

# Transport on cuda:0
(run_one transport_mh_jan28 cuda:0 /mnt/ssdB/erik/cupid_data/outputs/train/jan28/jan28_train_diffusion_unet_lowdim_transport_mh_0) &
PID_T=$!
# Square on cuda:1
(run_one square_mh_feb5 cuda:1 /mnt/ssdB/erik/cupid_data/outputs/train/feb5/feb5_train_diffusion_unet_lowdim_square_mh_0) &
PID_S=$!
# Wait for both
wait $PID_T $PID_S
echo "[transport, square] done — starting lift"
# Lift on cuda:0
run_one lift_mh_jan26 cuda:0 /mnt/ssdB/erik/cupid_data/outputs/train/jan26/jan26_train_diffusion_unet_lowdim_lift_mh_0
echo "[ALL DONE]"
