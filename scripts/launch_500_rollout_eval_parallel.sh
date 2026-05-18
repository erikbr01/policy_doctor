#!/usr/bin/env bash
# Parallel 500-rollout eval. Static chunk plan (no bash array gymnastics).
# Existing episodes in <task>_n500/episodes/ are preserved (the original
# serial process used seed=1000000+i; chunks below start at 2_000_000+ to
# stay disjoint).
#
# After all chunks complete, run scripts/merge_eval_chunks.py to assemble
# the final <task>_n500/episodes/ with sequential filenames (symlinks).

set -uo pipefail
WORKTREE="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_ROOT=/mnt/ssdB/erik/cupid_data/outputs/eval_save_episodes/graph_simplification
LOG_DIR=/mnt/ssdB/erik/cupid_data/graph_simplification/logs/eval_500_parallel
mkdir -p "$EVAL_ROOT" "$LOG_DIR"

CKPT=best

# Static chunk plan. Each entry: task | gpu | seed | n_ep | chunk_id | train_dir
# Distribute: 11 chunks (4 transport, 3 square, 4 lift); roughly balanced
# across 2 GPUs (6 on cuda:0, 5 on cuda:1).
TRANSPORT_TRAIN=/mnt/ssdB/erik/cupid_data/outputs/train/jan28/jan28_train_diffusion_unet_lowdim_transport_mh_0
SQUARE_TRAIN=/mnt/ssdB/erik/cupid_data/outputs/train/feb5/feb5_train_diffusion_unet_lowdim_square_mh_0
LIFT_TRAIN=/mnt/ssdB/erik/cupid_data/outputs/train/jan26/jan26_train_diffusion_unet_lowdim_lift_mh_0

# Order: alternate gpu:0 / gpu:1 so launch ordering balances startup load.
JOBS=(
  "transport_mh_jan28|cuda:0|2000000|105|chunk0|$TRANSPORT_TRAIN"
  "transport_mh_jan28|cuda:1|2200000|105|chunk1|$TRANSPORT_TRAIN"
  "transport_mh_jan28|cuda:0|2400000|105|chunk2|$TRANSPORT_TRAIN"
  "transport_mh_jan28|cuda:1|2600000|102|chunk3|$TRANSPORT_TRAIN"
  "square_mh_feb5|cuda:0|3000000|94|chunk0|$SQUARE_TRAIN"
  "square_mh_feb5|cuda:1|3200000|94|chunk1|$SQUARE_TRAIN"
  "square_mh_feb5|cuda:0|3400000|94|chunk2|$SQUARE_TRAIN"
  "lift_mh_jan26|cuda:1|4000000|125|chunk0|$LIFT_TRAIN"
  "lift_mh_jan26|cuda:0|4200000|125|chunk1|$LIFT_TRAIN"
  "lift_mh_jan26|cuda:1|4400000|125|chunk2|$LIFT_TRAIN"
  "lift_mh_jan26|cuda:0|4600000|125|chunk3|$LIFT_TRAIN"
)

echo "=== Launching ${#JOBS[@]} parallel eval chunks ==="
PIDS=()
for j in "${JOBS[@]}"; do
  IFS='|' read -r task device seed n_ep chunk train_dir <<< "$j"
  out_dir="$EVAL_ROOT/${task}_n500_${chunk}"
  log="$LOG_DIR/${task}_${chunk}.log"
  if [[ -d "$out_dir" ]] && [[ -d "$out_dir/episodes" ]] && [[ "$(ls -1 $out_dir/episodes/ 2>/dev/null | wc -l)" -ge "$n_ep" ]]; then
    echo "[skip] $out_dir (already has ≥ $n_ep eps)"
    continue
  fi
  echo "[launch] $task $chunk on $device  (seed=$seed n_ep=$n_ep)"
  (
    cd "$WORKTREE/third_party/cupid"
    conda run -n mimicgen_torch2 --no-capture-output python eval_save_episodes.py \
      --output_dir "$out_dir" \
      --train_dir "$train_dir" \
      --train_ckpt "$CKPT" \
      --num_episodes "$n_ep" \
      --test_start_seed "$seed" \
      --device "$device" \
      --n_test_vis 0 \
      --overwrite false \
      > "$log" 2>&1
    echo "[done] $task $chunk (rc=$?)"
  ) &
  PIDS+=($!)
done

echo "=== Launched ${#PIDS[@]} background processes ==="
for pid in "${PIDS[@]}"; do
  wait "$pid"
done
echo "=== ALL CHUNKS DONE ==="
