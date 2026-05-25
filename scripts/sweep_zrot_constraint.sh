#!/bin/bash
# Find the tightest z_rot constraint where MimicGen replay still succeeds.
# Runs run_mimicgen_generate.py with varying z_rot ranges on one kitchen seed,
# reports success/attempt ratio per range.
set -e
export HDF5_USE_FILE_LOCKING=FALSE
CUPID="/home/erbauer/policy_doctor/third_party/cupid"
SEED="$CUPID/data/pipeline_runs/mimicgen_kitchen_d1_may20_seed1_d100_bread_constrained/mimicgen_budget_rep_sweep/mimicgen_random_budget100_rep1/select_mimicgen_seed/seed.hdf5"
NUM_TRIALS=20
BASE_OUT="/tmp/zrot_sweep"
mkdir -p "$BASE_OUT"

run_one() {
  local label="$1"
  local z_rot_json="$2"
  local out="$BASE_OUT/$label"
  rm -rf "$out"
  mkdir -p "$out"
  local ranges="{\"bread\":{\"x\":[-0.04,0.04],\"y\":[-0.04,0.04],\"z_rot\":$z_rot_json},\"pot\":{\"x\":null,\"y\":null,\"z_rot\":null},\"stove\":{\"x\":null,\"y\":null,\"z_rot\":null}}"
  echo ""
  echo "=== z_rot=$label ranges=$ranges ==="
  conda run -n mimicgen_torch2 --no-capture-output python "$CUPID/../../scripts/run_mimicgen_generate.py" \
    --seed_hdf5 "$SEED" --output_dir "$out" \
    --task_name kitchen --env_interface_name MG_Kitchen --env_interface_type robosuite \
    --num_trials "$NUM_TRIALS" --action_noise 0.05 \
    --subtask_term_offset_lo 0 --subtask_term_offset_hi 0 --nn_k 1 \
    --num_interpolation_steps 5 --num_fixed_steps 0 \
    --fix_initial_object_poses \
    --object_pose_ranges "$ranges" \
    > "$out/log.txt" 2>&1 || true
  local n_succ=$(grep -oE '"num_success"[: ]+[0-9]+' "$out/stats.json" 2>/dev/null | head -1 | awk '{print $NF}')
  local n_att=$(grep -oE '"num_attempts"[: ]+[0-9]+' "$out/stats.json" 2>/dev/null | head -1 | awk '{print $NF}')
  echo "  z_rot=$label  →  $n_succ / $n_att"
}

run_one "null"  "null"
run_one "0.05"  "[-0.05,0.05]"
run_one "0.10"  "[-0.10,0.10]"
run_one "0.20"  "[-0.20,0.20]"
run_one "0.35"  "[-0.35,0.35]"
run_one "0.524" "[-0.524,0.524]"

echo ""
echo "=== Summary ==="
for label in null 0.05 0.10 0.20 0.35 0.524; do
  s="$BASE_OUT/$label/stats.json"
  if [ -f "$s" ]; then
    n_succ=$(grep -oE '"num_success"[: ]+[0-9]+' "$s" 2>/dev/null | head -1 | awk '{print $NF}')
    n_att=$(grep -oE '"num_attempts"[: ]+[0-9]+' "$s" 2>/dev/null | head -1 | awk '{print $NF}')
    pct=$(python3 -c "print(f'{int('$n_succ')/int('$n_att')*100:.0f}%' if int('$n_att')>0 else 'NA')" 2>/dev/null)
    printf "  z_rot=%-8s %s/%s  (%s)\n" "$label" "$n_succ" "$n_att" "$pct"
  else
    printf "  z_rot=%-8s NO STATS\n" "$label"
  fi
done
