#!/usr/bin/env bash
# Shared helpers for experiment runners (sourced by other scripts).
set -euo pipefail

experiments_lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DOCTOR_ROOT="$(cd "${experiments_lib_dir}/../.." && pwd)"

# Cupid data root: sibling "cupid" directory next to policy_doctor.
# Override with CUPID_DATA_ROOT env var if your checkout layout differs.
CUPID_DATA_ROOT="${CUPID_DATA_ROOT:-$(dirname "$POLICY_DOCTOR_ROOT")/cupid}"

# Code dir to cd into so diffusion_policy is on sys.path.
CUPID_CODE_DIR="$POLICY_DOCTOR_ROOT/third_party/cupid"

run_pipeline() {
  local env_name="${1:?conda env}"
  shift
  cd "$POLICY_DOCTOR_ROOT"
  export PYTHONPATH=.
  exec conda run -n "$env_name" --no-capture-output \
    python -m policy_doctor.scripts.run_pipeline "$@"
}

# run_monitor <output_dir> <train_dir> <ckpt> <infembed_fit> <infembed_npz>
#             <clustering_dir> <num_episodes> <device> [episodes_dir]
run_monitor() {
  local output_dir="$1" train_dir="$2" ckpt="$3" fit="$4" npz="$5"
  local clustering_dir="$6" num_eps="$7" device="$8"
  local episodes_dir="${9:-}"

  local extra_args=()
  [[ -n "$episodes_dir" ]] && extra_args+=(--episodes_dir "$episodes_dir")

  # Run from CUPID_DATA_ROOT so checkpoint-embedded relative paths (e.g.
  # data/robomimic/datasets/...) resolve correctly; PYTHONPATH brings in
  # diffusion_policy from the code-only third_party/cupid checkout.
  cd "$CUPID_DATA_ROOT"
  PYTHONPATH="$POLICY_DOCTOR_ROOT:$CUPID_CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  exec conda run -n cupid --no-capture-output \
    python "$POLICY_DOCTOR_ROOT/scripts/monitor_online.py" \
      --output_dir "$output_dir" \
      --train_dir "$train_dir" \
      --train_ckpt "$ckpt" \
      --infembed_fit "$fit" \
      --infembed_npz "$npz" \
      --clustering_dir "$clustering_dir" \
      --num_episodes "$num_eps" \
      --device "$device" \
      --verbose \
      "${extra_args[@]}"
}

# run_dagger <task> <output_dir> <train_dir> <ckpt> <infembed_fit> <infembed_npz>
#            <clustering_dir> <num_episodes> <device> <dagger_config>
run_dagger() {
  local task="$1" output_dir="$2" train_dir="$3" ckpt="$4" fit="$5" npz="$6"
  local clustering_dir="$7" num_eps="$8" device="$9" dagger_cfg="${10}"

  cd "$CUPID_DATA_ROOT"
  PYTHONPATH="$POLICY_DOCTOR_ROOT:$CUPID_CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  exec conda run -n cupid --no-capture-output \
    python "$POLICY_DOCTOR_ROOT/scripts/run_dagger.py" \
      --task "$task" \
      --output_dir "$output_dir" \
      --train_dir "$train_dir" \
      --train_ckpt "$ckpt" \
      --infembed_fit "$fit" \
      --infembed_npz "$npz" \
      --clustering_dir "$clustering_dir" \
      --num_episodes "$num_eps" \
      --device "$device" \
      --dagger_config "$dagger_cfg"
}
