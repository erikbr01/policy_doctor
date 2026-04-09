#!/usr/bin/env bash
# Launch Policy Doctor baseline pipelines in parallel (one process per seed).
#
# Replicates the mar27 baseline setup:
#   - experiment=mar27_baseline
#   - steps=[train_baseline,eval_policies,train_attribution]
#   - unique run_name per seed (avoids pipeline run_dir collisions)
#   - seeds 0,1 -> cuda:0 ; seed 2 -> cuda:1 (two GPUs, two jobs on GPU 0)
#
# Usage:
#   ./scripts/train/launch_mar27_baseline_pipeline.sh
#   EXPERIMENT=my_exp ./scripts/train/launch_mar27_baseline_pipeline.sh
#
# Logs: /tmp/pipeline_${LOG_PREFIX}_seed{0,1,2}.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_SH="${CONDA_SH:-/home/erbauer/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-cupid}"

EXPERIMENT="${EXPERIMENT:-mar27_baseline}"
LOG_PREFIX="${LOG_PREFIX:-${EXPERIMENT}}"
STEPS="${STEPS:-[train_baseline,eval_policies,train_attribution]}"

# Seeds to run (space-separated).
SEEDS="${SEEDS:-0 1 2}"

# shellcheck disable=SC2207
SEED_ARRAY=(${SEEDS})

launch_one() {
  local seed="$1"
  local device="$2"
  local log="/tmp/pipeline_${LOG_PREFIX}_seed${seed}.log"

  echo "Launching seed=${seed} device=${device} log=${log}"

  nohup bash -c "
    set -euo pipefail
    source '${CONDA_SH}'
    conda activate '${CONDA_ENV}'
    cd '${REPO_ROOT}'
    export PYTHONPATH='${REPO_ROOT}'
    python -m policy_doctor.scripts.run_pipeline \\
      experiment=${EXPERIMENT} \\
      run_name=${EXPERIMENT}_seed${seed} \\
      seeds=[${seed}] \\
      device=${device} \\
      steps=${STEPS}
  " > "${log}" 2>&1 &
}

for seed in "${SEED_ARRAY[@]}"; do
  if [[ "${seed}" -lt 2 ]]; then
    dev="cuda:0"
  else
    dev="cuda:1"
  fi
  launch_one "${seed}" "${dev}"
done

echo "Started ${#SEED_ARRAY[@]} background pipelines. Tail logs:"
for seed in "${SEED_ARRAY[@]}"; do
  echo "  tail -f /tmp/pipeline_${LOG_PREFIX}_seed${seed}.log"
done
