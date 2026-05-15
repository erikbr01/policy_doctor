#!/usr/bin/env bash
# scripts/run_sweep_ordered.sh
#
# Run the MimicGen budget sweep (phase-2 only) in demo-count order:
#   1. d60  — seeds 0, 1, 2  (seeds 1+2 may already be running; skip_if_done handles that)
#   2. d100 — seeds 0, 1, 2
#   3. d300 — seeds 0, 1, 2
#
# Phase-1 (train_baseline ... run_clustering) must already be complete for all
# (seed, demos) combinations before this script is run.
#
# Usage:
#   nohup bash -lc "
#     source ~/miniforge3/etc/profile.d/conda.sh
#     cd /path/to/worktree
#     ./scripts/run_sweep_ordered.sh
#   " > /tmp/run_sweep_ordered.log 2>&1 &
#   echo "PID=$!"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIPE_BASE="${WORKTREE_ROOT}/third_party/cupid/data/pipeline_runs"
LOG="/tmp/run_sweep_ordered.log"
EXPERIMENT="mimicgen_square_sweep_apr26"

run_phase2() {
    local seed=$1
    local n_demos=$2
    local td="apr26_sweep_demos${n_demos}"
    local run_dir="data/pipeline_runs/mimicgen_square_apr26_sweep_seed${seed}_demos${n_demos}"
    echo "[$(date '+%H:%M PDT')] phase-2 START  seed=${seed} demos=${n_demos}" | tee -a "${LOG}"
    conda run -n policy_doctor --no-capture-output \
        python -m policy_doctor.scripts.run_pipeline \
            data_source=mimicgen_square \
            "experiment=${EXPERIMENT}" \
            "run_dir=${run_dir}" \
            "seeds=[${seed}]" \
            "train_date=${td}" \
            "evaluation.train_date=${td}" \
            "evaluation.eval_date=${td}" \
            "attribution.train_date=${td}" \
            "attribution.eval_date=${td}" \
            "baseline.max_train_episodes=${n_demos}" \
            steps='[mimicgen_budget_sweep]'
    echo "[$(date '+%H:%M PDT')] phase-2 DONE   seed=${seed} demos=${n_demos}" | tee -a "${LOG}"
}

wait_all_done() {
    local n_demos=$1
    echo "[$(date '+%H:%M PDT')] Waiting for all seeds (0 1 2) demos=${n_demos} to finish..." | tee -a "${LOG}"
    while true; do
        all_done=true
        for seed in 0 1 2; do
            local sentinel="${PIPE_BASE}/mimicgen_square_apr26_sweep_seed${seed}_demos${n_demos}/mimicgen_budget_sweep/done"
            if [ ! -f "${sentinel}" ]; then
                all_done=false
                break
            fi
        done
        ${all_done} && break
        sleep 60
    done
    echo "[$(date '+%H:%M PDT')] All seeds done for demos=${n_demos}." | tee -a "${LOG}"
}

cd "${WORKTREE_ROOT}"

echo "[$(date '+%H:%M PDT')] === Ordered sweep started ===" | tee -a "${LOG}"

# -----------------------------------------------------------------------
# d60: seed0 only — seeds 1 and 2 are already running via their own
# orchestrators; skip_if_done ensures no double-work if they finish first.
# -----------------------------------------------------------------------
echo "[$(date '+%H:%M PDT')] --- d60: launching seed0 ---" | tee -a "${LOG}"
run_phase2 0 60

# Wait for seeds 1 and 2 d60 to also finish before moving on
wait_all_done 60

# -----------------------------------------------------------------------
# d100: all seeds sequentially
# -----------------------------------------------------------------------
echo "[$(date '+%H:%M PDT')] --- d100: launching seeds 0 1 2 ---" | tee -a "${LOG}"
for seed in 0 1 2; do
    run_phase2 "${seed}" 100
done

# -----------------------------------------------------------------------
# d300: all seeds sequentially
# -----------------------------------------------------------------------
echo "[$(date '+%H:%M PDT')] --- d300: launching seeds 0 1 2 ---" | tee -a "${LOG}"
for seed in 0 1 2; do
    run_phase2 "${seed}" 300
done

echo "[$(date '+%H:%M PDT')] === All sweeps complete ===" | tee -a "${LOG}"
