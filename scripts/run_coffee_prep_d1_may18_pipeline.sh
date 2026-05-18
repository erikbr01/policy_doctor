#!/usr/bin/env bash
# scripts/run_coffee_prep_d1_may18_pipeline.sh
#
# Runs Phase 1 (pipeline) for the coffee prep D1 may18 sweep.
# Waits for the D1 pool HDF5 to be ready before starting.
# Phase 0 (pool generation) is handled separately by generate_coffee_prep_d1_pool.py.

set -euo pipefail

WORKTREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${WORKTREE}/logs/coffee_prep_d1_may18_pipeline.log"
POOL_HDF5="${WORKTREE}/third_party/cupid/data/source/mimicgen/core_datasets/coffee_preparation_d1/demo.hdf5"

mkdir -p "${WORKTREE}/logs"
cd "${WORKTREE}"

echo "[$(date '+%H:%M')] === Coffee Prep D1 May18 pipeline starting ===" | tee -a "${LOG}"

# --- Wait for pool to be ready ---
if [ ! -f "${POOL_HDF5}" ]; then
    echo "[$(date '+%H:%M')] Waiting for D1 pool HDF5: ${POOL_HDF5}" | tee -a "${LOG}"
    until [ -f "${POOL_HDF5}" ]; do
        sleep 60
        echo "[$(date '+%H:%M')] Still waiting for pool..." | tee -a "${LOG}"
    done
    echo "[$(date '+%H:%M')] Pool ready!" | tee -a "${LOG}"
fi

DEMO_COUNT=$(conda run -n policy_doctor --no-capture-output python3 -c "
import h5py
f = h5py.File('${POOL_HDF5}', 'r')
print(len([k for k in f['data'].keys() if k.startswith('demo_')]))
f.close()
" 2>/dev/null || echo "?")
echo "[$(date '+%H:%M')] Pool has ${DEMO_COUNT} demos" | tee -a "${LOG}"

# --- Phase 1a: Baseline training ---
echo "[$(date '+%H:%M')] --- Phase 1a: train_baseline ---" | tee -a "${LOG}"
conda run -n policy_doctor --no-capture-output \
    python -m policy_doctor.scripts.run_pipeline \
        data_source=mimicgen_coffee_preparation \
        "experiment=mimicgen_coffee_prep_d1_may18_d100_mug_constrained" \
        steps='[train_baseline]' \
    2>&1 | tee -a "${LOG}"

# --- Phase 1b: Evaluate baseline ---
echo "[$(date '+%H:%M')] --- Phase 1b: eval_baseline ---" | tee -a "${LOG}"
conda run -n policy_doctor --no-capture-output \
    python -m policy_doctor.scripts.run_pipeline \
        data_source=mimicgen_coffee_preparation \
        "experiment=mimicgen_coffee_prep_d1_may18_d100_mug_constrained" \
        steps='[eval_baseline]' \
    2>&1 | tee -a "${LOG}"

# --- Phase 1c: InfEmbed attribution ---
echo "[$(date '+%H:%M')] --- Phase 1c: compute_infembed ---" | tee -a "${LOG}"
conda run -n policy_doctor --no-capture-output \
    python -m policy_doctor.scripts.run_pipeline \
        data_source=mimicgen_coffee_preparation \
        "experiment=mimicgen_coffee_prep_d1_may18_d100_mug_constrained" \
        steps='[compute_infembed]' \
    2>&1 | tee -a "${LOG}"

# --- Phase 1d: Clustering ---
echo "[$(date '+%H:%M')] --- Phase 1d: run_clustering ---" | tee -a "${LOG}"
conda run -n policy_doctor --no-capture-output \
    python -m policy_doctor.scripts.run_pipeline \
        data_source=mimicgen_coffee_preparation \
        "experiment=mimicgen_coffee_prep_d1_may18_d100_mug_constrained" \
        steps='[run_clustering]' \
    2>&1 | tee -a "${LOG}"

# --- Phase 1e: Budget rep sweep (27 arms: 3 seeds × 3 budgets × 3 heuristics) ---
echo "[$(date '+%H:%M')] --- Phase 1e: mimicgen_budget_rep_sweep (27 arms, 3 slots) ---" | tee -a "${LOG}"
conda run -n policy_doctor --no-capture-output \
    python -m policy_doctor.scripts.run_pipeline \
        data_source=mimicgen_coffee_preparation \
        "experiment=mimicgen_coffee_prep_d1_may18_d100_mug_constrained" \
        steps='[mimicgen_budget_rep_sweep]' \
    2>&1 | tee -a "${LOG}"

echo "[$(date '+%H:%M')] === Coffee Prep D1 May18 pipeline complete ===" | tee -a "${LOG}"
