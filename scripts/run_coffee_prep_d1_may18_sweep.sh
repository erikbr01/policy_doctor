#!/usr/bin/env bash
# scripts/run_coffee_prep_d1_may18_sweep.sh
#
# MimicGen budget × replicate sweep — CoffeePreparation D1, May 18 2026.
# 100-demo baseline, mug tightly constrained (±40mm x/y, ±30° z_rot).
# Three fixed selection seeds (rep_seeds: [1, 2, 3]).
# Generation budgets: [100, 300, 500].
# Device: cuda:0 only, 3 concurrent slots.
#
# Phase 0: Generate D1 pool from source demos (prerequisite for baseline training).
#   100+ successes needed; targets 300 with guarantee=True.
#   Output: data/source/mimicgen/core_datasets/coffee_preparation_d1/demo.hdf5
#
# Phase 1: Full pipeline:
#   train_baseline → eval_baseline → compute_infembed → run_clustering
#   → mimicgen_budget_rep_sweep (3 seeds × 3 budgets × 3 heuristics = 27 arms)
#
# Usage:
#   nohup bash -lc "
#     source ~/miniforge3/etc/profile.d/conda.sh
#     cd /home/erbauer/refactor_cupid/policy_doctor/.claude/worktrees/feat+mimicgen-traj-pipeline
#     ./scripts/run_coffee_prep_d1_may18_sweep.sh
#   " &
#   echo "PID=$!"

set -euo pipefail

WORKTREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${WORKTREE}/logs/coffee_prep_d1_may18_sweep.log"
POOL_DIR="${WORKTREE}/third_party/cupid/data/source/mimicgen/core_datasets/coffee_preparation_d1"

mkdir -p "${WORKTREE}/logs"
cd "${WORKTREE}"

echo "[$(date '+%H:%M')] === CoffeePrep D1 May18 sweep started ===" | tee -a "${LOG}"

# ---------------------------------------------------------------------------
# Phase 0: Generate D1 pool (if not already done)
# ---------------------------------------------------------------------------
if [ -f "${POOL_DIR}/demo.hdf5" ]; then
    DEMO_COUNT=$(conda run -n policy_doctor --no-capture-output python3 -c "
import h5py
f = h5py.File('${POOL_DIR}/demo.hdf5', 'r')
print(len([k for k in f['data'].keys() if k.startswith('demo_')]))
f.close()
" 2>/dev/null || echo "0")
    echo "[$(date '+%H:%M')] Phase 0: D1 pool exists with ${DEMO_COUNT} demos — skipping generation" | tee -a "${LOG}"
else
    echo "[$(date '+%H:%M')] Phase 0: Generating CoffeePrep D1 pool (target: 300 successes) ..." | tee -a "${LOG}"
    conda run -n mimicgen_torch2 --no-capture-output \
        python scripts/generate_coffee_prep_d1_pool.py \
            --output_dir "${POOL_DIR}" \
            --n_success 300 \
        2>&1 | tee -a "${LOG}"
    echo "[$(date '+%H:%M')] Phase 0: D1 pool generation complete" | tee -a "${LOG}"
fi

# ---------------------------------------------------------------------------
# Phase 1: Baseline training
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M')] --- Phase 1a: train_baseline ---" | tee -a "${LOG}"
conda run -n policy_doctor --no-capture-output \
    python -m policy_doctor.scripts.run_pipeline \
        data_source=mimicgen_coffee_preparation \
        "experiment=mimicgen_coffee_prep_d1_may18_d100_mug_constrained" \
        steps='[train_baseline]' \
    2>&1 | tee -a "${LOG}"

# ---------------------------------------------------------------------------
# Phase 1b: Evaluate baseline
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M')] --- Phase 1b: eval_baseline ---" | tee -a "${LOG}"
conda run -n policy_doctor --no-capture-output \
    python -m policy_doctor.scripts.run_pipeline \
        data_source=mimicgen_coffee_preparation \
        "experiment=mimicgen_coffee_prep_d1_may18_d100_mug_constrained" \
        steps='[eval_baseline]' \
    2>&1 | tee -a "${LOG}"

# ---------------------------------------------------------------------------
# Phase 1c: InfEmbed attribution
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M')] --- Phase 1c: compute_infembed ---" | tee -a "${LOG}"
conda run -n policy_doctor --no-capture-output \
    python -m policy_doctor.scripts.run_pipeline \
        data_source=mimicgen_coffee_preparation \
        "experiment=mimicgen_coffee_prep_d1_may18_d100_mug_constrained" \
        steps='[compute_infembed]' \
    2>&1 | tee -a "${LOG}"

# ---------------------------------------------------------------------------
# Phase 1d: Clustering
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M')] --- Phase 1d: run_clustering ---" | tee -a "${LOG}"
conda run -n policy_doctor --no-capture-output \
    python -m policy_doctor.scripts.run_pipeline \
        data_source=mimicgen_coffee_preparation \
        "experiment=mimicgen_coffee_prep_d1_may18_d100_mug_constrained" \
        steps='[run_clustering]' \
    2>&1 | tee -a "${LOG}"

# ---------------------------------------------------------------------------
# Phase 1e: Budget × rep sweep (27 arms: 3 seeds × 3 budgets × 3 heuristics)
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M')] --- Phase 1e: mimicgen_budget_rep_sweep ---" | tee -a "${LOG}"
conda run -n policy_doctor --no-capture-output \
    python -m policy_doctor.scripts.run_pipeline \
        data_source=mimicgen_coffee_preparation \
        "experiment=mimicgen_coffee_prep_d1_may18_d100_mug_constrained" \
        steps='[mimicgen_budget_rep_sweep]' \
    2>&1 | tee -a "${LOG}"

echo "[$(date '+%H:%M')] === CoffeePrep D1 May18 sweep complete ===" | tee -a "${LOG}"
