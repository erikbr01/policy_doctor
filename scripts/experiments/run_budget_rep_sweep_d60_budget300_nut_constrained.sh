#!/usr/bin/env bash
# scripts/run_budget_rep_sweep_d60_budget300_nut_constrained.sh
#
# Run the MimicGen budget × replicate sweep for the 60-demo baseline with
# nut-constrained initial poses and generation budget=300.
#
# Identical to run_budget_rep_sweep_d60_nut_constrained.sh except the
# generation budget is raised from 100 to 300 episodes per arm.
#
# Reuses the same upstream d60 seed=1 run_clustering (via symlink in the
# new constrained run dir) and the same baseline rollouts for seed selection.
#
#   Phase A — mimicgen_budget_sweep (rep-1, random_seed=null):
#     3 heuristics × budget=300 = 3 arms
#
#   Phase B — mimicgen_budget_rep_sweep (rep-2/3, random_seed=1/2):
#     3 heuristics × budget=300 × 2 rep_seeds = 6 arms
#
# Total: 9 arms.  Device pool: 2 concurrent slots (cuda:0 ×2).
#
# Usage:
#   nohup bash -lc "
#     source ~/miniforge3/etc/profile.d/conda.sh
#     cd /home/erbauer/refactor_cupid/policy_doctor/.claude/worktrees/feat+mimicgen-eef-pipeline
#     ./scripts/run_budget_rep_sweep_d60_budget300_nut_constrained.sh
#   " &
#   echo "PID=$!"

set -euo pipefail

WORKTREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${WORKTREE}/logs/budget_rep_sweep_d60_budget300_nut_constrained.log"

cd "${WORKTREE}"

echo "[$(date '+%H:%M %Z')] === Budget rep sweep d60 budget300 nut-constrained started ===" | tee -a "${LOG}"

echo "[$(date '+%H:%M %Z')] --- Phase A: budget sweep (rep-1) ---" | tee -a "${LOG}"
conda run -n policy_doctor --no-capture-output \
    python -m policy_doctor.scripts.run_pipeline \
        data_source=mimicgen_square \
        "experiment=mimicgen_square_rep_sweep_apr26_d60_budget300_nut_constrained" \
        steps='[mimicgen_budget_sweep]' \
    2>&1 | tee -a "${LOG}"

echo "[$(date '+%H:%M %Z')] --- Phase B: rep sweep (rep-2/3) ---" | tee -a "${LOG}"
conda run -n policy_doctor --no-capture-output \
    python -m policy_doctor.scripts.run_pipeline \
        data_source=mimicgen_square \
        "experiment=mimicgen_square_rep_sweep_apr26_d60_budget300_nut_constrained" \
        steps='[mimicgen_budget_rep_sweep]' \
    2>&1 | tee -a "${LOG}"

echo "[$(date '+%H:%M %Z')] === Budget rep sweep d60 budget300 nut-constrained complete ===" | tee -a "${LOG}"
