#!/usr/bin/env bash
# scripts/run_budget_rep_sweep_d300.sh
#
# Run the MimicGen budget × replicate sweep for the 300-demo baseline.
#
# Reuses the completed seed=1 d300 pipeline run (baseline policy trained on
# 300 demos, clustering, behavior graph all done).  Runs two phases:
#
#   Phase A — mimicgen_budget_sweep (rep-1, random_seed=null):
#     3 heuristics × 2 budgets [100, 500] = 6 arms
#
#   Phase B — mimicgen_budget_rep_sweep (rep-2/3, random_seed=1/2):
#     3 heuristics × 2 budgets × 2 rep_seeds = 12 arms
#
# Total: 18 arms.  Device pool: 4 concurrent slots (cuda:0 ×2, cuda:1 ×2).
# Rough wall-clock: ~8–16hr depending on budget distribution.
#
# Usage:
#   nohup bash -lc "
#     source ~/miniforge3/etc/profile.d/conda.sh
#     cd /home/erbauer/refactor_cupid/policy_doctor/.claude/worktrees/feat+mimicgen-eef-pipeline
#     ./scripts/run_budget_rep_sweep_d300.sh
#   " &
#   echo "PID=$!"

set -euo pipefail

WORKTREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${WORKTREE}/logs/budget_rep_sweep_d300.log"

cd "${WORKTREE}"

echo "[$(date '+%H:%M PDT')] === Budget rep sweep d300 started ===" | tee -a "${LOG}"

echo "[$(date '+%H:%M PDT')] --- Phase A: budget sweep (rep-1) ---" | tee -a "${LOG}"
conda run -n policy_doctor --no-capture-output \
    python -m policy_doctor.scripts.run_pipeline \
        data_source=mimicgen_square \
        "experiment=mimicgen_square_rep_sweep_apr26_d300" \
        steps='[mimicgen_budget_sweep]' \
    2>&1 | tee -a "${LOG}"

echo "[$(date '+%H:%M PDT')] --- Phase B: rep sweep (rep-2/3) ---" | tee -a "${LOG}"
conda run -n policy_doctor --no-capture-output \
    python -m policy_doctor.scripts.run_pipeline \
        data_source=mimicgen_square \
        "experiment=mimicgen_square_rep_sweep_apr26_d300" \
        steps='[mimicgen_budget_rep_sweep]' \
    2>&1 | tee -a "${LOG}"

echo "[$(date '+%H:%M PDT')] === Budget rep sweep d300 complete ===" | tee -a "${LOG}"
