#!/usr/bin/env bash
# scripts/run_budget_rep_sweep.sh
#
# Run rep-2 and rep-3 arms for the apr26 budget sweep.
#
# Reuses the completed seed=1 d60 pipeline run (baseline, clustering, behavior
# graph all done). Only the seed-draw random_seed is varied (1 and 2), giving a
# controlled heuristic comparison that matches the apr23 replicate design.
#
# Rep-1 arms are already done in the apr26 sweep — skip_if_done skips them.
# New arm dirs: mimicgen_{heuristic}_budget{N}_rep{1,2}
#
# Total new arms: 3 heuristics × 4 budgets × 2 new reps = 24 arms
# Device pool: 4 concurrent slots (cuda:0 ×2, cuda:1 ×2)
# Rough wall-clock: ~12–24hr depending on budget distribution
#
# Usage:
#   nohup bash -lc "
#     source ~/miniforge3/etc/profile.d/conda.sh
#     cd /home/erbauer/refactor_cupid/policy_doctor/.claude/worktrees/feat+mimicgen-eef-pipeline
#     ./scripts/run_budget_rep_sweep.sh
#   " &
#   echo "PID=$!"

set -euo pipefail

WORKTREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${WORKTREE}/logs/budget_rep_sweep.log"

cd "${WORKTREE}"

echo "[$(date '+%H:%M PDT')] === Budget rep sweep started ===" | tee -a "${LOG}"

conda run -n policy_doctor --no-capture-output \
    python -m policy_doctor.scripts.run_pipeline \
        data_source=mimicgen_square \
        "experiment=mimicgen_square_rep_sweep_apr26" \
        steps='[mimicgen_budget_rep_sweep]' \
    2>&1 | tee -a "${LOG}"

echo "[$(date '+%H:%M PDT')] === Budget rep sweep complete ===" | tee -a "${LOG}"
