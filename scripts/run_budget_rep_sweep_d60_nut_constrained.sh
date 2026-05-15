#!/usr/bin/env bash
# scripts/run_budget_rep_sweep_d60_nut_constrained.sh
#
# Run the MimicGen budget × replicate sweep for the 60-demo baseline with
# nut-constrained initial poses.
#
# Identical experiment design to run_budget_rep_sweep.sh (d60 unconstrained)
# except that MimicGen generation constrains the nut's initial pose:
#   x: ±50 mm, y: ±200 mm, z_rot: ±90° (offsets from seed demo pose).
# The peg remains fully unconstrained (D1 full randomisation).
#
# Uses the same upstream run (seed=1 d60) via a run_clustering symlink in the
# new constrained run dir — only the generation arms are new.
#
#   Phase A — mimicgen_budget_sweep (rep-1, random_seed=null):
#     3 heuristics × 4 budgets [20, 100, 500, 1000] = 12 arms
#
#   Phase B — mimicgen_budget_rep_sweep (rep-2/3, random_seed=1/2):
#     3 heuristics × 4 budgets × 2 rep_seeds = 24 arms
#
# Total: 36 arms.  Device pool: 4 concurrent slots (cuda:0 ×2, cuda:1 ×2).
# Rough wall-clock: ~24–48hr depending on budget distribution.
#
# Usage:
#   nohup bash -lc "
#     source ~/miniforge3/etc/profile.d/conda.sh
#     cd /home/erbauer/refactor_cupid/policy_doctor/.claude/worktrees/feat+mimicgen-eef-pipeline
#     ./scripts/run_budget_rep_sweep_d60_nut_constrained.sh
#   " &
#   echo "PID=$!"

set -euo pipefail

WORKTREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${WORKTREE}/logs/budget_rep_sweep_d60_nut_constrained.log"

cd "${WORKTREE}"

echo "[$(date '+%H:%M PDT')] === Budget rep sweep d60 nut-constrained started ===" | tee -a "${LOG}"

echo "[$(date '+%H:%M PDT')] --- Phase A: budget sweep (rep-1) ---" | tee -a "${LOG}"
conda run -n policy_doctor --no-capture-output \
    python -m policy_doctor.scripts.run_pipeline \
        data_source=mimicgen_square \
        "experiment=mimicgen_square_rep_sweep_apr26_d60_nut_constrained" \
        steps='[mimicgen_budget_sweep]' \
    2>&1 | tee -a "${LOG}"

echo "[$(date '+%H:%M PDT')] --- Phase B: rep sweep (rep-2/3) ---" | tee -a "${LOG}"
conda run -n policy_doctor --no-capture-output \
    python -m policy_doctor.scripts.run_pipeline \
        data_source=mimicgen_square \
        "experiment=mimicgen_square_rep_sweep_apr26_d60_nut_constrained" \
        steps='[mimicgen_budget_rep_sweep]' \
    2>&1 | tee -a "${LOG}"

echo "[$(date '+%H:%M PDT')] === Budget rep sweep d60 nut-constrained complete ===" | tee -a "${LOG}"
