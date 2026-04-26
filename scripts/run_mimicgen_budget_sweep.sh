#!/usr/bin/env bash
# scripts/run_mimicgen_budget_sweep.sh
#
# Runs the full MimicGen budget sweep experiment end-to-end.
#
# For each baseline seed the script:
#   Phase 1 — trains a baseline policy, runs eval rollouts, computes InfEmbed
#             attribution, and runs clustering (shared by all budget arms).
#   Phase 2 — runs all 11 budget arms sequentially, each reusing the Phase 1
#             clustering result and writing to a separate namespace in run_dir.
#
# The outer seed loop varies the dataset seed used when training the baseline
# policy (task.dataset.seed=<seed>), giving statistically independent replicates
# of the full pipeline.  This is analogous to the SEEDS loop in
# third_party/cupid/scripts/train/train_policies.sh but kept lightweight:
# only the seed is passed as a CLI override; all other config lives in the
# experiment YAML.
#
# Environment variables (all optional — defaults shown):
#   TASK     task name used to select data_source and experiment YAML  [square]
#   SEEDS    space-separated list of baseline seeds                    [0 1 2]
#   DEVICE   CUDA device passed to the pipeline                        [cuda:0]
#   DATE     train/eval date tag; must match the experiment YAML       [apr25_sweep]
#
# Usage:
#   # Default (square, seeds 0–2, cuda:0)
#   ./scripts/run_mimicgen_budget_sweep.sh
#
#   # Override task, seeds or device
#   TASK=square SEEDS="0 1" DEVICE=cuda:1 ./scripts/run_mimicgen_budget_sweep.sh
#
#   # Background with logging
#   nohup bash -lc "
#     source ~/miniforge3/etc/profile.d/conda.sh
#     cd /path/to/worktree
#     TASK=square SEEDS='0 1 2' DEVICE=cuda:0 ./scripts/run_mimicgen_budget_sweep.sh
#   " > /tmp/budget_sweep.log 2>&1 &
#   echo "PID=$!"
#
# Resume behaviour:
#   Each pipeline step writes a <step>/done sentinel on completion.
#   Re-running this script will skip already-finished steps automatically.
#   To force a step to re-run, delete its done file:
#     rm <run_dir>/mimicgen_behavior_graph_budget200/done

set -euo pipefail

TASK="${TASK:-square}"
SEEDS="${SEEDS:-0 1 2}"
DEVICE="${DEVICE:-cuda:0}"
DATE="${DATE:-apr25}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENT="mimicgen_${TASK}_sweep_${DATE}"

UPSTREAM_STEPS="[train_baseline,eval_policies,train_attribution,finalize_attribution,compute_infembed,run_clustering]"
BUDGET_STEPS="[mimicgen_budget_sweep]"

cd "${WORKTREE_ROOT}"

echo "=================================================="
echo "  MimicGen budget sweep"
echo "  task=${TASK}  seeds=(${SEEDS})  device=${DEVICE}"
echo "  experiment=${EXPERIMENT}"
echo "=================================================="

for seed in ${SEEDS}; do
    run_dir="data/pipeline_runs/mimicgen_${TASK}_${DATE}_sweep_seed${seed}"

    echo ""
    echo "--------------------------------------------------"
    echo "  Seed ${seed} | run_dir=${run_dir}"
    echo "--------------------------------------------------"

    # ------------------------------------------------------------------
    # Phase 1: upstream steps — one-time cost per seed
    # Trains the baseline policy with dataset seed=${seed}, runs 100-ep
    # eval rollouts, computes InfEmbed attributions, and clusters rollouts
    # into a behaviour graph.  Results are cached; safe to re-run.
    # ------------------------------------------------------------------
    echo "[seed=${seed}] Phase 1 — upstream (train_baseline → clustering)"
    conda run -n policy_doctor --no-capture-output \
        python -m policy_doctor.scripts.run_pipeline \
            data_source=mimicgen_"${TASK}" \
            experiment="${EXPERIMENT}" \
            run_dir="${run_dir}" \
            seeds="[${seed}]" \
            device="${DEVICE}" \
            skip_if_done=true \
            steps="${UPSTREAM_STEPS}"

    # ------------------------------------------------------------------
    # Phase 2: budget sweep — 33 arms (3 heuristics × 11 budgets) sharing
    # the Phase 1 clustering result.  Each arm selects seed trajectories,
    # runs MimicGen generation, trains on original+generated data, and
    # evaluates 5 top-k checkpoints × 500 episodes.  Arms run concurrently
    # according to the devices pool defined in the experiment YAML.
    # ------------------------------------------------------------------
    echo "[seed=${seed}] Phase 2 — budget arms (budgets: 20 100 200…1000)"
    conda run -n policy_doctor --no-capture-output \
        python -m policy_doctor.scripts.run_pipeline \
            data_source=mimicgen_"${TASK}" \
            experiment="${EXPERIMENT}" \
            run_dir="${run_dir}" \
            seeds="[${seed}]" \
            steps="${BUDGET_STEPS}"

    echo "[seed=${seed}] done."
done

echo ""
echo "Budget sweep complete — all seeds finished."
