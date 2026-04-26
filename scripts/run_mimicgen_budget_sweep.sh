#!/usr/bin/env bash
# scripts/run_mimicgen_budget_sweep.sh
#
# Runs the full MimicGen budget sweep experiment end-to-end.
#
# For each (seed × n_demos) combination the script:
#   Phase 1 — trains a baseline policy, runs eval rollouts, computes InfEmbed
#             attribution, and runs clustering (shared by all budget arms).
#   Phase 2 — runs all budget arms sequentially, each reusing the Phase 1
#             clustering result and writing to a separate namespace in run_dir.
#
# The outer seed loop varies the dataset seed used when training the baseline
# policy (task.dataset.seed=<seed>), giving statistically independent replicates.
#
# The optional N_DEMOS dimension sweeps over baseline training set sizes to test
# how the amount of source data affects the value of MimicGen augmentation.
# When N_DEMOS is unset, the demo count from the experiment YAML is used and
# no suffix is added to run_dir (backward compatible with prior runs).
#
# Environment variables (all optional — defaults shown):
#   TASK           task name used to select data_source and experiment YAML [square]
#   SEEDS          space-separated list of baseline seeds                   [0 1 2]
#   DEVICE         CUDA device passed to the pipeline                       [cuda:0]
#   DATE           train/eval date tag; must match the experiment YAML      [apr25]
#
#   N_DEMOS        space-separated explicit list of demo counts             [unset]
#                  e.g. N_DEMOS="60 100 300"
#   N_DEMOS_START  ) range alternative to N_DEMOS: generates               [unset]
#   N_DEMOS_STOP   )   seq N_DEMOS_START N_DEMOS_STEP N_DEMOS_STOP
#   N_DEMOS_STEP   )   e.g. START=100 STOP=1000 STEP=100 → 100 200 … 1000 [unset]
#
#   When N_DEMOS / range is set, each run_dir gets a _demos<N> suffix and
#   baseline.max_train_episodes=<N> is passed to the pipeline.
#   When neither is set, the YAML default applies and no suffix is added.
#
# Usage:
#   # Default: seeds 0-2, demo count from YAML, no demos sweep
#   ./scripts/run_mimicgen_budget_sweep.sh
#
#   # Explicit demo list
#   N_DEMOS="60 100 300" ./scripts/run_mimicgen_budget_sweep.sh
#
#   # Range-based demo sweep (100, 200, ..., 1000)
#   N_DEMOS_START=100 N_DEMOS_STOP=1000 N_DEMOS_STEP=100 \
#     ./scripts/run_mimicgen_budget_sweep.sh
#
#   # Combined: single seed, two demo counts, second GPU
#   SEEDS="0" N_DEMOS="60 300" DEVICE=cuda:1 \
#     ./scripts/run_mimicgen_budget_sweep.sh
#
#   # Background with logging
#   nohup bash -lc "
#     source ~/miniforge3/etc/profile.d/conda.sh
#     cd /path/to/worktree
#     TASK=square SEEDS='0 1 2' N_DEMOS='60 100 300' DEVICE=cuda:0 \
#       ./scripts/run_mimicgen_budget_sweep.sh
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

# ---------------------------------------------------------------------------
# Build the demo-count list from N_DEMOS (list) or N_DEMOS_START/STOP/STEP (range).
# When neither is provided N_DEMOS_LIST is empty — one iteration with no suffix.
# ---------------------------------------------------------------------------
if [ -n "${N_DEMOS_START:-}" ]; then
    _stop="${N_DEMOS_STOP:-${N_DEMOS_START}}"
    _step="${N_DEMOS_STEP:-1}"
    N_DEMOS_LIST="$(seq "${N_DEMOS_START}" "${_step}" "${_stop}")"
elif [ -n "${N_DEMOS:-}" ]; then
    N_DEMOS_LIST="${N_DEMOS}"
else
    N_DEMOS_LIST=""
fi

cd "${WORKTREE_ROOT}"

echo "=================================================="
echo "  MimicGen budget sweep"
echo "  task=${TASK}  seeds=(${SEEDS})  device=${DEVICE}"
echo "  experiment=${EXPERIMENT}"
if [ -n "${N_DEMOS_LIST}" ]; then
    echo "  n_demos sweep: $(echo "${N_DEMOS_LIST}" | tr '\n' ' ')"
else
    echo "  n_demos: from experiment YAML (no sweep)"
fi
echo "=================================================="

# ---------------------------------------------------------------------------
# Outer loop: seeds × demo counts
# ---------------------------------------------------------------------------
for seed in ${SEEDS}; do

    # When N_DEMOS_LIST is empty we do one pass with no suffix / no override.
    _demos_iter="${N_DEMOS_LIST:-__default__}"

    for n_demos in ${_demos_iter}; do

        # Determine run_dir suffix and optional baseline override.
        if [ "${n_demos}" = "__default__" ]; then
            run_dir="data/pipeline_runs/mimicgen_${TASK}_${DATE}_sweep_seed${seed}"
            demos_override=""
        else
            run_dir="data/pipeline_runs/mimicgen_${TASK}_${DATE}_sweep_seed${seed}_demos${n_demos}"
            demos_override="baseline.max_train_episodes=${n_demos}"
        fi

        echo ""
        echo "--------------------------------------------------"
        echo "  Seed ${seed} | n_demos=${n_demos} | run_dir=${run_dir}"
        echo "--------------------------------------------------"

        # ------------------------------------------------------------------
        # Phase 1: upstream steps — one-time cost per (seed, n_demos)
        # Trains the baseline policy, runs eval rollouts, computes InfEmbed
        # attributions, and clusters rollouts into a behaviour graph.
        # Results are cached via done sentinels; safe to re-run.
        # ------------------------------------------------------------------
        echo "[seed=${seed} n_demos=${n_demos}] Phase 1 — upstream (train_baseline → clustering)"
        conda run -n policy_doctor --no-capture-output \
            python -m policy_doctor.scripts.run_pipeline \
                data_source=mimicgen_"${TASK}" \
                experiment="${EXPERIMENT}" \
                run_dir="${run_dir}" \
                seeds="[${seed}]" \
                device="${DEVICE}" \
                skip_if_done=true \
                ${demos_override:+"${demos_override}"} \
                steps="${UPSTREAM_STEPS}"

        # ------------------------------------------------------------------
        # Phase 2: budget sweep — arms share the Phase 1 clustering result.
        # Each arm selects seed trajectories, runs MimicGen generation, trains
        # on original+generated data, and evaluates top-k checkpoints.
        # Arms run concurrently per the devices pool in the experiment YAML.
        # ------------------------------------------------------------------
        echo "[seed=${seed} n_demos=${n_demos}] Phase 2 — budget arms"
        conda run -n policy_doctor --no-capture-output \
            python -m policy_doctor.scripts.run_pipeline \
                data_source=mimicgen_"${TASK}" \
                experiment="${EXPERIMENT}" \
                run_dir="${run_dir}" \
                seeds="[${seed}]" \
                ${demos_override:+"${demos_override}"} \
                steps="${BUDGET_STEPS}"

        echo "[seed=${seed} n_demos=${n_demos}] done."
    done
done

echo ""
echo "Budget sweep complete — all (seed × n_demos) combinations finished."
