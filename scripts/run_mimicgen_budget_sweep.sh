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
# Demo-count sweep and Phase 1 concurrency are configured entirely in the
# experiment YAML:
#
#   n_demos_sweep:
#     demo_counts: [60, 100, 300]   # explicit list  (OR start/stop/step)
#
#   phase1_devices: [cuda:0, cuda:0, cuda:1, cuda:1]  # concurrent Phase 1 slots
#
# When n_demos_sweep is absent the experiment YAML's baseline.max_train_episodes
# is used directly and no _demos<N> suffix is added to run_dir.
#
# Phase 1 jobs for all (seed × n_demos) combinations are launched in parallel
# up to len(phase1_devices) at a time, each assigned to its pool device.
# Phase 2 for each combination is started sequentially after all Phase 1 jobs
# finish; Phase 2 has its own internal concurrency (mimicgen_budget_sweep.devices).
#
# Environment variables (all optional — defaults shown):
#   TASK           task name used to select data_source and experiment YAML [square]
#   SEEDS          space-separated list of baseline seeds                   [0 1 2]
#   DATE           train/eval date tag; must match the experiment YAML      [apr26]
#
# Usage:
#   # Default: seeds 0-2, demo counts and devices from YAML
#   ./scripts/run_mimicgen_budget_sweep.sh
#
#   # Override task or seeds
#   TASK=square SEEDS="0 1" ./scripts/run_mimicgen_budget_sweep.sh
#
#   # Background with logging
#   nohup bash -lc "
#     source ~/miniforge3/etc/profile.d/conda.sh
#     cd /path/to/worktree
#     TASK=square SEEDS='0 1 2' ./scripts/run_mimicgen_budget_sweep.sh
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
DATE="${DATE:-apr26}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENT="mimicgen_${TASK}_sweep_${DATE}"
EXPERIMENT_YAML="${WORKTREE_ROOT}/policy_doctor/configs/experiment/${EXPERIMENT}.yaml"

UPSTREAM_STEPS="[train_baseline,eval_policies,train_attribution,finalize_attribution,compute_infembed,run_clustering]"
BUDGET_STEPS="[mimicgen_budget_sweep]"

cd "${WORKTREE_ROOT}"

# ---------------------------------------------------------------------------
# Read demo counts and Phase 1 devices from the experiment YAML.
# ---------------------------------------------------------------------------
echo "Reading sweep configuration from ${EXPERIMENT_YAML} ..."

N_DEMOS_LIST=$(conda run -n policy_doctor --no-capture-output python -c "
import sys
sys.path.insert(0, '.')
from policy_doctor.curation_pipeline.steps.mimicgen_budget_sweep import _resolve_demo_counts
from omegaconf import OmegaConf
cfg = OmegaConf.load('${EXPERIMENT_YAML}')
counts = _resolve_demo_counts(OmegaConf.select(cfg, 'n_demos_sweep'))
print(' '.join(str(c) for c in counts) if counts else '')
" 2>/dev/null)

PHASE1_DEVICES=$(conda run -n policy_doctor --no-capture-output python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${EXPERIMENT_YAML}')
devices = list(OmegaConf.select(cfg, 'phase1_devices') or ['cuda:0'])
print(' '.join(devices))
" 2>/dev/null)

echo "=================================================="
echo "  MimicGen budget sweep"
echo "  task=${TASK}  seeds=(${SEEDS})"
echo "  experiment=${EXPERIMENT}"
if [ -n "${N_DEMOS_LIST}" ]; then
    echo "  n_demos sweep: ${N_DEMOS_LIST}"
else
    echo "  n_demos: from experiment YAML baseline.max_train_episodes"
fi
echo "  phase1_devices: ${PHASE1_DEVICES}"
echo "=================================================="

# ---------------------------------------------------------------------------
# Build the full list of (seed, n_demos) combinations.
# ---------------------------------------------------------------------------
_demos_iter="${N_DEMOS_LIST:-__default__}"

declare -a ALL_SEEDS=()
declare -a ALL_NDEMOS=()
declare -a ALL_RUN_DIRS=()
declare -a ALL_DEMO_OVERRIDES=()
declare -a ALL_DATE_OVERRIDES=()

for seed in ${SEEDS}; do
    for n_demos in ${_demos_iter}; do
        if [ "${n_demos}" = "__default__" ]; then
            run_dir="data/pipeline_runs/mimicgen_${TASK}_${DATE}_sweep_seed${seed}"
            # No demo override — use baseline.max_train_episodes from YAML.
            # date_override left empty; experiment YAML's evaluation/attribution
            # sub-sections provide the correct train_date for this case.
            demos_override=""
            date_override=""
        else
            run_dir="data/pipeline_runs/mimicgen_${TASK}_${DATE}_sweep_seed${seed}_demos${n_demos}"
            # Include demo count in train_date so checkpoints land in a unique
            # path per (seed, n_demos) — prevents clobbering across demo arms.
            # Must override evaluation.* and attribution.* explicitly because the
            # data_source YAML sets evaluation.train_date=jan18 via its defaults
            # chain, which takes precedence over the top-level train_date CLI arg.
            _td="${DATE}_sweep_demos${n_demos}"
            demos_override="baseline.max_train_episodes=${n_demos}"
            date_override="train_date=${_td} evaluation.train_date=${_td} evaluation.eval_date=${_td} attribution.train_date=${_td} attribution.eval_date=${_td}"
        fi
        ALL_SEEDS+=("${seed}")
        ALL_NDEMOS+=("${n_demos}")
        ALL_RUN_DIRS+=("${run_dir}")
        ALL_DEMO_OVERRIDES+=("${demos_override}")
        ALL_DATE_OVERRIDES+=("${date_override}")
    done
done

TOTAL_COMBOS="${#ALL_SEEDS[@]}"
echo "Total (seed × n_demos) combinations: ${TOTAL_COMBOS}"

# ---------------------------------------------------------------------------
# Phase 1: run all (seed, n_demos) combinations in parallel with device pool.
# ---------------------------------------------------------------------------
echo ""
echo "========== Phase 1: upstream steps (parallel) =========="

read -ra PHASE1_DEVICE_ARR <<< "${PHASE1_DEVICES}"
N_SLOTS="${#PHASE1_DEVICE_ARR[@]}"

# Per-slot PID tracking (0 = free slot)
declare -a SLOT_PIDS
for i in $(seq 0 $((N_SLOTS - 1))); do SLOT_PIDS[$i]=0; done

# Map from combo index → slot (for logging)
declare -a COMBO_SLOT

# Find a free device slot; echoes slot index; waits if all are busy.
find_free_slot() {
    while true; do
        for i in $(seq 0 $((N_SLOTS - 1))); do
            local pid="${SLOT_PIDS[$i]}"
            if [ "$pid" -eq 0 ]; then
                echo "$i"; return
            fi
            # Check if process finished
            if ! kill -0 "$pid" 2>/dev/null; then
                # Reap exit code — fail fast on Phase 1 errors
                if ! wait "$pid"; then
                    echo "ERROR: Phase 1 job (PID=$pid, slot=$i) failed." >&2
                    exit 1
                fi
                SLOT_PIDS[$i]=0
                echo "$i"; return
            fi
        done
        sleep 10
    done
}

declare -a PHASE1_PIDS=()

for idx in $(seq 0 $((TOTAL_COMBOS - 1))); do
    seed="${ALL_SEEDS[$idx]}"
    n_demos="${ALL_NDEMOS[$idx]}"
    run_dir="${ALL_RUN_DIRS[$idx]}"
    demos_override="${ALL_DEMO_OVERRIDES[$idx]}"
    date_override="${ALL_DATE_OVERRIDES[$idx]}"

    slot=$(find_free_slot)
    device="${PHASE1_DEVICE_ARR[$slot]}"

    echo "[Phase 1 | slot=${slot} device=${device}] seed=${seed} n_demos=${n_demos} → ${run_dir}"

    # Stagger starts by 30 s so each WandB service has time to bind its socket
    # before the next parallel training job tries to start one.
    [ "${idx}" -gt 0 ] && sleep 30

    (
        conda run -n policy_doctor --no-capture-output \
            python -m policy_doctor.scripts.run_pipeline \
                data_source=mimicgen_"${TASK}" \
                experiment="${EXPERIMENT}" \
                run_dir="${run_dir}" \
                seeds="[${seed}]" \
                device="${device}" \
                skip_if_done=true \
                ${date_override} \
                ${demos_override:+"${demos_override}"} \
                steps="${UPSTREAM_STEPS}"
    ) &

    SLOT_PIDS[$slot]=$!
    PHASE1_PIDS+=($!)
    COMBO_SLOT[$idx]=$slot
done

echo "Waiting for all Phase 1 jobs to complete ..."
for pid in "${PHASE1_PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: a Phase 1 job (PID=$pid) failed." >&2
        exit 1
    fi
done
echo "All Phase 1 jobs done."

# ---------------------------------------------------------------------------
# Phase 2: budget sweep — sequential over combos; each arm uses its own pool.
# ---------------------------------------------------------------------------
echo ""
echo "========== Phase 2: budget sweep (sequential over combos) =========="

for idx in $(seq 0 $((TOTAL_COMBOS - 1))); do
    seed="${ALL_SEEDS[$idx]}"
    n_demos="${ALL_NDEMOS[$idx]}"
    run_dir="${ALL_RUN_DIRS[$idx]}"
    demos_override="${ALL_DEMO_OVERRIDES[$idx]}"
    date_override="${ALL_DATE_OVERRIDES[$idx]}"

    echo ""
    echo "--------------------------------------------------"
    echo "  Phase 2 | seed=${seed} n_demos=${n_demos} | ${run_dir}"
    echo "--------------------------------------------------"

    conda run -n policy_doctor --no-capture-output \
        python -m policy_doctor.scripts.run_pipeline \
            data_source=mimicgen_"${TASK}" \
            experiment="${EXPERIMENT}" \
            run_dir="${run_dir}" \
            seeds="[${seed}]" \
            ${date_override:+"${date_override}"} \
            ${demos_override:+"${demos_override}"} \
            steps="${BUDGET_STEPS}"

    echo "[seed=${seed} n_demos=${n_demos}] Phase 2 done."
done

echo ""
echo "Budget sweep complete — all (seed × n_demos) combinations finished."
