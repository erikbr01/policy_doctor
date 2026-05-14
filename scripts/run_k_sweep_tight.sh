#!/usr/bin/env bash
# scripts/run_k_sweep_tight.sh
#
# K-robustness sweep: run the tight-constraint pipeline at K=5,10,15,20,25
# to show that behavior-graph seed selection is robust to the choice of K.
#
# Step 1 — run_clustering with clustering_n_clusters_sweep on the BASE run dir.
#          UMAP fits once; K-means re-fits for each K. Writes k{K}/ subdirs.
#
# Step 2 — for each K, run select_mimicgen_seed + generate + train in a
#          separate run dir that points back at the base clustering result
#          via clustering_run_dir.
#
# Usage:
#   nohup bash -lc "
#     source ~/miniforge3/etc/profile.d/conda.sh
#     cd /home/erbauer/policy_doctor
#     ./scripts/run_k_sweep_tight.sh
#   " &
#   echo "PID=$!"
#
# The pipeline auto-skips completed steps via sentinel files.
# Re-running this script resumes from the last incomplete step.

set -euo pipefail

WORKTREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${WORKTREE}/logs/k_sweep_tight.log"
mkdir -p "${WORKTREE}/logs"

cd "${WORKTREE}"

BASE_EXPERIMENT="mimicgen_square_rep_sweep_apr26_d60_budget300_nut_constrained_tight"
BASE_RUN="mimicgen_square_apr26_seed1_d60_budget300_nut_constrained_tight"
# clustering_run_dir is passed as a relative path so select_mimicgen_seed resolves
# it against REPO_ROOT (third_party/cupid/), matching where the pipeline writes run dirs.
BASE_RUN_REL="data/pipeline_runs/${BASE_RUN}"
K_VALUES=(5 10 15 20 25)

# The infembed NPZ files are in ~/data/cupid_data (not under REPO_ROOT).
# Pass evaluation.eval_output_dir as an absolute path so run_clustering and
# select_mimicgen_seed find the infembed embeddings and rollouts.hdf5.
EVAL_OUTPUT_DIR="${HOME}/data/cupid_data/outputs/eval_save_episodes"

echo "[$(date '+%H:%M %Z')] === K-sweep tight started ===" | tee -a "${LOG}"
echo "[$(date '+%H:%M %Z')] base run (relative): ${BASE_RUN_REL}" | tee -a "${LOG}"
echo "[$(date '+%H:%M %Z')] K values: ${K_VALUES[*]}" | tee -a "${LOG}"

# ---------------------------------------------------------------------------
# Step 1: Run clustering with sweep (UMAP once, K-means for each K)
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M %Z')] --- Step 1: run_clustering with K sweep ---" | tee -a "${LOG}"
conda run -n policy_doctor --no-capture-output \
    python -m policy_doctor.scripts.run_pipeline \
        data_source=mimicgen_square \
        "experiment=${BASE_EXPERIMENT}" \
        "steps=[run_clustering]" \
        "+clustering_n_clusters_sweep=[5,10,15,20,25]" \
        "clustering_n_clusters=15" \
        "~evaluation.eval_output_dir" \
        "+evaluation.eval_output_dir=${EVAL_OUTPUT_DIR}" \
    2>&1 | tee -a "${LOG}"
echo "[$(date '+%H:%M %Z')] Step 1 complete." | tee -a "${LOG}"

# ---------------------------------------------------------------------------
# Step 2: For each K, run downstream steps in their own run dir
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M %Z')] --- Step 2: per-K downstream pipeline ---" | tee -a "${LOG}"

for K in "${K_VALUES[@]}"; do
    RUN_NAME="${BASE_RUN}_k${K}"
    echo "[$(date '+%H:%M %Z')] --- K=${K}: run_name=${RUN_NAME} ---" | tee -a "${LOG}"

    # Phase A: rep-1
    conda run -n policy_doctor --no-capture-output \
        python -m policy_doctor.scripts.run_pipeline \
            data_source=mimicgen_square \
            "experiment=${BASE_EXPERIMENT}" \
            steps='[mimicgen_budget_sweep]' \
            "clustering_n_clusters=${K}" \
            "+clustering_run_dir=${BASE_RUN_REL}" \
            "run_dir=data/pipeline_runs/${RUN_NAME}" \
            "~evaluation.eval_output_dir" \
            "+evaluation.eval_output_dir=${EVAL_OUTPUT_DIR}" \
        2>&1 | tee -a "${LOG}"

    # Phase B: rep-2/3
    conda run -n policy_doctor --no-capture-output \
        python -m policy_doctor.scripts.run_pipeline \
            data_source=mimicgen_square \
            "experiment=${BASE_EXPERIMENT}" \
            steps='[mimicgen_budget_rep_sweep]' \
            "clustering_n_clusters=${K}" \
            "+clustering_run_dir=${BASE_RUN_REL}" \
            "run_dir=data/pipeline_runs/${RUN_NAME}" \
            "~evaluation.eval_output_dir" \
            "+evaluation.eval_output_dir=${EVAL_OUTPUT_DIR}" \
        2>&1 | tee -a "${LOG}"

    echo "[$(date '+%H:%M %Z')] K=${K} complete." | tee -a "${LOG}"
done

echo "[$(date '+%H:%M %Z')] === K-sweep tight complete ===" | tee -a "${LOG}"
echo "" | tee -a "${LOG}"
REPO_ROOT="${WORKTREE}/third_party/cupid"
echo "Aggregate results with:" | tee -a "${LOG}"
echo "python scripts/aggregate_sweep_results.py --k-sweep \\" | tee -a "${LOG}"
for K in "${K_VALUES[@]}"; do
    echo "  --run-dir ${REPO_ROOT}/data/pipeline_runs/${BASE_RUN}_k${K} \\" | tee -a "${LOG}"
done
