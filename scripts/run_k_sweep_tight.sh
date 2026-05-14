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
#   # Run both steps end-to-end (takes many hours):
#   nohup bash -lc "
#     source ~/miniforge3/etc/profile.d/conda.sh
#     cd /home/erbauer/policy_doctor
#     ./scripts/run_k_sweep_tight.sh
#   " &
#   echo "PID=$!"
#
#   # Step 1 only (if clustering already done, skip with SKIP_CLUSTERING=1):
#   SKIP_CLUSTERING=1 ./scripts/run_k_sweep_tight.sh

set -euo pipefail

WORKTREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${WORKTREE}/logs/k_sweep_tight.log"
mkdir -p "${WORKTREE}/logs"

cd "${WORKTREE}"

BASE_EXPERIMENT="mimicgen_square_rep_sweep_apr26_d60_budget300_nut_constrained_tight"
BASE_RUN="mimicgen_square_apr26_seed1_d60_budget300_nut_constrained_tight"
BASE_RUN_ABS="/mnt/ssdB/erik/cupid_data/pipeline_runs/${BASE_RUN}"
K_VALUES=(5 10 15 20 25)

echo "[$(date '+%H:%M %Z')] === K-sweep tight started ===" | tee -a "${LOG}"
echo "[$(date '+%H:%M %Z')] base run: ${BASE_RUN_ABS}" | tee -a "${LOG}"
echo "[$(date '+%H:%M %Z')] K values: ${K_VALUES[*]}" | tee -a "${LOG}"

# ---------------------------------------------------------------------------
# Step 1: Run clustering with sweep (UMAP once, K-means for each K)
# ---------------------------------------------------------------------------
if [[ "${SKIP_CLUSTERING:-0}" == "1" ]]; then
    echo "[$(date '+%H:%M %Z')] Skipping Step 1 (SKIP_CLUSTERING=1)" | tee -a "${LOG}"
else
    echo "[$(date '+%H:%M %Z')] --- Step 1: run_clustering with K sweep ---" | tee -a "${LOG}"
    conda run -n policy_doctor --no-capture-output \
        python -m policy_doctor.scripts.run_pipeline \
            data_source=mimicgen_square \
            "+experiment=${BASE_EXPERIMENT}" \
            steps='[run_clustering]' \
            "clustering_n_clusters_sweep=[5,10,15,20,25]" \
            clustering_n_clusters=15 \
        2>&1 | tee -a "${LOG}"
    echo "[$(date '+%H:%M %Z')] Step 1 complete." | tee -a "${LOG}"
fi

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
            "+experiment=${BASE_EXPERIMENT}" \
            steps='[mimicgen_budget_sweep]' \
            "clustering_n_clusters=${K}" \
            "clustering_run_dir=${BASE_RUN_ABS}" \
            "run_name=${RUN_NAME}" \
        2>&1 | tee -a "${LOG}"

    # Phase B: rep-2/3
    conda run -n policy_doctor --no-capture-output \
        python -m policy_doctor.scripts.run_pipeline \
            data_source=mimicgen_square \
            "+experiment=${BASE_EXPERIMENT}" \
            steps='[mimicgen_budget_rep_sweep]' \
            "clustering_n_clusters=${K}" \
            "clustering_run_dir=${BASE_RUN_ABS}" \
            "run_name=${RUN_NAME}" \
        2>&1 | tee -a "${LOG}"

    echo "[$(date '+%H:%M %Z')] K=${K} complete." | tee -a "${LOG}"
done

echo "[$(date '+%H:%M %Z')] === K-sweep tight complete ===" | tee -a "${LOG}"
echo "" | tee -a "${LOG}"
echo "Aggregate results with:" | tee -a "${LOG}"
DIRS=""
for K in "${K_VALUES[@]}"; do
    DIRS="${DIRS}    /mnt/ssdB/erik/cupid_data/pipeline_runs/${BASE_RUN}_k${K} \\"
    echo "  /mnt/ssdB/erik/cupid_data/pipeline_runs/${BASE_RUN}_k${K}" | tee -a "${LOG}"
done
echo "" | tee -a "${LOG}"
echo "python scripts/aggregate_sweep_results.py --k-sweep \\" | tee -a "${LOG}"
echo "  --run-dir /mnt/ssdB/erik/cupid_data/pipeline_runs/${BASE_RUN}_k5 \\" | tee -a "${LOG}"
echo "  --run-dir /mnt/ssdB/erik/cupid_data/pipeline_runs/${BASE_RUN}_k10 \\" | tee -a "${LOG}"
echo "  --run-dir /mnt/ssdB/erik/cupid_data/pipeline_runs/${BASE_RUN}_k15 \\" | tee -a "${LOG}"
echo "  --run-dir /mnt/ssdB/erik/cupid_data/pipeline_runs/${BASE_RUN}_k20 \\" | tee -a "${LOG}"
echo "  --run-dir /mnt/ssdB/erik/cupid_data/pipeline_runs/${BASE_RUN}_k25" | tee -a "${LOG}"
