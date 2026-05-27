#!/usr/bin/env bash
# K-sweep v2 for E1 transport_mh r512 seed0:
#   - centroid-proximal example selection (embeddings_reduced.npy in each clustering dir)
#   - n_repetitions=3 with majority vote
#   - global episode-disjoint planner (blocks cross-cluster episode-cue confound)
#
# Runs sequentially to avoid GPU contention. Total ~2.5h on Qwen3-VL-8B.
# Override the GPU index by exporting CUDA_VISIBLE_DEVICES before invoking.
set -eu

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES

PY=/home/erbauer/miniforge3/envs/policy_doctor/bin/python
SCRIPT=/home/erbauer/refactor_cupid/e1-cluster-validation/scripts/run_e1_transport_r512_qwen.py

declare -A CLUSTERING_DIR=(
  [20]=/tmp/transport_mh_seed0_r512_clustering
  [15]=/tmp/transport_mh_seed0_r512_clustering_k15
  [10]=/tmp/transport_mh_seed0_r512_clustering_k10
)

for K in 20 15 10; do
  OUT="experiments/e1_transport_r512_seed0_qwen3vl8b_k${K}_v2"
  CDIR="${CLUSTERING_DIR[$K]}"
  echo "===== K=${K} (clustering_dir=${CDIR} → ${OUT}) ====="
  "$PY" "$SCRIPT" \
    --clustering_dir "$CDIR" \
    --out_dir "$OUT" \
    --max_clusters "$K" \
    --n_example 3 \
    --n_query 3 \
    --n_repetitions 3 \
    --global_episode_disjoint
done

echo
echo "===== K-sweep done. Summaries: ====="
for K in 20 15 10; do
  M="experiments/e1_transport_r512_seed0_qwen3vl8b_k${K}_v2/metrics.json"
  if [ -f "$M" ]; then
    "$PY" -c "import json; m=json.load(open('$M')); print('K=${K}', 'top1=%.3f' % m['top1_accuracy'], 'p=%.2e' % m['binomial_test_pvalue'])"
  fi
done
