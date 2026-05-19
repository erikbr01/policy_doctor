"""Cluster pi0.5 Libero policy embeddings and save results for streamlit.

Usage:
    conda activate policy_doctor
    python scripts/cluster_pi05_libero.py --eval-dir /tmp/pi05_libero_spatial --n-clusters 15
    python scripts/cluster_pi05_libero.py --eval-dir /tmp/pi05_libero_object --n-clusters 15
    python scripts/cluster_pi05_libero.py --eval-dir /tmp/pi05_libero_goal  --n-clusters 15

The clustering result is saved to <eval_dir>/run_clustering/clustering/k<k>/ in the
standard policy_doctor format, ready for the streamlit behavior graph app.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import yaml

from policy_doctor.behaviors.clustering import (
    fit_cluster_kmeans,
    fit_normalize_embeddings,
    fit_reduce_dimensions,
)
from policy_doctor.data.clustering_loader import save_clustering_models
from policy_doctor.data.slice_representations import PolicyEmbeddingRepresentation, SliceWindowParams
from influence_visualizer.clustering_results import save_clustering_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster pi0.5 Libero policy embeddings")
    parser.add_argument("--eval-dir", required=True, type=pathlib.Path)
    parser.add_argument("--window-width", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--aggregation", default="mean")
    parser.add_argument("--n-clusters", type=int, default=15)
    parser.add_argument("--umap-n-components", type=int, default=50)
    parser.add_argument("--normalize", default="standard")
    parser.add_argument("--umap-prescale", default="standard")
    parser.add_argument("--umap-random-state", type=int, default=42)
    parser.add_argument("--experiment-name", default="pi05")
    args = parser.parse_args()

    eval_dir = args.eval_dir.resolve()
    print(f"Eval dir:  {eval_dir}")

    # Load embeddings via the existing PolicyEmbeddingRepresentation
    rep = PolicyEmbeddingRepresentation()
    params = SliceWindowParams(
        window_width=args.window_width,
        stride=args.stride,
        aggregation=args.aggregation,
    )
    print(f"Loading embeddings (layer=pi05, window={args.window_width}, stride={args.stride}, agg={args.aggregation})")
    embeddings_arr, all_metadata = rep.extract(eval_dir, params, layer="pi05")
    print(f"  Slice embeddings: {embeddings_arr.shape}")

    # Read metadata to report success rate
    meta_path = eval_dir / "episodes" / "metadata.yaml"
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    n_ep = len(meta["episode_lengths"])
    n_succ = sum(meta.get("episode_successes", []))
    print(f"  Episodes: {n_ep}, success rate: {100*n_succ/n_ep:.1f}%")

    # Normalize
    print(f"  Normalizing: {args.normalize}")
    embeddings_norm, normalizer_model = fit_normalize_embeddings(embeddings_arr, method=args.normalize)

    # Pre-UMAP scaling
    print(f"  Pre-UMAP scaling: {args.umap_prescale}")
    embeddings_scaled, prescaler_model = fit_normalize_embeddings(embeddings_norm, method=args.umap_prescale)

    # UMAP
    print(f"  UMAP: {embeddings_scaled.shape[1]}d -> {args.umap_n_components}d")
    embeddings_reduced, umap_model = fit_reduce_dimensions(
        embeddings_scaled,
        method="umap",
        n_components=args.umap_n_components,
        random_state=args.umap_random_state,
    )

    # KMeans
    k = args.n_clusters
    print(f"  K-Means: k={k}")
    labels, kmeans_model = fit_cluster_kmeans(embeddings_reduced, n_clusters=k)
    n_actual = len(set(labels) - {-1})
    print(f"  Clusters: {n_actual}, noise: {int((labels == -1).sum())}")

    # Save clustering result
    output_dir = eval_dir / "run_clustering" / "clustering" / f"k{k}"
    clustering_name = f"{args.experiment_name}_kmeans_k{k}"
    result_dir = save_clustering_result(
        name=clustering_name,
        cluster_labels=labels,
        metadata=all_metadata,
        algorithm="kmeans",
        scaling=args.normalize,
        influence_source="policy_emb",
        representation="sliding_window",
        level="rollout",
        n_clusters=n_actual,
        n_samples=len(labels),
        embeddings_reduced=embeddings_reduced,
        output_dir=output_dir,
    )
    models_path = save_clustering_models(
        result_dir=result_dir,
        normalizer=normalizer_model,
        normalizer_method=args.normalize,
        prescaler=prescaler_model,
        prescaler_method=args.umap_prescale,
        reducer=umap_model,
        reducer_method="umap",
        kmeans=kmeans_model,
    )
    print(f"\nSaved clustering: {result_dir}")
    print(f"Saved models:     {models_path}")
    print(f"\nStreamlit config: set  eval_dir: {eval_dir}")
    print(f"Clustering tab:  'Policy embeddings', layer='pi05'")
    print(f"Load from disk:  {result_dir}")


if __name__ == "__main__":
    main()
