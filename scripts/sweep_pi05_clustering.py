"""Sweep clustering hyperparameters (K, window, stride) for pi05 Libero suites.

Saves each result to influence_visualizer/configs/pi05_libero_<suite>/clustering/
with naming `policy_emb_bottleneck_plan_t0_w{W}_s{S}_seed0_kmeans_k{K}` so the
graph demo app's slug parser picks them up as the existing "policy_emb_bottleneck_plan_t0"
representation type.

Usage:
    conda activate policy_doctor
    python scripts/sweep_pi05_clustering.py
"""
from __future__ import annotations

import pathlib

import numpy as np

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------
_K_VALUES   = [8, 12, 15, 20]
_WS_PAIRS   = [(3, 1), (5, 2), (8, 3)]   # (window_width, stride)
_UMAP_DIM   = 30
_NORMALIZE  = "standard"
_AGG        = "mean"
_UMAP_RS    = 0                            # seed for reproducibility
_REP        = "policy_emb_bottleneck_plan_t0"   # matches _OFFICIAL_REPS in graph_demo

_SUITES = ["libero_spatial", "libero_object", "libero_goal"]


def run_sweep(suite: str, eval_dir: pathlib.Path, iv_configs: pathlib.Path) -> None:
    from policy_doctor.behaviors.clustering import (
        fit_cluster_kmeans,
        fit_normalize_embeddings,
        fit_reduce_dimensions,
    )
    from policy_doctor.data.clustering_loader import save_clustering_models
    from policy_doctor.data.slice_representations import PolicyEmbeddingRepresentation, SliceWindowParams
    from policy_doctor.influence.clustering_io import save_clustering_result

    task_config = f"pi05_{suite}"
    clust_root = iv_configs / task_config / "clustering"
    clust_root.mkdir(parents=True, exist_ok=True)

    rep = PolicyEmbeddingRepresentation()

    for w, s in _WS_PAIRS:
        print(f"\n  [{suite}] window={w}, stride={s} — loading embeddings …")
        params = SliceWindowParams(window_width=w, stride=s, aggregation=_AGG)
        emb_raw, metadata = rep.extract(eval_dir, params, layer="pi05")
        print(f"    raw: {emb_raw.shape}")

        emb_norm, norm_model = fit_normalize_embeddings(emb_raw,  method=_NORMALIZE)
        emb_pre,  pre_model  = fit_normalize_embeddings(emb_norm, method=_NORMALIZE)
        print(f"    UMAP {emb_pre.shape[1]}d → {_UMAP_DIM}d …")
        emb_red, umap_model = fit_reduce_dimensions(
            emb_pre, method="umap", n_components=_UMAP_DIM,
            random_state=_UMAP_RS, n_jobs=-1,
        )
        print(f"    UMAP done: {emb_red.shape}")

        for k in _K_VALUES:
            # Name matches slug pattern: {rep}_w{W}_s{S}_seed{SEED}_kmeans_k{K}
            name = f"{_REP}_w{w}_s{s}_seed{_UMAP_RS}_kmeans_k{k}"
            result_path = clust_root / name
            if result_path.exists():
                print(f"    k={k}: EXISTS — skipping")
                continue

            print(f"    k={k}: KMeans …")
            labels, km_model = fit_cluster_kmeans(emb_red, n_clusters=k)
            n_actual = len(set(labels) - {-1})

            result_dir = save_clustering_result(
                name=name,
                cluster_labels=labels,
                metadata=metadata,
                algorithm="kmeans",
                scaling=_NORMALIZE,
                influence_source="policy_emb",
                representation="sliding_window",
                level="rollout",
                n_clusters=n_actual,
                n_samples=len(labels),
                embeddings_reduced=emb_red,
                output_dir=clust_root,
            )
            save_clustering_models(
                result_dir=result_dir,
                normalizer=norm_model,    normalizer_method=_NORMALIZE,
                prescaler=pre_model,      prescaler_method=_NORMALIZE,
                reducer=umap_model,       reducer_method="umap",
                kmeans=km_model,
            )
            print(f"    k={k}: saved → {result_dir.name}  ({n_actual} clusters)")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--suites", nargs="+", default=_SUITES)
    parser.add_argument("--data-base", default="data/pi05_libero")
    args = parser.parse_args()

    data_base = pathlib.Path(args.data_base).resolve()
    iv_configs = (
        pathlib.Path(__file__).resolve().parents[1]
        / "third_party" / "influence_visualizer" / "configs"
    )
    print(f"Data base:   {data_base}")
    print(f"IV configs:  {iv_configs}")
    print(f"Grid: K={_K_VALUES}, (w,s)={_WS_PAIRS}")

    for suite in args.suites:
        eval_dir = data_base / suite
        if not eval_dir.exists():
            print(f"[SKIP] {suite}: eval dir not found ({eval_dir})")
            continue
        if not (eval_dir / "policy_embeddings" / "pi05.npz").exists():
            print(f"[SKIP] {suite}: pi05.npz not found")
            continue
        print(f"\n{'='*60}")
        print(f"Suite: {suite}")
        print(f"{'='*60}")
        run_sweep(suite, eval_dir, iv_configs)

    print("\nAll sweeps done.")
    print(f"Results in: {iv_configs}/pi05_<suite>/clustering/")
    print("\nTo launch the graph demo:")
    print("  streamlit run policy_doctor/streamlit_app/demo_app/Home.py")


if __name__ == "__main__":
    main()
