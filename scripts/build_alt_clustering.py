"""Build a clustering directory using any registered ``SliceRepresentation``.

The output directory has the same layout as the existing E1-compatible
clustering dirs:

    <out_dir>/
      manifest.yaml
      cluster_labels.npy
      metadata.json
      embeddings_reduced.npy        (UMAP-reduced features)
      clustering_models.pkl         (fitted scaler + reducer + kmeans)

so the existing E1 evaluation pipeline (``run_e1_transport_r512_qwen.py``,
``validate_cluster_coherence_vlm`` Hydra step, etc.) can consume the result
without modification.

Example:

    python scripts/build_alt_clustering.py \\
      --representation state_action \\
      --eval_dir /mnt/ssdB/.../mar27_train_diffusion_unet_lowdim_transport_mh_0_r512x512/latest \\
      --window_width 5 --stride 2 --aggregation mean \\
      --umap_n_components 100 --n_clusters 10 \\
      --prescale standard \\
      --out_dir /tmp/sa_k10_mean
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# Ensure the active worktree's policy_doctor wins over a sibling pip install.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--representation",
        required=True,
        help="Registered slice representation name (e.g. infembed, state, state_action).",
    )
    ap.add_argument(
        "--eval_dir",
        required=True,
        help="Eval episodes directory containing episodes/metadata.yaml and ep*.pkl files.",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--window_width", type=int, default=5)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--aggregation", default="sum",
                    choices=["sum", "mean", "max", "min", "std", "median"])

    # State / state_action specific kwargs (ignored by other reps)
    ap.add_argument("--obs_strategy", default="current",
                    choices=["current", "full_history"])
    ap.add_argument("--action_strategy", default="executed",
                    choices=["executed", "full_plan"])

    # Downstream pipeline knobs
    ap.add_argument("--normalize", default="none",
                    choices=["none", "standard", "l2"])
    ap.add_argument("--prescale", default="standard",
                    choices=["none", "standard", "l2"])
    ap.add_argument("--reducer", default="umap", choices=["umap", "pca", "none"])
    ap.add_argument("--umap_n_components", type=int, default=100)
    ap.add_argument("--umap_n_jobs", type=int, default=-1)
    ap.add_argument("--cluster_method", default="kmeans", choices=["kmeans"])
    ap.add_argument("--n_clusters", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--task_config", default="alt_clustering_runtime",
        help="Free-form label written into the manifest for traceability.",
    )
    return ap.parse_args()


def _build_per_representation_kwargs(
    representation: str, args: argparse.Namespace,
) -> Dict[str, Any]:
    """Pick the subset of CLI flags relevant to this representation."""
    if representation == "state":
        return {"obs_strategy": args.obs_strategy}
    if representation == "state_action":
        return {
            "obs_strategy": args.obs_strategy,
            "action_strategy": args.action_strategy,
        }
    return {}


def main() -> int:
    args = _parse_args()
    import joblib

    from policy_doctor.behaviors.clustering import (
        fit_cluster_kmeans,
        fit_normalize_embeddings,
        fit_reduce_dimensions,
    )
    from policy_doctor.data.clustering_loader import ClusteringModels
    from policy_doctor.data.slice_representations import (
        SliceWindowParams,
        get_slice_representation,
    )

    rep = get_slice_representation(args.representation)
    params = SliceWindowParams(
        window_width=args.window_width,
        stride=args.stride,
        aggregation=args.aggregation,
    )
    rep_kwargs = _build_per_representation_kwargs(args.representation, args)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[alt_clustering] rep={rep.name}  eval_dir={args.eval_dir}\n"
        f"  window_width={args.window_width} stride={args.stride} "
        f"aggregation={args.aggregation}  rep_kwargs={rep_kwargs}\n"
        f"  normalize={args.normalize} prescale={args.prescale} "
        f"reducer={args.reducer}({args.umap_n_components}d) "
        f"cluster={args.cluster_method}(K={args.n_clusters}, seed={args.seed})\n"
        f"  out_dir={out_dir}",
        flush=True,
    )

    t0 = time.time()
    features, metadata = rep.extract(
        pathlib.Path(args.eval_dir), params, **rep_kwargs,
    )
    t_extract = time.time() - t0
    print(f"  features: {features.shape}  ({t_extract:.1f}s)")
    if len(features) != len(metadata):
        raise RuntimeError(
            f"shape mismatch: features {features.shape} vs metadata len {len(metadata)}"
        )

    # Normalize → prescale (mirroring RunClusteringStep)
    norm_x, normalizer = fit_normalize_embeddings(features, method=args.normalize)
    scaled_x, prescaler = fit_normalize_embeddings(norm_x, method=args.prescale)

    # Reduce
    if args.reducer == "none":
        emb_red, reducer = scaled_x.astype(np.float32, copy=False), None
        reducer_method = "none"
    else:
        t0 = time.time()
        emb_red, reducer = fit_reduce_dimensions(
            scaled_x,
            method=args.reducer,
            n_components=args.umap_n_components,
            n_jobs=args.umap_n_jobs,
        )
        emb_red = emb_red.astype(np.float32, copy=False)
        print(f"  reducer ({args.reducer}, {args.umap_n_components}d): {time.time() - t0:.1f}s")
        reducer_method = args.reducer

    # Cluster
    if args.cluster_method == "kmeans":
        labels, kmeans = fit_cluster_kmeans(emb_red, n_clusters=args.n_clusters)
    else:
        raise NotImplementedError(args.cluster_method)
    n_actual = int(len(set(labels.tolist()) - {-1}))
    print(f"  clusters formed: {n_actual} (requested {args.n_clusters})")

    # Persist artifacts (E1-compatible layout)
    np.save(out_dir / "cluster_labels.npy", labels.astype(np.int32))
    np.save(out_dir / "embeddings_reduced.npy", emb_red.astype(np.float32))

    # metadata.json: list of per-slice dicts (the planner uses these)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(_metadata_to_jsonable(metadata), f)

    # manifest.yaml: same shape as run_clustering output, plus rep fingerprint
    manifest: Dict[str, Any] = {
        "algorithm": args.cluster_method,
        "scaling": args.normalize,
        "umap_prescale": args.prescale,
        "influence_source": rep.name,           # back-compat: code that reads this still works
        "representation": "sliding_window",     # back-compat field
        "slice_representation": rep.name,       # new authoritative field
        "level": "rollout",
        "n_clusters": n_actual,
        "n_samples": int(len(labels)),
        "umap_n_components": int(args.umap_n_components),
        "window_width": int(args.window_width),
        "stride": int(args.stride),
        "aggregation": args.aggregation,
        "task_config": args.task_config,
        "eval_dir": str(args.eval_dir),
        "seed": int(args.seed),
        "rep_kwargs": rep_kwargs,
    }
    with open(out_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f, default_flow_style=False, sort_keys=False)

    # clustering_models.pkl: same dataclass the existing pipeline uses
    models = ClusteringModels(
        normalizer=normalizer,
        normalizer_method=args.normalize,
        prescaler=prescaler,
        prescaler_method=args.prescale,
        reducer=reducer,
        reducer_method=reducer_method,
        kmeans=kmeans,
    )
    joblib.dump(models, out_dir / "clustering_models.pkl")

    print(f"  wrote: {out_dir}/{{manifest.yaml, cluster_labels.npy, "
          "metadata.json, embeddings_reduced.npy, clustering_models.pkl}}")
    print(f"[alt_clustering] done in {time.time() - t0:.1f}s")
    return 0


def _metadata_to_jsonable(metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for m in metadata:
        item: Dict[str, Any] = {}
        for k, v in m.items():
            if isinstance(v, (np.integer,)):
                item[k] = int(v)
            elif isinstance(v, (np.floating,)):
                item[k] = float(v)
            elif isinstance(v, np.ndarray):
                item[k] = v.tolist()
            else:
                item[k] = v
        out.append(item)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
