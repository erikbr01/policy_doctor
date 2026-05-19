"""Build a clustering directory using any registered ``SliceRepresentation``.

Three modes:

  --timestep_embed_only
      Extract per-timestep features → normalise → UMAP.
      No windowing, no kmeans.  Writes the shared trunk:
        <out_dir>/timestep_embeddings.npy   (N_timesteps, umap_dim)
        <out_dir>/ep_meta.json              (episode_lengths, episode_successes)
        <out_dir>/embedding_models.pkl      (scaler + UMAP reducer, no kmeans)
        <out_dir>/embed_manifest.yaml

  --timestep_embed_dir <trunk_dir>  [+ --window_width --stride --aggregation --n_clusters]
      Load the trunk, apply windowing on the UMAP-reduced per-timestep embeddings,
      run kmeans.  Writes a full E1-compatible clustering dir.  Fast (seconds).

  (neither flag — original full pipeline)
      Extract windowed slice features → normalise → UMAP → kmeans.
      Backward-compatible with existing code.

E1-compatible output layout (modes 2 and 3):
    <out_dir>/manifest.yaml
    <out_dir>/cluster_labels.npy
    <out_dir>/metadata.json
    <out_dir>/embeddings_reduced.npy
    <out_dir>/clustering_models.pkl
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

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--representation", required=True,
                    help="Registered slice representation (infembed, state, state_action).")
    ap.add_argument("--eval_dir", required=True,
                    help="Eval episodes directory (episodes/metadata.yaml + ep*.pkl).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--window_width", type=int, default=5)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--aggregation", default="mean",
                    choices=["sum", "mean", "max", "min", "std", "median"])
    ap.add_argument("--obs_strategy", default="current",
                    choices=["current", "full_history"])
    ap.add_argument("--action_strategy", default="executed",
                    choices=["executed", "full_plan"])
    ap.add_argument("--layer", default="bottleneck",
                    help="For policy_emb: which embedding file to load (e.g. bottleneck, "
                         "bottleneck_t0, bottleneck_plan_t0).")
    ap.add_argument("--n_svd_components", type=int, default=200,
                    help="For trak: TruncatedSVD output dim (full matrix → SVD → UMAP).")
    ap.add_argument("--svd_seed", type=int, default=42,
                    help="For trak: random state for TruncatedSVD.")
    ap.add_argument("--normalize", default="none", choices=["none", "standard", "l2"])
    ap.add_argument("--prescale", default="standard", choices=["none", "standard", "l2"])
    ap.add_argument("--reducer", default="umap", choices=["umap", "pca", "none"])
    ap.add_argument("--umap_n_components", type=int, default=50,
                    help="UMAP target dim. Capped at (feature_dim - 1) automatically.")
    ap.add_argument("--umap_n_jobs", type=int, default=-1)
    ap.add_argument("--umap_init", default="spectral",
                    help="UMAP initialization method ('spectral' or 'random'). "
                         "Use 'random' to skip spectral embedding; much faster on large disconnected graphs.")
    ap.add_argument("--cluster_method", default="kmeans", choices=["kmeans"])
    ap.add_argument("--n_clusters", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task_config", default="alt_clustering_runtime")
    ap.add_argument("--timestep_embed_only", action="store_true",
                    help="Trunk mode: extract per-timestep features + UMAP, no windowing/kmeans.")
    ap.add_argument("--timestep_embed_dir", default=None, metavar="TRUNK_DIR",
                    help="Branch mode: load per-timestep UMAP trunk, apply windowing, run kmeans.")
    return ap.parse_args()


def _rep_kwargs(representation: str, args: argparse.Namespace) -> Dict[str, Any]:
    if representation == "state":
        return {"obs_strategy": args.obs_strategy}
    if representation == "state_action":
        return {"obs_strategy": args.obs_strategy, "action_strategy": args.action_strategy}
    if representation == "policy_emb":
        return {"layer": args.layer}
    if representation == "trak":
        return {"n_svd_components": args.n_svd_components,
                "svd_seed": args.svd_seed}
    return {}


def _save_clustering(
    out_dir: pathlib.Path,
    emb_red: np.ndarray,
    metadata: List[Dict[str, Any]],
    labels: np.ndarray,
    kmeans: Any,
    normalizer: Any,
    normalizer_method: str,
    prescaler: Any,
    prescaler_method: str,
    reducer: Any,
    reducer_method: str,
    manifest_extra: Dict[str, Any],
) -> None:
    import joblib
    from policy_doctor.data.clustering_loader import ClusteringModels

    np.save(out_dir / "cluster_labels.npy", labels.astype(np.int32))
    np.save(out_dir / "embeddings_reduced.npy", emb_red.astype(np.float32))
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(_to_jsonable(metadata), f)
    with open(out_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest_extra, f, default_flow_style=False, sort_keys=False)
    joblib.dump(
        ClusteringModels(
            normalizer=normalizer, normalizer_method=normalizer_method,
            prescaler=prescaler, prescaler_method=prescaler_method,
            reducer=reducer, reducer_method=reducer_method,
            kmeans=kmeans,
        ),
        out_dir / "clustering_models.pkl",
    )


def main() -> int:
    args = _parse_args()
    import joblib

    from policy_doctor.behaviors.clustering import (
        fit_cluster_kmeans, fit_normalize_embeddings, fit_reduce_dimensions,
    )
    from policy_doctor.data.slice_representations import get_slice_representation

    rep = get_slice_representation(args.representation)
    kw = _rep_kwargs(args.representation, args)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── MODE A: UMAP trunk ────────────────────────────────────────────────────
    if args.timestep_embed_only:
        print(f"[embed-trunk] {rep.name}  eval={args.eval_dir}", flush=True)
        per_ts, ep_lens, ep_succ = rep.extract_per_timestep(
            pathlib.Path(args.eval_dir), **kw,
        )
        print(f"  per-timestep: {per_ts.shape}")
        norm_x, normalizer = fit_normalize_embeddings(per_ts, method=args.normalize)
        scaled_x, prescaler = fit_normalize_embeddings(norm_x, method=args.prescale)
        n_comp = min(args.umap_n_components, per_ts.shape[1] - 1)
        t1 = time.time()
        emb_ts, reducer = fit_reduce_dimensions(
            scaled_x, method=args.reducer, n_components=n_comp,
            n_jobs=args.umap_n_jobs, init=args.umap_init,
        )
        emb_ts = emb_ts.astype(np.float32)
        print(f"  UMAP {per_ts.shape[1]}→{n_comp}d: {time.time()-t1:.1f}s")
        np.save(out_dir / "timestep_embeddings.npy", emb_ts)
        with open(out_dir / "ep_meta.json", "w") as f:
            json.dump({
                "episode_lengths": ep_lens,
                "episode_successes": [bool(s) if s is not None else None for s in ep_succ],
            }, f)
        from policy_doctor.data.clustering_loader import ClusteringModels
        joblib.dump(
            ClusteringModels(
                normalizer=normalizer, normalizer_method=args.normalize,
                prescaler=prescaler, prescaler_method=args.prescale,
                reducer=reducer, reducer_method=args.reducer,
                kmeans=None,
            ),
            out_dir / "embedding_models.pkl",
        )
        yaml.safe_dump({
            "timestep_embed_only": True,
            "slice_representation": rep.name,
            "umap_n_components": int(n_comp),
            "umap_prescale": args.prescale,
            "n_timesteps": int(len(emb_ts)),
            "eval_dir": str(args.eval_dir),
            "seed": int(args.seed),
            "rep_kwargs": kw,
        }, open(out_dir / "embed_manifest.yaml", "w"), default_flow_style=False, sort_keys=False)
        print(f"  [embed-trunk] done {time.time()-t0:.1f}s → {out_dir}", flush=True)
        return 0

    # ── MODE B: windowing + kmeans branch ────────────────────────────────────
    if args.timestep_embed_dir is not None:
        if args.n_clusters is None:
            raise ValueError("--n_clusters required with --timestep_embed_dir")
        from policy_doctor.data.clustering_embeddings import (
            build_windows_from_rollout_timestep_embeddings,
        )
        src = pathlib.Path(args.timestep_embed_dir)
        print(f"[embed-branch] trunk={src.name}  w={args.window_width} s={args.stride} "
              f"agg={args.aggregation}  K={args.n_clusters}", flush=True)
        emb_ts = np.load(src / "timestep_embeddings.npy").astype(np.float32)
        ep_meta = json.load(open(src / "ep_meta.json"))
        src_models = joblib.load(src / "embedding_models.pkl")
        emb_red, metadata = build_windows_from_rollout_timestep_embeddings(
            emb_ts,
            ep_meta["episode_lengths"],
            ep_meta["episode_successes"],
            args.window_width, args.stride, args.aggregation,
        )
        emb_red = np.asarray(emb_red, dtype=np.float32)
        print(f"  slices: {emb_red.shape}")
        labels, kmeans = fit_cluster_kmeans(emb_red, n_clusters=args.n_clusters)
        n_actual = int(len(set(labels.tolist()) - {-1}))
        print(f"  clusters: {n_actual}")
        _save_clustering(
            out_dir, emb_red, metadata, labels, kmeans,
            src_models.normalizer, args.normalize,
            src_models.prescaler, args.prescale,
            src_models.reducer, src_models.reducer_method,
            manifest_extra={
                "algorithm": "kmeans",
                "scaling": args.normalize,
                "umap_prescale": args.prescale,
                "influence_source": rep.name,
                "representation": "sliding_window",
                "slice_representation": rep.name,
                "level": "rollout",
                "n_clusters": n_actual,
                "n_samples": int(len(labels)),
                "umap_n_components": int(emb_red.shape[1]),
                "window_width": int(args.window_width),
                "stride": int(args.stride),
                "aggregation": args.aggregation,
                "task_config": args.task_config,
                "eval_dir": str(args.eval_dir),
                "seed": int(args.seed),
                "rep_kwargs": kw,
                "timestep_embed_dir": str(src),
            },
        )
        print(f"  [embed-branch] done {time.time()-t0:.1f}s → {out_dir}", flush=True)
        return 0

    # ── MODE C: full pipeline (original, backward-compatible) ─────────────────
    if args.n_clusters is None:
        raise ValueError("--n_clusters required in full-pipeline mode")
    from policy_doctor.data.slice_representations import SliceWindowParams
    params = SliceWindowParams(
        window_width=args.window_width, stride=args.stride, aggregation=args.aggregation,
    )
    print(f"[alt_clustering] {rep.name}  w={args.window_width} s={args.stride} "
          f"agg={args.aggregation}  K={args.n_clusters}", flush=True)
    features, metadata = rep.extract(pathlib.Path(args.eval_dir), params, **kw)
    print(f"  features: {features.shape}")
    norm_x, normalizer = fit_normalize_embeddings(features, method=args.normalize)
    scaled_x, prescaler = fit_normalize_embeddings(norm_x, method=args.prescale)
    if args.reducer == "none":
        emb_red, reducer, reducer_method = scaled_x.astype(np.float32), None, "none"
    else:
        t1 = time.time()
        n_comp = min(args.umap_n_components, features.shape[1] - 1)
        emb_red, reducer = fit_reduce_dimensions(
            scaled_x, method=args.reducer, n_components=n_comp,
            n_jobs=args.umap_n_jobs, init=args.umap_init,
        )
        emb_red = emb_red.astype(np.float32)
        reducer_method = args.reducer
        print(f"  UMAP {features.shape[1]}→{n_comp}d: {time.time()-t1:.1f}s")
    labels, kmeans = fit_cluster_kmeans(emb_red, n_clusters=args.n_clusters)
    n_actual = int(len(set(labels.tolist()) - {-1}))
    print(f"  clusters: {n_actual}")
    _save_clustering(
        out_dir, emb_red, metadata, labels, kmeans,
        normalizer, args.normalize,
        prescaler, args.prescale,
        reducer, reducer_method,
        manifest_extra={
            "algorithm": "kmeans",
            "scaling": args.normalize,
            "umap_prescale": args.prescale,
            "influence_source": rep.name,
            "representation": "sliding_window",
            "slice_representation": rep.name,
            "level": "rollout",
            "n_clusters": n_actual,
            "n_samples": int(len(labels)),
            "umap_n_components": int(emb_red.shape[1]),
            "window_width": int(args.window_width),
            "stride": int(args.stride),
            "aggregation": args.aggregation,
            "task_config": args.task_config,
            "eval_dir": str(args.eval_dir),
            "seed": int(args.seed),
            "rep_kwargs": kw,
        },
    )
    print(f"  [alt_clustering] done {time.time()-t0:.1f}s → {out_dir}", flush=True)
    return 0


def _to_jsonable(metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for m in metadata:
        item: Dict[str, Any] = {}
        for k, v in m.items():
            if isinstance(v, np.integer):
                item[k] = int(v)
            elif isinstance(v, np.floating):
                item[k] = float(v)
            elif isinstance(v, np.ndarray):
                item[k] = v.tolist()
            else:
                item[k] = v
        out.append(item)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
