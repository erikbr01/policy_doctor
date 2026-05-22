"""Batch-compute data_support.json for all policy_emb clusterings in a task.

The naive per-clustering invocation re-fits a joint UMAP every time
(~140 s on the transport sweep) even though the rollout + demo windows
are identical for all clusterings that share `(window_width, stride,
aggregation, layer)`.  Same data → same joint UMAP → same per-slice
metric values; only the per-cluster aggregation differs between K
sweeps and between prep modes (umap_first vs agg_first).

This script groups clusterings by that key and amortises the expensive
work, then writes a `data_support.json` into each clustering directory.

Run from project root in the `policy_doctor` env, e.g.:

    python scripts/batch_data_support.py --task transport_mh_jan28 \\
        --radius 1.0 --knn_k 10 --umap_n_components 10
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import yaml

_WORKTREE = pathlib.Path(__file__).resolve().parents[1]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))
for _m in [k for k in list(sys.modules) if k.startswith("policy_doctor")]:
    _file = getattr(sys.modules.get(_m), "__file__", None) or ""
    if _file and str(_WORKTREE) not in _file:
        del sys.modules[_m]

from policy_doctor.behaviors.data_support import (
    aggregate_per_cluster,
    compute_all_metrics,
    fit_joint_umap,
)
from policy_doctor.behaviors.behavior_graph import (
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
)
from policy_doctor.data.clustering_embeddings import (
    build_windows_from_rollout_timestep_embeddings,
)


_SPECIAL_IDS = (SUCCESS_NODE_ID, FAILURE_NODE_ID, START_NODE_ID, END_NODE_ID)


TASK_REGISTRY: Dict[str, Dict[str, str]] = {
    "transport_mh_jan28": {
        "train_subdir":  "jan28/jan28_train_diffusion_unet_lowdim_transport_mh_0",
        "eval_subdir":   "jan28/jan28_train_diffusion_unet_lowdim_transport_mh_0",
        "sweep_subdir":  "transport_mh_jan28",
    },
    "lift_mh_jan26": {
        "train_subdir":  "jan26/jan26_train_diffusion_unet_lowdim_lift_mh_0",
        "eval_subdir":   "jan26/jan26_train_diffusion_unet_lowdim_lift_mh_0",
        "sweep_subdir":  "lift_mh_jan26",
    },
    "square_mh_feb5": {
        "train_subdir":  "feb5/feb5_train_diffusion_unet_lowdim_square_mh_0",
        "eval_subdir":   "feb5/feb5_train_diffusion_unet_lowdim_square_mh_0",
        "sweep_subdir":  "square_mh_feb5",
    },
}

DEFAULT_METRICS = (
    "count_in_radius",
    "binary_coverage",
    "knn_mean_distance",
    "knn_max_distance",
    "kde_log_density",
)


def _read_yaml(p: pathlib.Path) -> Dict:
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _find_policy_emb_clusterings(sweep_root: pathlib.Path) -> List[Tuple[pathlib.Path, Dict, str]]:
    """Return (clustering_dir, manifest, prep_mode) for every policy_emb clustering."""
    out: List[Tuple[pathlib.Path, Dict, str]] = []
    for prep in ("umap_first", "agg_first"):
        prep_dir = sweep_root / prep
        if not prep_dir.is_dir():
            continue
        for d in sorted(prep_dir.iterdir()):
            if not (d / "cluster_labels.npy").exists():
                continue
            manifest = _read_yaml(d / "manifest.yaml")
            if str(manifest.get("influence_source") or "") != "policy_emb":
                continue
            out.append((d, manifest, prep))
    return out


def _window_key(manifest: Dict) -> Tuple[int, int, str, str]:
    return (
        int(manifest.get("window_width") or 5),
        int(manifest.get("stride") or 2),
        str(manifest.get("aggregation") or "mean"),
        str(manifest.get("policy_emb_layer") or "bottleneck_plan_t0"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_REGISTRY))
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--knn_k", type=int, default=10)
    ap.add_argument("--kde_bandwidth", default="scott")
    ap.add_argument("--umap_n_components", type=int, default=10)
    ap.add_argument("--umap_random_state", type=int, default=0)
    ap.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute even if data_support.json already exists.")
    args = ap.parse_args()

    spec = TASK_REGISTRY[args.task]
    train_dir = _WORKTREE / "third_party" / "cupid" / "data" / "outputs" / "train" / spec["train_subdir"]
    eval_dir = (
        _WORKTREE / "third_party" / "cupid" / "data" / "outputs" / "eval_save_episodes"
        / spec["eval_subdir"] / "latest"
    )
    sweep_root = _WORKTREE / "data" / "demo_sweep" / spec["sweep_subdir"] / "run_clustering" / "clustering"

    print(f"=== Task: {args.task} ===")
    print(f"  train_dir: {train_dir}")
    print(f"  eval_dir:  {eval_dir}")
    print(f"  sweep:     {sweep_root}")

    clusterings = _find_policy_emb_clusterings(sweep_root)
    print(f"  policy_emb clusterings: {len(clusterings)}")
    if not clusterings:
        print("  nothing to do.")
        return

    # Group by (w, s, agg, layer) — joint UMAP and per-slice metrics are
    # identical inside a group; only cluster_labels.npy differs.
    groups: Dict[Tuple[int, int, str, str], List[Tuple[pathlib.Path, Dict, str]]] = defaultdict(list)
    for d, m, prep in clusterings:
        groups[_window_key(m)].append((d, m, prep))
    print(f"  unique (w, s, agg, layer) groups: {len(groups)}")

    # ── Cache: per-timestep embeddings keyed by layer (we usually have one). ──
    rollout_cache: Dict[str, Tuple[np.ndarray, List[int], List]] = {}
    demo_cache:    Dict[str, Tuple[np.ndarray, List[int]]] = {}

    def _load_rollout(layer: str):
        if layer not in rollout_cache:
            emb_path = eval_dir / "policy_embeddings" / f"{layer}.npz"
            with np.load(emb_path) as f:
                arr = np.asarray(f["rollout_embeddings"], dtype=np.float32)
            ep_meta = _read_yaml(eval_dir / "episodes" / "metadata.yaml")
            rollout_cache[layer] = (
                arr,
                list(ep_meta.get("episode_lengths") or []),
                list(ep_meta.get("episode_successes") or []),
            )
        return rollout_cache[layer]

    def _load_demo(layer: str):
        if layer not in demo_cache:
            emb_path = train_dir / "policy_embeddings_demos" / f"{layer}.npz"
            with np.load(emb_path) as f:
                arr = np.asarray(f["demo_embeddings"], dtype=np.float32)
                ep_lens = np.asarray(f["episode_lengths"], dtype=np.int64).tolist()
            demo_cache[layer] = (arr, ep_lens)
        return demo_cache[layer]

    total_written = 0
    total_skipped = 0
    for gi, (key, members) in enumerate(sorted(groups.items()), 1):
        w, s, agg, layer = key
        print(f"\n── Group {gi}/{len(groups)}: w={w} s={s} agg={agg} layer={layer} "
              f"({len(members)} clusterings)")

        # If every member already has data_support.json and we're not
        # overwriting, skip the expensive UMAP fit altogether.
        if not args.overwrite and all((d / "data_support.json").exists() for d, _, _ in members):
            print("  all members already have data_support.json — skipping group.")
            total_skipped += len(members)
            continue

        rollout_per_ts, ep_lens, ep_succ = _load_rollout(layer)
        demo_per_ts, demo_ep_lens = _load_demo(layer)
        print(f"  rollout per-ts {rollout_per_ts.shape}, demo per-ts {demo_per_ts.shape}")

        rollout_windows, _ = build_windows_from_rollout_timestep_embeddings(
            rollout_per_ts, ep_lens, ep_succ,
            window_width=w, stride=s, aggregation=agg,
        )
        demo_windows, _ = build_windows_from_rollout_timestep_embeddings(
            demo_per_ts, demo_ep_lens, [None] * len(demo_ep_lens),
            window_width=w, stride=s, aggregation=agg,
        )
        print(f"  rollout windows {rollout_windows.shape}, demo windows {demo_windows.shape}")

        t0 = time.time()
        joint = fit_joint_umap(
            demo_windows,
            rollout_windows,
            n_components=args.umap_n_components,
            random_state=args.umap_random_state,
        )
        print(f"  joint UMAP fit in {time.time()-t0:.1f}s")

        t0 = time.time()
        per_metric, _ = compute_all_metrics(
            joint.demo_reduced,
            joint.rollout_reduced,
            metrics=args.metrics,
            radius=args.radius,
            knn_k=args.knn_k,
            kde_bandwidth=args.kde_bandwidth,
        )
        print(f"  metrics computed in {time.time()-t0:.1f}s")

        for clu_dir, manifest, prep in members:
            out_path = clu_dir / "data_support.json"
            if out_path.exists() and not args.overwrite:
                print(f"    [skip] {prep}/{clu_dir.name} — already has data_support.json")
                total_skipped += 1
                continue
            cluster_labels = np.load(clu_dir / "cluster_labels.npy").astype(np.int64)
            if cluster_labels.shape[0] != rollout_windows.shape[0]:
                print(f"    [warn] {prep}/{clu_dir.name} — window count mismatch "
                      f"(labels={cluster_labels.shape[0]} vs windows={rollout_windows.shape[0]}). "
                      f"Skipping.")
                continue
            out_metrics: Dict[str, Dict[str, Dict]] = {}
            for mname, vals in per_metric.items():
                per_cluster = aggregate_per_cluster(
                    vals, cluster_labels, exclude_labels=_SPECIAL_IDS,
                )
                out_metrics[mname] = {str(cid): rec for cid, rec in per_cluster.items()}
            payload = {
                "_config": {
                    "radius": args.radius,
                    "metrics": args.metrics,
                    "knn_k": args.knn_k,
                    "kde_bandwidth": args.kde_bandwidth,
                    "umap_n_components": args.umap_n_components,
                    "umap_random_state": args.umap_random_state,
                    "umap_normalize": "standard",
                    "umap_refit": True,
                    "demo_embeddings_path": str(train_dir / "policy_embeddings_demos" / f"{layer}.npz"),
                    "rollout_embeddings_path": str(eval_dir / "policy_embeddings" / f"{layer}.npz"),
                    "layer": layer,
                    "n_demo_windows": int(demo_windows.shape[0]),
                    "n_rollout_windows": int(rollout_windows.shape[0]),
                    "prep_mode": prep,
                    "window_width": w,
                    "stride": s,
                    "aggregation": agg,
                },
                "metrics": out_metrics,
            }
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"    [write] {prep}/{clu_dir.name} → data_support.json")
            total_written += 1

    print(f"\n=== Task {args.task} done: written={total_written} skipped={total_skipped} ===")


if __name__ == "__main__":
    main()
