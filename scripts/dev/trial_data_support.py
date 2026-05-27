"""One-off trial of the data-support pipeline on transport_mh_jan28 seed 0.

Loads an existing policy_emb_bottleneck_plan_t0 clustering, the rollout
per-timestep embeddings, and the freshly-extracted demo embeddings, then
runs the compute_data_support inner loop directly (no pipeline orchestration
required).

Run from project root in the `policy_doctor` env:

    python scripts/trial_data_support.py [--clustering_dir PATH] [--radius 1.0]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

# Make the worktree importable even if a stale editable install of
# policy_doctor sits earlier on sys.path.
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


def _read_yaml(p: pathlib.Path):
    import yaml
    with open(p) as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--clustering_dir",
        default=str(
            _WORKTREE
            / "data/demo_sweep/transport_mh_jan28/run_clustering/clustering"
            / "umap_first/policy_emb_bottleneck_plan_t0_seed0_kmeans_k10"
        ),
    )
    ap.add_argument(
        "--eval_dir",
        default=str(
            _WORKTREE
            / "third_party/cupid/data/outputs/eval_save_episodes/jan28"
            / "jan28_train_diffusion_unet_lowdim_transport_mh_0/latest"
        ),
    )
    ap.add_argument(
        "--train_dir",
        default=str(
            _WORKTREE
            / "third_party/cupid/data/outputs/train/jan28"
            / "jan28_train_diffusion_unet_lowdim_transport_mh_0"
        ),
    )
    ap.add_argument("--layer", default=None,
                    help="Defaults to manifest.policy_emb_layer / "
                         "manifest.rep_id suffix.")
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--knn_k", type=int, default=10)
    ap.add_argument("--kde_bandwidth", default="scott")
    ap.add_argument("--umap_n_components", type=int, default=10)
    ap.add_argument("--umap_random_state", type=int, default=0)
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "count_in_radius",
            "binary_coverage",
            "knn_mean_distance",
            "knn_max_distance",
            "kde_log_density",
        ],
    )
    ap.add_argument("--save_joint_umap", action="store_true")
    args = ap.parse_args()

    clu_dir = pathlib.Path(args.clustering_dir)
    eval_dir = pathlib.Path(args.eval_dir)
    train_dir = pathlib.Path(args.train_dir)

    manifest = _read_yaml(clu_dir / "manifest.yaml")
    source = str(manifest.get("influence_source") or "")
    if source != "policy_emb":
        raise SystemExit(f"Clustering source is {source!r}, not policy_emb; aborting.")

    # Pick a layer name.
    layer = args.layer or manifest.get("policy_emb_layer")
    if not layer:
        rep_id = str(manifest.get("rep_id") or "")
        if rep_id.startswith("policy_emb_"):
            layer = rep_id[len("policy_emb_"):]
        else:
            layer = "bottleneck_plan_t0"
    print(f"Layer: {layer}")

    window_width = int(manifest.get("window_width") or 5)
    stride = int(manifest.get("stride") or 2)
    aggregation = str(manifest.get("aggregation") or "mean")
    print(f"Window: width={window_width} stride={stride} agg={aggregation}")

    # --- Load rollout embeddings + episode meta ---
    rollout_emb_path = eval_dir / "policy_embeddings" / f"{layer}.npz"
    print(f"Rollout embeddings: {rollout_emb_path}")
    with np.load(rollout_emb_path) as f:
        rollout_per_ts = np.asarray(f["rollout_embeddings"], dtype=np.float32)
    ep_meta = _read_yaml(eval_dir / "episodes" / "metadata.yaml")
    ep_lens = ep_meta["episode_lengths"]
    ep_succ = ep_meta.get("episode_successes", [None] * len(ep_lens))
    print(f"  rollout per-timestep: {rollout_per_ts.shape}, "
          f"episodes: {len(ep_lens)}, total_ts: {sum(ep_lens)}")

    # --- Load demo embeddings ---
    demo_emb_path = train_dir / "policy_embeddings_demos" / f"{layer}.npz"
    print(f"Demo embeddings: {demo_emb_path}")
    with np.load(demo_emb_path) as f:
        demo_per_ts = np.asarray(f["demo_embeddings"], dtype=np.float32)
        demo_ep_lens = np.asarray(f["episode_lengths"], dtype=np.int64).tolist()
    print(f"  demo per-timestep: {demo_per_ts.shape}, demos: {len(demo_ep_lens)}, "
          f"total_ts: {int(sum(demo_ep_lens))}")

    # --- Load cluster labels ---
    cluster_labels = np.load(clu_dir / "cluster_labels.npy").astype(np.int64)
    print(f"  cluster_labels: {cluster_labels.shape}, "
          f"unique: {len(np.unique(cluster_labels))}")

    # --- Window-aggregate both ---
    print("Building rollout windows ...")
    rollout_windows, _ = build_windows_from_rollout_timestep_embeddings(
        rollout_per_ts, ep_lens, ep_succ,
        window_width=window_width, stride=stride, aggregation=aggregation,
    )
    print(f"  rollout windows: {rollout_windows.shape}")
    print("Building demo windows ...")
    demo_windows, _ = build_windows_from_rollout_timestep_embeddings(
        demo_per_ts, demo_ep_lens, [None] * len(demo_ep_lens),
        window_width=window_width, stride=stride, aggregation=aggregation,
    )
    print(f"  demo windows:    {demo_windows.shape}")

    if rollout_windows.shape[0] != cluster_labels.shape[0]:
        raise SystemExit(
            f"Window count mismatch: {rollout_windows.shape[0]} rollout windows "
            f"vs {cluster_labels.shape[0]} cluster labels."
        )

    # --- Joint UMAP ---
    print(f"Joint UMAP: {demo_windows.shape[1]}d -> {args.umap_n_components}d "
          f"(demos+rollouts = {demo_windows.shape[0] + rollout_windows.shape[0]})")
    t0 = time.time()
    joint = fit_joint_umap(
        demo_windows,
        rollout_windows,
        n_components=args.umap_n_components,
        random_state=args.umap_random_state,
    )
    print(f"  fitted in {time.time()-t0:.1f}s")

    # --- Metrics ---
    print(f"Metrics: {args.metrics}  radius={args.radius}  knn_k={args.knn_k}")
    t0 = time.time()
    per_metric, _ = compute_all_metrics(
        joint.demo_reduced,
        joint.rollout_reduced,
        metrics=args.metrics,
        radius=args.radius,
        knn_k=args.knn_k,
        kde_bandwidth=args.kde_bandwidth,
    )
    print(f"  computed in {time.time()-t0:.1f}s")

    # Brief per-metric global summary for quick sanity-check.
    for mname, vals in per_metric.items():
        print(f"  {mname}: min={vals.min():.3f} med={np.median(vals):.3f} "
              f"max={vals.max():.3f}  (nonzero={int(np.sum(vals != 0))}/{vals.size})")

    # --- Per-cluster aggregation ---
    out_metrics = {}
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
            "demo_embeddings_path": str(demo_emb_path),
            "rollout_embeddings_path": str(rollout_emb_path),
            "layer": layer,
            "n_demo_windows": int(demo_windows.shape[0]),
            "n_rollout_windows": int(rollout_windows.shape[0]),
        },
        "metrics": out_metrics,
    }

    out_path = clu_dir / "data_support.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")

    if args.save_joint_umap:
        import joblib
        joblib.dump(joint.umap_model, clu_dir / "joint_umap.joblib")
        print(f"Wrote {clu_dir / 'joint_umap.joblib'}")


if __name__ == "__main__":
    main()
