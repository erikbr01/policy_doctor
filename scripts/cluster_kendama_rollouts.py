"""Compute state-based behavior clustering for kendama rollouts.

Reads joint_positions from each episode's trajectory.hdf5, computes
sliding-window embeddings, runs UMAP + K-means, and saves standard
policy_doctor clustering artifacts.

Also builds the study MP4 index (symlinks exterior.mp4 into study_mp4s).

Usage (policy_doctor env):
    conda activate policy_doctor
    python scripts/cluster_kendama_rollouts.py \\
        --rollouts /mnt/ssdB/erik/rollouts/rollouts_kendama_latest_may19 \\
        --out_dir  data/demo_sweep/kendama_may22/run_clustering/clustering/state_w20_s10_k8 \\
        --mp4_out  /tmp/study_mp4s/kendama_may22 \\
        -K 8 --window 20 --stride 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))


def _load_episodes(rollouts_dir: Path) -> list[dict]:
    """Load all episodes that have a trajectory.hdf5 and meta.json."""
    import h5py

    episodes = []
    for ep_dir in sorted(rollouts_dir.iterdir()):
        hdf5 = ep_dir / "trajectory.hdf5"
        meta_path = ep_dir / "meta.json"
        if not hdf5.exists() or not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        # Some episodes have `success: null` (run aborted before label assigned).
        raw_success = meta.get("success")
        success = bool(raw_success) and float(raw_success) > 0
        n_steps = int(meta.get("n_steps", 0))
        with h5py.File(hdf5, "r") as f:
            jp = f["data/demo_0/obs/joint_positions"][:]  # (T, 7)
        episodes.append(
            dict(
                ep_dir=ep_dir,
                joint_positions=jp.astype(np.float32),
                success=success,
                n_steps=n_steps,
            )
        )
    return episodes


def _build_windows(
    episodes: list[dict],
    window: int,
    stride: int,
) -> tuple[np.ndarray, list[dict]]:
    """Sliding-window embeddings from joint_positions."""
    rows, meta = [], []
    for ep_idx, ep in enumerate(episodes):
        jp = ep["joint_positions"]  # (T, 7)
        T = len(jp)
        t = 0
        while t + window <= T:
            chunk = jp[t : t + window]  # (W, 7)
            rows.append(chunk.flatten())  # W*7
            meta.append(
                dict(
                    rollout_idx=ep_idx,
                    window_start=t,
                    window_end=t + window,
                    window_width=window,
                    success=ep["success"],
                )
            )
            t += stride

    X = np.stack(rows, axis=0).astype(np.float32)
    return X, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="/mnt/ssdB/erik/rollouts/rollouts_kendama_latest_may19")
    ap.add_argument(
        "--out_root",
        default="data/demo_sweep/kendama_may22/run_clustering/clustering/aggregate_first",
        help="Parent dir under data/demo_sweep/<task>/.../clustering/. The "
             "slug is derived from W, S, and the final K.",
    )
    ap.add_argument("--mp4_out", default="/tmp/study_mp4s/kendama_may22")
    ap.add_argument("-K", "--n_clusters", type=int, default=8,
                    help="Fixed K. Ignored if --auto_k is set.")
    ap.add_argument("--auto_k", action="store_true",
                    help="Pick K by silhouette over [k_min, k_max] using "
                         "policy_doctor.behaviors.graph_simplification.auto_k_kmeans.")
    ap.add_argument("--k_min", type=int, default=4)
    ap.add_argument("--k_max", type=int, default=15)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no_mp4_index",
        action="store_true",
        help="Skip the symlink-creation + index.json step (section 6). Use "
             "this when running alongside transcode_kendama_videos.py — the "
             "transcoder owns /tmp/study_mp4s/kendama_may22/ in that flow.",
    )
    args = ap.parse_args()

    rollouts_dir = Path(args.rollouts)
    mp4_out = Path(args.mp4_out)
    # Defer out_dir creation until we know the final K (auto_k may pick it).

    mp4_out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load episodes ──────────────────────────────────────────────────────
    print(f"Loading episodes from {rollouts_dir} …")
    episodes = _load_episodes(rollouts_dir)
    print(f"  {len(episodes)} episodes, {sum(e['success'] for e in episodes):.0f} successes")

    # ── 2. Sliding-window embeddings ──────────────────────────────────────────
    print(f"Building windows (W={args.window}, S={args.stride}) …")
    X, meta = _build_windows(episodes, args.window, args.stride)
    print(f"  {len(X)} windows, {X.shape[1]}-dim features")

    # ── 3. Normalise + UMAP ───────────────────────────────────────────────────
    from sklearn.preprocessing import StandardScaler
    from umap import UMAP

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Running UMAP to 2D …")
    umap2 = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                 random_state=args.seed, verbose=False)
    coords = umap2.fit_transform(X_scaled).astype(np.float32)

    # ── 4. K-means (fixed K, or silhouette sweep) ────────────────────────────
    if args.auto_k:
        from policy_doctor.behaviors.graph_simplification import auto_k_kmeans

        print(f"Silhouette sweep K ∈ [{args.k_min}, {args.k_max}] …")
        best_labels, best_k, scores = auto_k_kmeans(
            X_scaled, k_range=(args.k_min, args.k_max), random_state=args.seed,
        )
        labels = best_labels.astype(np.int32)
        K = best_k
        print(f"  scores: {scores}")
        print(f"  best K = {best_k}  (silhouette {scores[best_k]:.4f})")
        silhouette_scores = scores
    else:
        from sklearn.cluster import KMeans

        K = args.n_clusters
        print(f"Running K-means (K={K}) …")
        km = KMeans(n_clusters=K, random_state=args.seed, n_init=10)
        labels = km.fit_predict(X_scaled).astype(np.int32)
        silhouette_scores = None
    print(f"  cluster sizes: {np.bincount(labels).tolist()}")

    # Now we know K — build the slug + output dir.
    out_dir = _REPO_ROOT / args.out_root / (
        f"state_full_history_w{args.window}_s{args.stride}"
        f"_seed{args.seed}_kmeans_k{K}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 5. Save clustering artifacts ──────────────────────────────────────────
    np.save(out_dir / "cluster_labels.npy", labels)
    np.save(out_dir / "embeddings_reduced.npy", coords)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    manifest = dict(
        algorithm="kmeans",
        scaling="none",
        umap_prescale="standard",
        influence_source="state",
        representation="sliding_window",
        slice_representation="state",
        level="rollout",
        n_clusters=K,
        n_samples=len(X),
        auto_k=args.auto_k,
        silhouette_scores=silhouette_scores,
        window_width=args.window,
        stride=args.stride,
        aggregation="mean",
        task_config="kendama_may22",
        seed=args.seed,
        # Window aggregation happens before UMAP → demo app reads this as
        # ordering "aggregate_first" (sweep_analysis label) and groups
        # accordingly in the picker.
        pipeline_steps=["embed", "window", "umap", "kmeans"],
    )
    (out_dir / "manifest.yaml").write_text(yaml.dump(manifest))
    print(f"Clustering saved to {out_dir}")

    from policy_doctor.curation_pipeline.steps.compute_clustering_metrics import (
        _compute_for_dir,
    )

    metrics = _compute_for_dir(out_dir)
    if metrics is not None:
        sil = metrics.get("silhouette_mean")
        sil_s = f"{sil:.4f}" if sil is not None else "n/a"
        print(f"  metrics.json written (silhouette_mean={sil_s})")

    # ── 6. Build study MP4 index (skipped under --no_mp4_index) ──────────────
    if args.no_mp4_index:
        print("Skipping MP4 index/symlink step (--no_mp4_index set).")
    else:
        print(f"Building MP4 index in {mp4_out} …")
        index_eps = []
        for ep_idx, ep in enumerate(episodes):
            src = ep["ep_dir"] / "exterior.mp4"
            suffix = "succ" if ep["success"] else "fail"
            dst_name = f"ep{ep_idx:04d}_{suffix}.mp4"
            dst = mp4_out / dst_name
            if not dst.exists():
                dst.symlink_to(src)
            index_eps.append(
                dict(
                    index=ep_idx,
                    path=str(dst),
                    frame_count=ep["n_steps"],
                    success=ep["success"],
                )
            )

        index = {"episodes": index_eps}
        (mp4_out / "index.json").write_text(json.dumps(index, indent=2))
        print(f"  {len(index_eps)} episodes indexed")

    # ── 7. Print task YAML snippet ────────────────────────────────────────────
    print("\n=== Add to tasks/kendama_may22.yaml ===")
    print(f"mp4_dir: {mp4_out}")
    print(f"clustering_dir: {out_dir}")
    print("==========================================")


if __name__ == "__main__":
    main()
