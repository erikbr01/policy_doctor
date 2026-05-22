"""Compute state-based behavior clustering for kendama rollouts.

Reads joint_positions from each episode's trajectory.hdf5, computes
sliding-window embeddings, runs UMAP + K-means, and saves standard
policy_doctor clustering artifacts.

Also builds the study MP4 index (symlinks exterior.mp4 into study_mp4s).

Usage (policy_doctor env):
    conda activate policy_doctor
    python scripts/cluster_kendama_rollouts.py \\
        --rollouts /mnt/ssdB/erik/rollouts/rollouts_kendama_latest_may19 \\
        --out_dir  data/demo_sweep/kendama_may20/run_clustering/clustering/state_w20_s10_k8 \\
        --mp4_out  /tmp/study_mp4s/kendama_may20 \\
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
        success = float(meta.get("success", 0)) > 0
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
    ap.add_argument("--out_dir", default="data/demo_sweep/kendama_may20/run_clustering/clustering/state_w20_s10_k8")
    ap.add_argument("--mp4_out", default="/tmp/study_mp4s/kendama_may20")
    ap.add_argument("-K", "--n_clusters", type=int, default=8)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rollouts_dir = Path(args.rollouts)
    out_dir = _REPO_ROOT / args.out_dir
    mp4_out = Path(args.mp4_out)
    K = args.n_clusters

    out_dir.mkdir(parents=True, exist_ok=True)
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

    # ── 4. K-means on normalised features ────────────────────────────────────
    from sklearn.cluster import KMeans

    print(f"Running K-means (K={K}) …")
    km = KMeans(n_clusters=K, random_state=args.seed, n_init=10)
    labels = km.fit_predict(X_scaled).astype(np.int32)
    print(f"  cluster sizes: {np.bincount(labels).tolist()}")

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
        window_width=args.window,
        stride=args.stride,
        aggregation="mean",
        task_config="kendama_may20",
        seed=args.seed,
    )
    (out_dir / "manifest.yaml").write_text(yaml.dump(manifest))
    print(f"Clustering saved to {out_dir}")

    # ── 6. Build study MP4 index ──────────────────────────────────────────────
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

    # ── 7. Print session YAML snippet ─────────────────────────────────────────
    print("\n=== Add to sessions/kendama_may20.yaml ===")
    print(f"mp4_dir: {mp4_out}")
    print(f"clustering_dir: {out_dir}")
    print("==========================================")


if __name__ == "__main__":
    main()
