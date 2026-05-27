"""Compute state-based behavior clustering for kendama rollouts.

Thin wrapper around ``kendama_clustering_lib`` + ``build_kendama_clustering``.
For a full W×S×K sweep, prefer::

    python scripts/run_kendama_clustering_sweep.py \\
        --spec sweep_specs/kendama_baseline250_policy_emb.yaml

Single combo (state)::

    python scripts/cluster_kendama_rollouts.py \\
        --rollouts /mnt/ssdB/erik/rollouts/rollouts_kendama_baseline_250_may22_final \\
        -K 8 --window 15 --stride 7
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="/mnt/ssdB/erik/rollouts/rollouts_kendama_baseline_250_may22_final")
    ap.add_argument(
        "--out_root",
        default="data/demo_sweep/kendama_baseline250/run_clustering/clustering/aggregate_first",
    )
    ap.add_argument("--mp4_out", default="/tmp/study_mp4s/kendama_may22")
    ap.add_argument("-K", "--n_clusters", type=int, default=8)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task_config", default="kendama_baseline250")
    ap.add_argument("--no_mp4_index", action="store_true")
    args = ap.parse_args()

    trunk = _REPO_ROOT / args.out_root / "_trunks" / "state"
    builder = _REPO_ROOT / "scripts" / "build_kendama_clustering.py"
    python = sys.executable

    if not (trunk / "timestep_cache.npz").exists():
        subprocess.check_call([
            python, str(builder),
            "--timestep_embed_only",
            "--representation", "state",
            "--rollouts", args.rollouts,
            "--out_dir", str(trunk),
            "--task_config", args.task_config,
        ])

    from scripts.kendama_clustering_lib import slug_for_combo

    slug = slug_for_combo(
        representation="state",
        window=args.window,
        stride=args.stride,
        n_clusters=args.n_clusters,
        seed=args.seed,
    )
    out_dir = _REPO_ROOT / args.out_root / slug
    subprocess.check_call([
        python, str(builder),
        "--timestep_embed_dir", str(trunk),
        "--rollouts", args.rollouts,
        "--out_dir", str(out_dir),
        "--window_width", str(args.window),
        "--stride", str(args.stride),
        "--n_clusters", str(args.n_clusters),
        "--seed", str(args.seed),
        "--task_config", args.task_config,
    ])

    if not args.no_mp4_index:
        _build_mp4_index(Path(args.rollouts), Path(args.mp4_out))

    print(f"\nClustering saved to {out_dir}")
    print(f"mp4_dir: {args.mp4_out}")
    print(f"clustering_dir: {out_dir}")


def _build_mp4_index(rollouts_dir: Path, mp4_out: Path) -> None:
    import json

    from scripts.kendama_clustering_lib import load_episodes

    mp4_out.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes(rollouts_dir, mode="state")
    index_eps = []
    for ep_idx, ep in enumerate(episodes):
        src = ep.ep_dir / "exterior.mp4"
        suffix = "succ" if ep.success else "fail"
        dst = mp4_out / f"ep{ep_idx:04d}_{suffix}.mp4"
        if not dst.exists() and src.exists():
            dst.symlink_to(src)
        index_eps.append(dict(
            index=ep_idx,
            path=str(dst),
            frame_count=ep.n_steps,
            success=ep.success,
        ))
    (mp4_out / "index.json").write_text(json.dumps({"episodes": index_eps}, indent=2))


if __name__ == "__main__":
    main()
