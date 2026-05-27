"""Cluster kendama rollouts by policy-bottleneck embeddings.

Thin wrapper around ``kendama_clustering_lib``. For a full W×S×K sweep, use::

    python scripts/run_kendama_clustering_sweep.py \\
        --spec sweep_specs/kendama_baseline250_policy_emb.yaml

Single combo::

    python scripts/cluster_kendama_policy_embeddings.py \\
        --rollouts /mnt/ssdB/erik/rollouts/rollouts_kendama_baseline_250_may22_final \\
        --ckpt /mnt/ssdB/erik/rollouts/baseline_250_demos.ckpt \\
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
    ap.add_argument("--ckpt", default="/mnt/ssdB/erik/rollouts/baseline_250_demos.ckpt")
    ap.add_argument(
        "--out_root",
        default="data/demo_sweep/kendama_baseline250/run_clustering/clustering/aggregate_first",
    )
    ap.add_argument("--layer", default="bottleneck_plan_t0")
    ap.add_argument("-K", "--n_clusters", type=int, default=8)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task_config", default="kendama_baseline250")
    args = ap.parse_args()

    trunk = _REPO_ROOT / args.out_root / "_trunks" / "policy_emb"
    builder = _REPO_ROOT / "scripts" / "build_kendama_clustering.py"
    python = sys.executable

    if not (trunk / "timestep_cache.npz").exists():
        subprocess.check_call([
            python, str(builder),
            "--timestep_embed_only",
            "--representation", "policy_emb",
            "--rollouts", args.rollouts,
            "--ckpt", args.ckpt,
            "--layer", args.layer,
            "--out_dir", str(trunk),
            "--batch_size", str(args.batch_size),
            "--device", args.device,
            "--task_config", args.task_config,
        ])

    from scripts.kendama_clustering_lib import slug_for_combo

    slug = slug_for_combo(
        representation="policy_emb",
        window=args.window,
        stride=args.stride,
        n_clusters=args.n_clusters,
        seed=args.seed,
        layer=args.layer,
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
    print(f"\nClustering saved to {out_dir}")


if __name__ == "__main__":
    main()
