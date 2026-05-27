"""Build one Kendama clustering dir from cached per-timestep features.

Used by ``run_clustering_sweep.py`` (via ``--builder``) after the embed trunk
is built. Loads the timestep cache, aggregates windows for one (W, S), runs
UMAP once if needed, and K-means for one K.

Usage:

    # Build embed trunk once:
    python scripts/build_kendama_clustering.py \\
        --timestep_embed_only \\
        --rollouts /mnt/ssdB/erik/rollouts/rollouts_kendama_baseline_250_may22_final \\
        --ckpt /mnt/ssdB/erik/rollouts/baseline_250_demos.ckpt \\
        --out_dir data/demo_sweep/kendama_baseline250/_trunks/policy_emb

    # Branch: one W/S/K combo (fast):
    python scripts/build_kendama_clustering.py \\
        --timestep_embed_dir data/demo_sweep/kendama_baseline250/_trunks/policy_emb \\
        --out_dir data/demo_sweep/.../policy_emb_... \\
        --window_width 15 --stride 7 --n_clusters 8 \\
        --rollouts /mnt/ssdB/erik/rollouts/rollouts_kendama_baseline_250_may22_final
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.kendama_clustering_lib import (  # noqa: E402
    aggregate_windows,
    build_manifest,
    cluster_coords,
    extract_policy_timesteps,
    extract_state_action_timesteps,
    extract_state_timesteps,
    fit_umap_coords,
    load_episodes,
    load_timestep_cache,
    save_clustering_dir,
    save_timestep_cache,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--representation", default="policy_emb",
                    choices=["state", "state_action", "policy_emb"])
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--ckpt", default=None, help="Required for policy_emb trunk mode.")
    ap.add_argument("--layer", default="bottleneck_plan_t0")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--window_width", type=int, default=15)
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--n_clusters", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task_config", default="kendama_baseline250")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--timestep_embed_only", action="store_true")
    ap.add_argument("--timestep_embed_dir", default=None)
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    rollouts_dir = Path(args.rollouts)
    out_dir = Path(args.out_dir)
    t0 = time.time()

    if args.timestep_embed_only:
        print(f"[kendama-trunk] {args.representation}  rollouts={rollouts_dir}", flush=True)
        if args.representation == "state":
            episodes = load_episodes(rollouts_dir, mode="state")
            per_ts, successes = extract_state_timesteps(episodes)
            meta = dict(representation="state", rollouts=str(rollouts_dir))
        elif args.representation == "state_action":
            episodes = load_episodes(rollouts_dir, mode="state_action")
            per_ts, successes = extract_state_action_timesteps(episodes)
            meta = dict(representation="state_action", rollouts=str(rollouts_dir))
        else:
            if not args.ckpt:
                raise SystemExit("--ckpt required for policy_emb trunk mode")
            import torch

            episodes = load_episodes(rollouts_dir, mode="policy")
            per_ts, successes = extract_policy_timesteps(
                episodes,
                ckpt_path=Path(args.ckpt),
                layer=args.layer,
                batch_size=args.batch_size,
                device=torch.device(args.device),
            )
            meta = dict(
                representation="policy_emb",
                rollouts=str(rollouts_dir),
                ckpt=str(args.ckpt),
                layer=args.layer,
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_path = out_dir / "timestep_cache.npz"
        save_timestep_cache(cache_path, per_ts=per_ts, successes=successes,
                            representation=args.representation, meta=meta)
        manifest = dict(
            timestep_embed_only=True,
            representation=args.representation,
            n_episodes=len(per_ts),
            n_timesteps=int(sum(len(e) for e in per_ts)),
            **meta,
        )
        (out_dir / "embed_manifest.yaml").write_text(yaml.dump(manifest, sort_keys=False))
        print(f"  {len(per_ts)} episodes, cache={cache_path.name}", flush=True)
        print(f"  [kendama-trunk] done {time.time()-t0:.1f}s → {out_dir}", flush=True)
        return 0

    if args.timestep_embed_dir is None:
        raise SystemExit("Use --timestep_embed_only or --timestep_embed_dir")
    if args.n_clusters is None:
        raise SystemExit("--n_clusters required in branch mode")

    trunk = Path(args.timestep_embed_dir)
    cache_path = trunk / "timestep_cache.npz"
    embed_manifest = yaml.safe_load((trunk / "embed_manifest.yaml").read_text())
    representation = embed_manifest["representation"]
    layer = embed_manifest.get("layer")

    print(
        f"[kendama-branch] w={args.window_width} s={args.stride} K={args.n_clusters}",
        flush=True,
    )
    per_ts, successes, _ = load_timestep_cache(cache_path)
    features, metadata = aggregate_windows(
        per_ts, successes,
        window=args.window_width,
        stride=args.stride,
        representation=representation,
    )
    if len(features) == 0:
        raise SystemExit(f"No windows for W={args.window_width} S={args.stride}")
    coords = fit_umap_coords(features, seed=args.seed)
    labels = cluster_coords(coords, n_clusters=args.n_clusters, seed=args.seed)
    manifest = build_manifest(
        representation=representation,
        n_clusters=args.n_clusters,
        n_samples=len(features),
        window=args.window_width,
        stride=args.stride,
        seed=args.seed,
        task_config=args.task_config,
        rollouts=str(rollouts_dir),
        ckpt=embed_manifest.get("ckpt"),
        layer=layer,
    )
    save_clustering_dir(out_dir, labels=labels, coords=coords, metadata=metadata, manifest=manifest)
    print(f"  slices={len(features)}  sizes={list(__import__('numpy').bincount(labels))}", flush=True)
    print(f"  [kendama-branch] done {time.time()-t0:.1f}s → {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
