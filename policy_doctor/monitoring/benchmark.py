"""Benchmark timing and storage for the runtime monitor.

Measures how fast each stage is (gradient, projection, scoring, assignment)
and how much disk space the required cached artifacts occupy.

Usage (cupid conda env):

    python -m policy_doctor.monitoring.benchmark \\
        --checkpoint third_party/cupid/data/outputs/train/.../checkpoints/latest.ckpt \\
        --trak_dir <eval_dir>/<exp_name> \\                    # optional
        --infembed_fit <eval_dir>/<exp_name>/infembed_fit.pt \\ # optional
        --infembed_npz <eval_dir>/<exp_name>/infembed_embeddings.npz \\
        --clustering_dir third_party/influence_visualizer/configs/<task>/clustering/<slug> \\
        --obs_dim 20 --action_dim 14 \\
        --obs_horizon 2 --action_horizon 8 \\
        --n_samples 50 --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_mb(path: Optional[Path]) -> str:
    if path is None or not Path(path).exists():
        return "N/A"
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) if Path(path).is_dir() else Path(path).stat().st_size
    return f"{size / 1e6:.1f} MB"


def _random_batch(obs_dim: int, action_dim: int, obs_horizon: int, action_horizon: int, num_ts: int, device: torch.device) -> dict:
    return {
        "obs": torch.randn(1, obs_horizon, obs_dim, device=device),
        "action": torch.randn(1, action_horizon, action_dim, device=device),
        "timesteps": torch.randint(100, (1, num_ts), device=device).long(),
    }


def _benchmark_scorer(scorer, batches, label: str) -> dict:
    embed_times = []
    score_times = []
    for batch in batches:
        t0 = time.perf_counter()
        embedding = scorer.embed(batch)
        t1 = time.perf_counter()
        _ = scorer.score(batch)
        t2 = time.perf_counter()
        embed_times.append((t1 - t0) * 1e3)
        score_times.append((t2 - t1) * 1e3)

    n = len(embed_times)
    print(f"\n=== {label} (n={n}) ===")
    print(f"  embed (gradient+project): {np.mean(embed_times):.1f} ± {np.std(embed_times):.1f} ms")
    print(f"  score (dot product):      {np.mean(score_times):.1f} ± {np.std(score_times):.1f} ms")
    print(f"  total:                    {np.mean(embed_times) + np.mean(score_times):.1f} ms / sample")
    return {"embed_ms": embed_times, "score_ms": score_times}


def _benchmark_assigner(assigner, embeddings) -> list:
    times = []
    for emb in embeddings:
        t0 = time.perf_counter()
        assigner.assign(emb)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
    print(f"\n=== Graph assignment (n={len(times)}) ===")
    print(f"  assign (nearest centroid): {np.mean(times):.3f} ± {np.std(times):.3f} ms")
    return times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Policy .ckpt path")
    parser.add_argument("--trak_dir", default=None, help="TRAK save_dir (optional)")
    parser.add_argument("--trak_model_id", type=int, default=0)
    parser.add_argument("--infembed_fit", default=None, help="Path to infembed_fit.pt (optional)")
    parser.add_argument("--infembed_npz", default=None, help="Path to infembed_embeddings.npz")
    parser.add_argument("--clustering_dir", default=None, help="Path to clustering result dir (for assigner)")
    parser.add_argument("--model_keys", default=None, help="Comma-separated model_keys (same as attribution run)")
    parser.add_argument("--loss_fn", default="square")
    parser.add_argument("--num_diffusion_timesteps", type=int, default=8)
    parser.add_argument("--obs_dim", type=int, required=True, help="Observation state dim Do")
    parser.add_argument("--action_dim", type=int, required=True, help="Action dim Da")
    parser.add_argument("--obs_horizon", type=int, default=2, help="Observation horizon To")
    parser.add_argument("--action_horizon", type=int, default=8, help="Action horizon Ta")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of random samples to benchmark")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations (excluded from stats)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    device = torch.device(args.device)

    # --- Generate random batches ---
    all_batches = [
        _random_batch(
            args.obs_dim, args.action_dim, args.obs_horizon, args.action_horizon,
            args.num_diffusion_timesteps, device,
        )
        for _ in range(args.n_samples + args.warmup)
    ]

    results = {}

    # --- TRAK ---
    if args.trak_dir is not None:
        print("Loading TRAKStreamScorer...")
        from policy_doctor.monitoring.trak_scorer import TRAKStreamScorer
        trak_scorer = TRAKStreamScorer(
            checkpoint=args.checkpoint,
            trak_save_dir=args.trak_dir,
            model_id=args.trak_model_id,
            grad_wrt=[k.strip() for k in args.model_keys.split(",")] if args.model_keys else None,
            num_diffusion_timesteps=args.num_diffusion_timesteps,
            loss_fn=args.loss_fn,
            device=args.device,
        )
        # Warmup
        for b in all_batches[:args.warmup]:
            trak_scorer.embed(b)
        results["trak"] = _benchmark_scorer(trak_scorer, all_batches[args.warmup:], "TRAK")

    # --- InfEmbed ---
    if args.infembed_fit is not None and args.infembed_npz is not None:
        print("Loading InfEmbedStreamScorer...")
        from policy_doctor.monitoring.infembed_scorer import InfEmbedStreamScorer
        ie_scorer = InfEmbedStreamScorer(
            checkpoint=args.checkpoint,
            infembed_fit_path=args.infembed_fit,
            infembed_embeddings_path=args.infembed_npz,
            model_keys=args.model_keys,
            num_diffusion_timesteps=args.num_diffusion_timesteps,
            loss_fn=args.loss_fn,
            device=args.device,
        )
        for b in all_batches[:args.warmup]:
            ie_scorer.embed(b)
        results["infembed"] = _benchmark_scorer(ie_scorer, all_batches[args.warmup:], "InfEmbed")

        # --- Graph assignment (uses InfEmbed embeddings) ---
        if args.clustering_dir is not None:
            print("Building NearestCentroidAssigner...")
            from policy_doctor.behaviors.behavior_graph import BehaviorGraph
            from policy_doctor.data.clustering_loader import load_clustering_result_from_path
            from policy_doctor.monitoring.graph_assigner import NearestCentroidAssigner

            labels, meta, manifest = load_clustering_result_from_path(Path(args.clustering_dir))
            graph = BehaviorGraph.from_cluster_assignments(
                cluster_labels=labels,
                metadata=meta,
                level=manifest.get("level", "rollout"),
            )
            rollout_embs = ie_scorer.rollout_embeddings
            if rollout_embs is None:
                print("  [skip] rollout_embeddings not in npz; cannot build assigner")
            else:
                assigner = NearestCentroidAssigner(
                    rollout_embeddings=rollout_embs,
                    cluster_labels=labels,
                    graph=graph,
                )
                sample_embeddings = [
                    ie_scorer.embed(all_batches[args.warmup + i])
                    for i in range(min(args.n_samples, 20))
                ]
                results["assign"] = _benchmark_assigner(assigner, sample_embeddings)

    # --- Storage report ---
    print("\n=== Storage requirements ===")
    trak_dir = Path(args.trak_dir) if args.trak_dir else None
    if trak_dir:
        features_path = trak_dir / str(args.trak_model_id) / "features.mmap"
        print(f"  TRAK features.mmap:          {_file_mb(features_path)}")
        print(f"  TRAK save_dir (total):       {_file_mb(trak_dir)}")
    if args.infembed_fit:
        print(f"  InfEmbed fit.pt:             {_file_mb(Path(args.infembed_fit))}")
    if args.infembed_npz:
        print(f"  InfEmbed embeddings.npz:     {_file_mb(Path(args.infembed_npz))}")
    print(f"  Policy checkpoint:           {_file_mb(Path(args.checkpoint))}")

    print("\nDone.")
    return results


if __name__ == "__main__":
    main()
