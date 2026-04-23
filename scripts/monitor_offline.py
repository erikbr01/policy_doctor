#!/usr/bin/env python3
"""Offline behavior monitor: classify each timestep of a saved rollout or demonstration.

Reads obs/action from a rollout pickle (eval_save_episodes.py output) or an HDF5
demonstration file, runs InfEmbed scoring + behavior graph assignment on each timestep,
and prints or saves a per-timestep assignment table.

Mode is auto-detected:
  --episode <pkl>    → mode=rollout (data is already in policy format)
  --hdf5 <file>      → mode=demo    (applies rotation transform if abs_action=True)

Both modes read abs_action, rotation_rep, obs_keys, and horizon lengths directly from
the policy checkpoint config, so no extra flags are needed.

Usage (cupid conda env):

    # From a rollout pickle
    python scripts/monitor_offline.py \\
        --episode <output_dir>/episodes/ep0000_succ.pkl \\
        --checkpoint <train_dir>/checkpoints/latest.ckpt \\
        --infembed_fit <eval_dir>/infembed_fit.pt \\
        --infembed_npz <eval_dir>/infembed_embeddings.npz \\
        --clustering_dir <configs_root>/clustering/<slug> \\
        --output assignments.csv

    # From an HDF5 demonstration
    python scripts/monitor_offline.py \\
        --hdf5 <path>/dataset.hdf5 --demo demo_0 \\
        --checkpoint <train_dir>/checkpoints/latest.ckpt \\
        --infembed_fit <eval_dir>/infembed_fit.pt \\
        --infembed_npz <eval_dir>/infembed_embeddings.npz \\
        --clustering_dir <configs_root>/clustering/<slug>
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from collections import Counter
from pathlib import Path


def _build_classifier(args, mode: str):
    from policy_doctor.monitoring.trajectory_classifier import TrajectoryClassifier
    return TrajectoryClassifier.from_checkpoint(
        checkpoint=args.checkpoint,
        infembed_fit_path=args.infembed_fit,
        infembed_embeddings_path=args.infembed_npz,
        clustering_dir=args.clustering_dir,
        mode=mode,
        device=args.device,
    )


def _results_to_rows(timesteps_and_results, episode: int = 0):
    rows = []
    for t, r in timesteps_and_results:
        rows.append({
            "episode": episode,
            "timestep": t,
            "cluster_id": r.assignment.cluster_id if r.assignment else "",
            "node_id": r.assignment.node_id if r.assignment else "",
            "node_name": r.assignment.node_name if r.assignment else "",
            "distance": f"{r.assignment.distance:.4f}" if r.assignment else "",
            "total_ms": f"{r.timing_ms.get('total_ms', 0):.1f}",
        })
    return rows


def _print_table(rows):
    if not rows:
        print("(no results)")
        return
    header = list(rows[0].keys())
    widths = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in header}
    sep = "  "
    fmt = sep.join(f"{{:<{widths[k]}}}" for k in header)
    print(fmt.format(*header))
    print(sep.join("-" * widths[k] for k in header))
    for r in rows:
        print(fmt.format(*[str(r[k]) for k in header]))


def _print_summary(rows):
    counts = Counter(r["node_name"] for r in rows if r["node_name"])
    total = len(rows)
    print("\nNode distribution:")
    for name, n in counts.most_common():
        pct = 100.0 * n / total
        print(f"  {name:<30s}  {n:4d} steps  ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Classify each timestep of a rollout or demonstration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument("--episode", metavar="PKL",
                     help="Rollout pickle from eval_save_episodes.py")
    inp.add_argument("--hdf5", metavar="HDF5",
                     help="HDF5 demonstration file")

    parser.add_argument("--demo", default="demo_0",
                        help="Demo key inside HDF5 data/ group (default: demo_0)")

    parser.add_argument("--checkpoint", required=True, metavar="CKPT",
                        help="Policy checkpoint (.ckpt)")
    parser.add_argument("--infembed_fit", required=True, metavar="PT",
                        help="infembed_fit.pt path")
    parser.add_argument("--infembed_npz", required=True, metavar="NPZ",
                        help="infembed_embeddings.npz path")
    parser.add_argument("--clustering_dir", required=True, metavar="DIR",
                        help="Clustering result directory")

    parser.add_argument("--output", metavar="CSV", default=None,
                        help="Save assignments to this CSV path (default: print only)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episode_idx", type=int, default=0, metavar="N",
                        help="Episode index written to the 'episode' column in the output CSV "
                             "(default: 0). Useful when appending results from multiple episodes.")

    args = parser.parse_args()

    mode = "demo" if args.hdf5 else "rollout"
    print(f"Building classifier (mode={mode})...")
    classifier = _build_classifier(args, mode)

    if args.episode:
        print(f"\nLoading episode: {args.episode}")
        with open(args.episode, "rb") as f:
            episode_df = pickle.load(f)
        print(f"  {len(episode_df)} timesteps")
        results = classifier.classify_episode_from_pkl(episode_df)

    else:
        import h5py
        print(f"\nLoading HDF5: {args.hdf5}  demo={args.demo}")
        with h5py.File(args.hdf5, "r") as f:
            demo_key = f"data/{args.demo}"
            if demo_key not in f:
                sys.exit(f"Demo key {demo_key!r} not found. Available: {list(f['data'].keys())[:5]}")
            demo_group = f[demo_key]
            T = len(demo_group["actions"])
            print(f"  {T} timesteps")
            results = classifier.classify_demo_from_hdf5(demo_group)

    print(f"\nClassified {len(results)} timesteps\n")
    rows = _results_to_rows(results, episode=args.episode_idx)
    _print_table(rows)
    _print_summary(rows)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
