#!/usr/bin/env python3
"""Online behavior monitor: run policy eval with per-timestep behavior classification.

Mirrors eval_save_episodes.py but wraps the policy in MonitoredPolicy so that each
predict_action() call is classified and logged in real time. Results are saved as a
CSV alongside the standard episode output.

The script must be run from third_party/cupid/ (or with PYTHONPATH including
diffusion_policy), because it loads the policy workspace via hydra.

Usage (cupid conda env, run from third_party/cupid/):

    python ../../scripts/monitor_online.py \\
        --output_dir /tmp/monitor_run \\
        --train_dir <train_dir> \\
        --train_ckpt best \\
        --infembed_fit <eval_dir>/infembed_fit.pt \\
        --infembed_npz <eval_dir>/infembed_embeddings.npz \\
        --clustering_dir <configs_root>/clustering/<slug> \\
        --num_episodes 10 \\
        --verbose True
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import shutil
import sys
from collections import Counter
from pathlib import Path

# Ensure the repo root (two levels above this script's dir) is on sys.path so
# that policy_doctor is importable when the script is run from third_party/cupid/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import Optional

import click
import dill
import hydra
import torch

from diffusion_policy.common.device_util import get_device
from diffusion_policy.common.trak_util import get_best_checkpoint, get_index_checkpoint
from diffusion_policy.workspace.base_workspace import BaseWorkspace


@click.command()
@click.option("--output_dir", required=True, help="Directory for episode output and monitor CSV")
@click.option("--train_dir", required=True, help="Training output directory (contains checkpoints/)")
@click.option("--train_ckpt", required=True, help="Checkpoint: 'best', an epoch int, or filename stem")
@click.option("--infembed_fit", required=True, help="Path to infembed_fit.pt")
@click.option("--infembed_npz", required=True, help="Path to infembed_embeddings.npz")
@click.option("--clustering_dir", required=True, help="Path to clustering result directory")
@click.option("--num_episodes", type=int, default=10, help="Number of test episodes to run")
@click.option("--test_start_seed", type=int, default=100000, help="Random seed for test envs")
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing output_dir")
@click.option("--device", default="cuda:0")
@click.option("--verbose", is_flag=True, default=False,
              help="Print node assignment after each predict_action() call")
@click.option(
    "--episodes_dir", default=None,
    help="Path to eval episodes directory (contains metadata.yaml with episode_lengths). "
         "Required when clustering was done at the window level and rollout_embeddings "
         "from the npz are at the timestep level (NearestCentroidAssigner fallback only).",
)
def main(
    output_dir: str,
    train_dir: str,
    train_ckpt: str,
    infembed_fit: str,
    infembed_npz: str,
    clustering_dir: str,
    num_episodes: int,
    test_start_seed: int,
    overwrite: bool,
    device: str,
    verbose: bool,
    episodes_dir: Optional[str],
):
    # ------------------------------------------------------------------
    # Load checkpoint (mirrors eval_save_episodes.py)
    # ------------------------------------------------------------------
    checkpoint_dir = pathlib.Path(train_dir) / "checkpoints"
    checkpoints = list(checkpoint_dir.iterdir())
    if train_ckpt.isdigit():
        checkpoint = get_index_checkpoint(checkpoints, int(train_ckpt))
    elif train_ckpt == "best":
        checkpoint = get_best_checkpoint(checkpoints)
    else:
        checkpoint = checkpoint_dir / f"{train_ckpt}.ckpt"
    print(f"Checkpoint: {checkpoint}")

    if os.path.exists(output_dir):
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise click.UsageError(f"Output path {output_dir} already exists. Pass --overwrite to replace.")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    payload = torch.load(open(str(checkpoint), "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)

    cfg.task.env_runner.n_envs = 1
    cfg.task.env_runner.n_train = 0
    cfg.task.env_runner.n_train_vis = 0
    cfg.task.env_runner.n_test = num_episodes
    cfg.task.env_runner.n_test_vis = 0  # disable offscreen video capture
    cfg.task.env_runner.test_start_seed = test_start_seed

    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if getattr(cfg.training, "use_ema", False):
        policy = workspace.ema_model

    torch_device = get_device(device)
    policy.to(torch_device)
    policy.eval()

    # ------------------------------------------------------------------
    # Build classifier + wrap policy
    # ------------------------------------------------------------------
    from policy_doctor.monitoring.monitored_policy import MonitoredPolicy
    from policy_doctor.monitoring.trajectory_classifier import TrajectoryClassifier

    print("\nBuilding TrajectoryClassifier...")
    classifier = TrajectoryClassifier.from_checkpoint(
        checkpoint=str(checkpoint),
        infembed_fit_path=infembed_fit,
        infembed_embeddings_path=infembed_npz,
        clustering_dir=clustering_dir,
        mode="rollout",
        device=device,
        episodes_dir=episodes_dir,
    )
    monitored = MonitoredPolicy(policy=policy, classifier=classifier, verbose=verbose)

    # ------------------------------------------------------------------
    # Run eval
    # ------------------------------------------------------------------
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir,
        save_episodes=True,
    )
    print(f"\nRunning {num_episodes} episodes...")
    runner_log = env_runner.run(monitored)

    # ------------------------------------------------------------------
    # Save monitor assignments CSV
    # ------------------------------------------------------------------
    results_path = Path(output_dir) / "monitor_assignments.csv"
    if monitored.episode_results:
        fieldnames = [k for k in monitored.episode_results[0] if k != "result"]
        with open(results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in monitored.episode_results:
                writer.writerow({k: entry[k] for k in fieldnames})
        print(f"\nMonitor assignments → {results_path}  ({len(monitored.episode_results)} rows)")

        total = len(monitored.episode_results)
        counts = Counter(e["node_name"] for e in monitored.episode_results)
        print("\nNode distribution across all episodes:")
        for name, n in counts.most_common():
            print(f"  {name:<30s}  {n:4d} steps  ({100*n/total:.1f}%)")

    # ------------------------------------------------------------------
    # Save eval log (mirrors eval_save_episodes.py)
    # ------------------------------------------------------------------
    try:
        import wandb
        json_log = {
            k: v._path if isinstance(v, wandb.sdk.data_types.video.Video) else v
            for k, v in runner_log.items()
        }
    except ImportError:
        json_log = {k: str(v) for k, v in runner_log.items()}

    log_path = Path(output_dir) / "eval_log.json"
    json.dump(json_log, open(log_path, "w"), indent=2, sort_keys=True)
    print(f"Eval log → {log_path}")


if __name__ == "__main__":
    main()
