#!/usr/bin/env python
"""CLI for interactive DAgger rollouts on robocasa with behavior graph intervention timing.

Usage:
    python scripts/run_dagger_robocasa.py \
      --train_dir third_party/cupid/data/outputs/train/... \
      --train_ckpt best \
      --infembed_fit /path/to/infembed_fit.pt \
      --infembed_npz /path/to/infembed_embeddings.npz \
      --clustering_dir /path/to/clustering/result \
      --dataset_path data/robocasa/datasets/kitchen_lowdim_merged.hdf5 \
      --output_dir /tmp/dagger_rollout \
      --num_episodes 5 \
      --intervention_threshold 0.0 \
      --device auto

Environment: requires cupid_torch2 conda env (PyTorch 2.x, torch.func for InfEmbed)
"""

from __future__ import annotations

import os
from pathlib import Path

import click
import torch


def auto_device() -> str:
    """Auto-detect device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@click.command()
@click.option(
    "--train_dir",
    required=True,
    type=click.Path(exists=True),
    help="Training output directory (contains checkpoint/)",
)
@click.option(
    "--train_ckpt",
    default="best",
    help="Checkpoint name ('best', 'latest', or epoch number)",
)
@click.option(
    "--infembed_fit",
    required=True,
    type=click.Path(exists=True),
    help="Path to infembed_fit.pt",
)
@click.option(
    "--infembed_npz",
    required=True,
    type=click.Path(exists=True),
    help="Path to infembed_embeddings.npz",
)
@click.option(
    "--clustering_dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to clustering result directory",
)
@click.option(
    "--dataset_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to robomimic HDF5 dataset (kitchen_lowdim_merged.hdf5)",
)
@click.option(
    "--output_dir",
    required=True,
    type=click.Path(),
    help="Output directory for DAgger episodes",
)
@click.option(
    "--num_episodes",
    default=5,
    type=int,
    help="Number of DAgger episodes to run",
)
@click.option(
    "--intervention_threshold",
    default=0.0,
    type=float,
    help="V-value threshold for automatic intervention (lower = more sensitive)",
)
@click.option(
    "--device",
    default="auto",
    type=str,
    help="PyTorch device ('cuda:0', 'mps', 'cpu', or 'auto')",
)
@click.option(
    "--no_visualization",
    is_flag=True,
    help="Disable live matplotlib visualization",
)
@click.option(
    "--dagger_config",
    default="keyboard_default",
    type=str,
    help="DAgger config preset (keyboard_default, spacemouse_default, defaults, etc.)",
)
def main(
    train_dir: str,
    train_ckpt: str,
    infembed_fit: str,
    infembed_npz: str,
    clustering_dir: str,
    dataset_path: str,
    output_dir: str,
    num_episodes: int,
    intervention_threshold: float,
    device: str,
    no_visualization: bool,
    dagger_config: str,
) -> None:
    """Run DAgger episodes on robocasa with behavior graph intervention timing."""

    from robomimic.utils.env_utils import EnvUtils
    from robomimic.utils.file_utils import FileUtils
    from robomimic.utils.obs_utils import ObsUtils

    from policy_doctor.behaviors.behavior_graph import BehaviorGraph
    from policy_doctor.behaviors.behavior_values import get_behavior_graph_and_slice_values
    from policy_doctor.data.adapters import ensure_robocasa_on_path
    from policy_doctor.data.clustering_loader import load_clustering_result_from_path
    from policy_doctor.envs import (
        DAggerVisualizer,
        RobomimicDAggerEnv,
        RobomimicDAggerRunner,
    )
    from policy_doctor.envs.dagger_config import (
        create_intervention_device,
        get_intervention_threshold,
        load_dagger_config,
    )
    from policy_doctor.gym_util.multistep_wrapper import MultiStepWrapper
    from policy_doctor.monitoring.intervention import NodeValueThresholdRule
    from policy_doctor.monitoring.monitored_policy import MonitoredPolicy
    from policy_doctor.monitoring.trajectory_classifier import TrajectoryClassifier
    from policy_doctor.env.robomimic.robomimic_lowdim_wrapper import (
        RobomimicLowdimWrapper,
    )

    # Auto-detect device if requested
    if device == "auto":
        device = auto_device()
    print(f"Using device: {device}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load TrajectoryClassifier and build MonitoredPolicy ---
    print("Loading checkpoint and behavior graph...")
    classifier = TrajectoryClassifier.from_checkpoint(
        checkpoint=str(Path(train_dir) / f"checkpoints/{train_ckpt}.ckpt"),
        infembed_fit_path=infembed_fit,
        infembed_embeddings_path=infembed_npz,
        clustering_dir=clustering_dir,
        mode="rollout",
        device=device,
    )

    graph, node_values = get_behavior_graph_and_slice_values(
        clustering_dir=clustering_dir,
        rollout_embeddings_path=None,
    )

    # Load DAgger config
    print(f"Loading DAgger config: {dagger_config}")
    dagger_cfg = load_dagger_config(dagger_config)

    # Use config threshold if not overridden via CLI
    if intervention_threshold == 0.0:  # default value
        intervention_threshold = get_intervention_threshold(dagger_cfg)
    print(f"Intervention threshold: {intervention_threshold}")

    intervention_rule = NodeValueThresholdRule(
        node_values=node_values, threshold=intervention_threshold
    )
    monitored_policy = MonitoredPolicy(
        policy=classifier.monitor.scorer._policy,
        classifier=classifier,
        intervention_rule=intervention_rule,
    )

    # --- Create robocasa environment ---
    print("Setting up robocasa environment...")
    ensure_robocasa_on_path()

    dataset_path = Path(dataset_path)
    env_meta = FileUtils.get_env_metadata_from_dataset(str(dataset_path))

    # Extract obs_keys from checkpoint config
    obs_keys = classifier._obs_keys or [
        "object",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ]
    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": obs_keys})

    def create_env():
        """Create one instance of the robocasa env stack."""
        robomimic_env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=True,
            use_image_obs=False,
        )
        lowdim_wrapper = RobomimicLowdimWrapper(
            env=robomimic_env, obs_keys=obs_keys, init_state=None
        )
        multistep_wrapper = MultiStepWrapper(
            lowdim_wrapper,
            n_obs_steps=classifier.n_obs_steps,
            n_action_steps=classifier.n_action_steps,
            max_episode_steps=500,
        )
        dagger_env = RobomimicDAggerEnv(
            inner_env=multistep_wrapper,
            obs_keys=obs_keys,
            output_dir=output_dir,
        )
        return dagger_env

    env = create_env()

    # --- Create intervention device and visualizer ---
    device_type = dagger_cfg.get("device", "keyboard")
    print(f"Initializing {device_type} intervention device...")
    try:
        intervention_device = create_intervention_device(dagger_cfg)
        if device_type == "keyboard":
            print("  Press SPACE to toggle human/robot control")
            print("  W/S/A/D/Q/E: move arm")
            print("  G/H: gripper close/open")
            print("  I/K/J/L: move base")
        elif device_type == "spacemouse":
            print("  Use SpaceMouse 6-DOF input")
            print("  Left button (hold): gripper close")
            print("  Right button: toggle human/robot control")
    except Exception as e:
        print(f"Error creating intervention device: {e}")
        raise

    visualizer = None
    viz_cfg = dagger_cfg.get("visualization", {})
    if not no_visualization and viz_cfg.get("enabled", True):
        try:
            visualizer = DAggerVisualizer(
                camera_names=viz_cfg.get("camera_names", ["agentview"]),
                figsize=tuple(viz_cfg.get("figsize", [8, 5])),
            )
            print("Visualization enabled")
        except Exception as e:
            print(f"Failed to create visualizer: {e}")

    # --- Run DAgger episodes ---
    print("\n" + "=" * 60)
    print("Starting DAgger rollouts (press Ctrl+C to stop)")
    print("=" * 60)

    runner = RobomimicDAggerRunner(
        monitored_policy=monitored_policy,
        env=env,
        intervention_device=intervention_device,
        n_obs_steps=classifier.n_obs_steps,
        n_action_steps=classifier.n_action_steps,
        max_steps=500,
        output_dir=output_dir,
        visualizer=visualizer,
    )

    try:
        records = runner.run(num_episodes)
    except KeyboardInterrupt:
        print("\nInterrupted by user")

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"DAgger rollouts saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
