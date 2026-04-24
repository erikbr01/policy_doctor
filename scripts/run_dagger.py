#!/usr/bin/env python
"""Generic CLI for DAgger rollouts on any robomimic-compatible environment.

Works with: robomimic, kitchen, robocasa, libero, blockpush, mimicgen, etc.

Usage:
    # Kitchen manipulation (square stacking)
    python scripts/run_dagger.py \
      --task square_mh \
      --train_dir third_party/cupid/data/outputs/train/square_mh/... \
      --train_ckpt best \
      --infembed_fit /path/to/infembed_fit.pt \
      --infembed_npz /path/to/infembed_embeddings.npz \
      --clustering_dir /path/to/clustering/result \
      --output_dir /tmp/dagger_square \
      --num_episodes 5

    # Robocasa (kitchen pick-and-place)
    python scripts/run_dagger.py \
      --task robocasa_layout_lowdim \
      --train_dir third_party/cupid/data/outputs/train/robocasa_layout_lowdim/... \
      --train_ckpt best \
      --infembed_fit /path/to/infembed_fit.pt \
      --infembed_npz /path/to/infembed_embeddings.npz \
      --clustering_dir /path/to/clustering/result \
      --output_dir /tmp/dagger_robocasa \
      --num_episodes 5

    # Kitchen transport task
    python scripts/run_dagger.py \
      --task transport_mh \
      --train_dir third_party/cupid/data/outputs/train/transport_mh/... \
      ...

Environment: requires cupid_torch2 conda env (PyTorch 2.x, torch.func for InfEmbed)
"""

from __future__ import annotations

from pathlib import Path

import click
import torch

TASK_CONFIG = {
    "square_mh": {
        "dataset_path": "data/source/kitchen_square.hdf5",
        "obs_keys": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
    },
    "lift_mh": {
        "dataset_path": "data/source/kitchen_lift.hdf5",
        "obs_keys": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
    },
    "transport_mh": {
        "dataset_path": "data/source/kitchen_transport.hdf5",
        "obs_keys": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
    },
    "robocasa_layout_lowdim": {
        "dataset_path": "data/robocasa/datasets/kitchen_lowdim_merged.hdf5",
        "obs_keys": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
    },
}


def auto_device() -> str:
    """Auto-detect device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@click.command()
@click.option(
    "--task",
    required=True,
    type=click.Choice(list(TASK_CONFIG.keys())),
    help="Task name (square_mh, lift_mh, transport_mh, robocasa_layout_lowdim, etc.)",
)
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
    default=None,
    type=click.Path(exists=True),
    help="Dataset HDF5 path (inferred from task if not provided)",
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
    task: str,
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
    """Run DAgger episodes on any robomimic-compatible environment with behavior graph intervention timing."""

    from robomimic.utils.env_utils import EnvUtils
    from robomimic.utils.file_utils import FileUtils
    from robomimic.utils.obs_utils import ObsUtils

    from policy_doctor.behaviors.behavior_graph import BehaviorGraph
    from policy_doctor.behaviors.behavior_values import get_behavior_graph_and_slice_values
    from policy_doctor.data.adapters import ensure_robocasa_on_path
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

    # Get task config
    task_cfg = TASK_CONFIG.get(task)
    if not task_cfg:
        click.echo(f"Unknown task: {task}")
        raise click.Abort()

    if dataset_path is None:
        dataset_path = task_cfg["dataset_path"]
    obs_keys = task_cfg["obs_keys"]

    # Auto-detect device if requested
    if device == "auto":
        device = auto_device()
    click.echo(f"Using device: {device}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure robocasa on path if using robocasa
    if "robocasa" in task.lower():
        ensure_robocasa_on_path()

    # --- Load TrajectoryClassifier and build MonitoredPolicy ---
    click.echo(f"Loading checkpoint and behavior graph for task: {task}")
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
    click.echo(f"Loading DAgger config: {dagger_config}")
    dagger_cfg = load_dagger_config(dagger_config)

    # Use config threshold if not overridden via CLI
    if intervention_threshold == 0.0:  # default value
        intervention_threshold = get_intervention_threshold(dagger_cfg)
    click.echo(f"Intervention threshold: {intervention_threshold}")

    intervention_rule = NodeValueThresholdRule(
        node_values=node_values, threshold=intervention_threshold
    )
    monitored_policy = MonitoredPolicy(
        policy=classifier.monitor.scorer._policy,
        classifier=classifier,
        intervention_rule=intervention_rule,
    )

    # --- Create robomimic environment ---
    click.echo(f"Setting up {task} environment...")

    dataset_path = Path(dataset_path)
    env_meta = FileUtils.get_env_metadata_from_dataset(str(dataset_path))

    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": obs_keys})

    def create_env():
        """Create one instance of the environment stack."""
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
    click.echo(f"Initializing {device_type} intervention device...")
    try:
        intervention_device = create_intervention_device(dagger_cfg)
        if device_type == "keyboard":
            click.echo("  Press SPACE to toggle human/robot control")
            click.echo("  W/S/A/D/Q/E: move arm (z, x, y)")
            click.echo("  G/H: gripper close/open")
            click.echo("  I/K/J/L: move base (x, rotation)")
        elif device_type == "spacemouse":
            click.echo("  Use SpaceMouse 6-DOF input")
            click.echo("  Left button (hold): gripper close")
            click.echo("  Right button: toggle human/robot control")
    except Exception as e:
        click.echo(f"Error creating intervention device: {e}", err=True)
        raise click.Abort()

    visualizer = None
    viz_cfg = dagger_cfg.get("visualization", {})
    if not no_visualization and viz_cfg.get("enabled", True):
        try:
            visualizer = DAggerVisualizer(
                camera_names=viz_cfg.get("camera_names", ["agentview"]),
                figsize=tuple(viz_cfg.get("figsize", [8, 5])),
            )
            click.echo("Visualization enabled")
        except Exception as e:
            click.echo(f"Warning: Failed to create visualizer: {e}")

    # --- Run DAgger episodes ---
    click.echo("\n" + "=" * 60)
    click.echo(f"Starting DAgger rollouts for task: {task}")
    click.echo("=" * 60)

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
        click.echo("\nInterrupted by user")

    # --- Summary ---
    click.echo("\n" + "=" * 60)
    click.echo(f"DAgger rollouts saved to: {output_dir}")
    click.echo("\nTo convert episodes to training dataset:")
    click.echo(f"  python scripts/build_dagger_dataset.py \\")
    click.echo(f"    --episodes_dir {output_dir} \\")
    click.echo(f"    --output_hdf5 data/{task}_dagger.hdf5 \\")
    click.echo(f"    --filter_human_only")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
