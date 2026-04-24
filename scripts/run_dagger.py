#!/usr/bin/env python
"""Generic CLI for DAgger rollouts on any robomimic-compatible environment.

Works with: robomimic, kitchen, robocasa, libero, blockpush, mimicgen, etc.

Run from third_party/cupid/ (diffusion_policy must be on PYTHONPATH):

    # Kitchen square stacking (square_mh_feb5)
    python ../../scripts/run_dagger.py \
      --task square_mh \
      --train_dir /abs/path/to/cupid/data/outputs/train/feb5/feb5_train_diffusion_unet_lowdim_square_mh_0 \
      --train_ckpt best \
      --infembed_fit /abs/path/to/infembed_fit.pt \
      --infembed_npz /abs/path/to/infembed_embeddings.npz \
      --clustering_dir /abs/path/to/clustering/result \
      --output_dir /tmp/dagger_square \
      --num_episodes 5

    # Robocasa (kitchen pick-and-place)
    python ../../scripts/run_dagger.py \
      --task robocasa_layout_lowdim \
      --train_dir /abs/path/to/train_dir \
      ...

Environment: requires cupid_torch2 conda env (PyTorch 2.x, torch.func for InfEmbed)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure policy_doctor is importable when the script is run from third_party/cupid/
_PD_ROOT = Path(__file__).resolve().parent.parent
if str(_PD_ROOT) not in sys.path:
    sys.path.insert(0, str(_PD_ROOT))

import click
import torch

from policy_doctor.paths import REPO_ROOT

TASK_CONFIG = {
    "square_mh": {
        "dataset_path": str(REPO_ROOT / "data" / "robomimic" / "datasets" / "square" / "mh" / "low_dim_abs.hdf5"),
        "obs_keys": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
    },
    "lift_mh": {
        "dataset_path": str(REPO_ROOT / "data" / "robomimic" / "datasets" / "lift" / "mh" / "low_dim_abs.hdf5"),
        "obs_keys": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
    },
    "transport_mh": {
        "dataset_path": str(REPO_ROOT / "data" / "robomimic" / "datasets" / "transport" / "mh" / "low_dim_abs.hdf5"),
        "obs_keys": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
    },
    "robocasa_layout_lowdim": {
        "dataset_path": str(REPO_ROOT / "data" / "robocasa" / "datasets" / "kitchen_lowdim_merged.hdf5"),
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


def resolve_checkpoint(train_dir: str, train_ckpt: str) -> Path:
    """Resolve 'best', 'latest', or epoch number to an actual .ckpt path."""
    import re
    checkpoint_dir = Path(train_dir) / "checkpoints"
    if train_ckpt == "best":
        best, best_score = None, -1.0
        for p in checkpoint_dir.iterdir():
            m = re.search(r"test_mean_score=(\d+\.\d+)", p.name)
            if m and float(m.group(1)) > best_score:
                best_score, best = float(m.group(1)), p
        if best is not None:
            return best
        return checkpoint_dir / "latest.ckpt"
    elif train_ckpt.isdigit():
        target = int(train_ckpt)
        best, best_dist = None, float("inf")
        for p in checkpoint_dir.iterdir():
            m = re.search(r"epoch=(\d+)", p.name)
            if m and abs(int(m.group(1)) - target) < best_dist:
                best_dist, best = abs(int(m.group(1)) - target), p
        if best is not None:
            return best
    return checkpoint_dir / f"{train_ckpt}.ckpt"


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
@click.option(
    "--episodes_dir",
    default=None,
    type=click.Path(exists=True),
    help="Directory with metadata.yaml for window-level NearestCentroidAssigner (required for sliding-window clustering).",
)
@click.option(
    "--no_monitor",
    is_flag=True,
    help="Skip InfEmbed monitoring entirely — just run the raw policy. "
         "Removes the per-step gradient pass so inference is ~2x faster.",
)
@click.option(
    "--server_url",
    default=None,
    help="HTTP policy server URL (e.g. http://localhost:5001). "
         "When set, policy inference is delegated to the server process "
         "instead of running locally. Start the server with "
         "run_policy_server_square_feb5.sh first.",
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
    episodes_dir: str,
    no_monitor: bool,
    server_url: str,
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
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
    from policy_doctor.monitoring.intervention import NodeValueThresholdRule
    from policy_doctor.monitoring.monitored_policy import MonitoredPolicy
    from policy_doctor.monitoring.trajectory_classifier import TrajectoryClassifier
    from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import (
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

    # --- Load policy (+ optionally the full monitoring stack) ---
    checkpoint = resolve_checkpoint(train_dir, train_ckpt)
    click.echo(f"Checkpoint: {checkpoint}")

    if server_url:
        # HTTP server mode: delegate all inference to the running server process.
        # No local model loading — the server handles its own MPS context.
        from policy_doctor.envs.policy_server import PolicyClient
        import requests as _requests
        click.echo(f"Using policy server at {server_url}")
        try:
            resp = _requests.get(f"{server_url}/health", timeout=5)
            info = resp.json()
            click.echo(f"  server status: {info.get('status')} | device: {info.get('device')}")
        except Exception as e:
            raise click.ClickException(
                f"Cannot reach policy server at {server_url}: {e}\n"
                f"Start it with: ./scripts/experiments/run_policy_server_square_feb5.sh"
            )

        # Read abs_action from checkpoint config for the action transform
        import dill as _dill, torch as _torch
        payload = _torch.load(open(str(checkpoint), "rb"), pickle_module=_dill, map_location="cpu")
        from omegaconf import OmegaConf as _OC
        abs_action = bool(_OC.select(payload["cfg"], "task.dataset.abs_action") or False)
        n_obs_steps_val = int(_OC.select(payload["cfg"], "n_obs_steps") or 2)
        n_action_steps_val = int(_OC.select(payload["cfg"], "n_action_steps") or 8)
        del payload

        monitored_policy = PolicyClient(url=server_url)
        monitored_policy.start()
        classifier_n_obs_steps = n_obs_steps_val
        classifier_n_action_steps = n_action_steps_val
        dagger_cfg = load_dagger_config(dagger_config)
    elif no_monitor:
        # Bare-policy mode: load checkpoint directly, skip InfEmbed entirely.
        click.echo("Monitor disabled — loading bare policy (no InfEmbed gradient pass)")
        import dill as _dill
        import hydra as _hydra
        import torch as _torch
        from diffusion_policy.common.trak_util import get_best_checkpoint
        payload = _torch.load(open(str(checkpoint), "rb"), pickle_module=_dill)
        cfg = payload["cfg"]
        cls = _hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=str(output_dir))
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        raw_policy = workspace.ema_model if getattr(cfg.training, "use_ema", False) else workspace.model
        raw_policy.to(device)
        raw_policy.eval()

        abs_action_flag = bool(getattr(getattr(cfg, "task", None) and getattr(cfg.task, "dataset", None) and cfg.task.dataset, "abs_action", False))

        from omegaconf import OmegaConf as _OC
        abs_action_flag = bool(_OC.select(cfg, "task.dataset.abs_action") or False)
        n_obs_steps_val = int(_OC.select(cfg, "n_obs_steps") or 2)
        n_action_steps_val = int(_OC.select(cfg, "n_action_steps") or 16)

        class _BareMonitor:
            """Thin shim so RobomimicDAggerRunner works without monitoring."""
            def __init__(self): self.episode_results = []
            def reset(self): self.episode_results = []
            def predict_action(self, obs_dict): return raw_policy.predict_action(obs_dict)

        monitored_policy = _BareMonitor()
        classifier_n_obs_steps = n_obs_steps_val
        classifier_n_action_steps = n_action_steps_val
        abs_action = abs_action_flag
    else:
        click.echo(f"Loading checkpoint and behavior graph for task: {task}")
        classifier = TrajectoryClassifier.from_checkpoint(
            checkpoint=str(checkpoint),
            infembed_fit_path=infembed_fit,
            infembed_embeddings_path=infembed_npz,
            clustering_dir=clustering_dir,
            mode="rollout",
            device=device,
            episodes_dir=episodes_dir,
        )

        from policy_doctor.data.clustering_loader import load_clustering_result_from_path
        import pathlib
        cluster_labels, cluster_metadata, _ = load_clustering_result_from_path(pathlib.Path(clustering_dir))
        graph, node_values, _, _ = get_behavior_graph_and_slice_values(
            cluster_labels=cluster_labels,
            metadata=cluster_metadata,
        )

        # Load DAgger config
        click.echo(f"Loading DAgger config: {dagger_config}")
        dagger_cfg = load_dagger_config(dagger_config)

        if intervention_threshold == 0.0:
            intervention_threshold = get_intervention_threshold(dagger_cfg)
        click.echo(f"Intervention threshold: {intervention_threshold}")

        intervention_rule = NodeValueThresholdRule(
            node_values=node_values, threshold=intervention_threshold
        )
        monitored_policy = MonitoredPolicy(
            policy=classifier.monitor.scorer.policy,
            classifier=classifier,
            intervention_rule=intervention_rule,
        )
        classifier_n_obs_steps = classifier.n_obs_steps
        classifier_n_action_steps = classifier.n_action_steps
        abs_action = classifier.abs_action

    if not no_monitor:
        # Load DAgger config for device/viz settings (already done above for monitor path)
        pass
    else:
        dagger_cfg = load_dagger_config(dagger_config)

    # --- Create robomimic environment ---
    click.echo(f"Setting up {task} environment...")

    dataset_path = Path(dataset_path)
    env_meta = FileUtils.get_env_metadata_from_dataset(str(dataset_path))

    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": obs_keys})

    # Build action transformer: policy outputs rotation_6d, env expects axis_angle
    rotation_transformer = None
    if abs_action:
        from diffusion_policy.model.common.rotation_transformer import RotationTransformer
        rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")
        # robosuite 1.5 uses input_type="absolute" instead of control_delta=False
        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False

    def convert_action(action_10d) -> "np.ndarray":
        """Convert rotation_6d action (10-dim) to axis_angle (7-dim) for robosuite."""
        import numpy as _np
        if rotation_transformer is None:
            return action_10d
        pos = action_10d[..., :3]
        rot = action_10d[..., 3:9]
        gripper = action_10d[..., [9]]
        rot_aa = rotation_transformer.inverse(rot)
        return _np.concatenate([pos, rot_aa, gripper], axis=-1)

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
        # Do NOT wrap with MultiStepWrapper: the DAgger runner steps one action
        # at a time (to allow mid-chunk intervention), so it handles obs stacking
        # manually via its own obs_queue deque.
        dagger_env = RobomimicDAggerEnv(
            inner_env=lowdim_wrapper,
            obs_keys=obs_keys,
            output_dir=output_dir,
        )
        return dagger_env

    env = create_env()

    # --- Create intervention device and visualizer ---
    # When monitoring is disabled there is no behavior-graph trigger, so we
    # default to passthrough (autonomous rollout) to avoid pynput accessibility
    # issues on macOS.
    if no_monitor and dagger_cfg.get("device", "keyboard") == "keyboard":
        dagger_cfg = dict(dagger_cfg)  # copy so we don't mutate the loaded config
        dagger_cfg["device"] = "passthrough"
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
        n_obs_steps=classifier_n_obs_steps,
        n_action_steps=classifier_n_action_steps,
        max_steps=500,
        output_dir=output_dir,
        visualizer=visualizer,
        action_transform=convert_action,
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
