#!/usr/bin/env python
"""DAgger rollouts on any robomimic-compatible environment.

Works with: robomimic, robocasa, libero, blockpush, mimicgen, etc.
Run from third_party/cupid/ (diffusion_policy must be on PYTHONPATH).

Keyboard-only (no checkpoint):
    python ../../scripts/run_dagger.py task=square_mh no_monitor=true \\
      viz_url=http://localhost:5002 output_dir=/tmp/dagger_test

Bare policy (checkpoint, no InfEmbed):
    python ../../scripts/run_dagger.py task=square_mh no_monitor=true \\
      train_dir=/path/to/train_dir output_dir=/tmp/dagger_test

Full monitored DAgger:
    python ../../scripts/run_dagger.py task=square_mh \\
      train_dir=... infembed_fit=... infembed_npz=... clustering_dir=... \\
      output_dir=/tmp/dagger

Environment: cupid conda env (PyTorch 2.x, diffusion_policy on PYTHONPATH)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure policy_doctor and diffusion_policy are importable from any working directory.
_PD_ROOT = Path(__file__).resolve().parent.parent
_CUPID_ROOT = _PD_ROOT / "third_party" / "cupid"
for _p in [str(_PD_ROOT), str(_CUPID_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from policy_doctor.paths import CONFIGS_DIR, REPO_ROOT

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
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_checkpoint(train_dir: str, train_ckpt: str) -> Path:
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


@hydra.main(config_path=str(CONFIGS_DIR), config_name="dagger_run", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    from robomimic.utils.env_utils import EnvUtils
    from robomimic.utils.file_utils import FileUtils
    from robomimic.utils.obs_utils import ObsUtils

    from policy_doctor.behaviors.behavior_values import get_behavior_graph_and_slice_values
    from policy_doctor.data.adapters import ensure_robocasa_on_path
    from policy_doctor.envs import (
        DAggerVisualizer,
        RobomimicDAggerEnv,
        RobomimicDAggerRunner,
        BarePolicy,
    )
    from policy_doctor.envs.dagger_config import (
        create_intervention_device,
        get_intervention_threshold,
        load_dagger_config,
    )
    from policy_doctor.monitoring.intervention import NodeValueThresholdRule
    from policy_doctor.monitoring.monitored_policy import MonitoredPolicy
    from policy_doctor.monitoring.trajectory_classifier import TrajectoryClassifier
    from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper

    task = cfg.task
    train_dir = cfg.train_dir
    train_ckpt = cfg.train_ckpt
    infembed_fit = cfg.infembed_fit
    infembed_npz = cfg.infembed_npz
    clustering_dir = cfg.clustering_dir
    dataset_path = cfg.dataset_path
    output_dir = cfg.output_dir
    num_episodes = cfg.num_episodes
    intervention_threshold = cfg.intervention_threshold
    device = cfg.device
    no_monitor = cfg.no_monitor
    no_visualization = cfg.no_visualization
    viz_url = cfg.viz_url
    dagger_config = cfg.dagger_config
    episodes_dir = cfg.episodes_dir
    server_url = cfg.server_url

    # Validate required args for full monitoring mode
    if not no_monitor and not server_url:
        missing = [k for k, v in [
            ("infembed_fit", infembed_fit),
            ("infembed_npz", infembed_npz),
            ("clustering_dir", clustering_dir),
            ("train_dir", train_dir),
        ] if v is None]
        if missing:
            raise ValueError(
                f"Missing required config keys for monitored mode: {missing}\n"
                f"Set no_monitor=true to run without the behavior graph."
            )

    task_cfg = TASK_CONFIG.get(task)
    if task_cfg is None:
        raise ValueError(f"Unknown task '{task}'. Choices: {list(TASK_CONFIG)}")

    if dataset_path is None:
        dataset_path = task_cfg["dataset_path"]
    obs_keys = task_cfg["obs_keys"]

    if device == "auto":
        device = auto_device()
    print(f"Using device: {device}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "robocasa" in task.lower():
        ensure_robocasa_on_path()

    # --- Load policy ---
    if server_url:
        from policy_doctor.envs.policy_server import PolicyClient
        import requests as _requests
        print(f"Using policy server at {server_url}")
        resp = _requests.get(f"{server_url}/health", timeout=5)
        info = resp.json()
        print(f"  status: {info.get('status')} | device: {info.get('device')}")

        checkpoint = resolve_checkpoint(train_dir, train_ckpt)
        import dill as _dill, torch as _torch
        payload = _torch.load(open(str(checkpoint), "rb"), pickle_module=_dill, map_location="cpu")
        _OC = OmegaConf
        abs_action = bool(_OC.select(payload["cfg"], "task.dataset.abs_action") or False)
        classifier_n_obs_steps = int(_OC.select(payload["cfg"], "n_obs_steps") or 2)
        classifier_n_action_steps = int(_OC.select(payload["cfg"], "n_action_steps") or 8)
        del payload

        monitored_policy = PolicyClient(url=server_url)
        monitored_policy.start()
        dagger_cfg = load_dagger_config(dagger_config)

    elif no_monitor:
        dagger_cfg = load_dagger_config(dagger_config)
        if train_dir is None:
            print("No checkpoint — keyboard-only teleoperation (human drives entire episode)")
            monitored_policy = None
            classifier_n_obs_steps = 2
            classifier_n_action_steps = 8
            abs_action = False
        else:
            print("Monitor disabled — loading bare policy")
            checkpoint = resolve_checkpoint(train_dir, train_ckpt)
            print(f"Checkpoint: {checkpoint}")
            import dill as _dill, hydra as _hydra, torch as _torch
            payload = _torch.load(open(str(checkpoint), "rb"), pickle_module=_dill)
            cfg_ckpt = payload["cfg"]
            cls = _hydra.utils.get_class(cfg_ckpt._target_)
            workspace = cls(cfg_ckpt, output_dir=str(output_dir))
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)
            raw_policy = workspace.ema_model if getattr(cfg_ckpt.training, "use_ema", False) else workspace.model
            raw_policy.to(device)
            raw_policy.eval()
            from omegaconf import OmegaConf as _OC
            abs_action = bool(_OC.select(cfg_ckpt, "task.dataset.abs_action") or False)
            classifier_n_obs_steps = int(_OC.select(cfg_ckpt, "n_obs_steps") or 2)
            classifier_n_action_steps = int(_OC.select(cfg_ckpt, "n_action_steps") or 16)
            monitored_policy = BarePolicy(raw_policy)

    else:
        print(f"Loading checkpoint and behavior graph for task: {task}")
        checkpoint = resolve_checkpoint(train_dir, train_ckpt)
        print(f"Checkpoint: {checkpoint}")
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
        cluster_labels, cluster_metadata, _ = load_clustering_result_from_path(Path(clustering_dir))
        _, node_values, _, _ = get_behavior_graph_and_slice_values(
            cluster_labels=cluster_labels,
            metadata=cluster_metadata,
        )
        dagger_cfg = load_dagger_config(dagger_config)
        if intervention_threshold == 0.0:
            intervention_threshold = get_intervention_threshold(dagger_cfg)
        print(f"Intervention threshold: {intervention_threshold}")
        monitored_policy = MonitoredPolicy(
            policy=classifier.monitor.scorer.policy,
            classifier=classifier,
            intervention_rule=NodeValueThresholdRule(node_values=node_values, threshold=intervention_threshold),
        )
        classifier_n_obs_steps = classifier.n_obs_steps
        classifier_n_action_steps = classifier.n_action_steps
        abs_action = classifier.abs_action

    # --- Build environment ---
    print(f"Setting up {task} environment...")
    dataset_path = Path(dataset_path)
    env_meta = FileUtils.get_env_metadata_from_dataset(str(dataset_path))
    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": obs_keys})

    rotation_transformer = None
    if abs_action:
        from diffusion_policy.model.common.rotation_transformer import RotationTransformer
        rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")
        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False

    def convert_action(action_10d):
        import numpy as _np
        if rotation_transformer is not None:
            # abs_action policy: [pos(3), rot_6d(6), gripper(1)] → axis_angle 7-dim
            pos = action_10d[..., :3]
            rot = action_10d[..., 3:9]
            gripper = action_10d[..., [9]]
            return _np.concatenate([pos, rotation_transformer.inverse(rot), gripper], axis=-1)
        else:
            # keyboard / delta policy: [pos(3), aa(3), gripper(1), base(3)] → strip base
            return action_10d[..., :7]

    robomimic_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=False,
    )
    lowdim_wrapper = RobomimicLowdimWrapper(env=robomimic_env, obs_keys=obs_keys, init_state=None)
    env = RobomimicDAggerEnv(inner_env=lowdim_wrapper, obs_keys=obs_keys, output_dir=output_dir)

    # --- Intervention device ---
    if viz_url:
        from policy_doctor.envs.intervention_device import HTTPInterventionDevice
        intervention_device = HTTPInterventionDevice(server_url=viz_url)
        device_type = "http"
        print(f"Intervention via viz server ({viz_url})")
        print("  Space: toggle  W/S/A/D/Q/E: arm  G/H: gripper")
    else:
        device_type = dagger_cfg.get("device", "keyboard")
        print(f"Initializing {device_type} intervention device...")
        intervention_device = create_intervention_device(dagger_cfg)

    if device_type == "keyboard":
        print("  Space: toggle human/robot  W/S/A/D/Q/E: arm  G/H: gripper  I/K/J/L: base")
    elif device_type == "spacemouse":
        print("  SpaceMouse: 6-DOF  Left btn: gripper  Right btn: toggle")

    # --- Visualizer ---
    visualizer = None
    viz_cfg = dagger_cfg.get("visualization", {})
    if not no_visualization and (viz_url or viz_cfg.get("enabled", True)):
        try:
            kw = dict(camera_names=viz_cfg.get("camera_names", ["agentview"]),
                      figsize=tuple(viz_cfg.get("figsize", [8, 5])))
            if viz_url:
                kw["server_url"] = viz_url
            visualizer = DAggerVisualizer(**kw)
            print("Visualization enabled")
        except Exception as e:
            print(f"Warning: visualizer failed: {e}")

    # --- Run ---
    print("\n" + "=" * 60)
    print(f"Starting DAgger rollouts: {task}")
    print("=" * 60)

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
        runner.run(num_episodes)
    except KeyboardInterrupt:
        print("\nInterrupted by user")

    print("\n" + "=" * 60)
    print(f"Episodes saved to: {output_dir}")
    print(f"Convert to HDF5:  python scripts/build_dagger_dataset.py "
          f"--episodes_dir {output_dir} --output_hdf5 data/{task}_dagger.hdf5 --filter_human_only")
    print("=" * 60)


if __name__ == "__main__":
    main()
