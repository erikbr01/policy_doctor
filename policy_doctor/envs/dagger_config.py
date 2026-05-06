"""Configuration loading for DAgger interventions and devices."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml


def get_config_dir() -> Path:
    """Get the DAgger config directory."""
    return Path(__file__).parent.parent / "configs" / "dagger"


def load_dagger_config(config_name: str) -> dict[str, Any]:
    """Load a DAgger configuration by name.

    Parameters
    ----------
    config_name : str
        Config name (e.g., 'defaults', 'keyboard_default', 'spacemouse_default').

    Returns
    -------
    config : dict
        Parsed YAML configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    config_path = get_config_dir() / f"{config_name}.yaml"
    if not config_path.exists():
        available = [f.stem for f in get_config_dir().glob("*.yaml")]
        raise FileNotFoundError(
            f"Config '{config_name}' not found at {config_path}. "
            f"Available: {', '.join(available)}"
        )

    with open(config_path) as f:
        return yaml.safe_load(f)


def deep_merge_dict(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``src`` into ``dst`` (in-place). Non-dict values replace."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge_dict(dst[k], v)
        else:
            dst[k] = v
    return dst


def merge_data_collection_task_into_dagger_cfg(
    dagger_cfg: dict[str, Any], task_cfg: Optional[dict[str, Any]]
) -> dict[str, Any]:
    """Overlay task YAML sections onto a dagger preset (task wins per-key).

    Reads ``configs/data_collection/tasks/<task>.yaml``. Merged top-level keys:
    ``pygame``, ``spacemouse``, ``visualization`` (nested dicts are deep-merged).
    """
    if not task_cfg:
        return dagger_cfg
    for key in ("pygame", "spacemouse", "visualization"):
        overlay = task_cfg.get(key)
        if overlay:
            deep_merge_dict(dagger_cfg.setdefault(key, {}), overlay)
    return dagger_cfg


def resolve_dagger_config_with_task(
    dagger_config_name: str, task_cfg: Optional[dict[str, Any]]
) -> dict[str, Any]:
    cfg = load_dagger_config(dagger_config_name)
    merge_data_collection_task_into_dagger_cfg(cfg, task_cfg)
    return cfg


def load_merged_dagger_config(
    dagger_config_name: str, task: Optional[str]
) -> dict[str, Any]:
    """Load a dagger preset and overlay optional data_collection task (viz_server / helpers)."""
    cfg = load_dagger_config(dagger_config_name)
    if task:
        from policy_doctor.envs.data_collection_config import load_data_collection_task_config

        merge_data_collection_task_into_dagger_cfg(cfg, load_data_collection_task_config(task))
    return cfg


def build_spacemouse_spatial_matrices(dagger_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """3×3 translation and rotation mixes from ``spacemouse.spatial_mapping`` (defaults: identity)."""
    params = dagger_cfg.get("spacemouse", {})
    spatial = params.get("spatial_mapping") or {}
    trans = spatial.get("translation_mix")
    rot = spatial.get("rotation_mix")
    if trans is None:
        trans_arr = np.eye(3, dtype=np.float64)
    else:
        trans_arr = np.asarray(trans, dtype=np.float64)
        if trans_arr.shape != (3, 3):
            raise ValueError(
                "spacemouse.spatial_mapping.translation_mix must be a 3×3 matrix "
                "(rows map [tx, ty, tz] desk frame → arm x,y,z)"
            )
    if rot is None:
        rot_arr = np.eye(3, dtype=np.float64)
    else:
        rot_arr = np.asarray(rot, dtype=np.float64)
        if rot_arr.shape != (3, 3):
            raise ValueError(
                "spacemouse.spatial_mapping.rotation_mix must be a 3×3 matrix "
                "(rows map wire-order [r0, r1, r2] from decode → OSC [roll, pitch, yaw])"
            )
    return trans_arr, rot_arr


def _parse_pygame_rotation_sources(params: dict[str, Any]) -> tuple[str, str]:
    rm = params.get("rotation_mapping") or {}
    yaw = str(rm.get("yaw_source", "right_stick_x")).strip().lower()
    pitch = str(rm.get("pitch_source", "trigger_diff")).strip().lower()
    allowed_rot = frozenset({"right_stick_x", "trigger_diff"})
    if yaw not in allowed_rot:
        raise ValueError(f"pygame.rotation_mapping.yaw_source must be one of {sorted(allowed_rot)}")
    if pitch not in allowed_rot | {"none"}:
        raise ValueError(
            "pygame.rotation_mapping.pitch_source must be "
            "right_stick_x | trigger_diff | none"
        )
    if yaw == pitch and yaw != "none":
        raise ValueError("pygame.rotation_mapping: yaw_source and pitch_source must differ")
    return yaw, pitch


def build_pygame_controller_kwargs(dagger_cfg: dict[str, Any]) -> dict[str, Any]:
    """Keyword args for ``PygameControllerInterventionDevice`` from ``pygame:`` in dagger YAML."""
    params = dagger_cfg.get("pygame", {})
    btn_kw = {
        k: params[k]
        for k in (
            "button_gripper_close",
            "button_gripper_open",
            "button_reset",
            "button_toggle",
        )
        if k in params
    }
    spatial = params.get("spatial_mapping") or {}
    mix = spatial.get("left_stick_xy_mix")
    if mix is None:
        mix_arr = np.eye(2, dtype=np.float64)
    else:
        mix_arr = np.asarray(mix, dtype=np.float64)
        if mix_arr.shape != (2, 2):
            raise ValueError(
                "pygame.spatial_mapping.left_stick_xy_mix must be a 2×2 array "
                "(rows map [lx, ly] into [action_x, action_y])"
            )

    yaw_src, pitch_src = _parse_pygame_rotation_sources(params)

    return {
        "controller_index": params.get("controller_index", 0),
        "deadzone": params.get("deadzone", 0.15),
        "scale_position": params.get("scale_position", 1.0),
        "scale_rotation": params.get("scale_rotation", 1.0),
        "axis_left_x": params.get("axis_left_x", 0),
        "axis_left_y": params.get("axis_left_y", 1),
        "axis_right_x": params.get("axis_right_x", 2),
        "axis_right_y": params.get("axis_right_y", 3),
        "axis_left_trigger": params.get("axis_left_trigger", 4),
        "axis_right_trigger": params.get("axis_right_trigger", 5),
        "controller_layout": params.get("controller_layout", "auto"),
        "left_stick_xy_mix": mix_arr,
        "yaw_source": yaw_src,
        "pitch_source": pitch_src,
        **btn_kw,
    }


def create_intervention_device(config: dict[str, Any]) -> Any:
    """Create an InterventionDevice from config.

    Parameters
    ----------
    config : dict
        DAgger configuration dict (from load_dagger_config).

    Returns
    -------
    device : InterventionDevice
        Configured device instance (KeyboardInterventionDevice, SpaceMouseInterventionDevice, etc.).
    """
    from policy_doctor.envs.intervention_device import (
        KeyboardInterventionDevice,
        PassthroughInterventionDevice,
        PygameControllerInterventionDevice,
        RandomInterventionDevice,
        SpaceMouseInterventionDevice,
        XboxControllerInterventionDevice,
    )

    device_type = config.get("device", "keyboard")

    if device_type == "keyboard":
        params = config.get("keyboard", {})
        return KeyboardInterventionDevice(
            step_size=params.get("step_size", 0.05)
        )

    elif device_type == "spacemouse":
        params = config.get("spacemouse", {})
        trans_mix, rot_mix = build_spacemouse_spatial_matrices(config)
        return SpaceMouseInterventionDevice(
            vendor_id=params.get("vendor_id", 9583),
            product_id=params.get("product_id", 50741),
            deadzone=params.get("deadzone", 0.1),
            scale_position=params.get("scale_position", 125.0),
            scale_rotation=params.get("scale_rotation", 50.0),
            translation_mix=trans_mix,
            rotation_mix=rot_mix,
        )

    elif device_type == "xbox":
        params = config.get("xbox", {})
        return XboxControllerInterventionDevice(
            controller_index=params.get("controller_index", 0),
            deadzone=params.get("deadzone", 0.15),
            scale_position=params.get("scale_position", 1.0),
            scale_rotation=params.get("scale_rotation", 1.0),
        )

    elif device_type == "pygame":
        return PygameControllerInterventionDevice(**build_pygame_controller_kwargs(config))

    elif device_type == "passthrough":
        return PassthroughInterventionDevice()

    elif device_type == "random":
        params = config.get("random", {})
        return RandomInterventionDevice(
            action_dim=params.get("action_dim", 10),
            scale=params.get("scale", 1.0),
            seed=params.get("seed"),
        )

    else:
        raise ValueError(f"Unknown device type: {device_type}")


def get_intervention_threshold(config: dict[str, Any]) -> float:
    """Extract intervention threshold from config.

    Parameters
    ----------
    config : dict
        DAgger configuration dict.

    Returns
    -------
    threshold : float
        V-value threshold for NodeValueThresholdRule.
    """
    intervention_cfg = config.get("intervention", {})
    return intervention_cfg.get("node_value_threshold", 0.0)
