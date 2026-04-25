"""Configuration loading for DAgger interventions and devices."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

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
        return SpaceMouseInterventionDevice(
            vendor_id=params.get("vendor_id", 9583),
            product_id=params.get("product_id", 50741),
            deadzone=params.get("deadzone", 0.1),
            scale_position=params.get("scale_position", 125.0),
            scale_rotation=params.get("scale_rotation", 50.0),
        )

    elif device_type == "xbox":
        params = config.get("xbox", {})
        return XboxControllerInterventionDevice(
            controller_index=params.get("controller_index", 0),
            deadzone=params.get("deadzone", 0.15),
            scale_position=params.get("scale_position", 1.0),
            scale_rotation=params.get("scale_rotation", 1.0),
        )

    elif device_type == "passthrough":
        return PassthroughInterventionDevice()

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
