"""Domain-neutral types for MimicGen seed trajectories and env bindings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np


class SimulationBackend(str, Enum):
    """Where the rollout was collected / which materialization path to use.

    Several simulators expose a robomimic-compatible HDF5 layout via official or
    community env wrappers; keep distinct enum values so experiment code can
    select task configs and bindings without conflating dataset layout with
    simulator identity.
    """

    ROBO_MIMIC = "robomimic"
    LIBERO_ROBO_MIMIC = "libero_robomimic"
    ROBOCASA_ROBO_MIMIC = "robocasa_robomimic"


@dataclass(frozen=True)
class MimicGenBinding:
    """Registered MimicGen environment interface (see ``mimicgen.env_interfaces``)."""

    env_interface_name: str
    env_interface_type: str

    def as_prepare_kwargs(self) -> dict[str, str]:
        return {
            "env_interface_name": self.env_interface_name,
            "env_interface_type": self.env_interface_type,
        }


@dataclass(frozen=True)
class PolicyRolloutTrajectory:
    """Canonical low-dimensional rollout for MimicGen source preparation.

    ``env_meta`` / ``model_xml`` should describe the **same** MuJoCo/robosuite stack that
    will replay the file (cupid vs mimicgen is a real distinction for embedded XML).

    This matches what ``mimicgen.scripts.prepare_src_dataset`` consumes: per-timestep
    simulator states and actions, plus robomimic ``env_meta`` and optional robosuite
    ``model_file`` XML on the episode.
    """

    states: np.ndarray
    actions: np.ndarray
    env_meta: Mapping[str, Any]
    model_file: str | None = None

    def __post_init__(self) -> None:
        validate_policy_rollout_trajectory(self)


def validate_policy_rollout_trajectory(traj: PolicyRolloutTrajectory) -> None:
    """Raise ``ValueError`` if the rollout cannot be written as a MimicGen source demo."""

    if traj.states.ndim != 2:
        raise ValueError(f"states must be 2-D (T, D), got shape {traj.states.shape}")
    if traj.actions.ndim != 2:
        raise ValueError(f"actions must be 2-D (T, A), got shape {traj.actions.shape}")
    t_s, t_a = traj.states.shape[0], traj.actions.shape[0]
    if t_s != t_a:
        raise ValueError(f"states length {t_s} must match actions length {t_a}")
    if t_s == 0:
        raise ValueError("trajectory must contain at least one timestep")
    if not isinstance(traj.env_meta, Mapping):
        raise ValueError("env_meta must be a mapping (robomimic env metadata dict)")
    if not traj.env_meta:
        raise ValueError("env_meta must be non-empty")
