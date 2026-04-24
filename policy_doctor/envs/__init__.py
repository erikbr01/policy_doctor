"""DAgger-style interactive evaluation environments for robomimic-based tasks."""

from policy_doctor.envs.robomimic_dagger_env import RobomimicDAggerEnv
# Backward compatibility alias
RobocasaDAggerEnv = RobomimicDAggerEnv
from policy_doctor.envs.intervention_device import (
    InterventionDevice,
    KeyboardInterventionDevice,
    PassthroughInterventionDevice,
)
from policy_doctor.envs.dagger_runner import RobocasaDAggerRunner, EpisodeRecord
from policy_doctor.envs.visualization import DAggerVisualizer

__all__ = [
    "RobomimicDAggerEnv",
    "RobocasaDAggerEnv",  # backward compat
    "InterventionDevice",
    "KeyboardInterventionDevice",
    "PassthroughInterventionDevice",
    "RobocasaDAggerRunner",
    "EpisodeRecord",
    "DAggerVisualizer",
]
