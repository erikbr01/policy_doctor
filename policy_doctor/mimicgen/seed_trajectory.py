"""Production types for MimicGen seed trajectories.

A :class:`MimicGenSeedTrajectory` can be created from either a human
demonstration stored in a robomimic-style HDF5 or from a policy rollout saved
by ``eval_save_episodes``.  Both share the same on-disk layout (``data/<demo>/
states``, ``actions``, ``model_file`` attr, ``data`` attrs ``env_args``/
``total``) so the factory methods differ only in the ``source`` label they
attach.

The class bridges cleanly to the existing
:class:`~tests.support.mimicgen_seed.schema.PolicyRolloutTrajectory` via
``to_policy_rollout_trajectory`` / ``from_policy_rollout_trajectory``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    # Avoid hard import of test-support at runtime; only used for type hints.
    pass


class SeedSource(str, Enum):
    """Origin of the seed trajectory."""

    DEMONSTRATION = "demonstration"
    ROLLOUT = "rollout"


@dataclass(frozen=True)
class MimicGenSeedTrajectory:
    """Canonical seed trajectory for MimicGen source preparation.

    Both demonstration slices and policy rollouts share the same robomimic HDF5
    layout, so a single type covers both use-cases.  Use :attr:`source` to
    distinguish them in experiment bookkeeping.

    Fields mirror what ``mimicgen.scripts.prepare_src_dataset`` needs:
    * ``states``     — per-timestep simulator state vectors, shape ``(T, D)``.
    * ``actions``    — per-timestep action vectors, shape ``(T, A)``.
    * ``env_meta``   — robomimic environment metadata dict (``env_name``,
                       ``type``, ``env_kwargs``).
    * ``model_file`` — optional MuJoCo MJCF XML string stored as the episode
                       ``model_file`` attribute.  Required by NVlabs source HDF5;
                       may be ``None`` for custom environments.
    * ``source``     — :class:`SeedSource` tag.
    """

    states: np.ndarray
    actions: np.ndarray
    env_meta: Mapping[str, Any]
    model_file: str | None
    source: SeedSource

    def __post_init__(self) -> None:
        _validate(self)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_robomimic_hdf5_demo(
        cls,
        hdf5_path: Path | str,
        demo_key: str = "demo_0",
    ) -> "MimicGenSeedTrajectory":
        """Load a single demo from a robomimic / MimicGen source HDF5.

        Reads ``data/<demo_key>/states``, ``data/<demo_key>/actions``, the
        optional ``model_file`` episode attribute, and the ``env_args`` dataset
        attribute.

        Args:
            hdf5_path: Path to the HDF5 file.
            demo_key:  Key under ``data/`` (e.g. ``"demo_0"``).

        Returns:
            :class:`MimicGenSeedTrajectory` with ``source=DEMONSTRATION``.
        """
        return _load_from_hdf5(hdf5_path, demo_key, SeedSource.DEMONSTRATION)

    @classmethod
    def from_rollout_hdf5(
        cls,
        hdf5_path: Path | str,
        demo_key: str = "demo_0",
    ) -> "MimicGenSeedTrajectory":
        """Load a single episode from an ``eval_save_episodes`` rollout HDF5.

        The layout is identical to the robomimic source format, so this is a
        thin alias of :meth:`from_robomimic_hdf5_demo` with the source tag set
        to :attr:`SeedSource.ROLLOUT`.

        Args:
            hdf5_path: Path to the HDF5 file.
            demo_key:  Key under ``data/`` (e.g. ``"demo_0"``).

        Returns:
            :class:`MimicGenSeedTrajectory` with ``source=ROLLOUT``.
        """
        return _load_from_hdf5(hdf5_path, demo_key, SeedSource.ROLLOUT)

    @classmethod
    def from_policy_rollout_trajectory(
        cls,
        traj: Any,
        source: SeedSource = SeedSource.ROLLOUT,
    ) -> "MimicGenSeedTrajectory":
        """Bridge from :class:`tests.support.mimicgen_seed.schema.PolicyRolloutTrajectory`.

        Accepts any object with ``states``, ``actions``, ``env_meta``, and
        optional ``model_file`` attributes (duck-typed).
        """
        return cls(
            states=np.asarray(traj.states, dtype=np.float32),
            actions=np.asarray(traj.actions, dtype=np.float32),
            env_meta=dict(traj.env_meta),
            model_file=getattr(traj, "model_file", None),
            source=source,
        )

    # ------------------------------------------------------------------
    # Bridge back to existing test-support type
    # ------------------------------------------------------------------

    def to_policy_rollout_trajectory(self) -> Any:
        """Convert to :class:`tests.support.mimicgen_seed.schema.PolicyRolloutTrajectory`.

        Useful for feeding into the existing
        :class:`tests.support.mimicgen_seed.robomimic_source.RobomimicSeedMaterializer`
        or other legacy test-support helpers.
        """
        from tests.support.mimicgen_seed.schema import PolicyRolloutTrajectory

        return PolicyRolloutTrajectory(
            states=self.states,
            actions=self.actions,
            env_meta=self.env_meta,
            model_file=self.model_file,
        )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _load_from_hdf5(
    hdf5_path: Path | str,
    demo_key: str,
    source: SeedSource,
) -> MimicGenSeedTrajectory:
    import h5py

    hdf5_path = Path(hdf5_path)
    with h5py.File(hdf5_path, "r") as f:
        data_grp = f["data"]
        env_args_raw = data_grp.attrs.get("env_args", "{}")
        try:
            env_meta: dict = json.loads(env_args_raw)
        except (json.JSONDecodeError, TypeError):
            env_meta = {}

        ep = data_grp[demo_key]
        states = np.array(ep["states"], dtype=np.float32)
        actions = np.array(ep["actions"], dtype=np.float32)
        model_file: str | None = ep.attrs.get("model_file", None)
        if isinstance(model_file, bytes):
            model_file = model_file.decode("utf-8")

    return MimicGenSeedTrajectory(
        states=states,
        actions=actions,
        env_meta=env_meta,
        model_file=model_file,
        source=source,
    )


def _validate(traj: MimicGenSeedTrajectory) -> None:
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
        raise ValueError("env_meta must be a Mapping")
    if not traj.env_meta:
        raise ValueError("env_meta must be non-empty")
