"""Write robomimic-format HDF5 source datasets from :class:`PolicyRolloutTrajectory`.

Pure I/O + layout; safe from any Python env with ``h5py``. Downstream
``prepare_src_dataset`` should run under the **mimicgen** stack for NVlabs datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import h5py
import numpy as np

from tests.support.mimicgen_seed.schema import PolicyRolloutTrajectory, SimulationBackend


@runtime_checkable
class SeedDatasetMaterializer(Protocol):
    """Pluggable writer for a single source demonstration (extend for other sim backends)."""

    backend: SimulationBackend

    def write_source_dataset(
        self,
        trajectory: PolicyRolloutTrajectory,
        output_path: Path,
        *,
        demo_key: str = "demo_0",
    ) -> None:
        ...


def _env_args_json(env_meta: dict) -> str:
    return json.dumps(env_meta, indent=4)


class RobomimicSeedMaterializer:
    """Materialize one or more demos into a robomimic-style HDF5 MimicGen can prepare.

    Layout follows ``mimicgen.utils.file_utils.write_demo_to_hdf5`` / standard robomimic
    datasets: ``data/<demo>/actions``, ``data/<demo>/states``, episode attrs
    ``model_file`` (robosuite) and ``num_samples``, and ``data`` attrs ``total``,
    ``env_args``.
    """

    backend = SimulationBackend.ROBO_MIMIC

    def write_source_dataset(
        self,
        trajectory: PolicyRolloutTrajectory,
        output_path: Path,
        *,
        demo_key: str = "demo_0",
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        env_meta = dict(trajectory.env_meta)
        actions = np.asarray(trajectory.actions, dtype=np.float32)
        states = np.asarray(trajectory.states, dtype=np.float32)
        total = int(actions.shape[0])

        with h5py.File(output_path, "w") as f:
            data = f.create_group("data")
            data.attrs["total"] = np.int64(total)
            data.attrs["env_args"] = _env_args_json(env_meta)

            ep = data.create_group(demo_key)
            ep.create_dataset("actions", data=actions, compression="gzip")
            ep.create_dataset("states", data=states, compression="gzip")
            ep.attrs["num_samples"] = np.int64(total)
            if trajectory.model_file is not None:
                ep.attrs["model_file"] = trajectory.model_file


class LiberoRobomimicSeedMaterializer(RobomimicSeedMaterializer):
    """Same HDF5 layout as robomimic; use for LIBERO-collected rollouts once wrapped."""

    backend = SimulationBackend.LIBERO_ROBO_MIMIC


class RobocasaRobomimicSeedMaterializer(RobomimicSeedMaterializer):
    """Same HDF5 layout; use when RoboCasa exposes a robomimic-compatible env_meta."""

    backend = SimulationBackend.ROBOCASA_ROBO_MIMIC
