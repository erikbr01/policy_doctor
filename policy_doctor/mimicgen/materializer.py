"""Write robomimic-format HDF5 source datasets for MimicGen.

Pure I/O using only ``h5py`` and ``numpy`` — safe from any Python environment.
The resulting file is designed to be passed to
``mimicgen.scripts.prepare_src_dataset`` running under the **mimicgen** conda
stack (MuJoCo 2.3.x, pinned robosuite / robomimic).

Layout follows ``mimicgen.utils.file_utils.write_demo_to_hdf5`` / standard
robomimic datasets::

    data/
        <demo_key>/
            actions     (T, A) float32  gzip-compressed
            states      (T, D) float32  gzip-compressed
            attrs: num_samples, model_file (optional)
        attrs: total (int64), env_args (JSON string)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import h5py
import numpy as np


@runtime_checkable
class SeedDatasetMaterializer(Protocol):
    """Pluggable writer for a single source demonstration.

    Extend this protocol for other simulation backends (e.g. LIBERO, RoboCasa)
    that produce a different ``env_meta`` schema but the same HDF5 layout.
    """

    def write_source_dataset(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        env_meta: Mapping[str, Any],
        output_path: Path,
        *,
        demo_key: str = "demo_0",
        model_file: str | None = None,
    ) -> None:
        ...


class RobomimicSeedMaterializer:
    """Materialize a trajectory into a robomimic-style HDF5 for MimicGen.

    This is the production version of the class that was previously only in
    ``tests/support/mimicgen_seed/robomimic_source.py``.  The test-support
    module re-exports from here for backward compatibility.

    Example::

        mat = RobomimicSeedMaterializer()
        mat.write_source_dataset(
            states=traj.states,
            actions=traj.actions,
            env_meta=traj.env_meta,
            output_path=Path("/tmp/seed.hdf5"),
            model_file=traj.model_file,
        )
    """

    def write_source_dataset(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        env_meta: Mapping[str, Any],
        output_path: Path,
        *,
        demo_key: str = "demo_0",
        model_file: str | None = None,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        actions = np.asarray(actions, dtype=np.float32)
        states = np.asarray(states, dtype=np.float32)
        total = int(actions.shape[0])

        with h5py.File(output_path, "w") as f:
            data = f.create_group("data")
            data.attrs["total"] = np.int64(total)
            data.attrs["env_args"] = json.dumps(dict(env_meta), indent=4)

            ep = data.create_group(demo_key)
            ep.create_dataset("actions", data=actions, compression="gzip")
            ep.create_dataset("states", data=states, compression="gzip")
            ep.attrs["num_samples"] = np.int64(total)
            if model_file is not None:
                ep.attrs["model_file"] = model_file


def materialize_seed_trajectory(
    traj: Any,
    output_path: Path | str,
    *,
    demo_key: str = "demo_0",
    materializer: RobomimicSeedMaterializer | None = None,
) -> Path:
    """Write a :class:`~policy_doctor.mimicgen.seed_trajectory.MimicGenSeedTrajectory` to disk.

    Accepts any object with ``states``, ``actions``, ``env_meta``, and optional
    ``model_file`` attributes (duck-typed), including
    :class:`~policy_doctor.mimicgen.seed_trajectory.MimicGenSeedTrajectory` and
    the legacy :class:`~tests.support.mimicgen_seed.schema.PolicyRolloutTrajectory`.
    """
    mat = materializer or RobomimicSeedMaterializer()
    out = Path(output_path)
    mat.write_source_dataset(
        states=traj.states,
        actions=traj.actions,
        env_meta=traj.env_meta,
        output_path=out,
        demo_key=demo_key,
        model_file=getattr(traj, "model_file", None),
    )
    return out.resolve()
