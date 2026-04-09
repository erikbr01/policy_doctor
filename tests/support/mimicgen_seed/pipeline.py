"""Glue: vendored MimicGen on ``sys.path`` + ``prepare_src_dataset`` entry point.

Test support only — not part of the ``policy_doctor`` package.

Stack note: ``run_mimicgen_prepare_src_dataset`` replays HDF5 through robosuite/MuJoCo and
expects the **mimicgen** conda stack (MuJoCo ~2.3.2, pinned robosuite/robomimic; see
``environment_mimicgen.yaml``). NVlabs Hugging Face source demos embed MJCF for that stack;
running prepare from the **cupid** env (MuJoCo 3.x) typically fails with parser errors.

``materialize_robomimic_seed_hdf5`` only needs Python + ``h5py``/``numpy`` and is safe from
any env; feed its output into prepare after ``conda activate mimicgen`` (or equivalent).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

from policy_doctor.paths import MIMICGEN_CONDA_ENV_NAME, MIMICGEN_ROOT

from tests.support.mimicgen_seed.robomimic_source import RobomimicSeedMaterializer
from tests.support.mimicgen_seed.schema import MimicGenBinding, PolicyRolloutTrajectory

_ROBOMIMIC_BASE_ENV_PATCHED = False


def _apply_robomimic_base_env_shim() -> None:
    """MimicGen calls ``env.base_env``; robomimic 0.2+ ``EnvRobosuite`` only exposes ``.env``.

    Older robomimic commits used in the official MimicGen stack may already define
    ``base_env``; in that case this is a no-op.
    """
    global _ROBOMIMIC_BASE_ENV_PATCHED
    if _ROBOMIMIC_BASE_ENV_PATCHED:
        return
    try:
        import robomimic.envs.env_robosuite as er
    except ImportError:
        return
    if not hasattr(er.EnvRobosuite, "base_env"):
        er.EnvRobosuite.base_env = property(lambda self: self.env)  # type: ignore[attr-defined]
    _ROBOMIMIC_BASE_ENV_PATCHED = True


def ensure_mimicgen_importable() -> Path:
    """Put the vendored MimicGen repo root on ``sys.path`` so ``import mimicgen`` works.

    Prefer ``pip install -e third_party/mimicgen`` in the **mimicgen** conda env when
    running prepare/generate; this helper also supports editable policy_doctor usage
    (extra ``sys.path`` insert is harmless if the package is already installed).
    """
    root = MIMICGEN_ROOT.resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"MimicGen submodule missing at {root}. Clone with: "
            "git submodule update --init --recursive"
        )
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    _apply_robomimic_base_env_shim()
    return root


def _warn_mujoco_stack_for_nvlabs_source_data() -> None:
    """Emit a one-time hint if MuJoCo major != 2 (NVlabs HF demos target 2.3.x)."""
    try:
        import mujoco
    except ImportError:
        return
    try:
        major = int(str(mujoco.__version__).split(".", 1)[0])
    except ValueError:
        return
    if major == 2:
        return
    warnings.warn(
        "NVlabs MimicGen Hugging Face source datasets embed MJCF for MuJoCo 2.3.x; "
        f"this interpreter has mujoco {mujoco.__version__}. "
        f"Use `conda activate {MIMICGEN_CONDA_ENV_NAME}` (see environment_mimicgen.yaml) "
        "for prepare/generate on those files, or re-record demos on your current sim stack.",
        UserWarning,
        stacklevel=3,
    )


def materialize_robomimic_seed_hdf5(
    trajectory: PolicyRolloutTrajectory,
    output_hdf5: Path | str,
    *,
    demo_key: str = "demo_0",
    materializer: RobomimicSeedMaterializer | None = None,
) -> Path:
    """Write a policy rollout to ``output_hdf5`` in robomimic source format.

    Environment-agnostic (``h5py``/``numpy`` only). The resulting file is intended for
    ``run_mimicgen_prepare_src_dataset`` under the mimicgen sim stack unless you control
    both the HDF5 ``env_meta``/model XML and installed MuJoCo.
    """
    out = Path(output_hdf5)
    mat = materializer or RobomimicSeedMaterializer()
    mat.write_source_dataset(trajectory, out, demo_key=demo_key)
    return out.resolve()


def run_mimicgen_prepare_src_dataset(
    dataset_path: Path | str,
    binding: MimicGenBinding,
    *,
    filter_key: str | None = None,
    n: int | None = None,
    output_path: Path | str | None = None,
) -> None:
    """Replay the dataset through the simulator and attach ``datagen_info`` (MimicGen).

    Intended for the **mimicgen** conda environment (pinned MuJoCo/robosuite/robomimic).
    Requires ``robomimic``, ``robosuite``, and MimicGen dependencies. Mutates
    ``dataset_path`` in place unless ``output_path`` is set.

    If ``mujoco`` is importable and its major version is not 2, a :class:`UserWarning`
    is issued (HF source data is built for MuJoCo 2.3.x).
    """
    ensure_mimicgen_importable()
    _warn_mujoco_stack_for_nvlabs_source_data()
    from mimicgen.scripts.prepare_src_dataset import prepare_src_dataset

    prepare_src_dataset(
        dataset_path=str(dataset_path),
        filter_key=filter_key,
        n=n,
        output_path=str(output_path) if output_path is not None else None,
        **binding.as_prepare_kwargs(),
    )
