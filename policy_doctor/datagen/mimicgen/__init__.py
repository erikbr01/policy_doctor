"""MimicGen seed-demo helpers (robomimic HDF5 → ``prepare_src_dataset``).

Workflow split across conda stacks (see repo README):

- **cupid**: replay policy rollouts (e.g. ``replay_mar27_transport``) and anything that
  needs diffusion_policy + MuJoCo 3.x.
- **mimicgen**: ``run_mimicgen_prepare_src_dataset`` / MimicGen generate on NVlabs-style
  data (MuJoCo ~2.3.2).
- **policy_doctor**: schema, materialization to HDF5, and tests; ``materialize_*`` only
  needs ``h5py``/``numpy``.
"""

from policy_doctor.datagen.mimicgen.pipeline import (
    ensure_mimicgen_importable,
    materialize_robomimic_seed_hdf5,
    run_mimicgen_prepare_src_dataset,
)
from policy_doctor.datagen.mimicgen.robomimic_source import (
    LiberoRobomimicSeedMaterializer,
    RobocasaRobomimicSeedMaterializer,
    RobomimicSeedMaterializer,
)
from policy_doctor.datagen.mimicgen.schema import (
    MimicGenBinding,
    PolicyRolloutTrajectory,
    SimulationBackend,
    validate_policy_rollout_trajectory,
)

__all__ = [
    "LiberoRobomimicSeedMaterializer",
    "MimicGenBinding",
    "PolicyRolloutTrajectory",
    "RobocasaRobomimicSeedMaterializer",
    "RobomimicSeedMaterializer",
    "SimulationBackend",
    "ensure_mimicgen_importable",
    "materialize_robomimic_seed_hdf5",
    "run_mimicgen_prepare_src_dataset",
    "validate_policy_rollout_trajectory",
]
