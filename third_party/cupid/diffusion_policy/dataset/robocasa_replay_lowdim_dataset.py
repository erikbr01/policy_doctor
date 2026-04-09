"""RoboCasa kitchen demos in robomimic-layout HDF5 — same loading as Robomimic, explicit adapter name."""

from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import (
    RobomimicReplayLowdimDataset,
)


class RobocasaReplayLowdimDataset(RobomimicReplayLowdimDataset):
    """Dataset adapter for merged RoboCasa low-dim robomimic HDF5."""

    pass
