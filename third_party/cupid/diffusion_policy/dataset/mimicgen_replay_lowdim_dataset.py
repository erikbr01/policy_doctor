"""MimicGen demos in robomimic HDF5 layout — same loading as Robomimic, explicit adapter name."""

from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import (
    RobomimicReplayLowdimDataset,
)


class MimicgenReplayLowdimDataset(RobomimicReplayLowdimDataset):
    """Dataset adapter for MimicGen-exported low-dim HDF5 (``demo_*``, ``obs``, ``actions``)."""

    pass
