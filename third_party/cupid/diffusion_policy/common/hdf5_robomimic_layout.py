"""HDF5 layout helpers for robomimic and MimicGen-merged datasets (h5py only)."""

from __future__ import annotations

from typing import List

import h5py


def sorted_robomimic_demo_keys(data_grp: h5py.Group) -> List[str]:
    """Return ``data/demo_*`` keys sorted by numeric suffix.

    Standard Robomimic datasets and MimicGen outputs (including ``merge_all_hdf5``) use
    this layout. Sorting avoids assuming contiguous ``demo_0`` … ``demo_{N-1}`` when the
    group has gaps or arbitrary key order.
    """

    def _suffix_int(k: str) -> int:
        try:
            return int(k.split("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    keys = [
        k
        for k in data_grp.keys()
        if isinstance(k, str) and k.startswith("demo_") and _suffix_int(k) >= 0
    ]
    keys.sort(key=_suffix_int)
    return keys
