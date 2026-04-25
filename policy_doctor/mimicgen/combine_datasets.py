"""Combine an original robomimic HDF5 dataset with MimicGen-generated demos.

The combined dataset is written to a new file so that neither source is
modified.  All demos from *generated_path* are appended to a copy of
*original_path* under new keys (``demo_{N}``, ``demo_{N+1}``, …) to avoid
collisions with the existing ``demo_0`` … ``demo_{N-1}`` keys.

Attributes preserved from the original:
- All per-demo groups (``data/demo_*``) and their datasets / attributes.
- ``data.attrs`` (including ``env_args``).  ``total`` is updated to reflect
  the combined count.
- Any top-level groups outside of ``data`` (e.g. ``mask``).

Generated demos are copied verbatim from the generated HDF5; only their
group name changes to avoid collision.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py


def combine_hdf5_datasets(
    original_path: Path,
    generated_path: Path,
    output_path: Path,
) -> int:
    """Copy *original_path* to *output_path*, then append all demos from *generated_path*.

    Args:
        original_path:  Path to the original training HDF5 (not modified).
        generated_path: Path to the MimicGen-generated HDF5 (not modified).
        output_path:    Destination path for the combined HDF5.  Created (or
                        overwritten) by this function.

    Returns:
        Total number of demos in the combined dataset.

    Raises:
        FileNotFoundError: If either source path does not exist.
        KeyError:          If the HDF5 files do not have a ``"data"`` group.
    """
    original_path = Path(original_path)
    generated_path = Path(generated_path)
    output_path = Path(output_path)

    if not original_path.exists():
        raise FileNotFoundError(f"Original dataset not found: {original_path}")
    if not generated_path.exists():
        raise FileNotFoundError(f"Generated dataset not found: {generated_path}")

    # Start from a clean copy of the original.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original_path, output_path)

    with h5py.File(generated_path, "r") as gen_f:
        gen_demo_keys = sorted(
            k for k in gen_f["data"].keys() if k.startswith("demo_")
        )

    def _count_demos(path: Path) -> int:
        with h5py.File(path, "r") as f:
            return sum(1 for k in f["data"].keys() if k.startswith("demo_"))

    if not gen_demo_keys:
        # Nothing to append — return the original demo count unchanged.
        return _count_demos(output_path)

    with h5py.File(output_path, "a") as out_f:
        n_existing = sum(1 for k in out_f["data"].keys() if k.startswith("demo_"))

        with h5py.File(generated_path, "r") as gen_f:
            for i, demo_key in enumerate(gen_demo_keys):
                new_key = f"demo_{n_existing + i}"
                _copy_group(gen_f[f"data/{demo_key}"], out_f["data"].require_group(new_key))

        total = sum(1 for k in out_f["data"].keys() if k.startswith("demo_"))
        out_f["data"].attrs["total"] = total

    return _count_demos(output_path)


def _copy_group(src: "h5py.Group", dst: "h5py.Group") -> None:
    """Recursively copy *src* group into *dst*, reading data as numpy arrays.

    Using ``gen_f.copy()`` (which calls HDF5's ``H5Ocopy()``) can silently
    corrupt compressed chunks when the writer and reader use different HDF5
    library versions.  This manual copy re-encodes each dataset from its
    decompressed numpy form, which is portable across library versions.
    """
    # Copy group-level attributes
    for attr_key, attr_val in src.attrs.items():
        dst.attrs[attr_key] = attr_val

    for name, item in src.items():
        if isinstance(item, h5py.Group):
            child = dst.require_group(name)
            _copy_group(item, child)
        else:
            # Read dataset as numpy array and write with matching settings
            data = item[()]
            kwargs: dict = {}
            if item.chunks is not None:
                kwargs["chunks"] = item.chunks
            if item.compression is not None:
                kwargs["compression"] = item.compression
                if item.compression_opts is not None:
                    kwargs["compression_opts"] = item.compression_opts
            ds = dst.create_dataset(name, data=data, **kwargs)
            # Copy dataset-level attributes
            for attr_key, attr_val in item.attrs.items():
                ds.attrs[attr_key] = attr_val
