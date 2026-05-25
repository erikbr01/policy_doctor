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
    max_original_demos: int | None = None,
) -> int:
    """Copy *original_path* to *output_path*, then append all demos from *generated_path*.

    Args:
        original_path:      Path to the original training HDF5 (not modified).
        generated_path:     Path to the MimicGen-generated HDF5 (not modified).
        output_path:        Destination path for the combined HDF5.  Created (or
                            overwritten) by this function.
        max_original_demos: If set, take only the first N original demos (sorted
                            by demo_key, matching cupid's max_train_episodes
                            slicing). Use this to match the baseline's
                            max_train_episodes so the combined arm trains on the
                            same baseline subset + the generated demos, instead
                            of inheriting the full source dataset.

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

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if max_original_demos is None:
        # Whole-source copy — fast path via shutil.
        shutil.copy2(original_path, output_path)
    else:
        # Subset copy — materialize only the first max_original_demos demos.
        with h5py.File(original_path, "r") as orig_f, h5py.File(output_path, "w") as out_f:
            out_data = out_f.create_group("data")
            for attr_key, attr_val in orig_f["data"].attrs.items():
                out_data.attrs[attr_key] = attr_val
            orig_demo_keys = sorted(
                k for k in orig_f["data"].keys() if k.startswith("demo_")
            )[:max_original_demos]
            for k in orig_demo_keys:
                orig_f.copy(f"data/{k}", out_data, name=k)
            # Preserve top-level groups beside "data" (e.g. "mask") if present.
            for top in orig_f.keys():
                if top != "data" and top not in out_f:
                    orig_f.copy(top, out_f, name=top)

    with h5py.File(generated_path, "r") as gen_f:
        gen_demo_keys = sorted(
            k for k in gen_f["data"].keys() if k.startswith("demo_")
        )

    def _count_demos(path: Path) -> int:
        with h5py.File(path, "r") as f:
            return sum(1 for k in f["data"].keys() if k.startswith("demo_"))

    if not gen_demo_keys:
        return _count_demos(output_path)

    with h5py.File(output_path, "a") as out_f:
        # Compute the next index from the MAX numeric demo index, not the
        # count: sorted demo keys are lexicographic (demo_100 < demo_2), so
        # taking the first N keys can include high-numbered ones. Using count
        # alone could collide on append.
        def _demo_idx(k: str) -> int:
            try:
                return int(k.split("_", 1)[1])
            except (IndexError, ValueError):
                return -1
        existing_idxs = [
            _demo_idx(k) for k in out_f["data"].keys() if k.startswith("demo_")
        ]
        next_idx = (max(existing_idxs) + 1) if existing_idxs else 0

        with h5py.File(generated_path, "r") as gen_f:
            for i, demo_key in enumerate(gen_demo_keys):
                new_key = f"demo_{next_idx + i}"
                gen_f.copy(f"data/{demo_key}", out_f["data"], name=new_key)

    return _count_demos(output_path)
