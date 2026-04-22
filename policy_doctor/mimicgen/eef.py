"""Extract end-effector trajectories from MimicGen HDF5 files.

After running ``prepare_src_dataset`` and ``generate_dataset``, both the
prepared source demo and generated demos contain a ``datagen_info/eef_pose``
dataset per episode.  ``eef_pose`` is a ``(T, 4, 4)`` homogeneous
transformation matrix; the XYZ translation is the last column of the rotation
block: ``eef_pose[:, :3, 3]``.

Functions here are pure numpy + h5py — safe from any Python environment.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def extract_eef_xyz_from_hdf5(hdf5_path: Path | str) -> list[np.ndarray]:
    """Return per-demo end-effector XYZ trajectories from a prepared/generated HDF5.

    Reads ``data/<demo_key>/datagen_info/eef_pose`` for every episode in the
    file and extracts the translation column.

    Args:
        hdf5_path: Path to the HDF5 file (prepared source or generated demo).

    Returns:
        List of ``(T, 3)`` float32 arrays, one per demo, in ``demo_*`` key order.
        Returns an empty list if no ``datagen_info/eef_pose`` is found.
    """
    hdf5_path = Path(hdf5_path)
    results: list[np.ndarray] = []

    with h5py.File(hdf5_path, "r") as f:
        data_grp = f.get("data")
        if data_grp is None:
            return results

        demo_keys = sorted(k for k in data_grp.keys() if k.startswith("demo_"))
        for key in demo_keys:
            ep = data_grp[key]
            eef_pose_key = "datagen_info/eef_pose"
            if eef_pose_key not in ep:
                continue
            eef_pose = np.array(ep[eef_pose_key], dtype=np.float32)
            # eef_pose shape: (T, 4, 4) — translation is column index 3 of the 3x3 rotation
            if eef_pose.ndim == 3 and eef_pose.shape[1:] == (4, 4):
                xyz = eef_pose[:, :3, 3]  # (T, 3)
            elif eef_pose.ndim == 2 and eef_pose.shape[1] >= 3:
                # Some variants store a flat (T, 7) pose (pos + quat); take first 3
                xyz = eef_pose[:, :3]
            else:
                continue
            results.append(xyz)

    return results


def extract_object_pose_t0_from_hdf5(
    hdf5_path: Path | str,
    object_name: str = "square_nut",
) -> list[np.ndarray]:
    """Return the initial (t=0) pose of *object_name* for every demo.

    Reads ``data/<demo_key>/datagen_info/object_poses/<object_name>[0]``.

    Args:
        hdf5_path:   Path to the prepared/generated HDF5.
        object_name: Key under ``datagen_info/object_poses`` (e.g. ``"square_nut"``).

    Returns:
        List of ``(4, 4)`` float32 homogeneous matrices, one per demo.
        Returns an empty list if the object key is not present.
    """
    hdf5_path = Path(hdf5_path)
    results: list[np.ndarray] = []
    with h5py.File(hdf5_path, "r") as f:
        data_grp = f.get("data")
        if data_grp is None:
            return results
        demo_keys = sorted(k for k in data_grp.keys() if k.startswith("demo_"))
        for key in demo_keys:
            ep = data_grp[key]
            pose_key = f"datagen_info/object_poses/{object_name}"
            if pose_key not in ep:
                continue
            poses = np.array(ep[pose_key], dtype=np.float32)  # (T, 4, 4)
            if poses.ndim == 3 and poses.shape[1:] == (4, 4):
                results.append(poses[0])
    return results


def extract_eef_pose_at_lowest_z_from_hdf5(hdf5_path: Path | str) -> list[np.ndarray]:
    """Return the EEF pose ``(4, 4)`` at the timestep with minimum Z, per demo.

    Useful for visualising gripper approach orientation.

    Returns:
        List of ``(4, 4)`` float32 homogeneous matrices, one per demo.
        Returns an empty list if ``datagen_info/eef_pose`` is not present.
    """
    hdf5_path = Path(hdf5_path)
    results: list[np.ndarray] = []
    with h5py.File(hdf5_path, "r") as f:
        data_grp = f.get("data")
        if data_grp is None:
            return results
        demo_keys = sorted(k for k in data_grp.keys() if k.startswith("demo_"))
        for key in demo_keys:
            ep = data_grp[key]
            pose_key = "datagen_info/eef_pose"
            if pose_key not in ep:
                continue
            poses = np.array(ep[pose_key], dtype=np.float32)  # (T, 4, 4)
            if poses.ndim != 3 or poses.shape[1:] != (4, 4):
                continue
            z_vals = poses[:, 2, 3]  # Z translation at each timestep
            t_min_z = int(np.argmin(z_vals))
            results.append(poses[t_min_z])
    return results


def extract_initial_eef_xyz(hdf5_path: Path | str) -> np.ndarray:
    """Return the initial (t=0) EEF XYZ position for every demo.

    Convenience wrapper around :func:`extract_eef_xyz_from_hdf5`.

    Returns:
        ``(N, 3)`` float32 array of initial positions, one row per demo.
        Returns empty ``(0, 3)`` array if no trajectories are found.
    """
    trajs = extract_eef_xyz_from_hdf5(hdf5_path)
    if not trajs:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack([t[0] for t in trajs], axis=0)
