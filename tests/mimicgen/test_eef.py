"""Unit tests for policy_doctor.mimicgen.eef."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from policy_doctor.mimicgen.eef import extract_eef_xyz_from_hdf5, extract_initial_eef_xyz


def _write_hdf5_with_datagen_info(
    path: Path,
    eef_poses: list[np.ndarray],  # list of (T, 4, 4) arrays, one per demo
) -> None:
    """Write a minimal HDF5 with datagen_info/eef_pose for each demo."""
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["total"] = np.int64(sum(p.shape[0] for p in eef_poses))
        for i, eef_pose in enumerate(eef_poses):
            ep = data.create_group(f"demo_{i}")
            ep.attrs["num_samples"] = np.int64(eef_pose.shape[0])
            di = ep.create_group("datagen_info")
            di.create_dataset("eef_pose", data=eef_pose.astype(np.float32))


def _identity_pose_sequence(n_timesteps: int, tx: float = 0.1, ty: float = 0.2, tz: float = 0.3) -> np.ndarray:
    """Build (n_timesteps, 4, 4) identity rotation + constant translation."""
    poses = np.eye(4, dtype=np.float32)[np.newaxis].repeat(n_timesteps, axis=0)
    poses[:, 0, 3] = tx
    poses[:, 1, 3] = ty
    poses[:, 2, 3] = tz
    return poses


class TestExtractEefXyzFromHdf5(unittest.TestCase):
    def test_returns_xyz_from_4x4_pose(self):
        pose = _identity_pose_sequence(10, tx=0.5, ty=0.6, tz=0.7)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "gen.hdf5"
            _write_hdf5_with_datagen_info(p, [pose])
            result = extract_eef_xyz_from_hdf5(p)

        self.assertEqual(len(result), 1)
        xyz = result[0]
        self.assertEqual(xyz.shape, (10, 3))
        np.testing.assert_allclose(xyz[:, 0], 0.5, atol=1e-5)
        np.testing.assert_allclose(xyz[:, 1], 0.6, atol=1e-5)
        np.testing.assert_allclose(xyz[:, 2], 0.7, atol=1e-5)

    def test_multiple_demos(self):
        poses = [
            _identity_pose_sequence(5, tx=0.0),
            _identity_pose_sequence(8, tx=1.0),
            _identity_pose_sequence(3, tx=2.0),
        ]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "gen.hdf5"
            _write_hdf5_with_datagen_info(p, poses)
            result = extract_eef_xyz_from_hdf5(p)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].shape, (5, 3))
        self.assertEqual(result[1].shape, (8, 3))
        self.assertEqual(result[2].shape, (3, 3))

    def test_empty_file_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "empty.hdf5"
            with h5py.File(p, "w") as f:
                f.create_group("data")
            result = extract_eef_xyz_from_hdf5(p)
        self.assertEqual(result, [])

    def test_no_datagen_info_skipped(self):
        """Demos without datagen_info/eef_pose are skipped silently."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "no_di.hdf5"
            with h5py.File(p, "w") as f:
                data = f.create_group("data")
                ep = data.create_group("demo_0")
                ep.create_dataset("actions", data=np.zeros((3, 2)))
                ep.create_dataset("states", data=np.zeros((3, 4)))
            result = extract_eef_xyz_from_hdf5(p)
        self.assertEqual(result, [])

    def test_flat_7d_pose_fallback(self):
        """Also handles flat (T, 7) pose format (pos + quat)."""
        flat_pose = np.zeros((6, 7), dtype=np.float32)
        flat_pose[:, 0] = 0.1  # x
        flat_pose[:, 1] = 0.2  # y
        flat_pose[:, 2] = 0.3  # z
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "flat.hdf5"
            with h5py.File(p, "w") as f:
                data = f.create_group("data")
                ep = data.create_group("demo_0")
                di = ep.create_group("datagen_info")
                di.create_dataset("eef_pose", data=flat_pose)
            result = extract_eef_xyz_from_hdf5(p)

        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0][:, 0], 0.1, atol=1e-5)


class TestExtractInitialEefXyz(unittest.TestCase):
    def test_returns_n_by_3(self):
        poses = [_identity_pose_sequence(4, tx=float(i)) for i in range(3)]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "gen.hdf5"
            _write_hdf5_with_datagen_info(p, poses)
            t0 = extract_initial_eef_xyz(p)

        self.assertEqual(t0.shape, (3, 3))
        np.testing.assert_allclose(t0[:, 0], [0.0, 1.0, 2.0], atol=1e-5)

    def test_empty_returns_zeros(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "empty.hdf5"
            with h5py.File(p, "w") as f:
                f.create_group("data")
            t0 = extract_initial_eef_xyz(p)
        self.assertEqual(t0.shape, (0, 3))
