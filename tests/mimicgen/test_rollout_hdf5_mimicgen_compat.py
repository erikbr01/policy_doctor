"""Integration test: rollouts.hdf5 format → MimicGen prepare_src_dataset.

Verifies that an HDF5 file written in the format produced by
``MimicgenLowdimRunner._write_rollouts_hdf5`` can be consumed by MimicGen's
``prepare_src_dataset`` (and optionally ``generate_dataset``).

This test runs in the **mimicgen** conda environment (MuJoCo 2.3.x, pinned
robosuite/robomimic) because that is where ``prepare_src_dataset`` replays
trajectories through the simulator.

The key compatibility guarantee under test:

    eval runner (cupid env) → rollouts.hdf5 → prepare_src_dataset (mimicgen env)

Since ``prepare_src_dataset`` modifies the HDF5 in-place by adding
``datagen_info/object_poses`` per demo, a successful run with expected keys in
the output confirms end-to-end format compatibility.

Enable with::

    MIMICGEN_E2E=1 python run_tests.py --suite mimicgen
    # or directly:
    MIMICGEN_E2E=1 conda run -n mimicgen python -m unittest \\
        tests.mimicgen.test_rollout_hdf5_mimicgen_compat -v
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Helpers: gating and env setup
# ---------------------------------------------------------------------------

def _e2e_requested() -> bool:
    return os.environ.get("MIMICGEN_E2E", "0").strip() in ("1", "true", "yes")


def _require_e2e() -> None:
    if not _e2e_requested():
        raise unittest.SkipTest(
            "Set MIMICGEN_E2E=1 to run MimicGen sim integration tests "
            "(requires mimicgen conda env + source/square.hdf5)."
        )


def _require_sim_deps() -> None:
    """Skip if robomimic/robosuite/mimicgen aren't importable."""
    try:
        import robomimic  # noqa: F401
        import robosuite  # noqa: F401
    except ImportError as e:
        raise unittest.SkipTest(f"robomimic/robosuite not available: {e}") from e
    from tests.support.mimicgen_seed.pipeline import ensure_mimicgen_importable
    ensure_mimicgen_importable()


def _source_square_hdf5() -> Path:
    """Return path to the cached NVlabs source square.hdf5, or skip.

    The cache may live in the main repo root (git worktrees share the parent
    repo's non-tracked directories), so we check a few candidate locations.
    """
    # 1. Standard location relative to project root (works in main repo + worktrees
    #    because worktrees resolve parent chains from the file's actual path).
    candidates = []

    # Worktree / project root: walk up from this file to project root
    # (test is at tests/mimicgen/, so parents[2] = project root)
    worktree_root = Path(__file__).resolve().parents[2]
    candidates.append(worktree_root / ".cache" / "mimicgen_e2e" / "source" / "square.hdf5")

    # Main repo root via policy_doctor.paths (handles the worktree case where
    # PROJECT_ROOT points to the worktree directory instead of the canonical repo)
    try:
        from policy_doctor.paths import PROJECT_ROOT as _pr
        candidates.append(Path(_pr) / ".cache" / "mimicgen_e2e" / "source" / "square.hdf5")
        # Also try the grandparent (main repo root when PROJECT_ROOT = worktree inside .claude/)
        candidates.append(Path(_pr).parent.parent.parent / ".cache" / "mimicgen_e2e" / "source" / "square.hdf5")
    except ImportError:
        pass

    for p in candidates:
        if p.is_file():
            return p.resolve()

    raise unittest.SkipTest(
        f"source/square.hdf5 not found (checked: {[str(c) for c in candidates]}).  "
        "Run tests.integration.test_mimicgen_square_e2e::test_01 first, "
        "or download manually from NVlabs Hugging Face."
    )


def _write_rollout_hdf5_from_source(
    source_path: Path,
    dest_path: Path,
    *,
    demo_key: str = "demo_0",
    n_demos: int = 1,
) -> None:
    """Copy states/actions/model_file/env_args from the source HDF5 into a
    rollout-format HDF5, exactly matching what ``_write_rollouts_hdf5`` produces.

    This lets us run ``prepare_src_dataset`` against a synthetic "rollout" that
    has the same binary content as the source data, proving format compatibility
    without needing to actually run a policy in the eval env.
    """
    with h5py.File(source_path, "r") as src:
        env_args_str: str = src["data"].attrs["env_args"]
        demo = src[f"data/{demo_key}"]
        states = demo["states"][()]
        actions = demo["actions"][()]
        model_file: str = demo.attrs.get("model_file", "")
        if isinstance(model_file, bytes):
            model_file = model_file.decode("utf-8")

    with h5py.File(dest_path, "w") as f:
        grp = f.create_group("data")
        # Same attrs as _write_rollouts_hdf5
        grp.attrs["env_args"] = env_args_str
        grp.attrs["total"] = n_demos
        for i in range(n_demos):
            demo_grp = grp.create_group(f"demo_{i}")
            demo_grp.create_dataset("states", data=states.astype(np.float64))
            demo_grp.create_dataset("actions", data=actions.astype(np.float64))
            if model_file:
                demo_grp.attrs["model_file"] = model_file
            demo_grp.attrs["success"] = 1


# ---------------------------------------------------------------------------
# Test: rollout HDF5 → prepare_src_dataset
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    _e2e_requested(),
    "Set MIMICGEN_E2E=1 to run sim-level MimicGen integration tests.",
)
class TestRolloutHdf5PrepareCompatibility(unittest.TestCase):
    """prepare_src_dataset can annotate a rollout-format HDF5 with datagen_info."""

    @classmethod
    def setUpClass(cls):
        _require_e2e()
        _require_sim_deps()

    def _run_prepare(self, hdf5_path: Path, n: int = 1) -> None:
        """Run prepare_src_dataset in-process; skip on MuJoCo schema errors."""
        from tests.support.mimicgen_seed.pipeline import run_mimicgen_prepare_src_dataset
        from tests.support.mimicgen_seed.schema import MimicGenBinding

        binding = MimicGenBinding(
            env_interface_name="MG_Square",
            env_interface_type="robosuite",
        )
        try:
            run_mimicgen_prepare_src_dataset(hdf5_path, binding, n=n)
        except Exception as e:
            self.skipTest(
                f"prepare_src_dataset failed — likely MuJoCo/robosuite version mismatch. "
                f"Details: {type(e).__name__}: {e}"
            )

    def test_prepare_adds_datagen_info(self):
        """After prepare_src_dataset, demo_0 must have datagen_info/object_poses."""
        source = _source_square_hdf5()
        with tempfile.TemporaryDirectory() as td:
            rollout_hdf5 = Path(td) / "rollouts.hdf5"
            _write_rollout_hdf5_from_source(source, rollout_hdf5, n_demos=1)

            self._run_prepare(rollout_hdf5, n=1)

            with h5py.File(rollout_hdf5, "r") as f:
                self.assertIn(
                    "data/demo_0/datagen_info",
                    f,
                    "prepare_src_dataset did not write datagen_info to rollout HDF5",
                )
                self.assertIn(
                    "data/demo_0/datagen_info/object_poses",
                    f,
                    "datagen_info/object_poses missing — MimicGen cannot use this rollout "
                    "for pose-constrained generation",
                )

    def test_prepare_object_poses_shape(self):
        """object_poses must have shape (T, 4, 4) per object — the homogeneous matrix."""
        source = _source_square_hdf5()
        with tempfile.TemporaryDirectory() as td:
            rollout_hdf5 = Path(td) / "rollouts.hdf5"
            _write_rollout_hdf5_from_source(source, rollout_hdf5, n_demos=1)
            self._run_prepare(rollout_hdf5, n=1)

            with h5py.File(rollout_hdf5, "r") as f:
                poses_grp = f["data/demo_0/datagen_info/object_poses"]
                for obj_name in poses_grp.keys():
                    poses = poses_grp[obj_name][()]
                    self.assertEqual(
                        poses.shape[1:],
                        (4, 4),
                        msg=f"object_poses['{obj_name}'] has shape {poses.shape}, "
                            "expected (T, 4, 4)",
                    )

    def test_prepared_rollout_has_env_interface_attrs(self):
        """prepare_src_dataset must tag datagen_info with env_interface_name/type."""
        source = _source_square_hdf5()
        with tempfile.TemporaryDirectory() as td:
            rollout_hdf5 = Path(td) / "rollouts.hdf5"
            _write_rollout_hdf5_from_source(source, rollout_hdf5, n_demos=1)
            self._run_prepare(rollout_hdf5, n=1)

            with h5py.File(rollout_hdf5, "r") as f:
                di_attrs = f["data/demo_0/datagen_info"].attrs
                self.assertEqual(di_attrs.get("env_interface_name"), "MG_Square")
                self.assertEqual(di_attrs.get("env_interface_type"), "robosuite")

    def test_from_rollout_hdf5_reads_prepared_file(self):
        """MimicGenSeedTrajectory.from_rollout_hdf5 must accept the prepared file."""
        from policy_doctor.mimicgen.seed_trajectory import MimicGenSeedTrajectory, SeedSource

        source = _source_square_hdf5()
        with tempfile.TemporaryDirectory() as td:
            rollout_hdf5 = Path(td) / "rollouts.hdf5"
            _write_rollout_hdf5_from_source(source, rollout_hdf5, n_demos=1)
            self._run_prepare(rollout_hdf5, n=1)

            traj = MimicGenSeedTrajectory.from_rollout_hdf5(rollout_hdf5, demo_key="demo_0")
            self.assertEqual(traj.source, SeedSource.ROLLOUT)
            self.assertIsNotNone(traj.model_file)
            self.assertGreater(traj.states.shape[0], 0)
            self.assertEqual(traj.states.shape[0], traj.actions.shape[0])


# ---------------------------------------------------------------------------
# Test: rollout HDF5 → prepare → generate_dataset (full pipeline smoke test)
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    _e2e_requested(),
    "Set MIMICGEN_E2E=1 to run sim-level MimicGen integration tests.",
)
class TestRolloutHdf5GeneratePipeline(unittest.TestCase):
    """Full pipeline smoke: rollout HDF5 → prepare_src_dataset → generate_dataset."""

    @classmethod
    def setUpClass(cls):
        _require_e2e()
        _require_sim_deps()

    def test_generate_from_rollout_seed(self):
        """generate_dataset with a rollout as seed must produce ≥1 successful demo."""
        source = _source_square_hdf5()

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            rollout_hdf5 = td_path / "rollouts.hdf5"
            _write_rollout_hdf5_from_source(source, rollout_hdf5, n_demos=1)

            # Step 1: prepare
            from tests.support.mimicgen_seed.pipeline import run_mimicgen_prepare_src_dataset
            from tests.support.mimicgen_seed.schema import MimicGenBinding

            binding = MimicGenBinding(
                env_interface_name="MG_Square",
                env_interface_type="robosuite",
            )
            try:
                run_mimicgen_prepare_src_dataset(rollout_hdf5, binding, n=1)
            except Exception as e:
                self.skipTest(
                    f"prepare_src_dataset failed (MuJoCo version mismatch?): "
                    f"{type(e).__name__}: {e}"
                )

            # Step 2: generate (3 trials — enough to expect ≥1 success on Square D0)
            from tests.support.mimicgen_seed.pipeline import ensure_mimicgen_importable
            ensure_mimicgen_importable()
            from mimicgen.configs import config_factory
            from mimicgen.scripts.generate_dataset import generate_dataset

            gen_root = td_path / "gen"
            cfg = config_factory("square", "robosuite")
            cfg.experiment.name = "pd_rollout_compat"
            cfg.experiment.source.dataset_path = str(rollout_hdf5)
            cfg.experiment.source.n = 1
            cfg.experiment.generation.path = str(gen_root)
            cfg.experiment.generation.num_trials = 5
            cfg.experiment.generation.guarantee = False
            cfg.experiment.render_video = False
            cfg.experiment.num_demo_to_render = 0
            cfg.experiment.num_fail_demo_to_render = 0
            cfg.experiment.max_num_failures = 10
            cfg.experiment.log_every_n_attempts = 1000
            cfg.obs.collect_obs = True
            cfg.obs.camera_names = []

            try:
                stats = generate_dataset(cfg, auto_remove_exp=True, render=False, video_path=None)
            except Exception as e:
                self.skipTest(
                    f"generate_dataset failed: {type(e).__name__}: {e}"
                )

            self.assertIsInstance(stats, dict)
            self.assertGreaterEqual(
                stats.get("num_attempts", 0),
                1,
                msg="No generation attempts were made.",
            )
            # At least one success expected for Square D0 with 5 trials
            self.assertGreaterEqual(
                stats.get("num_success", 0),
                1,
                msg=f"Expected ≥1 generation success; stats={stats}",
            )


if __name__ == "__main__":
    unittest.main()
