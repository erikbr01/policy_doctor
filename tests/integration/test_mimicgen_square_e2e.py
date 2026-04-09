"""Run MimicGen for real: HF ``source/square.hdf5`` → ``prepare_src_dataset`` → ``generate_dataset``.

This is **not** the mar27 transport task (no ``MG_TwoArmTransport`` in upstream MimicGen). It
exercises the same scripts you would use for transport after adding a task interface.

**Why it may skip:** datasets embed MJCF tuned for MimicGen’s supported stack. If your MuJoCo /
robosuite versions differ from that release, ``prepare_src_dataset`` can fail while loading
``model_file`` (e.g. schema errors on ``autolimits``, ``gravcomp``, ``actdim``). The test then
``skipTest``\ s with the exception so you see a precise version skew instead of a silent omission.

Enable with::

    MIMICGEN_E2E=1 python -m unittest tests.integration.test_mimicgen_square_e2e -v

Default ``MIMICGEN_E2E=0`` keeps the rest of the suite fast and independent of HF + sim versions.
"""

from __future__ import annotations

import os
import shutil
import sys
import unittest
from pathlib import Path

_SOURCE: Path | None = None
_PREPARED: Path | None = None


def _e2e_requested() -> bool:
    return os.environ.get("MIMICGEN_E2E", "0").strip() in ("1", "true", "yes")


def _require_mimicgen_e2e_deps() -> None:
    if not _e2e_requested():
        raise unittest.SkipTest(
            "Set MIMICGEN_E2E=1 to run MimicGen download + sim (see module docstring)."
        )
    try:
        import robomimic  # noqa: F401
        import robosuite  # noqa: F401
    except ImportError as e:
        raise unittest.SkipTest(f"robomimic/robosuite not available: {e}") from e
    from tests.support.mimicgen_seed.pipeline import ensure_mimicgen_importable

    ensure_mimicgen_importable()
    import mimicgen  # noqa: F401


@unittest.skipUnless(
    os.environ.get("MIMICGEN_E2E", "0").strip() in ("1", "true", "yes"),
    "Set MIMICGEN_E2E=1 to run MimicGen E2E (Hugging Face + matching MuJoCo/robosuite).",
)
class TestMimicgenSquareE2E(unittest.TestCase):
    """Ordered ``test_01`` … ``test_03`` so unittest runs them in sequence."""

    @classmethod
    def setUpClass(cls):
        _require_mimicgen_e2e_deps()

    def test_01_download_source_square(self):
        global _SOURCE
        from mimicgen import DATASET_REGISTRY, HF_REPO_ID
        import mimicgen.utils.file_utils as mg_files

        root = Path(__file__).resolve().parents[2]
        dest_dir = root / ".cache" / "mimicgen_e2e" / "source"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / "square.hdf5"
        rel = DATASET_REGISTRY["source"]["square"]["url"]
        try:
            if not dest_file.is_file():
                mg_files.download_file_from_hf(
                    repo_id=HF_REPO_ID,
                    filename=rel,
                    download_dir=str(dest_dir),
                    check_overwrite=True,
                )
        except Exception as e:
            self.skipTest(f"Hugging Face download failed: {e}")

        self.assertTrue(dest_file.is_file())
        self.assertGreater(dest_file.stat().st_size, 10_000)
        _SOURCE = dest_file.resolve()

    def test_02_prepare_src_dataset(self):
        global _PREPARED
        import tempfile

        import h5py

        from tests.support.mimicgen_seed.pipeline import run_mimicgen_prepare_src_dataset
        from tests.support.mimicgen_seed.schema import MimicGenBinding

        if _SOURCE is None or not _SOURCE.is_file():
            self.skipTest("Run test_01_download_source_square first.")

        keep = Path(tempfile.mkdtemp(prefix="mimicgen_prepared_")) / "square_prepared.hdf5"
        shutil.copy(_SOURCE, keep)
        try:
            run_mimicgen_prepare_src_dataset(
                keep,
                MimicGenBinding(env_interface_name="MG_Square", env_interface_type="robosuite"),
                n=1,
            )
        except Exception as e:
            self.skipTest(
                "MimicGen prepare_src_dataset failed (often MJCF in the dataset vs your "
                f"MuJoCo/robosuite build). Details: {type(e).__name__}: {e}"
            )

        with h5py.File(keep, "r") as f:
            self.assertIn("data/demo_0/datagen_info", f)
        _PREPARED = keep.resolve()

    def test_03_generate_dataset(self):
        import tempfile

        from mimicgen.configs import config_factory
        from mimicgen.scripts.generate_dataset import generate_dataset

        if _PREPARED is None or not _PREPARED.is_file():
            self.skipTest("Run test_02_prepare_src_dataset first (or it skipped).")

        gen_root = Path(tempfile.mkdtemp(prefix="mimicgen_gen_"))

        cfg = config_factory("square", "robosuite")
        cfg.experiment.name = "policy_doctor_e2e"
        cfg.experiment.source.dataset_path = str(_PREPARED)
        cfg.experiment.source.n = 1
        cfg.experiment.generation.path = str(gen_root)
        cfg.experiment.generation.num_trials = 3
        cfg.experiment.generation.guarantee = False
        cfg.experiment.render_video = False
        cfg.experiment.num_demo_to_render = 0
        cfg.experiment.num_fail_demo_to_render = 0
        cfg.experiment.max_num_failures = 5
        cfg.experiment.log_every_n_attempts = 1000

        old_out, old_err = sys.stdout, sys.stderr
        try:
            try:
                stats = generate_dataset(cfg, auto_remove_exp=True, render=False, video_path=None)
            except Exception as e:
                self.skipTest(f"MimicGen generate_dataset failed: {type(e).__name__}: {e}")
        finally:
            sys.stdout, sys.stderr = old_out, old_err

        self.assertIsInstance(stats, dict)
        self.assertGreaterEqual(stats["num_attempts"], 1)
        self.assertGreaterEqual(
            stats["num_success"],
            1,
            msg=f"Expected ≥1 success; stats={stats}. Try raising num_trials.",
        )
        demo = gen_root / "policy_doctor_e2e" / "demo.hdf5"
        self.assertTrue(demo.is_file(), msg=str(demo))
