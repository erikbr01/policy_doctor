"""Unit tests for cupid diffusion extra Hydra overrides."""

from __future__ import annotations

import unittest

try:
    from omegaconf import OmegaConf

    from policy_doctor.curation_pipeline.diffusion_overrides import (
        baseline_diffusion_extra_overrides,
    )
except ImportError:  # pragma: no cover
    OmegaConf = None  # type: ignore
    baseline_diffusion_extra_overrides = None  # type: ignore


@unittest.skipUnless(OmegaConf is not None, "omegaconf not installed")
class TestBaselineDiffusionExtraOverrides(unittest.TestCase):
    def test_empty_baseline(self):
        self.assertEqual(baseline_diffusion_extra_overrides(None), [])
        self.assertEqual(baseline_diffusion_extra_overrides({}), [])

    def test_compose_overrides_list(self):
        b = OmegaConf.create(
            {"diffusion_compose_overrides": ["task.name=foo", "training.lr=1e-4"]}
        )
        self.assertEqual(
            baseline_diffusion_extra_overrides(b),
            ["task.name=foo", "training.lr=1e-4"],
        )

    def test_dataset_path_shorthand(self):
        b = OmegaConf.create({"diffusion_dataset_path": "/tmp/demo.hdf5"})
        self.assertEqual(
            baseline_diffusion_extra_overrides(b),
            [
                "++task.dataset.dataset_path=/tmp/demo.hdf5",
                "++task.env_runner.dataset_path=/tmp/demo.hdf5",
            ],
        )


if __name__ == "__main__":
    unittest.main()
