"""Hydra composition: data_source profiles resolve expected stacks."""

from __future__ import annotations

import unittest

try:
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover
    hydra = None  # type: ignore
    GlobalHydra = None  # type: ignore
    OmegaConf = None  # type: ignore

from policy_doctor.paths import CONFIGS_DIR, ROBOCASA_ROOT


@unittest.skipUnless(hydra is not None, "hydra-core / omegaconf not installed")
class TestDataSourceHydraCompose(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        GlobalHydra.instance().clear()
        hydra.initialize_config_dir(
            config_dir=str(CONFIGS_DIR.resolve()),
            version_base=None,
        )

    @classmethod
    def tearDownClass(cls):
        GlobalHydra.instance().clear()

    def _compose(self, overrides: list[str]):
        cfg = hydra.compose(config_name="config", overrides=overrides)
        OmegaConf.resolve(cfg)
        return cfg

    def test_default_is_cupid_robomimic(self):
        cfg = self._compose([])
        self.assertEqual(OmegaConf.select(cfg, "data_source.id"), "cupid_robomimic")
        self.assertEqual(OmegaConf.select(cfg, "task"), "transport_mh")
        self.assertEqual(OmegaConf.select(cfg, "baseline.task"), "transport_mh")

    def test_mimicgen_square_profile(self):
        cfg = self._compose(["data_source=mimicgen_square"])
        self.assertEqual(OmegaConf.select(cfg, "data_source.id"), "mimicgen_square")
        self.assertEqual(OmegaConf.select(cfg, "task"), "square_mh_mimicgen")
        self.assertIn(
            "square_mimicgen_lowdim",
            OmegaConf.select(cfg, "baseline.config_dir"),
        )
        self.assertTrue(OmegaConf.select(cfg, "data_source.mimicgen_submodule"))

    def test_robocasa_layout_profile(self):
        cfg = self._compose(["data_source=robocasa_layout"])
        self.assertEqual(OmegaConf.select(cfg, "data_source.id"), "robocasa_layout")
        self.assertEqual(OmegaConf.select(cfg, "task"), "robocasa_layout_lowdim")
        self.assertTrue(OmegaConf.select(cfg, "data_source.robocasa_submodule"))
        self.assertTrue(ROBOCASA_ROOT.is_dir(), f"expected submodule at {ROBOCASA_ROOT}")

    def test_experiment_overrides_data_source(self):
        cfg = self._compose(
            [
                "data_source=mimicgen_square",
                "experiment=trak_filtering_mar13_p96",
            ]
        )
        # experiment sets task_config etc.; data_source should still set mimicgen id
        self.assertEqual(OmegaConf.select(cfg, "data_source.id"), "mimicgen_square")


if __name__ == "__main__":
    unittest.main()
