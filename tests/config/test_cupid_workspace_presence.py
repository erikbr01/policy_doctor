"""Sanity: cupid diffusion workspace YAMLs referenced by data_source profiles exist on disk."""

from __future__ import annotations

import unittest
from pathlib import Path

from policy_doctor.paths import CUPID_ROOT, PROJECT_ROOT


class TestCupidWorkspaceYamlPresent(unittest.TestCase):
    def test_mimicgen_square_workspace(self):
        p = (
            CUPID_ROOT
            / "configs/low_dim/square_mimicgen_lowdim/diffusion_policy_cnn/config.yaml"
        )
        self.assertTrue(
            p.is_file(),
            f"expected {p} (install third_party/cupid submodule)",
        )

    def test_robocasa_layout_workspace(self):
        p = (
            CUPID_ROOT
            / "configs/low_dim/robocasa_layout_lowdim/diffusion_policy_cnn/config.yaml"
        )
        self.assertTrue(p.is_file(), f"expected {p}")

    def test_robocasa_lerobot_image_workspace(self):
        p = (
            CUPID_ROOT
            / "configs/image/robocasa_lerobot_atomic/diffusion_policy_transformer/config.yaml"
        )
        self.assertTrue(p.is_file(), f"expected {p}")

    def test_policy_doctor_repo_root(self):
        self.assertTrue((PROJECT_ROOT / "pyproject.toml").is_file())


if __name__ == "__main__":
    unittest.main()
