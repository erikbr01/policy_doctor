"""Third-party adapter path helpers."""

from __future__ import annotations

import sys
import unittest

from policy_doctor.data.adapters import ensure_robocasa_on_path
from policy_doctor.paths import ROBOCASA_ROOT


class TestRobocasaAdapter(unittest.TestCase):
    def test_submodule_present(self):
        self.assertTrue(
            ROBOCASA_ROOT.is_dir(),
            f"clone submodule: git submodule update --init {ROBOCASA_ROOT}",
        )

    def test_ensure_on_path_idempotent(self):
        if not ROBOCASA_ROOT.is_dir():
            self.skipTest("robocasa submodule missing")
        ensure_robocasa_on_path()
        n = sys.path.count(str(ROBOCASA_ROOT.resolve()))
        ensure_robocasa_on_path()
        self.assertEqual(sys.path.count(str(ROBOCASA_ROOT.resolve())), n)
        self.assertEqual(ensure_robocasa_on_path(), ROBOCASA_ROOT.resolve())


if __name__ == "__main__":
    unittest.main()
