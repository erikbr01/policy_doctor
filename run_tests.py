"""Run policy_doctor tests with unittest (no pytest required).

Requires: conda env ``policy_doctor`` with editable installs
(``scripts/install_policy_doctor_env.sh``).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent


def run() -> int:
    loader = unittest.TestLoader()
    start_dir = str(_PROJECT_ROOT / "tests")
    suite = loader.discover(start_dir, pattern="test_*.py", top_level_dir=str(_PROJECT_ROOT))
    runner = unittest.runner.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run())
