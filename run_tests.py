"""Run policy_doctor tests with unittest (no pytest required).

Default: discover all tests under ``tests/`` (requires a single env with all deps).

Suite-scoped runs (use matching conda env — see ``scripts/run_tests_*.sh``):

* ``policy_doctor`` — package tests, curation pipeline integration, VLM, data, etc.
* ``cupid`` — mar27 transport + fingerprint tests (diffusion_policy on path).
* ``mimicgen`` — MimicGen seed tests and optional E2E.
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent

# Integration modules for the policy_doctor env (no cupid/mimicgen sim deps).
_ORCHESTRATION_INTEGRATION_MODULES = (
    "tests.integration.test_recreate_curation_config",
    "tests.integration.test_markov_property",
    "tests.integration.test_compare_iv_vs_policy_doctor",
    "tests.integration.test_diffusion_data_source_smoke",
)

_CUPID_TEST_MODULES = (
    "tests.integration.test_mar27_transport_mimicgen_pipeline",
    "tests.integration.test_fingerprint_episode_ends",
)

_MIMICGEN_TEST_MODULES = (
    "tests.test_mimicgen_seed_abstractions",
    "tests.integration.test_mimicgen_square_e2e",
)

_ORCHESTRATION_SUBPACKAGES = (
    "tests/config",
    "tests/curation",
    "tests/data",
    "tests/behaviors",
    "tests/computations",
    "tests/curation",
    "tests/plotting",
    "tests/vlm",
)


def _suite_policy_doctor() -> unittest.TestSuite:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for mod in _ORCHESTRATION_INTEGRATION_MODULES:
        suite.addTests(loader.loadTestsFromName(mod))
    for rel in _ORCHESTRATION_SUBPACKAGES:
        suite.addTests(
            loader.discover(
                str(_PROJECT_ROOT / rel),
                pattern="test_*.py",
                top_level_dir=str(_PROJECT_ROOT),
            )
        )
    return suite


def _suite_cupid() -> unittest.TestSuite:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for mod in _CUPID_TEST_MODULES:
        suite.addTests(loader.loadTestsFromName(mod))
    return suite


def _suite_mimicgen() -> unittest.TestSuite:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for mod in _MIMICGEN_TEST_MODULES:
        suite.addTests(loader.loadTestsFromName(mod))
    return suite


def _suite_discover_all() -> unittest.TestSuite:
    loader = unittest.TestLoader()
    return loader.discover(
        str(_PROJECT_ROOT / "tests"),
        pattern="test_*.py",
        top_level_dir=str(_PROJECT_ROOT),
    )


def run_suite(name: str) -> int:
    if name == "policy_doctor":
        suite = _suite_policy_doctor()
    elif name == "cupid":
        suite = _suite_cupid()
    elif name == "mimicgen":
        suite = _suite_mimicgen()
    elif name == "all":
        suite = _suite_discover_all()
    else:
        raise ValueError(f"Unknown suite: {name!r}")

    runner = unittest.runner.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--suite",
        choices=("all", "policy_doctor", "cupid", "mimicgen"),
        default="all",
        help="Test set to run (default: all — full discover; use stack-specific with matching conda env).",
    )
    args = p.parse_args(argv)
    return run_suite(args.suite)


if __name__ == "__main__":
    sys.exit(main())
