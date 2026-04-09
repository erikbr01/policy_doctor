"""Integration test: Markov property of real behavior graphs.

Loads clustering results produced by the pipeline and checks whether
the behavior graph transitions satisfy the first-order Markov assumption
using chi-squared, permutation exact, and permutation modal tests.
"""

import unittest
from pathlib import Path

import numpy as np

from policy_doctor.paths import iv_task_configs_base

_IV_CFG = iv_task_configs_base()

_CLUSTERING_BASE = _IV_CFG / "transport_mh_jan28" / "clustering"

_CLUSTERING_DIRS = {
    "seed0": _CLUSTERING_BASE / "trak_filtering_mar13_seed0_kmeans_k20",
    "seed1": _CLUSTERING_BASE / "trak_filtering_mar13_seed1_kmeans_k20",
    "seed2": _CLUSTERING_BASE / "trak_filtering_mar13_seed2_kmeans_k20",
}

SIGNIFICANCE_LEVEL = 0.05
N_PERMUTATIONS = 5000


def _require_clustering(clustering_dir: Path):
    if not clustering_dir.is_dir() or not (clustering_dir / "manifest.yaml").exists():
        raise unittest.SkipTest(f"Clustering not found: {clustering_dir}")


def _load_clustering(clustering_dir: Path):
    from policy_doctor.data.clustering_loader import load_clustering_result_from_path

    return load_clustering_result_from_path(clustering_dir)


def _print_result(tag: str, result: dict):
    violations = [
        (sid, r.p_value)
        for sid, r in result["per_state"].items()
        if r.testable and not r.markov_holds
    ]
    print(f"\n[{tag}] tested={result['num_states_tested']}, "
          f"untestable={result['num_states_untestable']}, "
          f"violations={len(violations)}")
    for sid, p in violations:
        print(f"  State {sid}: p={p:.4e}")
    if result["markov_holds"]:
        print("  => Markov HOLDS")
    elif result["markov_holds"] is not None:
        print("  => Markov VIOLATED")


class TestMarkovPropertyRealData(unittest.TestCase):
    """Run the Markov property test on real pipeline clustering results."""

    def _run_test(self, seed_key, exclude_terminals=True, method="chi2", **kw):
        clustering_dir = _CLUSTERING_DIRS[seed_key]
        _require_clustering(clustering_dir)
        cluster_labels, metadata, manifest = _load_clustering(clustering_dir)
        level = manifest.get("level", "rollout")

        from policy_doctor.behaviors.behavior_graph import (
            BehaviorGraph,
            test_markov_property,
        )

        graph = BehaviorGraph.from_cluster_assignments(
            cluster_labels, metadata, level=level,
        )
        result = test_markov_property(
            cluster_labels, metadata, level=level,
            significance_level=SIGNIFICANCE_LEVEL,
            exclude_terminals=exclude_terminals,
            method=method, **kw,
        )
        return graph, result

    # ------------------------------------------------------------------
    # Chi-squared (original)
    # ------------------------------------------------------------------

    def test_seed0_chi2(self):
        _, result = self._run_test("seed0")
        self.assertIsNotNone(result["markov_holds"])
        self.assertGreater(result["num_states_tested"], 0)
        _print_result("seed0 chi2", result)

    def test_seed1_chi2(self):
        _, result = self._run_test("seed1")
        self.assertIsNotNone(result["markov_holds"])
        _print_result("seed1 chi2", result)

    def test_seed2_chi2(self):
        _, result = self._run_test("seed2")
        self.assertIsNotNone(result["markov_holds"])
        _print_result("seed2 chi2", result)

    def test_seed0_chi2_with_terminals(self):
        _, result = self._run_test("seed0", exclude_terminals=False)
        self.assertIsNotNone(result["markov_holds"])
        _print_result("seed0 chi2+terminals", result)

    # ------------------------------------------------------------------
    # Exact (permutation independence test)
    # ------------------------------------------------------------------

    def test_seed0_exact(self):
        """Permutation exact test on seed-0. Should test at least as many
        states as chi2 (lower data threshold)."""
        _, r_chi2 = self._run_test("seed0", method="chi2")
        _, r_exact = self._run_test(
            "seed0", method="exact",
            n_permutations=N_PERMUTATIONS, random_state=42,
        )
        self.assertGreaterEqual(
            r_exact["num_states_tested"], r_chi2["num_states_tested"],
        )
        _print_result("seed0 exact", r_exact)

    def test_seed1_exact(self):
        _, result = self._run_test(
            "seed1", method="exact",
            n_permutations=N_PERMUTATIONS, random_state=42,
        )
        self.assertIsNotNone(result["markov_holds"])
        _print_result("seed1 exact", result)

    def test_seed2_exact(self):
        _, result = self._run_test(
            "seed2", method="exact",
            n_permutations=N_PERMUTATIONS, random_state=42,
        )
        self.assertIsNotNone(result["markov_holds"])
        _print_result("seed2 exact", result)

    # ------------------------------------------------------------------
    # Modal (permutation test on most-likely successor)
    # ------------------------------------------------------------------

    def test_seed0_modal(self):
        _, result = self._run_test(
            "seed0", method="modal",
            n_permutations=N_PERMUTATIONS, random_state=42,
        )
        self.assertIsNotNone(result["markov_holds"])
        _print_result("seed0 modal", result)

    def test_seed1_modal(self):
        _, result = self._run_test(
            "seed1", method="modal",
            n_permutations=N_PERMUTATIONS, random_state=42,
        )
        self.assertIsNotNone(result["markov_holds"])
        _print_result("seed1 modal", result)

    def test_seed2_modal(self):
        _, result = self._run_test(
            "seed2", method="modal",
            n_permutations=N_PERMUTATIONS, random_state=42,
        )
        self.assertIsNotNone(result["markov_holds"])
        _print_result("seed2 modal", result)

    # ------------------------------------------------------------------
    # Structural checks
    # ------------------------------------------------------------------

    def test_per_state_results_populated(self):
        """Every cluster should have a per-state entry when terminals included."""
        clustering_dir = _CLUSTERING_DIRS["seed0"]
        _require_clustering(clustering_dir)
        cluster_labels, metadata, manifest = _load_clustering(clustering_dir)

        from policy_doctor.behaviors.behavior_graph import test_markov_property

        result = test_markov_property(
            cluster_labels, metadata,
            level=manifest.get("level", "rollout"),
            exclude_terminals=False,
        )
        cluster_ids = set(int(c) for c in cluster_labels if c != -1)
        for cid in cluster_ids:
            self.assertIn(cid, result["per_state"])

    def test_cross_seed_cross_method_summary(self):
        """Summary across all seeds and methods."""
        methods = ["chi2", "exact", "modal"]
        summary = {}
        for seed_key in ["seed0", "seed1", "seed2"]:
            _require_clustering(_CLUSTERING_DIRS[seed_key])
            for method in methods:
                kw = {}
                if method != "chi2":
                    kw = dict(n_permutations=N_PERMUTATIONS, random_state=42)
                _, result = self._run_test(seed_key, method=method, **kw)
                violations = {
                    sid for sid, r in result["per_state"].items()
                    if r.testable and not r.markov_holds
                }
                key = f"{seed_key}/{method}"
                summary[key] = {
                    "tested": result["num_states_tested"],
                    "violations": len(violations),
                    "violating_states": sorted(violations),
                }

        print("\n[cross-seed × cross-method summary]")
        for key, info in sorted(summary.items()):
            print(f"  {key:16s}  tested={info['tested']:2d}  "
                  f"violations={info['violations']}  "
                  f"states={info['violating_states'] or '-'}")
