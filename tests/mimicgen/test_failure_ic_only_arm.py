"""Unit tests for MimicgenFailureICOnlyArmStep ablation.

Verifies that the IC-only arm:
  1. Has the correct name and cfg_overrides (subtask_constraint_idx=None)
  2. Is registered in the pipeline registry
  3. Does NOT emit chained-warp constraints when subtask_constraint_idx is None
     (simulated via the core condition in select_mimicgen_seed path_based branch)
"""

from __future__ import annotations

import unittest


class TestMimicgenFailureICOnlyArmClass(unittest.TestCase):
    """Basic attribute checks on the arm class."""

    def test_name(self):
        from policy_doctor.curation_pipeline.steps.mimicgen_arm import (
            MimicgenFailureICOnlyArmStep,
        )
        self.assertEqual(MimicgenFailureICOnlyArmStep.name, "mimicgen_failure_ic_only")

    def test_subtask_constraint_idx_is_none(self):
        from policy_doctor.curation_pipeline.steps.mimicgen_arm import (
            MimicgenFailureICOnlyArmStep,
        )
        overrides = MimicgenFailureICOnlyArmStep.cfg_overrides
        self.assertIn(
            "mimicgen_datagen.failure_analysis.subtask_constraint_idx", overrides
        )
        self.assertIsNone(
            overrides["mimicgen_datagen.failure_analysis.subtask_constraint_idx"]
        )

    def test_failure_analysis_enabled(self):
        from policy_doctor.curation_pipeline.steps.mimicgen_arm import (
            MimicgenFailureICOnlyArmStep,
        )
        self.assertTrue(
            MimicgenFailureICOnlyArmStep.cfg_overrides.get(
                "mimicgen_datagen.failure_analysis.enabled"
            )
        )

    def test_ic_constraint_enabled(self):
        from policy_doctor.curation_pipeline.steps.mimicgen_arm import (
            MimicgenFailureICOnlyArmStep,
        )
        self.assertTrue(
            MimicgenFailureICOnlyArmStep.cfg_overrides.get(
                "mimicgen_datagen.fix_initial_object_poses"
            )
        )

    def test_seed_heuristic_near_failure(self):
        from policy_doctor.curation_pipeline.steps.mimicgen_arm import (
            MimicgenFailureICOnlyArmStep,
        )
        self.assertEqual(
            MimicgenFailureICOnlyArmStep.cfg_overrides.get(
                "mimicgen_datagen.seed_selection_heuristic"
            ),
            "near_failure",
        )

    def test_differs_from_full_targeting_only_in_subtask_idx(self):
        from policy_doctor.curation_pipeline.steps.mimicgen_arm import (
            MimicgenFailureICOnlyArmStep,
            MimicgenFailureTargetingArmStep,
        )
        ft_overrides = MimicgenFailureTargetingArmStep.cfg_overrides
        ic_overrides = MimicgenFailureICOnlyArmStep.cfg_overrides

        # All keys shared between the two arms should be identical — only
        # "subtask_constraint_idx" distinguishes them.
        shared_keys = set(ft_overrides) & set(ic_overrides)
        for key in shared_keys:
            if key == "mimicgen_datagen.failure_analysis.subtask_constraint_idx":
                continue
            self.assertEqual(ft_overrides[key], ic_overrides[key], f"key={key}")

        # IC-only arm has the subtask_constraint_idx key; full targeting does not.
        self.assertIn(
            "mimicgen_datagen.failure_analysis.subtask_constraint_idx", ic_overrides
        )
        self.assertNotIn(
            "mimicgen_datagen.failure_analysis.subtask_constraint_idx", ft_overrides
        )


class TestPipelineRegistration(unittest.TestCase):
    """Checks that the step can be found in the pipeline registry."""

    def test_in_all_steps(self):
        from policy_doctor.curation_pipeline.pipeline import ALL_STEPS

        self.assertIn("mimicgen_failure_ic_only", ALL_STEPS)

    def test_in_registry(self):
        from policy_doctor.curation_pipeline.pipeline import _build_step_registry
        from policy_doctor.curation_pipeline.steps.mimicgen_arm import (
            MimicgenFailureICOnlyArmStep,
        )

        registry = _build_step_registry()
        self.assertIn("mimicgen_failure_ic_only", registry)
        self.assertIs(registry["mimicgen_failure_ic_only"], MimicgenFailureICOnlyArmStep)

    def test_ordered_after_failure_targeting(self):
        from policy_doctor.curation_pipeline.pipeline import ALL_STEPS

        ft_idx = ALL_STEPS.index("mimicgen_failure_targeting")
        ic_idx = ALL_STEPS.index("mimicgen_failure_ic_only")
        self.assertGreater(ic_idx, ft_idx)


class TestNoChainedWarpWhenSubtaskIdxNone(unittest.TestCase):
    """Simulates the path_based branch logic from select_mimicgen_seed.py.

    The core invariant: when cw_subtask_idx is None the condition
    ``resolved_subtask_idx is not None`` fails and chained_warp_for_path
    stays None, so per_seed_chained_warp_constraints is all-None and the
    key is never written to the result dict.
    """

    def _run_path_based_loop(self, subtask_constraint_idx, paths):
        """Minimal reproduction of the chained-warp-building loop in
        SelectMimicgenSeedStep._run_path_based (path_based branch).

        Returns the per_seed_chained_warp_constraints list.
        """
        from policy_doctor.mimicgen.chained_warp_generator import (
            cluster_to_chained_warp_constraint,
        )
        from policy_doctor.mimicgen.failure_targeting import DEFAULT_SQUARE_STATE_SCHEMA

        cw_subtask_idx: int | None = subtask_constraint_idx
        cw_subtask_idx_by_node: dict[int, int] = {}
        cw_slack_alpha = 1.5
        cw_slack_widen_factor = 2.0
        state_schema = DEFAULT_SQUARE_STATE_SCHEMA

        per_seed_chained_warp_constraints: list[dict | None] = []

        for path_entry in paths:
            intermediate_pool = path_entry.get("intermediate_pool")
            intermediate_node_id = path_entry.get("intermediate_node_id")

            resolved_subtask_idx = cw_subtask_idx
            if intermediate_node_id is not None and int(intermediate_node_id) in cw_subtask_idx_by_node:
                resolved_subtask_idx = cw_subtask_idx_by_node[int(intermediate_node_id)]

            chained_warp_for_path: dict | None = None
            if (
                resolved_subtask_idx is not None
                and intermediate_pool
                and intermediate_pool.get("clusters")
            ):
                mid_clusters = intermediate_pool["clusters"]
                dominant = max(mid_clusters, key=lambda c: int(c.get("n_states", 0)))
                chained_warp_for_path = cluster_to_chained_warp_constraint(
                    center_feature=dominant["center_feature"],
                    stddev_feature=dominant["stddev_feature"],
                    state_schema=state_schema,
                    subtask_idx=resolved_subtask_idx,
                    slack_alpha=cw_slack_alpha,
                    slack_widen_factor=cw_slack_widen_factor,
                )
            per_seed_chained_warp_constraints.append(chained_warp_for_path)

        return per_seed_chained_warp_constraints

    def _fake_paths(self, n: int = 3):
        """Return n fake path entries with synthetic 7-dim cluster features."""
        import numpy as np

        paths = []
        for i in range(n):
            center = np.zeros(7, dtype=np.float32)
            center[3] = 1.0  # qw=1 (identity quaternion)
            stddev = np.full(7, 0.01, dtype=np.float32)
            paths.append({
                "path_idx": i,
                "intermediate_node_id": 10 + i,
                "intermediate_pool": {
                    "clusters": [
                        {
                            "n_states": 5,
                            "center_feature": center.tolist(),
                            "stddev_feature": stddev.tolist(),
                        }
                    ]
                },
                "ic_pool": {
                    "clusters": [
                        {
                            "n_states": 3,
                            "center_feature": center.tolist(),
                            "stddev_feature": stddev.tolist(),
                        }
                    ]
                },
            })
        return paths

    def test_no_chained_warp_when_subtask_idx_is_none(self):
        paths = self._fake_paths(n=3)
        constraints = self._run_path_based_loop(
            subtask_constraint_idx=None, paths=paths
        )
        self.assertEqual(len(constraints), 3)
        self.assertTrue(
            all(c is None for c in constraints),
            f"Expected all None, got: {constraints}",
        )

    def test_chained_warp_emitted_when_subtask_idx_is_set(self):
        """Positive control: subtask_idx=0 produces non-None constraints."""
        paths = self._fake_paths(n=2)
        constraints = self._run_path_based_loop(
            subtask_constraint_idx=0, paths=paths
        )
        self.assertEqual(len(constraints), 2)
        self.assertTrue(
            all(c is not None for c in constraints),
            f"Expected all non-None, got: {constraints}",
        )
        # Each constraint should have the expected top-level keys.
        for c in constraints:
            self.assertIn("subtask_idx", c)
            self.assertEqual(c["subtask_idx"], 0)
            self.assertIn("target_pose", c)
            self.assertIn("slack", c)

    def test_result_key_absent_when_all_none(self):
        """If all constraints are None, per_seed_chained_warp_constraints is NOT
        written to the result dict (mirrors SelectMimicgenSeedStep logic)."""
        per_seed = [None, None, None]
        result: dict = {}
        if any(v is not None for v in per_seed):
            result["per_seed_chained_warp_constraints"] = per_seed
        self.assertNotIn("per_seed_chained_warp_constraints", result)


if __name__ == "__main__":
    unittest.main()
