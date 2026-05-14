"""Regression: GraphConditionInputBuilder must explicitly enumerate the
operator-forbidden terms in its system prompt.

The system message contains words like "cluster"/"node"/"graph" because it
explains the schema and the denylist to the VLM. What we DO NOT want is for
those words to leak into the operator-facing fields of the resulting
DemonstrationRequests — that's enforced at validation time. Here we just
check the system prompt does enumerate the forbidden terms so the VLM has
the prohibition in context.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from policy_doctor.behaviors.behavior_graph import BehaviorGraph
from policy_doctor.vlm.proposals.graph_representation.base import VLMArtefact
from policy_doctor.vlm.proposals.pool import RolloutPool
from policy_doctor.vlm.proposals.registry import get_vlm_input_builder
from policy_doctor.vlm.proposals.request import request_json_schema


def _write_fake_episodes(episodes_dir: Path):
    episodes_dir.mkdir(parents=True, exist_ok=True)
    successes = [True, False, True]
    lengths = [4, 4, 4]
    for i in range(len(successes)):
        df = pd.DataFrame({
            "sim_state": [np.zeros(4, dtype=np.float64) for _ in range(lengths[i])],
            "obs": [{} for _ in range(lengths[i])],
            "success": [successes[i]] * lengths[i],
        })
        df.to_pickle(str(episodes_dir / f"ep{i:04d}.pkl"))
    with open(episodes_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump({
            "episode_successes": successes, "episode_lengths": lengths,
        }, f)


def _make_graph() -> BehaviorGraph:
    labels = np.asarray([0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1], dtype=np.int64)
    metadata = []
    successes = [True, False, True]
    for ridx in range(3):
        for t in range(4):
            metadata.append({"rollout_idx": ridx, "timestep": t, "success": successes[ridx]})
    return BehaviorGraph.from_cluster_assignments(labels, metadata, level="rollout")


class TestForbiddenTermEnumeration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="test_leak_audit_"))
        eps = cls.tmp / "episodes"
        _write_fake_episodes(eps)
        cls.pool = RolloutPool.from_episodes_dir(eps)
        cls.graph = _make_graph()
        cls.artefact = VLMArtefact(images=[], text_blocks=["## Behavior nodes\n(stub)"])
        cls.builder = get_vlm_input_builder("graph_condition", {})
        cls.schema = request_json_schema(with_target_cluster=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _system_prompt_text(self) -> str:
        msgs = self.builder.build_messages(
            graph_artefact=self.artefact,
            pool=self.pool,
            condition="graph",
            n_requests_per_type={"full_trajectory": 1, "recovery": 1, "alternative_strategy": 1},
            json_schema=self.schema,
            history=None,
            task_hint="pick up the cube",
        )
        # First message is system.
        self.assertEqual(msgs[0].role, "system")
        return "\n".join(msgs[0].text_blocks)

    def test_forbidden_terms_are_listed(self):
        text = self._system_prompt_text()
        for term in ("cluster", "node", "graph"):
            self.assertIn(term, text.lower())

    def test_prompt_explains_operator_forbidden_context(self):
        """The forbidden terms should appear inside an instruction telling the
        VLM not to use them in operator-facing fields."""
        text = self._system_prompt_text().lower()
        # Look for an instruction-anchor phrase.
        self.assertIn("operator", text)
        self.assertIn("do not use", text)

    def test_full_set_of_forbidden_terms_present(self):
        text = self._system_prompt_text().lower()
        for term in (
            "cluster", "node", "graph", "umap", "kmeans", "centroid", "embedding",
        ):
            self.assertIn(term, text, f"forbidden term {term!r} not enumerated in system prompt")


class TestOutcomeOnlyHasNoGraphLeakage(unittest.TestCase):
    """Outcome-only condition's system prompt should still enumerate forbidden
    terms (operator-facing constraint applies in both conditions)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="test_leak_audit_oc_"))
        eps = cls.tmp / "episodes"
        _write_fake_episodes(eps)
        cls.pool = RolloutPool.from_episodes_dir(eps)
        cls.graph = _make_graph()
        cls.artefact = VLMArtefact(images=[], text_blocks=[])
        cls.builder = get_vlm_input_builder("outcome_condition", {})
        cls.schema = request_json_schema(with_target_cluster=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_outcome_prompt_enumerates_forbidden_terms(self):
        msgs = self.builder.build_messages(
            graph_artefact=self.artefact,
            pool=self.pool,
            condition="outcome_only",
            n_requests_per_type={"full_trajectory": 1, "recovery": 1, "alternative_strategy": 1},
            json_schema=self.schema,
            history=None,
            task_hint="pick up the cube",
        )
        sys_text = "\n".join(msgs[0].text_blocks).lower()
        for term in ("cluster", "node", "graph", "umap"):
            self.assertIn(term, sys_text)


if __name__ == "__main__":
    unittest.main()
