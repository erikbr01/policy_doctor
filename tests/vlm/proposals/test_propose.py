"""End-to-end Tier-0 plumbing tests for generate_proposals (mock backend)."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from policy_doctor.behaviors.behavior_graph import BehaviorGraph
from policy_doctor.vlm import get_vlm_backend
from policy_doctor.vlm.proposals.pool import RolloutPool
from policy_doctor.vlm.proposals.propose import generate_proposals


def _write_fake_episodes(episodes_dir: Path, successes, lengths):
    episodes_dir.mkdir(parents=True, exist_ok=True)
    n = len(successes)
    for i in range(n):
        T = lengths[i]
        df = pd.DataFrame({
            "sim_state": [np.zeros(4, dtype=np.float64) + i + t * 0.01 for t in range(T)],
            "obs": [{"object": np.zeros(3, dtype=np.float32)} for _ in range(T)],
            "success": [bool(successes[i])] * T,
        })
        df.to_pickle(str(episodes_dir / f"ep{i:04d}.pkl"))
    with open(episodes_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump({
            "episode_successes": [bool(s) for s in successes],
            "episode_lengths": [int(t) for t in lengths],
        }, f)


def _make_synthetic_graph() -> BehaviorGraph:
    """Tiny per-rollout graph with cluster ids 0,1 and outcomes."""
    # 5 rollouts, each with 4 timesteps. Mixed clusters and successes.
    cluster_seqs = [
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 1, 1],
    ]
    successes = [True, True, False, False, True]
    labels = []
    metadata = []
    for ridx, seq in enumerate(cluster_seqs):
        for t, lab in enumerate(seq):
            labels.append(int(lab))
            metadata.append({
                "rollout_idx": ridx,
                "timestep": t,
                "success": successes[ridx],
            })
    return BehaviorGraph.from_cluster_assignments(
        np.asarray(labels, dtype=np.int64),
        metadata,
        level="rollout",
    )


def _build_pool(tmp: Path) -> RolloutPool:
    eps = tmp / "episodes"
    successes = [True, True, False, False, True]
    lengths = [4, 4, 4, 4, 4]
    _write_fake_episodes(eps, successes, lengths)
    return RolloutPool.from_episodes_dir(eps)


class TestGenerateProposalsGraphCondition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="test_propose_graph_"))
        cls.pool = _build_pool(cls.tmp)
        cls.graph = _make_synthetic_graph()
        cls.backend = get_vlm_backend("mock")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_graph_condition_end_to_end(self):
        out_dir = self.tmp / "graph_run"
        result = generate_proposals(
            backend=self.backend,
            pool=self.pool,
            behavior_graph=self.graph,
            condition="graph",
            n_requests_per_type={"full_trajectory": 1, "recovery": 1, "alternative_strategy": 1},
            output_dir=out_dir,
            n_repetitions=2,
        )
        # Selected requests non-empty and tagged with condition + target_cluster.
        self.assertGreater(len(result.selected_requests), 0)
        for req in result.selected_requests:
            self.assertEqual(req.source_condition, "graph")
            self.assertIsNotNone(req.target_cluster)

        # Consistency metrics include the aggregation key.
        self.assertIn("aggregation", result.consistency_metrics)

        # Files written.
        self.assertTrue((out_dir / "run_1.json").exists())
        self.assertTrue((out_dir / "run_2.json").exists())
        self.assertTrue((out_dir / "selected_run.json").exists())
        self.assertTrue((out_dir / "consistency_metrics.json").exists())
        self.assertTrue((out_dir / "graph_artefact").is_dir())

        # selected_run.json content reflects ProposalBatchResult
        with open(out_dir / "selected_run.json") as f:
            sel = json.load(f)
        self.assertEqual(sel["condition"], "graph")
        self.assertEqual(sel["n_requests"], len(result.selected_requests))


class TestGenerateProposalsOutcomeOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="test_propose_outcome_"))
        cls.pool = _build_pool(cls.tmp)
        cls.graph = _make_synthetic_graph()
        cls.backend = get_vlm_backend("mock")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_outcome_condition_end_to_end(self):
        out_dir = self.tmp / "outcome_run"
        result = generate_proposals(
            backend=self.backend,
            pool=self.pool,
            behavior_graph=self.graph,
            condition="outcome_only",
            n_requests_per_type={"full_trajectory": 1, "recovery": 1, "alternative_strategy": 1},
            output_dir=out_dir,
            n_repetitions=2,
        )
        self.assertGreater(len(result.selected_requests), 0)
        for req in result.selected_requests:
            self.assertEqual(req.source_condition, "outcome_only")
            self.assertIsNone(req.target_cluster)


class TestGenerateProposalsBadCondition(unittest.TestCase):
    def test_unknown_condition_raises(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pool = _build_pool(tmp)
            graph = _make_synthetic_graph()
            backend = get_vlm_backend("mock")
            with self.assertRaises(ValueError):
                generate_proposals(
                    backend=backend,
                    pool=pool,
                    behavior_graph=graph,
                    condition="bogus",
                    n_requests_per_type={"full_trajectory": 1},
                    output_dir=tmp / "out",
                    n_repetitions=1,
                )


if __name__ == "__main__":
    unittest.main()
