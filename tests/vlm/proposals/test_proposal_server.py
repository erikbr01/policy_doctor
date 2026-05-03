"""Light Flask test-client tests for the E2 proposal server.

We boot the server in-process via ``boot(...)`` (mock backend) and exercise the
HTTP endpoints with ``app.test_client()`` — no real sockets.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from policy_doctor.vlm.proposals import server as ps_mod


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


def _write_fake_clustering(clustering_dir: Path):
    clustering_dir.mkdir(parents=True, exist_ok=True)
    # 5 rollouts, each 4 timesteps (matches our episodes), 2 cluster ids
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
    np.save(clustering_dir / "cluster_labels.npy", np.asarray(labels, dtype=np.int64))
    with open(clustering_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    with open(clustering_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump({"level": "rollout"}, f)


class TestProposalServer(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="test_propsrv_"))
        self.run_dir = self.tmp / "run"
        self.episodes_dir = self.tmp / "episodes"
        self.clustering_dir = self.tmp / "clustering"
        _write_fake_episodes(self.episodes_dir, [True, True, False, False, True], [4, 4, 4, 4, 4])
        _write_fake_clustering(self.clustering_dir)

        # Reset module globals before each boot, then build state.
        ps_mod._STATE = None
        ps_mod._BACKEND = None
        ps_mod._BACKEND_DEVICE = "cpu"
        ps_mod._CFG = {}

        ps_mod.boot(
            run_dir=self.run_dir,
            pool_episodes_dir=self.episodes_dir,
            clustering_dir=self.clustering_dir,
            chat_enabled=False,
            vlm_backend_name="mock",
            vlm_backend_params={},
            vlm_lifecycle="persistent",
        )
        self.app = ps_mod._make_app()
        self.client = self.app.test_client()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        ps_mod._STATE = None
        ps_mod._BACKEND = None

    def test_health_returns_ok_with_chat_disabled(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["chat_enabled"], False)
        self.assertEqual(body["pool_size"], 5)

    def test_chat_disabled_returns_405(self):
        resp = self.client.post("/chat", json={"text": "hi"})
        self.assertEqual(resp.status_code, 405)
        body = resp.get_json()
        self.assertIn("error", body)

    def test_pool_endpoint(self):
        resp = self.client.get("/pool")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["n_rollouts"], 5)
        self.assertEqual(len(body["rollouts"]), 5)

    def test_propose_grows_queue(self):
        n_before = len(ps_mod._STATE.queue)
        resp = self.client.post(
            "/propose",
            json={
                "condition": "graph",
                "n_requests_per_type": {
                    "full_trajectory": 1,
                    "recovery": 1,
                    "alternative_strategy": 1,
                },
                "n_repetitions": 1,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
        body = resp.get_json()
        self.assertEqual(body["condition"], "graph")
        self.assertGreater(body["n_added"], 0)
        n_after = len(ps_mod._STATE.queue)
        self.assertEqual(n_after, n_before + body["n_added"])

    def test_requests_active_returns_operator_view(self):
        # Seed the queue.
        self.client.post(
            "/propose",
            json={
                "condition": "graph",
                "n_requests_per_type": {
                    "full_trajectory": 1, "recovery": 1, "alternative_strategy": 1,
                },
                "n_repetitions": 1,
            },
        )

        resp = self.client.get("/requests/active")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertNotIn("target_cluster", body)
        self.assertNotIn("source_condition", body)
        # Operator-facing fields must be present.
        self.assertIn("request_id", body)
        self.assertIn("target_behavior", body)
        self.assertIn("initial_conditions", body)

    def test_post_result_marks_completed_even_if_pkl_missing(self):
        # Seed the queue and grab one pending id.
        self.client.post(
            "/propose",
            json={
                "condition": "graph",
                "n_requests_per_type": {
                    "full_trajectory": 1, "recovery": 1, "alternative_strategy": 1,
                },
                "n_repetitions": 1,
            },
        )
        self.assertGreater(len(ps_mod._STATE.queue), 0)
        rid = ps_mod._STATE.queue[0]

        # Use a non-existent demo pkl path; classifier is None so adherence is skipped
        # (server returns scored=False but request status -> 'completed').
        bogus_pkl = str(self.tmp / "no_such.pkl")
        resp = self.client.post(
            f"/requests/{rid}/result",
            json={"demo_pkl": bogus_pkl, "success": True},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["request_id"], rid)
        self.assertFalse(body["scored"])
        self.assertEqual(ps_mod._STATE.requests[rid].status, "completed")

    def test_post_result_unknown_id_returns_404(self):
        resp = self.client.post(
            "/requests/does-not-exist/result",
            json={"demo_pkl": "/tmp/x.pkl", "success": False},
        )
        self.assertEqual(resp.status_code, 404)


class TestProposalServerChatEnabled(unittest.TestCase):
    """When chat_enabled=True, /chat should NOT 405. Mock backend supports
    describe_slice, which is what the server falls back to when generate is
    absent."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="test_propsrv_chat_"))
        self.run_dir = self.tmp / "run"
        self.episodes_dir = self.tmp / "episodes"
        self.clustering_dir = self.tmp / "clustering"
        _write_fake_episodes(self.episodes_dir, [True, False], [4, 4])
        _write_fake_clustering_minimal(self.clustering_dir)

        ps_mod._STATE = None
        ps_mod._BACKEND = None
        ps_mod._BACKEND_DEVICE = "cpu"
        ps_mod._CFG = {}

        ps_mod.boot(
            run_dir=self.run_dir,
            pool_episodes_dir=self.episodes_dir,
            clustering_dir=self.clustering_dir,
            chat_enabled=True,
            vlm_backend_name="mock",
            vlm_backend_params={},
            vlm_lifecycle="persistent",
        )
        self.app = ps_mod._make_app()
        self.client = self.app.test_client()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        ps_mod._STATE = None
        ps_mod._BACKEND = None

    def test_health_chat_enabled_true(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["chat_enabled"], True)


def _write_fake_clustering_minimal(clustering_dir: Path):
    clustering_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    metadata = [
        {"rollout_idx": 0, "timestep": 0, "success": True},
        {"rollout_idx": 0, "timestep": 1, "success": True},
        {"rollout_idx": 1, "timestep": 0, "success": False},
        {"rollout_idx": 1, "timestep": 1, "success": False},
    ]
    np.save(clustering_dir / "cluster_labels.npy", labels)
    with open(clustering_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    with open(clustering_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump({"level": "rollout"}, f)


if __name__ == "__main__":
    unittest.main()
