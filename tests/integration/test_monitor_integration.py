"""Integration tests for the runtime behavior monitor using real jan28 artifacts.

Requires the cupid_torch2 conda env (diffusion_policy + infembed installed).
Run with:
    conda activate cupid_torch2
    python run_tests.py --suite cupid

Skipped automatically when artifacts are not present.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------

_BASE = Path("/mnt/ssdB/erik/cupid_data")
_TRAK_DIR = (
    _BASE
    / "outputs/eval_save_episodes/jan28"
    / "jan28_train_diffusion_unet_lowdim_transport_mh_0/latest"
    / "default_trak_results-proj_dim=4096-lambda_reg=0.0-num_ckpts=1-seed=0-loss_fn=square-num_timesteps=64"
)
_CHECKPOINT = (
    _BASE
    / "outputs/train/jan28"
    / "jan28_train_diffusion_unet_lowdim_transport_mh_0/checkpoints/latest.ckpt"
)
_INFEMBED_FIT = _TRAK_DIR / "infembed_fit.pt"
_INFEMBED_NPZ = _TRAK_DIR / "infembed_embeddings.npz"
_EPISODE_PKL = (
    _BASE
    / "outputs/eval_save_episodes/jan28"
    / "jan28_train_diffusion_unet_lowdim_transport_mh_0/latest/episodes/ep0000_succ.pkl"
)
_EPISODES_DIR = (
    _BASE
    / "outputs/eval_save_episodes/jan28"
    / "jan28_train_diffusion_unet_lowdim_transport_mh_0/latest/episodes"
)
_CLUSTERING_DIR = Path(
    "third_party/influence_visualizer/configs"
    "/transport_mh_jan28/clustering"
    "/trak_filtering_mar13_aggfix_seed0_kmeans_k20"
)

_ARTIFACTS_PRESENT = (
    _CHECKPOINT.exists()
    and _INFEMBED_FIT.exists()
    and _INFEMBED_NPZ.exists()
    and _EPISODE_PKL.exists()
    and (
        _CLUSTERING_DIR.exists()
        or (Path(__file__).parent.parent.parent / _CLUSTERING_DIR).exists()
    )
)

_SKIP_REASON = "jan28 transport_mh_0 artifacts not found — run in environment with /mnt/ssdB mounted"


def _compute_window_embeddings(rollout_embeddings: np.ndarray, metadata: list, episode_lengths: list) -> np.ndarray:
    """Compute window-level mean embeddings from per-timestep rollout embeddings.

    The clustering was done at "rollout" level using sliding-window means.
    ``metadata[i]`` has ``rollout_idx`` (episode index), ``window_start``,
    ``window_end`` (timestep range within that episode).
    """
    global_starts = np.cumsum([0] + list(episode_lengths[:-1]))
    window_embs = []
    for entry in metadata:
        ep = entry["rollout_idx"]
        gs = int(global_starts[ep]) + entry["window_start"]
        ge = int(global_starts[ep]) + entry["window_end"]
        window_embs.append(rollout_embeddings[gs:ge].mean(axis=0))
    return np.array(window_embs, dtype=np.float32)


def _resolve_clustering_dir() -> Path:
    if _CLUSTERING_DIR.is_absolute() and _CLUSTERING_DIR.exists():
        return _CLUSTERING_DIR
    repo_root = Path(__file__).parent.parent.parent
    resolved = repo_root / _CLUSTERING_DIR
    return resolved


# ---------------------------------------------------------------------------
# Shared scorer fixture (loaded once per class via setUpClass)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_ARTIFACTS_PRESENT, _SKIP_REASON)
class TestInfEmbedStreamScorer(unittest.TestCase):
    """InfEmbedStreamScorer loads artifacts and produces correct-shaped embeddings."""

    @classmethod
    def setUpClass(cls):
        from policy_doctor.monitoring.infembed_scorer import InfEmbedStreamScorer
        cls.scorer = InfEmbedStreamScorer(
            checkpoint=str(_CHECKPOINT),
            infembed_fit_path=str(_INFEMBED_FIT),
            infembed_embeddings_path=str(_INFEMBED_NPZ),
            device="cpu",
        )

    def _make_batch(self):
        """Synthetic (B=1, To=2, Do=59) obs and (B=1, Ta=16, Da=20) action batch."""
        import torch
        obs = torch.zeros(1, 2, 59)
        action = torch.zeros(1, 16, 20)
        timesteps = torch.zeros(1, 8, dtype=torch.long)
        return {"obs": obs, "action": action, "timesteps": timesteps}

    def test_embed_returns_correct_shape(self):
        batch = self._make_batch()
        embedding = self.scorer.embed(batch)
        self.assertEqual(embedding.ndim, 1)
        self.assertEqual(embedding.shape[0], 100)

    def test_embed_returns_float32(self):
        embedding = self.scorer.embed(self._make_batch())
        self.assertEqual(embedding.dtype, np.float32)

    def test_score_returns_correct_shape(self):
        batch = self._make_batch()
        scores = self.scorer.score(batch)
        self.assertEqual(scores.ndim, 1)
        # N_demo = 185942 for this artifact
        self.assertGreater(scores.shape[0], 0)

    def test_rollout_embeddings_shape(self):
        emb = self.scorer.rollout_embeddings
        self.assertEqual(emb.ndim, 2)
        self.assertEqual(emb.shape[1], 100)
        self.assertGreater(emb.shape[0], 0)

    def test_embed_is_deterministic_with_fixed_seed(self):
        import torch
        batch = self._make_batch()
        torch.manual_seed(0)
        e1 = self.scorer.embed(batch)
        torch.manual_seed(0)
        e2 = self.scorer.embed(batch)
        np.testing.assert_array_almost_equal(e1, e2, decimal=5)


@unittest.skipUnless(_ARTIFACTS_PRESENT, _SKIP_REASON)
class TestNearestCentroidAssigner(unittest.TestCase):
    """NearestCentroidAssigner assigns new embeddings to valid cluster IDs."""

    @classmethod
    def setUpClass(cls):
        import yaml
        from policy_doctor.behaviors.behavior_graph import BehaviorGraph
        from policy_doctor.data.clustering_loader import load_clustering_result_from_path
        from policy_doctor.monitoring.graph_assigner import NearestCentroidAssigner
        from policy_doctor.monitoring.infembed_scorer import InfEmbedStreamScorer

        # Load infembed embeddings
        data = np.load(_INFEMBED_NPZ, allow_pickle=False)
        rollout_embeddings = data["rollout_embeddings"].astype(np.float32)

        clustering_dir = _resolve_clustering_dir()
        labels, metadata, manifest = load_clustering_result_from_path(clustering_dir)
        graph = BehaviorGraph.from_cluster_assignments(
            labels, metadata, level=manifest.get("level", "rollout")
        )
        cls.n_clusters = int(labels.max()) + 1

        # The clustering was done at "rollout" level using window-mean embeddings.
        # Aggregate timestep-level rollout_embeddings into window embeddings.
        metadata_yaml = _EPISODES_DIR / "metadata.yaml"
        with open(metadata_yaml) as f:
            ep_meta = yaml.safe_load(f)
        episode_lengths = ep_meta["episode_lengths"]
        window_embeddings = _compute_window_embeddings(rollout_embeddings, metadata, episode_lengths)

        cls.assigner = NearestCentroidAssigner(
            rollout_embeddings=window_embeddings,
            cluster_labels=labels,
            graph=graph,
        )

    def test_assign_returns_valid_cluster_id(self):
        from policy_doctor.monitoring.base import AssignmentResult
        embedding = np.random.randn(100).astype(np.float32)
        result = self.assigner.assign(embedding)
        self.assertIsInstance(result, AssignmentResult)
        self.assertIn(result.cluster_id, range(self.n_clusters))

    def test_assign_distance_is_positive(self):
        embedding = np.random.randn(100).astype(np.float32)
        result = self.assigner.assign(embedding)
        self.assertGreater(result.distance, 0.0)

    def test_assign_node_name_is_nonempty_string(self):
        embedding = np.random.randn(100).astype(np.float32)
        result = self.assigner.assign(embedding)
        self.assertIsInstance(result.node_name, str)
        self.assertGreater(len(result.node_name), 0)

    def test_consistent_assignment_for_repeated_input(self):
        embedding = np.random.randn(100).astype(np.float32)
        r1 = self.assigner.assign(embedding)
        r2 = self.assigner.assign(embedding)
        self.assertEqual(r1.cluster_id, r2.cluster_id)


@unittest.skipUnless(_ARTIFACTS_PRESENT, _SKIP_REASON)
class TestStreamMonitorEndToEnd(unittest.TestCase):
    """StreamMonitor produces valid MonitorResult with timing for a real scorer."""

    @classmethod
    def setUpClass(cls):
        import yaml
        from policy_doctor.behaviors.behavior_graph import BehaviorGraph
        from policy_doctor.data.clustering_loader import load_clustering_result_from_path
        from policy_doctor.monitoring.graph_assigner import NearestCentroidAssigner
        from policy_doctor.monitoring.infembed_scorer import InfEmbedStreamScorer
        from policy_doctor.monitoring.stream_monitor import StreamMonitor

        scorer = InfEmbedStreamScorer(
            checkpoint=str(_CHECKPOINT),
            infembed_fit_path=str(_INFEMBED_FIT),
            infembed_embeddings_path=str(_INFEMBED_NPZ),
            device="cpu",
        )
        clustering_dir = _resolve_clustering_dir()
        labels, metadata, manifest = load_clustering_result_from_path(clustering_dir)
        graph = BehaviorGraph.from_cluster_assignments(
            labels, metadata, level=manifest.get("level", "rollout")
        )
        cls.n_clusters = int(labels.max()) + 1

        # Aggregate timestep-level embeddings into window embeddings for centroid computation
        with open(_EPISODES_DIR / "metadata.yaml") as f:
            ep_meta = yaml.safe_load(f)
        window_embeddings = _compute_window_embeddings(
            scorer.rollout_embeddings, metadata, ep_meta["episode_lengths"]
        )
        assigner = NearestCentroidAssigner(
            rollout_embeddings=window_embeddings,
            cluster_labels=labels,
            graph=graph,
        )
        cls.monitor = StreamMonitor(scorer=scorer, assigner=assigner)

    def test_process_sample_returns_monitor_result(self):
        from policy_doctor.monitoring.base import MonitorResult
        obs = np.zeros((2, 59), dtype=np.float32)
        action = np.zeros((16, 20), dtype=np.float32)
        result = self.monitor.process_sample(obs, action)
        self.assertIsInstance(result, MonitorResult)

    def test_embedding_shape(self):
        obs = np.zeros((2, 59), dtype=np.float32)
        action = np.zeros((16, 20), dtype=np.float32)
        result = self.monitor.process_sample(obs, action)
        self.assertEqual(result.embedding.shape, (100,))

    def test_influence_scores_shape(self):
        obs = np.zeros((2, 59), dtype=np.float32)
        action = np.zeros((16, 20), dtype=np.float32)
        result = self.monitor.process_sample(obs, action)
        self.assertGreater(result.influence_scores.shape[0], 0)

    def test_assignment_is_valid(self):
        obs = np.zeros((2, 59), dtype=np.float32)
        action = np.zeros((16, 20), dtype=np.float32)
        result = self.monitor.process_sample(obs, action)
        self.assertIsNotNone(result.assignment)
        self.assertIn(result.assignment.cluster_id, range(self.n_clusters))

    def test_timing_keys_present_and_positive(self):
        obs = np.zeros((2, 59), dtype=np.float32)
        action = np.zeros((16, 20), dtype=np.float32)
        result = self.monitor.process_sample(obs, action)
        for key in ("gradient_project_ms", "score_ms", "assign_ms", "total_ms"):
            self.assertIn(key, result.timing_ms)
            self.assertGreater(result.timing_ms[key], 0.0)

    def test_embed_only_skips_influence_scores(self):
        from policy_doctor.monitoring.base import MonitorResult
        obs = np.zeros((2, 59), dtype=np.float32)
        action = np.zeros((16, 20), dtype=np.float32)
        result = self.monitor.process_sample_embed_only(obs, action)
        self.assertIsInstance(result, MonitorResult)
        self.assertIsNone(result.influence_scores)


@unittest.skipUnless(_ARTIFACTS_PRESENT, _SKIP_REASON)
class TestTrajectoryClassifierFromCheckpoint(unittest.TestCase):
    """TrajectoryClassifier.from_checkpoint() reads config and classifies a real pkl.

    Uses NearestCentroidAssigner (no clustering_models.pkl for jan28 artifacts).
    Window embeddings are computed from the timestep-level rollout embeddings + metadata.
    """

    @classmethod
    def setUpClass(cls):
        import yaml
        from policy_doctor.behaviors.behavior_graph import BehaviorGraph
        from policy_doctor.data.clustering_loader import load_clustering_result_from_path
        from policy_doctor.monitoring.graph_assigner import NearestCentroidAssigner
        from policy_doctor.monitoring.infembed_scorer import InfEmbedStreamScorer
        from policy_doctor.monitoring.stream_monitor import StreamMonitor
        from policy_doctor.monitoring.trajectory_classifier import TrajectoryClassifier

        clustering_dir = _resolve_clustering_dir()
        labels, metadata, manifest = load_clustering_result_from_path(clustering_dir)
        graph = BehaviorGraph.from_cluster_assignments(
            labels, metadata, level=manifest.get("level", "rollout")
        )

        scorer = InfEmbedStreamScorer(
            checkpoint=str(_CHECKPOINT),
            infembed_fit_path=str(_INFEMBED_FIT),
            infembed_embeddings_path=str(_INFEMBED_NPZ),
            device="cpu",
        )
        with open(_EPISODES_DIR / "metadata.yaml") as f:
            ep_meta = yaml.safe_load(f)
        window_embeddings = _compute_window_embeddings(
            scorer.rollout_embeddings, metadata, ep_meta["episode_lengths"]
        )
        assigner = NearestCentroidAssigner(
            rollout_embeddings=window_embeddings,
            cluster_labels=labels,
            graph=graph,
        )
        monitor = StreamMonitor(scorer=scorer, assigner=assigner)

        import dill, torch
        payload = torch.load(str(_CHECKPOINT), map_location="cpu", pickle_module=dill)
        cfg = payload["cfg"]
        from omegaconf import OmegaConf
        dataset_cfg = cfg.task.dataset
        runner_cfg = cfg.task.env_runner
        n_obs_steps = int(OmegaConf.select(runner_cfg, "n_obs_steps") or 2)
        n_action_steps = int(OmegaConf.select(dataset_cfg, "horizon") or 16)

        cls.classifier = TrajectoryClassifier(
            monitor=monitor,
            mode="rollout",
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
        )

    def test_config_values(self):
        self.assertEqual(self.classifier.n_obs_steps, 2)
        self.assertEqual(self.classifier.n_action_steps, 16)

    def test_classify_single_sample(self):
        obs = np.zeros((2, 59), dtype=np.float32)
        action = np.zeros((16, 20), dtype=np.float32)
        result = self.classifier.classify_sample(obs, action)
        self.assertIsNotNone(result.assignment)
        self.assertIsInstance(result.assignment.node_name, str)

    def test_classify_episode_from_pkl(self):
        import pickle
        with open(_EPISODE_PKL, "rb") as f:
            episode_df = pickle.load(f)

        results = self.classifier.classify_episode_from_pkl(episode_df)
        # ep0000_succ.pkl has 86 timesteps
        self.assertEqual(len(results), len(episode_df))
        for t, r in results:
            self.assertIsNotNone(r.assignment)
            self.assertIsInstance(r.assignment.cluster_id, int)
            self.assertGreater(r.timing_ms["total_ms"], 0.0)

    def test_pkl_node_names_are_strings(self):
        import pickle
        with open(_EPISODE_PKL, "rb") as f:
            episode_df = pickle.load(f)
        results = self.classifier.classify_episode_from_pkl(episode_df)
        names = [r.assignment.node_name for _, r in results]
        self.assertTrue(all(isinstance(n, str) and len(n) > 0 for n in names))

    def test_classify_sequence_returns_correct_count(self):
        T = 10
        obs_seq = np.zeros((T, 59), dtype=np.float32)
        action_seq = np.zeros((T, 20), dtype=np.float32)
        results = self.classifier.classify_sequence(obs_seq, action_seq)
        # starts at t = n_obs_steps - 1 = 1
        self.assertEqual(len(results), T - (self.classifier.n_obs_steps - 1))


if __name__ == "__main__":
    unittest.main()
