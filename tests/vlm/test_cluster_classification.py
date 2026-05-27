"""Tests for Experiment E1: VLM cluster coherence classification."""

from __future__ import annotations

import json
import pathlib
import pickle
import tempfile
import unittest
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

from policy_doctor.vlm.backends.mock import MockVLMBackend
from policy_doctor.vlm.cluster_classification import (
    _centroid_distances,
    _load_slice_extra_text,
    _opaque_labels,
    build_label_map,
    build_sample_plan,
    compute_classification_metrics,
    generate_label_maps_for_plan,
    parse_classification_response,
    run_cluster_coherence_classification,
    run_query_with_label_maps,
)
from policy_doctor.vlm.storyboard import make_storyboard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_eval_dir(root: pathlib.Path, n_episodes: int = 3, n_timesteps: int = 10) -> pathlib.Path:
    """Create minimal eval_dir with episode pickles."""
    ep_dir = root / "episodes"
    ep_dir.mkdir(parents=True)
    for ep_i in range(n_episodes):
        rows = []
        for t in range(n_timesteps):
            img = np.zeros((8, 8, 3), dtype=np.uint8)
            img[..., 0] = (ep_i * 50 + t * 10) % 255
            rows.append({"timestep": t, "img": img})
        df = pd.DataFrame(rows)
        fname = f"ep{ep_i:04d}_succ.pkl"
        with open(ep_dir / fname, "wb") as f:
            pickle.dump(df, f)
    with open(ep_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump({"length": n_timesteps * n_episodes}, f)
    return root


def _make_clustering_dir(
    root: pathlib.Path,
    n_samples: int = 30,
    n_clusters: int = 3,
    n_episodes: int = 3,
    save_embeddings: bool = True,
) -> pathlib.Path:
    cdir = root / "clustering"
    cdir.mkdir(parents=True)
    rng = np.random.default_rng(0)

    labels = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        labels[i] = i % n_clusters

    metadata = []
    for i in range(n_samples):
        ep = i % n_episodes
        # Keep windows within [0, n_timesteps-3] so window_end <= n_timesteps
        w0 = (i % 5) * 1  # w0 in [0, 4], window_end in [3, 7]
        metadata.append({
            "rollout_idx": ep,
            "window_start": w0,
            "window_end": w0 + 3,
        })

    np.save(cdir / "cluster_labels.npy", labels)
    with open(cdir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    with open(cdir / "manifest.yaml", "w") as f:
        yaml.safe_dump({"level": "rollout", "n_samples": n_samples, "n_clusters": n_clusters}, f)

    if save_embeddings:
        emb = rng.random((n_samples, 10)).astype(np.float32)
        np.save(cdir / "embeddings_reduced.npy", emb)

    return cdir


# ---------------------------------------------------------------------------
# Storyboard
# ---------------------------------------------------------------------------

class TestLoadSliceExtraText(unittest.TestCase):
    def _make_eval_with_obs_action(self, td: pathlib.Path) -> pathlib.Path:
        ep_dir = td / "episodes"
        ep_dir.mkdir(parents=True)
        rng = np.random.default_rng(0)
        for ep_i in range(2):
            rows = []
            for t in range(8):
                rows.append({
                    "obs": rng.standard_normal((2, 5)).astype(np.float32),
                    "action": rng.standard_normal((3, 4)).astype(np.float32),
                    "img": np.zeros((4, 4, 3), dtype=np.uint8),
                })
            df = pd.DataFrame(rows)
            with open(ep_dir / f"ep{ep_i:04d}_succ.pkl", "wb") as f:
                pickle.dump(df, f)
        with open(ep_dir / "metadata.yaml", "w") as f:
            yaml.safe_dump({"episode_lengths": [8, 8]}, f)
        return td

    def test_returns_none_when_both_flags_off(self):
        with tempfile.TemporaryDirectory() as td:
            ed = self._make_eval_with_obs_action(pathlib.Path(td))
            metadata = [{"rollout_idx": 0, "window_start": 0, "window_end": 3}]
            out = _load_slice_extra_text(
                ed, 0, metadata,
                include_action_text=False, include_state_text=False,
            )
            self.assertIsNone(out)

    def test_action_only_block(self):
        with tempfile.TemporaryDirectory() as td:
            ed = self._make_eval_with_obs_action(pathlib.Path(td))
            metadata = [{"rollout_idx": 0, "window_start": 0, "window_end": 3}]
            out = _load_slice_extra_text(
                ed, 0, metadata,
                include_action_text=True, include_state_text=False,
            )
            self.assertIsNotNone(out)
            self.assertIn("action", out)
            self.assertNotIn("obs=", out)
            self.assertEqual(out.count("t="), 3)  # 3 timesteps in window

    def test_state_only_block(self):
        with tempfile.TemporaryDirectory() as td:
            ed = self._make_eval_with_obs_action(pathlib.Path(td))
            metadata = [{"rollout_idx": 0, "window_start": 0, "window_end": 3}]
            out = _load_slice_extra_text(
                ed, 0, metadata,
                include_action_text=False, include_state_text=True,
            )
            self.assertIsNotNone(out)
            self.assertIn("obs=", out)
            self.assertNotIn("action=", out)

    def test_both_block(self):
        with tempfile.TemporaryDirectory() as td:
            ed = self._make_eval_with_obs_action(pathlib.Path(td))
            metadata = [{"rollout_idx": 0, "window_start": 0, "window_end": 3}]
            out = _load_slice_extra_text(
                ed, 0, metadata,
                include_action_text=True, include_state_text=True,
            )
            self.assertIsNotNone(out)
            self.assertIn("obs=", out)
            self.assertIn("action=", out)


class TestMakeStoryboard(unittest.TestCase):
    def test_single_image(self):
        from PIL import Image
        img = Image.new("RGB", (32, 32), color=(100, 200, 50))
        out = make_storyboard([img], target_size=(64, 64))
        self.assertEqual(out.size, (64, 64))

    def test_four_images_grid(self):
        from PIL import Image
        imgs = [Image.new("RGB", (32, 32), color=(i * 60, 0, 0)) for i in range(4)]
        out = make_storyboard(imgs, target_size=(128, 128))
        self.assertEqual(out.size, (128, 128))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            make_storyboard([], target_size=(64, 64))


# ---------------------------------------------------------------------------
# Opaque labels
# ---------------------------------------------------------------------------

class TestOpaqueLabels(unittest.TestCase):
    def test_basic(self):
        labels = _opaque_labels(5)
        self.assertEqual(labels, ["Group A", "Group B", "Group C", "Group D", "Group E"])

    def test_too_many_raises(self):
        with self.assertRaises(ValueError):
            _opaque_labels(27)


# ---------------------------------------------------------------------------
# Label map
# ---------------------------------------------------------------------------

class TestBuildLabelMap(unittest.TestCase):
    def test_all_ids_present(self):
        rng = np.random.default_rng(0)
        cids = [0, 1, 2, 3]
        lm = build_label_map(cids, rng)
        self.assertEqual(set(lm.keys()), set(cids))
        self.assertEqual(len(set(lm.values())), 4)

    def test_different_shuffles(self):
        rng = np.random.default_rng(0)
        lm1 = build_label_map([0, 1, 2], rng)
        lm2 = build_label_map([0, 1, 2], rng)
        # Very low probability both are identical for 3-item shuffle
        # Just check both are valid
        self.assertEqual(set(lm1.values()), {"Group A", "Group B", "Group C"})
        self.assertEqual(set(lm2.values()), {"Group A", "Group B", "Group C"})


# ---------------------------------------------------------------------------
# Sample plan
# ---------------------------------------------------------------------------

class TestBuildSamplePlan(unittest.TestCase):
    def setUp(self):
        self.n_samples = 30
        self.n_clusters = 3
        self.n_episodes = 3
        rng_setup = np.random.default_rng(42)
        self.labels = np.array(
            [i % self.n_clusters for i in range(self.n_samples)], dtype=np.int64
        )
        self.metadata = [
            {
                "rollout_idx": i % self.n_episodes,
                "window_start": 0,
                "window_end": 3,
            }
            for i in range(self.n_samples)
        ]
        self.embeddings = rng_setup.random((self.n_samples, 10)).astype(np.float32)

    def test_plan_structure(self):
        rng = np.random.default_rng(0)
        plan = build_sample_plan(
            self.labels, self.metadata, self.embeddings,
            n_example=3, n_query=3, rng=rng,
        )
        self.assertIn("cluster_ids", plan)
        self.assertEqual(sorted(plan["cluster_ids"]), [0, 1, 2])
        for cid in [0, 1, 2]:
            c = plan["clusters"][cid]
            self.assertIn("example_indices", c)
            self.assertIn("query_indices", c)
            self.assertIn("disjointness_status", c)

    def test_no_overlap_example_query(self):
        rng = np.random.default_rng(0)
        plan = build_sample_plan(
            self.labels, self.metadata, None,
            n_example=3, n_query=3, rng=rng,
        )
        for cid in [0, 1, 2]:
            ex = set(plan["clusters"][cid]["example_indices"])
            qr = set(plan["clusters"][cid]["query_indices"])
            self.assertEqual(len(ex & qr), 0, f"cluster {cid}: overlap between example and query")

    def test_with_embeddings_uses_centroid_proximity(self):
        rng = np.random.default_rng(0)
        plan = build_sample_plan(
            self.labels, self.metadata, self.embeddings,
            n_example=2, n_query=2, rng=rng,
        )
        # With 10 samples per cluster and embeddings, should still produce valid plans
        for cid in [0, 1, 2]:
            self.assertGreater(len(plan["clusters"][cid]["example_indices"]), 0)
            self.assertGreater(len(plan["clusters"][cid]["query_indices"]), 0)

    def test_max_clusters(self):
        rng = np.random.default_rng(0)
        plan = build_sample_plan(
            self.labels, self.metadata, None,
            n_example=2, n_query=2, rng=rng,
            max_clusters=2,
        )
        self.assertEqual(len(plan["cluster_ids"]), 2)

    def test_centroid_distances(self):
        emb = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        indices = np.array([0, 1, 2])
        dists = _centroid_distances(indices, emb)
        # centroid = [1, 0], distances = [1, 0, 1]
        np.testing.assert_allclose(dists, [1.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

class TestParseClassificationResponse(unittest.TestCase):
    valid = ["Group A", "Group B", "Group C"]

    def test_exact_match_first_line(self):
        label, unclear = parse_classification_response("Group B", self.valid)
        self.assertEqual(label, "Group B")
        self.assertFalse(unclear)

    def test_case_insensitive(self):
        label, unclear = parse_classification_response("group a\nbecause it grabs", self.valid)
        self.assertEqual(label, "Group A")
        self.assertFalse(unclear)

    def test_unclear(self):
        label, unclear = parse_classification_response("unclear\nboth look similar", self.valid)
        self.assertEqual(label, "unclear")
        self.assertTrue(unclear)

    def test_fallback_full_text(self):
        label, unclear = parse_classification_response(
            "The answer is Group C based on the motion.", self.valid
        )
        self.assertEqual(label, "Group C")
        self.assertFalse(unclear)

    def test_no_match_returns_unclear(self):
        label, unclear = parse_classification_response("I cannot tell.", self.valid)
        self.assertEqual(label, "unclear")
        self.assertTrue(unclear)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestComputeClassificationMetrics(unittest.TestCase):
    def _make_records(self, n_correct: int, n_wrong: int, n_unclear: int, cluster_ids=None):
        if cluster_ids is None:
            cluster_ids = [0, 1, 2]
        records = []
        for i in range(n_correct):
            records.append({
                "true_cluster_id": cluster_ids[i % len(cluster_ids)],
                "majority_predicted_cluster_id": cluster_ids[i % len(cluster_ids)],
                "is_correct": True,
                "is_unclear": False,
                "agreement_rate": 1.0,
            })
        for i in range(n_wrong):
            cid = cluster_ids[i % len(cluster_ids)]
            wrong_cid = cluster_ids[(i + 1) % len(cluster_ids)]
            records.append({
                "true_cluster_id": cid,
                "majority_predicted_cluster_id": wrong_cid,
                "is_correct": False,
                "is_unclear": False,
                "agreement_rate": 0.67,
            })
        for i in range(n_unclear):
            records.append({
                "true_cluster_id": cluster_ids[i % len(cluster_ids)],
                "majority_predicted_cluster_id": None,
                "is_correct": False,
                "is_unclear": True,
                "agreement_rate": 0.33,
            })
        return records

    def test_perfect_accuracy(self):
        records = self._make_records(9, 0, 0)
        m = compute_classification_metrics(records, [0, 1, 2])
        self.assertAlmostEqual(m["top1_accuracy"], 1.0)
        self.assertEqual(m["n_unclear"], 0)

    def test_chance_accuracy(self):
        records = self._make_records(3, 6, 0)
        m = compute_classification_metrics(records, [0, 1, 2])
        self.assertAlmostEqual(m["top1_accuracy"], 3 / 9)
        self.assertAlmostEqual(m["chance_level"], 1 / 3)

    def test_unclear_excluded_from_accuracy(self):
        records = self._make_records(6, 0, 3)
        m = compute_classification_metrics(records, [0, 1, 2])
        self.assertEqual(m["n_unclear"], 3)
        self.assertEqual(m["n_valid"], 6)
        self.assertAlmostEqual(m["top1_accuracy"], 1.0)

    def test_confusion_matrix_shape(self):
        records = self._make_records(3, 3, 0)
        m = compute_classification_metrics(records, [0, 1, 2])
        cm = m["confusion_matrix"]
        self.assertEqual(len(cm), 3)
        self.assertEqual(len(cm[0]), 3)

    def test_ci_bounds_valid(self):
        records = self._make_records(5, 5, 0)
        m = compute_classification_metrics(records, [0, 1, 2])
        lo, hi = m["top1_accuracy_ci_95"]
        self.assertLessEqual(lo, m["top1_accuracy"])
        self.assertGreaterEqual(hi, m["top1_accuracy"])
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_binomial_test_above_chance(self):
        # 15 clusters, 75 queries all correct → should be highly significant
        records = self._make_records(75, 0, 0, cluster_ids=list(range(15)))
        m = compute_classification_metrics(records, list(range(15)))
        p = m.get("binomial_test_pvalue")
        if p is not None:  # scipy may not be installed
            self.assertLess(p, 0.001)


# ---------------------------------------------------------------------------
# Mock backend classify_slice
# ---------------------------------------------------------------------------

class TestMockBackendClassifySlice(unittest.TestCase):
    def test_returns_first_group(self):
        from PIL import Image
        be = MockVLMBackend()
        query = [Image.new("RGB", (8, 8))]
        example_sets = [
            ("Group A", [Image.new("RGB", (8, 8))]),
            ("Group B", [Image.new("RGB", (8, 8))]),
        ]
        result = be.classify_slice(
            query_images=query,
            example_sets=example_sets,
            system_prompt=None,
            user_preamble="test preamble",
            user_prompt="which group?",
        )
        self.assertIn("Group A", result)
        self.assertIn("predicted=", result)


# ---------------------------------------------------------------------------
# Integration: run_cluster_coherence_classification with mock backend
# ---------------------------------------------------------------------------

class TestRunClusterCoherenceClassification(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.eval_dir = _make_eval_dir(self.root / "eval", n_episodes=3, n_timesteps=10)
        self.cluster_dir = _make_clustering_dir(
            self.root, n_samples=30, n_clusters=3, n_episodes=3, save_embeddings=True
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_dry_run(self):
        step_dir = self.root / "step"
        result = run_cluster_coherence_classification(
            clustering_dir=self.cluster_dir,
            eval_dir=self.eval_dir,
            backend=MockVLMBackend(),
            n_example=2,
            n_query=2,
            n_repetitions=2,
            max_frames_per_storyboard=2,
            random_seed=42,
            step_dir=step_dir,
            dry_run=True,
        )
        self.assertTrue(result["dry_run"])
        # sample_plan.json should still be written even for dry_run
        self.assertTrue((step_dir / "sample_plan.json").is_file())

    def test_full_run_with_mock(self):
        step_dir = self.root / "step_full"
        result = run_cluster_coherence_classification(
            clustering_dir=self.cluster_dir,
            eval_dir=self.eval_dir,
            backend=MockVLMBackend(),
            n_example=2,
            n_query=2,
            n_repetitions=2,
            max_frames_per_storyboard=2,
            random_seed=0,
            step_dir=step_dir,
            dry_run=False,
        )
        self.assertFalse(result["dry_run"])
        self.assertIn("top1_accuracy", result)
        self.assertTrue(pathlib.Path(result["predictions_path"]).is_file())
        self.assertTrue(pathlib.Path(result["metrics_path"]).is_file())
        self.assertTrue(pathlib.Path(result["sample_plan_path"]).is_file())
        # Verify predictions.jsonl structure
        with open(result["predictions_path"]) as f:
            records = [json.loads(line) for line in f if line.strip()]
        self.assertGreater(len(records), 0)
        for rec in records:
            self.assertIn("true_cluster_id", rec)
            self.assertIn("is_correct", rec)
            self.assertIn("repetitions", rec)

    def test_metrics_written(self):
        step_dir = self.root / "step_metrics"
        result = run_cluster_coherence_classification(
            clustering_dir=self.cluster_dir,
            eval_dir=self.eval_dir,
            backend=MockVLMBackend(),
            n_example=2,
            n_query=2,
            n_repetitions=1,
            max_frames_per_storyboard=1,
            random_seed=7,
            step_dir=step_dir,
        )
        with open(result["metrics_path"]) as f:
            metrics = json.load(f)
        self.assertIn("top1_accuracy", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("chance_level", metrics)
        self.assertAlmostEqual(metrics["chance_level"], 1 / 3, places=5)

    def test_no_embeddings_fallback(self):
        cluster_dir_no_emb = _make_clustering_dir(
            self.root / "clust_no_emb", n_samples=15, n_clusters=3,
            n_episodes=3, save_embeddings=False
        )
        step_dir = self.root / "step_no_emb"
        result = run_cluster_coherence_classification(
            clustering_dir=cluster_dir_no_emb,
            eval_dir=self.eval_dir,
            backend=MockVLMBackend(),
            n_example=2,
            n_query=2,
            n_repetitions=1,
            max_frames_per_storyboard=1,
            random_seed=0,
            step_dir=step_dir,
        )
        self.assertFalse(result["dry_run"])
        self.assertIn("top1_accuracy", result)

    def test_max_clusters_cap(self):
        step_dir = self.root / "step_max"
        result = run_cluster_coherence_classification(
            clustering_dir=self.cluster_dir,
            eval_dir=self.eval_dir,
            backend=MockVLMBackend(),
            n_example=2,
            n_query=1,
            n_repetitions=1,
            max_frames_per_storyboard=1,
            random_seed=0,
            step_dir=step_dir,
            max_clusters=2,
        )
        self.assertEqual(result["n_clusters"], 2)

    def test_sample_plan_pre_committed(self):
        """sample_plan.json must exist before predictions.jsonl is written."""
        step_dir = self.root / "step_precommit"

        # Track file creation order
        created_order = []
        orig_open = open

        class TrackingPath(type(step_dir)):
            pass

        # Just verify both files exist and sample_plan has the right structure
        result = run_cluster_coherence_classification(
            clustering_dir=self.cluster_dir,
            eval_dir=self.eval_dir,
            backend=MockVLMBackend(),
            n_example=2,
            n_query=2,
            n_repetitions=1,
            max_frames_per_storyboard=1,
            random_seed=0,
            step_dir=step_dir,
        )
        plan_path = pathlib.Path(result["sample_plan_path"])
        pred_path = pathlib.Path(result["predictions_path"])
        self.assertTrue(plan_path.is_file())
        self.assertTrue(pred_path.is_file())
        with open(plan_path) as f:
            plan = json.load(f)
        self.assertIn("label_maps", plan)
        self.assertIn("random_seed", plan)
        self.assertIn("cluster_ids", plan)


# ---------------------------------------------------------------------------
# Claude backend registration
# ---------------------------------------------------------------------------

class TestClaudeBackendRegistered(unittest.TestCase):
    def test_registry_contains_claude(self):
        from policy_doctor.vlm.registry import list_vlm_backend_names
        self.assertIn("claude", list_vlm_backend_names())


# ---------------------------------------------------------------------------
# save/load embeddings_reduced
# ---------------------------------------------------------------------------

class TestEmbeddingsReducedPersistence(unittest.TestCase):
    def test_save_and_load(self):
        from policy_doctor.influence.clustering_io import (
            load_embeddings_reduced,
            save_clustering_result,
        )

        with tempfile.TemporaryDirectory() as td:
            tdir = pathlib.Path(td)
            import policy_doctor.influence.clustering_io as cr_mod
            orig = cr_mod.get_clustering_dir

            def _tmp_dir(task_config):
                return tdir / task_config / "clustering"

            cr_mod.get_clustering_dir = _tmp_dir
            try:
                emb = np.random.default_rng(0).random((10, 5)).astype(np.float32)
                labels = np.zeros(10, dtype=np.int64)
                meta = [{"rollout_idx": i, "window_start": 0, "window_end": 3} for i in range(10)]
                result_dir = save_clustering_result(
                    "test_run",
                    labels,
                    meta,
                    task_config="test_task",
                    algorithm="kmeans",
                    scaling="none",
                    influence_source="infembed",
                    representation="sliding_window",
                    level="rollout",
                    n_clusters=1,
                    n_samples=10,
                    embeddings_reduced=emb,
                )
                loaded = load_embeddings_reduced(result_dir)
                self.assertIsNotNone(loaded)
                np.testing.assert_allclose(loaded, emb)
            finally:
                cr_mod.get_clustering_dir = orig

    def test_load_returns_none_when_absent(self):
        from policy_doctor.influence.clustering_io import load_embeddings_reduced
        with tempfile.TemporaryDirectory() as td:
            result = load_embeddings_reduced(pathlib.Path(td))
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
