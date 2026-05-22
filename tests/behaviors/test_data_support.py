"""Tests for the data-support measurement module.

These exercise the pure-Python metric / aggregation logic on hand-built
embeddings — no policy checkpoints, no clustering pipeline. The
end-to-end test fits a real joint UMAP on synthetic embeddings to confirm
the full pipeline produces sensible per-cluster scores.
"""

import unittest

import numpy as np

from policy_doctor.behaviors.data_support import (
    aggregate_per_cluster,
    available_metrics,
    compute_all_metrics,
    fit_joint_umap,
)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        # 10 demo points clustered at origin; 3 rollout points: one inside,
        # one on the boundary, one far away.
        rng = np.random.default_rng(0)
        self.demo = rng.normal(loc=0.0, scale=0.1, size=(50, 4)).astype(np.float32)
        # demo std is 0.1, so 0.15 sits at the edge of the cloud (low but
        # non-zero local count); 10.0 is clearly OOD.
        self.rollout = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],     # well inside the demo cloud
                [0.15, 0.0, 0.0, 0.0],    # near the boundary, partial overlap
                [10.0, 10.0, 0.0, 0.0],   # far out of distribution
            ],
            dtype=np.float32,
        )

    def test_registry_lists_all_metrics(self):
        names = available_metrics()
        for required in (
            "count_in_radius",
            "binary_coverage",
            "knn_mean_distance",
            "knn_max_distance",
            "kde_log_density",
        ):
            self.assertIn(required, names)

    def test_count_in_radius_orders_by_density(self):
        out, _ = compute_all_metrics(
            self.demo, self.rollout,
            metrics=["count_in_radius"],
            radius=0.3, knn_k=5,
        )
        c = out["count_in_radius"]
        self.assertEqual(c.shape, (3,))
        # Inside ≥ boundary > far.
        self.assertGreaterEqual(c[0], c[1])
        self.assertGreater(c[1], c[2])
        self.assertEqual(int(c[2]), 0)

    def test_binary_coverage_is_zero_for_far_point(self):
        out, _ = compute_all_metrics(
            self.demo, self.rollout,
            metrics=["binary_coverage"],
            radius=0.3, knn_k=5,
        )
        v = out["binary_coverage"]
        self.assertEqual(v.shape, (3,))
        self.assertEqual(v[0], 1.0)
        self.assertEqual(v[2], 0.0)

    def test_knn_distance_orders_inside_below_far(self):
        out, _ = compute_all_metrics(
            self.demo, self.rollout,
            metrics=["knn_mean_distance", "knn_max_distance"],
            radius=0.3, knn_k=5,
        )
        for m in ("knn_mean_distance", "knn_max_distance"):
            d = out[m]
            self.assertEqual(d.shape, (3,))
            # Inside has smaller distance than far.
            self.assertLess(d[0], d[2])
            # max ≥ mean by construction.
        self.assertGreaterEqual(out["knn_max_distance"][0], out["knn_mean_distance"][0])

    def test_kde_log_density_finite(self):
        out, _ = compute_all_metrics(
            self.demo, self.rollout,
            metrics=["kde_log_density"],
            radius=0.3, knn_k=5,
            kde_bandwidth=0.1,
        )
        v = out["kde_log_density"]
        self.assertEqual(v.shape, (3,))
        self.assertTrue(np.all(np.isfinite(v)))
        # The KDE should rank the inside point as more likely than the far one.
        self.assertGreater(v[0], v[2])

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            compute_all_metrics(
                self.demo, self.rollout,
                metrics=["does_not_exist"],
                radius=0.3, knn_k=5,
            )


class TestAggregate(unittest.TestCase):
    def test_per_cluster_summary_stats(self):
        values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, -5.0], dtype=np.float64)
        labels = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
        out = aggregate_per_cluster(values, labels)
        self.assertEqual(set(out.keys()), {0, 1, 2})
        # Cluster 0: median of {1,2,3} = 2; mean = 2.
        self.assertAlmostEqual(out[0]["median"], 2.0)
        self.assertAlmostEqual(out[0]["mean"], 2.0)
        self.assertEqual(out[0]["n_slices"], 3)
        # Cluster 1: 10 / 20 -> mean 15, median 15.
        self.assertAlmostEqual(out[1]["mean"], 15.0)
        # Single-element cluster: q10 == q90 == value.
        self.assertEqual(out[2]["n_slices"], 1)
        self.assertAlmostEqual(out[2]["q10"], -5.0)
        self.assertAlmostEqual(out[2]["q90"], -5.0)

    def test_exclude_labels_skips_them(self):
        values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        labels = np.array([-2, 0, -1], dtype=np.int64)  # -2/-1 = terminal placeholders
        out = aggregate_per_cluster(values, labels, exclude_labels=(-1, -2))
        self.assertEqual(set(out.keys()), {0})

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            aggregate_per_cluster(
                np.array([1.0, 2.0]),
                np.array([0, 0, 1], dtype=np.int64),
            )


class TestEndToEnd(unittest.TestCase):
    """Joint UMAP + per-cluster scoring on synthetic embeddings.

    Two well-separated rollout clusters: one near the demo cloud, one far
    away.  After joint UMAP + count-in-radius, the supported cluster should
    score strictly higher than the unsupported one in median terms.
    """

    def test_supported_cluster_scores_higher(self):
        rng = np.random.default_rng(42)
        D = 16
        # Demo cloud at origin.
        demo = rng.normal(loc=0.0, scale=1.0, size=(120, D)).astype(np.float32)
        # Rollout cluster 0: overlapping with demos.
        roll_a = rng.normal(loc=0.0, scale=1.0, size=(40, D)).astype(np.float32)
        # Rollout cluster 1: far away from demos.
        roll_b = rng.normal(loc=8.0, scale=0.5, size=(40, D)).astype(np.float32)
        rollout = np.concatenate([roll_a, roll_b], axis=0)
        cluster_labels = np.array([0] * 40 + [1] * 40, dtype=np.int64)

        joint = fit_joint_umap(
            demo, rollout,
            n_components=3,
            n_neighbors=10,
            random_state=0,
        )
        per_metric, _ = compute_all_metrics(
            joint.demo_reduced,
            joint.rollout_reduced,
            metrics=["count_in_radius", "knn_mean_distance", "binary_coverage"],
            radius=1.5, knn_k=5,
        )
        per_cluster = aggregate_per_cluster(
            per_metric["count_in_radius"], cluster_labels
        )
        # Supported cluster 0 has more demos in its radius than unsupported cluster 1.
        self.assertGreater(per_cluster[0]["median"], per_cluster[1]["median"])

        per_cluster_knn = aggregate_per_cluster(
            per_metric["knn_mean_distance"], cluster_labels
        )
        # Supported cluster 0 has smaller kNN distance to demos than cluster 1.
        self.assertLess(per_cluster_knn[0]["median"], per_cluster_knn[1]["median"])

    def test_dimension_mismatch_raises(self):
        demo = np.zeros((10, 4), dtype=np.float32)
        rollout = np.zeros((5, 6), dtype=np.float32)
        with self.assertRaises(ValueError):
            fit_joint_umap(demo, rollout, n_components=2)


if __name__ == "__main__":
    unittest.main()
