"""Tests for clustering (Gaussian Mixture, PCA, normalize)."""

import unittest
import numpy as np

from policy_doctor.behaviors.clustering import (
    cluster_gaussian_mixture,
    normalize_embeddings,
    reduce_dimensions,
    run_clustering,
)


class TestClustering(unittest.TestCase):
    def test_normalize_embeddings_none(self):
        x = np.random.randn(10, 4).astype(np.float32)
        out = normalize_embeddings(x, method="none")
        np.testing.assert_array_almost_equal(out, x)

    def test_normalize_embeddings_standard(self):
        x = np.random.randn(10, 4).astype(np.float32)
        out = normalize_embeddings(x, method="standard")
        self.assertEqual(out.shape, x.shape)
        np.testing.assert_array_almost_equal(out.mean(axis=0), np.zeros(4), decimal=5)
        np.testing.assert_array_almost_equal(out.std(axis=0), np.ones(4), decimal=5)

    def test_reduce_dimensions_pca(self):
        x = np.random.randn(20, 5).astype(np.float32)
        out = reduce_dimensions(x, method="pca", n_components=2)
        self.assertEqual(out.shape, (20, 2))

    def test_cluster_gaussian_mixture(self):
        x = np.random.randn(30, 4).astype(np.float32)
        labels = cluster_gaussian_mixture(x, n_components=3)
        self.assertEqual(labels.shape, (30,))
        self.assertLessEqual(set(labels), set(range(3)))

    def test_run_clustering_gaussian_mixture(self):
        x = np.random.randn(25, 6).astype(np.float32)
        labels, coords_2d, metrics = run_clustering(
            x,
            method="gaussian_mixture",
            dim_reduce="pca",
            n_components_2d=2,
            normalize="none",
            n_components=3,
        )
        self.assertEqual(labels.shape, (25,))
        self.assertEqual(coords_2d.shape, (25, 2))
        self.assertIn("n_clusters", metrics)

    def test_run_clustering_output_compatible_with_pipeline(self):
        """Cluster assignments from run_clustering are valid for pipeline (BehaviorGraph, slice search)."""
        x = np.random.randn(20, 4).astype(np.float32)
        labels, coords_2d, metrics = run_clustering(
            x,
            method="gaussian_mixture",
            dim_reduce="pca",
            n_components_2d=2,
            normalize="none",
            n_components=2,
        )
        self.assertEqual(labels.shape, (20,), "One label per sample")
        self.assertTrue(np.issubdtype(labels.dtype, np.integer), "Labels must be integer for pipeline")
        self.assertTrue(np.all(labels >= -1), "Only -1 (noise) or non-negative cluster ids allowed")
        self.assertEqual(coords_2d.shape, (20, 2), "2D coords for visualization")
