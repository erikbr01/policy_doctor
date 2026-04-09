"""
Failure Clusterer: K-Means clustering of failure embeddings.

This module identifies distinct failure modes by clustering the influence
embeddings of failed rollouts. Each cluster center represents a "type of error."

Typical usage:
1. Filter eval embeddings to keep only failures (reward = -1)
2. Run K-Means clustering (k=5 to k=20 typically)
3. Each cluster center represents a failure mode
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch import Tensor


class FailureClusterer:
    """K-Means clustering for failure mode discovery.

    Clusters failure embeddings to identify distinct failure modes.
    Each cluster center represents a type of error.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 300,
    ):
        """Initialize FailureClusterer.

        Args:
            n_clusters: Number of clusters (failure modes) to identify.
            random_state: Random seed for reproducibility.
            n_init: Number of times to run K-means with different seeds.
            max_iter: Maximum iterations for K-means convergence.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter

        self.kmeans: Optional[KMeans] = None
        self.cluster_centers_: Optional[Tensor] = None
        self.labels_: Optional[Tensor] = None
        self.failure_indices_: Optional[Tensor] = None

    def fit(
        self,
        embeddings: Tensor,
        rewards: Optional[Tensor] = None,
        filter_failures: bool = True,
    ) -> "FailureClusterer":
        """Fit K-Means to failure embeddings.

        Args:
            embeddings: Embedding tensor [num_samples, embedding_dim].
            rewards: Optional reward tensor [num_samples]. If provided with
                filter_failures=True, only embeddings with reward < 0 are clustered.
            filter_failures: Whether to filter to only failure embeddings.

        Returns:
            self for method chaining.
        """
        # Convert to numpy for sklearn (ensure float32 for numerical stability).
        if isinstance(embeddings, Tensor):
            emb_np = embeddings.float().numpy().astype(np.float32)
        else:
            emb_np = np.asarray(embeddings, dtype=np.float32)

        # Filter to failures if requested.
        if filter_failures and rewards is not None:
            rewards_np = rewards.numpy() if isinstance(rewards, Tensor) else rewards
            failure_mask = rewards_np < 0
            self.failure_indices_ = torch.from_numpy(np.where(failure_mask)[0])
            emb_np = emb_np[failure_mask]
            self.logger.info(f"Filtered to {emb_np.shape[0]} failure embeddings")
        else:
            self.failure_indices_ = torch.arange(emb_np.shape[0])

        # Handle edge case: no samples to cluster.
        if emb_np.shape[0] == 0:
            self.logger.warning("No failure embeddings to cluster. Skipping clustering.")
            self.kmeans = None
            self.cluster_centers_ = None
            self.labels_ = None
            return self

        if emb_np.shape[0] < self.n_clusters:
            self.logger.warning(
                f"Number of samples ({emb_np.shape[0]}) is less than n_clusters ({self.n_clusters}). "
                f"Reducing n_clusters to {emb_np.shape[0]}."
            )
            self.n_clusters = emb_np.shape[0]

        # Fit K-Means.
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter,
        )
        self.kmeans.fit(emb_np)

        # Store results as tensors.
        self.cluster_centers_ = torch.from_numpy(self.kmeans.cluster_centers_).float()
        self.labels_ = torch.from_numpy(self.kmeans.labels_).long()

        # Compute metrics.
        if emb_np.shape[0] > self.n_clusters:
            silhouette = silhouette_score(emb_np, self.kmeans.labels_)
            self.logger.info(f"Clustering silhouette score: {silhouette:.4f}")

        # Log cluster sizes.
        unique, counts = np.unique(self.kmeans.labels_, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            self.logger.info(f"Cluster {cluster_id}: {count} failures")

        return self

    def predict(self, embeddings: Tensor) -> Tensor:
        """Predict cluster membership for new embeddings.

        Args:
            embeddings: Embedding tensor [num_samples, embedding_dim].

        Returns:
            Cluster labels [num_samples].
        """
        if self.kmeans is None:
            raise ValueError("Clusterer has not been fit. Call fit() first.")

        emb_np = embeddings.numpy() if isinstance(embeddings, Tensor) else embeddings
        labels = self.kmeans.predict(emb_np)
        return torch.from_numpy(labels).long()

    def get_cluster_centers(self) -> Tensor:
        """Get cluster centers as a tensor.

        Returns:
            Tensor of shape [n_clusters, embedding_dim].
        """
        if self.cluster_centers_ is None:
            raise ValueError("Clusterer has not been fit. Call fit() first.")
        return self.cluster_centers_

    def get_cluster_members(self, cluster_id: int) -> Tuple[Tensor, Tensor]:
        """Get indices and labels of members in a specific cluster.

        Args:
            cluster_id: The cluster to query.

        Returns:
            Tuple of (original_indices, cluster_local_indices).
        """
        if self.labels_ is None or self.failure_indices_ is None:
            raise ValueError("Clusterer has not been fit. Call fit() first.")

        cluster_mask = self.labels_ == cluster_id
        cluster_local_indices = torch.where(cluster_mask)[0]
        original_indices = self.failure_indices_[cluster_mask]

        return original_indices, cluster_local_indices

    def get_cluster_summary(self) -> Dict:
        """Get summary statistics for all clusters.

        Returns:
            Dictionary with cluster statistics.
        """
        if self.labels_ is None:
            raise ValueError("Clusterer has not been fit. Call fit() first.")

        labels_np = self.labels_.numpy()
        unique, counts = np.unique(labels_np, return_counts=True)

        return {
            "n_clusters": self.n_clusters,
            "cluster_sizes": dict(zip(unique.tolist(), counts.tolist())),
            "total_failures": len(labels_np),
            "inertia": self.kmeans.inertia_ if self.kmeans else None,
        }

    def find_optimal_k(
        self,
        embeddings: Tensor,
        rewards: Optional[Tensor] = None,
        k_range: Tuple[int, int] = (2, 20),
        method: str = "silhouette",
    ) -> Tuple[int, List[float]]:
        """Find optimal number of clusters using elbow or silhouette method.

        Args:
            embeddings: Embedding tensor [num_samples, embedding_dim].
            rewards: Optional reward tensor for filtering.
            k_range: Range of k values to try (min, max).
            method: "silhouette" or "elbow".

        Returns:
            Tuple of (optimal_k, scores_list).
        """
        # Prepare data.
        emb_np = embeddings.numpy() if isinstance(embeddings, Tensor) else embeddings
        if rewards is not None:
            rewards_np = rewards.numpy() if isinstance(rewards, Tensor) else rewards
            failure_mask = rewards_np < 0
            emb_np = emb_np[failure_mask]

        k_min, k_max = k_range
        k_max = min(k_max, emb_np.shape[0] - 1)  # Can't have more clusters than samples

        scores = []
        k_values = list(range(k_min, k_max + 1))

        for k in k_values:
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=self.n_init,
                max_iter=self.max_iter,
            )
            kmeans.fit(emb_np)

            if method == "silhouette":
                score = silhouette_score(emb_np, kmeans.labels_)
            else:  # elbow
                score = -kmeans.inertia_  # Negative because we want to maximize

            scores.append(score)
            self.logger.info(f"k={k}: {method} score = {score:.4f}")

        # Find optimal k.
        optimal_idx = np.argmax(scores)
        optimal_k = k_values[optimal_idx]

        self.logger.info(
            f"Optimal k={optimal_k} with {method} score={scores[optimal_idx]:.4f}"
        )

        return optimal_k, scores

    def save(self, path: str) -> None:
        """Save clusterer state to disk."""
        state = {
            "n_clusters": self.n_clusters,
            "cluster_centers_": self.cluster_centers_,
            "labels_": self.labels_,
            "failure_indices_": self.failure_indices_,
            "random_state": self.random_state,
        }
        if self.kmeans is not None:
            state["kmeans_inertia"] = self.kmeans.inertia_

        torch.save(state, path)
        self.logger.info(f"Saved clusterer to {path}")

    @classmethod
    def load(cls, path: str) -> "FailureClusterer":
        """Load clusterer from disk."""
        state = torch.load(path)
        instance = cls(
            n_clusters=state["n_clusters"],
            random_state=state["random_state"],
        )
        instance.cluster_centers_ = state["cluster_centers_"]
        instance.labels_ = state["labels_"]
        instance.failure_indices_ = state["failure_indices_"]
        return instance
