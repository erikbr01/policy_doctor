"""
Training Attributor: Links failure clusters to training demonstrations.

For each failure mode (cluster center), this module:
1. Computes cosine similarity between the cluster center and all training embeddings
2. Returns the top-k training examples with highest similarity
3. These are the training demonstrations "responsible" for that failure mode

The result answers: "Which training demonstrations pushed the model toward this type of failure?"
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class Attribution:
    """Result of attributing a failure cluster to training demonstrations."""

    cluster_id: int
    training_indices: List[int]
    similarity_scores: List[float]
    cluster_center: Tensor


class TrainingAttributor:
    """Attributes failure modes to training demonstrations via cosine similarity.

    For each cluster center (failure mode), finds the training demonstrations
    that are most similar in the influence embedding space.
    """

    def __init__(
        self,
        train_embeddings: Tensor,
        top_k: int = 10,
    ):
        """Initialize TrainingAttributor.

        Args:
            train_embeddings: Training embedding tensor [num_train, embedding_dim].
            top_k: Number of top training examples to return per cluster.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.train_embeddings = train_embeddings
        self.top_k = top_k

        # Pre-normalize training embeddings for efficient cosine similarity.
        self.train_embeddings_normalized = F.normalize(train_embeddings, p=2, dim=1)
        self.logger.info(
            f"Initialized attributor with {train_embeddings.shape[0]} training embeddings"
        )

    def attribute_cluster(
        self,
        cluster_center: Tensor,
        top_k: Optional[int] = None,
    ) -> Attribution:
        """Attribute a single failure cluster to training demonstrations.

        Args:
            cluster_center: Cluster center embedding [embedding_dim].
            top_k: Number of top examples to return (overrides default).

        Returns:
            Attribution object with training indices and similarity scores.
        """
        k = top_k if top_k is not None else self.top_k

        # Normalize cluster center.
        center_normalized = F.normalize(cluster_center.unsqueeze(0), p=2, dim=1)

        # Compute cosine similarity with all training embeddings.
        similarities = torch.mm(
            center_normalized, self.train_embeddings_normalized.T
        ).squeeze(0)

        # Get top-k.
        top_scores, top_indices = torch.topk(similarities, min(k, len(similarities)))

        return Attribution(
            cluster_id=-1,  # Will be set by caller
            training_indices=top_indices.tolist(),
            similarity_scores=top_scores.tolist(),
            cluster_center=cluster_center,
        )

    def attribute_all_clusters(
        self,
        cluster_centers: Tensor,
        top_k: Optional[int] = None,
    ) -> List[Attribution]:
        """Attribute all failure clusters to training demonstrations.

        Args:
            cluster_centers: Cluster centers [n_clusters, embedding_dim].
            top_k: Number of top examples per cluster.

        Returns:
            List of Attribution objects, one per cluster.
        """
        attributions = []

        for cluster_id in range(cluster_centers.shape[0]):
            center = cluster_centers[cluster_id]
            attr = self.attribute_cluster(center, top_k)
            attr.cluster_id = cluster_id
            attributions.append(attr)

            self.logger.info(
                f"Cluster {cluster_id}: top training indices {attr.training_indices[:5]}... "
                f"(max similarity: {attr.similarity_scores[0]:.4f})"
            )

        return attributions

    def compute_all_similarities(
        self,
        cluster_centers: Tensor,
    ) -> Tensor:
        """Compute similarity matrix between all clusters and all training demos.

        Args:
            cluster_centers: Cluster centers [n_clusters, embedding_dim].

        Returns:
            Similarity matrix [n_clusters, num_train].
        """
        centers_normalized = F.normalize(cluster_centers, p=2, dim=1)
        similarities = torch.mm(centers_normalized, self.train_embeddings_normalized.T)
        return similarities

    def get_training_demo_influence(
        self,
        train_idx: int,
        cluster_centers: Tensor,
    ) -> Dict[int, float]:
        """Get influence of a single training demo on each failure cluster.

        Args:
            train_idx: Index of training demonstration.
            cluster_centers: Cluster centers [n_clusters, embedding_dim].

        Returns:
            Dictionary mapping cluster_id -> similarity score.
        """
        train_emb = self.train_embeddings_normalized[train_idx]
        centers_normalized = F.normalize(cluster_centers, p=2, dim=1)

        similarities = torch.mm(centers_normalized, train_emb.unsqueeze(1)).squeeze(1)

        return {i: sim.item() for i, sim in enumerate(similarities)}

    def find_most_influential_demos(
        self,
        cluster_centers: Tensor,
        top_k: Optional[int] = None,
    ) -> Tuple[List[int], List[float]]:
        """Find training demos that are most influential across all failure clusters.

        This aggregates influence by summing similarity across all clusters.

        Args:
            cluster_centers: Cluster centers [n_clusters, embedding_dim].
            top_k: Number of top demos to return.

        Returns:
            Tuple of (demo_indices, aggregate_scores).
        """
        k = top_k if top_k is not None else self.top_k

        # Compute all similarities.
        similarities = self.compute_all_similarities(
            cluster_centers
        )  # [n_clusters, num_train]

        # Aggregate across clusters (sum or max).
        aggregate_scores = similarities.sum(dim=0)  # [num_train]

        # Get top-k.
        top_scores, top_indices = torch.topk(
            aggregate_scores, min(k, len(aggregate_scores))
        )

        return top_indices.tolist(), top_scores.tolist()

    def generate_report(
        self,
        attributions: List[Attribution],
        train_metadata: Optional[Dict] = None,
    ) -> str:
        """Generate a human-readable report of attributions.

        Args:
            attributions: List of Attribution objects from attribute_all_clusters.
            train_metadata: Optional metadata about training demonstrations.

        Returns:
            Formatted string report.
        """
        lines = ["=" * 60]
        lines.append("FAILURE MODE ATTRIBUTION REPORT")
        lines.append("=" * 60)
        lines.append("")

        for attr in attributions:
            lines.append(f"CLUSTER {attr.cluster_id}")
            lines.append("-" * 40)
            lines.append("Top responsible training demonstrations:")

            for rank, (idx, score) in enumerate(
                zip(attr.training_indices, attr.similarity_scores)
            ):
                demo_info = f"Demo {idx}"
                if train_metadata and "demo_names" in train_metadata:
                    demo_info = train_metadata["demo_names"].get(idx, demo_info)
                lines.append(f"  {rank + 1}. {demo_info} (similarity: {score:.4f})")

            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def save_attributions(
        self,
        attributions: List[Attribution],
        path: str,
    ) -> None:
        """Save attributions to disk."""
        data = {
            "attributions": [
                {
                    "cluster_id": a.cluster_id,
                    "training_indices": a.training_indices,
                    "similarity_scores": a.similarity_scores,
                    "cluster_center": a.cluster_center.cpu(),
                }
                for a in attributions
            ],
            "top_k": self.top_k,
            "num_train": self.train_embeddings.shape[0],
        }
        torch.save(data, path)
        self.logger.info(f"Saved attributions to {path}")

    @staticmethod
    def load_attributions(path: str) -> Tuple[List[Attribution], Dict]:
        """Load attributions from disk."""
        data = torch.load(path)
        attributions = [
            Attribution(
                cluster_id=a["cluster_id"],
                training_indices=a["training_indices"],
                similarity_scores=a["similarity_scores"],
                cluster_center=a["cluster_center"],
            )
            for a in data["attributions"]
        ]
        metadata = {"top_k": data["top_k"], "num_train": data["num_train"]}
        return attributions, metadata
