"""Behavior graph node assigner: maps an influence embedding to the nearest cluster.

Environment: usable from the ``policy_doctor`` conda env (no diffusion_policy needed).

The assigner is initialized with pre-computed rollout embeddings and cluster labels
(from ``infembed_embeddings.npz`` and ``cluster_labels.npy`` respectively).  It computes
per-cluster centroids in the raw InfEmbed embedding space (``projection_dim`` dimensions)
and assigns new samples by nearest-centroid L2 distance.

**Important caveat on cluster assignment accuracy:** The clustering pipeline reduces
embeddings via UMAP before clustering, but neither the UMAP model nor the cluster
model (KMeans / GMM) is currently persisted.  Centroids computed here are in the raw
InfEmbed embedding space, not the UMAP-reduced space.  This is an approximation that
works best when clustering was performed without dimensionality reduction (``dim_reduce``
argument = "none") or when the raw-space geometry closely tracks the UMAP geometry.
Saving the UMAP + cluster models during ``RunClusteringStep`` is the correct long-term
fix and is left as future work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from policy_doctor.behaviors.behavior_graph import BehaviorGraph
from policy_doctor.monitoring.base import AssignmentResult, GraphAssigner


class NearestCentroidAssigner(GraphAssigner):
    """Assign a new influence embedding to the nearest cluster centroid.

    Parameters
    ----------
    rollout_embeddings:
        Pre-computed InfEmbed embeddings for training rollout samples,
        shape ``(N_rollout, proj_dim)``.  Loaded from ``infembed_embeddings.npz``
        (key ``rollout_embeddings``).
    cluster_labels:
        Per-rollout-sample cluster assignments, shape ``(N_rollout,)``.
        Loaded from ``cluster_labels.npy`` in the clustering result directory.
        Label ``-1`` denotes noise (HDBSCAN) and is excluded from centroids.
    graph:
        The :class:`~policy_doctor.behaviors.behavior_graph.BehaviorGraph` built
        from the same clustering.  Used to look up node names and to resolve
        cluster IDs that may have been merged by degree-one pruning.
    cluster_id_to_node_id:
        Optional explicit mapping from cluster label to graph node ID.  When
        ``None`` (default), cluster ID == node ID (no pruning was applied).
    """

    def __init__(
        self,
        rollout_embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        graph: BehaviorGraph,
        cluster_id_to_node_id: Optional[Dict[int, int]] = None,
    ) -> None:
        self._graph = graph
        self._cluster_id_to_node_id = cluster_id_to_node_id or {}

        cluster_labels = np.asarray(cluster_labels, dtype=np.int32)
        rollout_embeddings = np.asarray(rollout_embeddings, dtype=np.float32)
        unique_ids = sorted(int(k) for k in np.unique(cluster_labels) if k != -1)

        centroids = np.zeros((len(unique_ids), rollout_embeddings.shape[1]), dtype=np.float32)
        for i, cid in enumerate(unique_ids):
            mask = cluster_labels == cid
            centroids[i] = rollout_embeddings[mask].mean(axis=0)

        self._cluster_ids = np.array(unique_ids, dtype=np.int32)
        self._centroids = centroids  # (K, proj_dim)

    @classmethod
    def from_paths(
        cls,
        rollout_embeddings: np.ndarray,
        clustering_dir: Union[str, Path],
        graph: BehaviorGraph,
        cluster_id_to_node_id: Optional[Dict[int, int]] = None,
    ) -> "NearestCentroidAssigner":
        """Convenience constructor that loads labels from a clustering result directory."""
        from policy_doctor.data.clustering_loader import load_clustering_result_from_path

        labels, _, _ = load_clustering_result_from_path(Path(clustering_dir))
        return cls(
            rollout_embeddings=rollout_embeddings,
            cluster_labels=labels,
            graph=graph,
            cluster_id_to_node_id=cluster_id_to_node_id,
        )

    def assign(self, embedding: np.ndarray) -> AssignmentResult:
        """Assign ``embedding`` to the nearest centroid.

        Args:
            embedding: ``(proj_dim,)`` float32 array in the InfEmbed embedding space.

        Returns:
            :class:`~policy_doctor.monitoring.base.AssignmentResult`.
        """
        embedding = np.asarray(embedding, dtype=np.float32)
        diffs = self._centroids - embedding[np.newaxis, :]  # (K, proj_dim)
        distances = np.linalg.norm(diffs, axis=1)           # (K,)
        best_i = int(np.argmin(distances))
        cluster_id = int(self._cluster_ids[best_i])
        node_id = self._cluster_id_to_node_id.get(cluster_id, cluster_id)
        node = self._graph.nodes.get(node_id)
        node_name = node.name if node is not None else f"Behavior {node_id}"
        return AssignmentResult(
            cluster_id=cluster_id,
            node_id=node_id,
            distance=float(distances[best_i]),
            node_name=node_name,
        )
