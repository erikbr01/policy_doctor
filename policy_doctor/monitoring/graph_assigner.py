"""Behavior graph node assigners: map an influence embedding to the nearest cluster.

Environment: usable from the ``policy_doctor`` conda env (no diffusion_policy needed).

Two assigners are provided:

* :class:`NearestCentroidAssigner` — approximation in raw InfEmbed embedding space.
  Does not require saved models; works even for clustering runs that predate model
  persistence.  Centroids are computed from ``rollout_embeddings`` + ``cluster_labels``.

* :class:`FittedModelAssigner` — exact assignment through the fitted pipeline.
  Requires ``clustering_models.pkl`` produced by :func:`~policy_doctor.data.clustering_loader.save_clustering_models`
  (available in ``RunClusteringStep`` runs after model saving was added).
  Applies normalizer → prescaler → UMAP → KMeans.predict in the exact same order as
  the clustering run, giving geometrically correct assignments.
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


class FittedModelAssigner(GraphAssigner):
    """Assign a new influence embedding to a cluster using the exact fitted pipeline.

    Applies the same normalizer → prescaler → UMAP reducer → KMeans.predict sequence
    that was used during the original ``RunClusteringStep``, giving geometrically
    correct cluster assignments for new data points.

    Requires ``clustering_models.pkl`` in the clustering result directory, which is
    saved automatically by ``RunClusteringStep`` (from the version that added model
    persistence).

    Parameters
    ----------
    models:
        :class:`~policy_doctor.data.clustering_loader.ClusteringModels` container with
        the fitted sklearn/UMAP objects.
    graph:
        :class:`~policy_doctor.behaviors.behavior_graph.BehaviorGraph` built from the
        same clustering run.
    cluster_id_to_node_id:
        Optional explicit mapping from cluster label to graph node ID (for pruned graphs).
    """

    def __init__(
        self,
        models: "ClusteringModels",  # noqa: F821
        graph: BehaviorGraph,
        cluster_id_to_node_id: Optional[Dict[int, int]] = None,
    ) -> None:
        self._models = models
        self._graph = graph
        self._cluster_id_to_node_id = cluster_id_to_node_id or {}

    @classmethod
    def from_paths(
        cls,
        clustering_dir: Union[str, Path],
        graph: BehaviorGraph,
        cluster_id_to_node_id: Optional[Dict[int, int]] = None,
    ) -> "FittedModelAssigner":
        """Load models from ``<clustering_dir>/clustering_models.pkl``."""
        from policy_doctor.data.clustering_loader import load_clustering_models

        models = load_clustering_models(Path(clustering_dir))
        return cls(models=models, graph=graph, cluster_id_to_node_id=cluster_id_to_node_id)

    def assign(self, embedding: np.ndarray) -> AssignmentResult:
        """Assign ``embedding`` to a cluster via the fitted pipeline.

        Args:
            embedding: ``(proj_dim,)`` float32 array in the raw InfEmbed embedding space.

        Returns:
            :class:`~policy_doctor.monitoring.base.AssignmentResult`.
        """
        x = np.asarray(embedding, dtype=np.float32).reshape(1, -1)  # (1, proj_dim)

        models = self._models
        if models.normalizer is not None:
            x = models.normalizer.transform(x).astype(np.float32)

        if models.prescaler is not None:
            x = models.prescaler.transform(x).astype(np.float32)

        if models.reducer is not None:
            x = models.reducer.transform(x).astype(np.float32)

        if models.kmeans is None:
            raise RuntimeError("FittedModelAssigner requires a fitted KMeans model.")
        cluster_id = int(models.kmeans.predict(x)[0])

        node_id = self._cluster_id_to_node_id.get(cluster_id, cluster_id)
        node = self._graph.nodes.get(node_id)
        node_name = node.name if node is not None else f"Behavior {node_id}"

        centroid = models.kmeans.cluster_centers_[cluster_id]
        distance = float(np.linalg.norm(x[0] - centroid))
        return AssignmentResult(
            cluster_id=cluster_id,
            node_id=node_id,
            distance=distance,
            node_name=node_name,
        )
