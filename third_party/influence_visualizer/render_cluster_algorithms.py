"""Clustering algorithms section for the clustering tab.

Applies actual clustering algorithms (KMeans, DBSCAN, etc.) to influence
embeddings, evaluates cluster quality, checks label coherency against human
annotations, and displays per-cluster video samples.
"""

from collections import Counter
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import streamlit as st

from influence_visualizer import plotting
from influence_visualizer.data_loader import InfluenceData
from influence_visualizer.render_heatmaps import SplitType, get_split_data

# ---------------------------------------------------------------------------
# Session state memory management
# ---------------------------------------------------------------------------

MAX_CLUSTERING_CACHES = 3

_CLUSTERING_CACHE_SUFFIXES = (
    "_embeddings",
    "_embeddings_original",
    "_metadata",
    "_cluster_labels",
    "_cluster_algorithm",
    "_cluster_scaling",
    "_embeddings_2d",
    "_tsne_cache",
    "_refined_embeddings",
    "_refined_metadata",
    "_refined_labels",
    "_refined_high_sil_clusters",
    "_refined_low_sil_clusters",
    "_refined_original_indices",
    "_refined_scaling",
    "_refined_dimred_used",
    "_refined_algorithm",
    "_sil_values",
    "_refine_sil_values",
    "_behavior_graph",
    "_video_exploration_shown",
)


def _register_clustering_cache(key_prefix: str, cache_key: str) -> None:
    """Track a clustering cache key and evict the oldest when over the limit.

    Keeps at most ``MAX_CLUSTERING_CACHES`` cache entries per key_prefix.
    Oldest entries (by insertion order) are evicted first.
    """
    tracker_key = f"{key_prefix}__cache_order"
    order: list = st.session_state.get(tracker_key, [])

    # Move to end if already tracked (re-used)
    if cache_key in order:
        order.remove(cache_key)
    order.append(cache_key)

    # Evict oldest
    while len(order) > MAX_CLUSTERING_CACHES:
        old_key = order.pop(0)
        for suffix in _CLUSTERING_CACHE_SUFFIXES:
            st.session_state.pop(f"{old_key}{suffix}", None)
        # Also clean any keys that start with old_key + "_cluster"
        for k in list(st.session_state.keys()):
            if isinstance(k, str) and k.startswith(f"{old_key}_"):
                del st.session_state[k]

    st.session_state[tracker_key] = order


# ---------------------------------------------------------------------------
# InfEmbed embedding loader
# ---------------------------------------------------------------------------


def load_infembed_embeddings(
    data: InfluenceData,
    split: SplitType,
    level: Literal["rollout", "demo"],
) -> Tuple[np.ndarray, List[Dict]]:
    """Load precomputed InfEmbed embeddings from disk (no aggregation).

    Expects npz at data.eval_dir / data.trak_exp_name / "infembed_embeddings.npz"
    with keys:
      - rollout_embeddings: (n_rollout_samples, D) one per timestep
      - demo_embeddings: (n_demo_samples, D), train then holdout order

    Returns (embeddings, metadata): one row per timestep for both rollout and demo (no aggregation).
    """
    if data.eval_dir is None or getattr(data, "trak_exp_name", None) is None:
        raise FileNotFoundError(
            "eval_dir or trak_exp_name not set; cannot locate InfEmbed file."
        )
    path = data.eval_dir / data.trak_exp_name / "infembed_embeddings.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"InfEmbed embeddings not found at {path}. "
            "Run the InfEmbed computation script first (see INFEMBED_INTEGRATION_PLAN.md)."
        )
    with np.load(path, allow_pickle=False) as f:
        rollout_emb = np.asarray(f["rollout_embeddings"])  # (n_rollout_samples, D)
        demo_emb = np.asarray(f["demo_embeddings"])        # (n_demo_samples, D)

    if level == "rollout":
        expected = sum(ep.num_samples for ep in data.rollout_episodes)
        if rollout_emb.shape[0] != expected:
            raise ValueError(
                f"InfEmbed rollout sample count mismatch: npz has {rollout_emb.shape[0]}, "
                f"expected {expected}. Run compute_infembed_embeddings.py with same eval_dir."
            )
        # One embedding per timestep (no aggregation)
        metadata = []
        for ep in data.rollout_episodes:
            for t in range(ep.sample_start_idx, ep.sample_end_idx):
                metadata.append({
                    "rollout_idx": ep.index,
                    "success": ep.success,
                    "num_samples": ep.num_samples,
                    "timestep": t - ep.sample_start_idx,
                })
        return rollout_emb, metadata

    # level == "demo": one embedding per timestep, only for the selected split (train/holdout/both)
    _, demo_episodes, ep_idxs, _ = get_split_data(data, split)
    embeddings_list = []
    metadata = []
    for demo_ep in demo_episodes:
        sl = slice(demo_ep.sample_start_idx, demo_ep.sample_end_idx)
        embeddings_list.append(demo_emb[sl])
        quality_label = "unknown"
        if data.demo_quality_labels is not None:
            quality_label = data.demo_quality_labels.get(demo_ep.index, "unknown")
        for t in range(demo_ep.sample_start_idx, demo_ep.sample_end_idx):
            metadata.append({
                "demo_idx": demo_ep.index,
                "quality_label": quality_label,
                "num_samples": demo_ep.num_samples,
                "raw_length": demo_ep.raw_length,
                "timestep": t - demo_ep.sample_start_idx,
            })
    return np.vstack(embeddings_list), metadata


def extract_infembed_sliding_window_embeddings(
    data: InfluenceData,
    split: SplitType,
    level: Literal["rollout", "demo"],
    window_width: int = 10,
    stride: int = 1,
    aggregation_method: Literal[
        "mean", "max", "min", "std", "sum", "median", "svd"
    ] = "mean",
    n_components: int = 10,
    annotations: Optional[Dict] = None,
    rollout_normalization: Literal["none", "center", "normalize"] = "none",
) -> Tuple[np.ndarray, List[Dict]]:
    """Extract sliding window embeddings from InfEmbed per-timestep embeddings.

    Mirrors the TRAK sliding window logic: for each rollout/demo episode,
    slide a window over timesteps and aggregate embeddings within each window
    (mean, max, svd, etc.). Same aggregation methods and metadata structure
    as extract_episode_sliding_window_embeddings for TRAK.

    Returns:
        Tuple of (embeddings, metadata) where metadata has rollout_idx/demo_idx,
        window_start, window_end, etc. for downstream compatibility.
    """
    if data.eval_dir is None or getattr(data, "trak_exp_name", None) is None:
        raise FileNotFoundError(
            "eval_dir or trak_exp_name not set; cannot locate InfEmbed file."
        )
    path = data.eval_dir / data.trak_exp_name / "infembed_embeddings.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"InfEmbed embeddings not found at {path}. "
            "Run the InfEmbed computation script first (see INFEMBED_INTEGRATION_PLAN.md)."
        )
    with np.load(path, allow_pickle=False) as f:
        rollout_emb = np.asarray(f["rollout_embeddings"])
        demo_emb = np.asarray(f["demo_embeddings"])

    # Validate alignment: InfEmbed npz order must match data.rollout_episodes
    # (both use eval_dir/episodes; BatchEpisodeDataset and build_rollout_sample_infos
    # read the same metadata.yaml and episode files)
    if level == "rollout":
        expected_samples = sum(ep.num_samples for ep in data.rollout_episodes)
        if rollout_emb.shape[0] != expected_samples:
            raise ValueError(
                f"InfEmbed rollout sample count mismatch: npz has {rollout_emb.shape[0]}, "
                f"expected {expected_samples} from rollout_episodes. "
                "Ensure compute_infembed_embeddings.py was run with the same eval_dir "
                "as the influence visualizer."
            )

    from influence_visualizer.render_annotation import (
        get_episode_annotations,
        get_label_for_frame,
    )

    all_embeddings = []
    all_metadata = []

    def _aggregate_window(
        window: np.ndarray,
        method: str,
        n_comp: int,
    ) -> np.ndarray:
        """Aggregate (window_width, D) window along axis=0."""
        if method == "svd":
            try:
                U, s, Vt = np.linalg.svd(window, full_matrices=False)
                if len(s) < n_comp:
                    emb = np.zeros(n_comp)
                    emb[: len(s)] = s
                else:
                    emb = s[:n_comp]
            except np.linalg.LinAlgError:
                emb = np.zeros(n_comp)
        elif method == "mean":
            emb = np.mean(window, axis=0)
        elif method == "max":
            emb = np.max(window, axis=0)
        elif method == "min":
            emb = np.min(window, axis=0)
        elif method == "std":
            emb = np.std(window, axis=0)
        elif method == "sum":
            emb = np.sum(window, axis=0)
        elif method == "median":
            emb = np.median(window, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        return emb

    if level == "rollout":
        for ep in data.rollout_episodes:
            sl = slice(ep.sample_start_idx, ep.sample_end_idx)
            ep_emb = rollout_emb[sl]  # (num_timesteps, D)
            num_timesteps = ep_emb.shape[0]
            if num_timesteps < window_width:
                continue

            annotation_labels_dict = {}
            if annotations:
                episode_annotations = get_episode_annotations(
                    annotations, str(ep.index), split="rollout"
                )
                for t in range(num_timesteps):
                    annotation_labels_dict[t] = get_label_for_frame(
                        t, episode_annotations
                    )

            for start_idx in range(0, num_timesteps - window_width + 1, stride):
                end_idx = start_idx + window_width
                window = ep_emb[start_idx:end_idx, :]  # (window_width, D)
                embedding = _aggregate_window(
                    window, aggregation_method, n_components
                )
                all_embeddings.append(embedding)

                middle_timestep = start_idx + window_width // 2
                annotation_label = annotation_labels_dict.get(
                    middle_timestep, "no label"
                )
                all_metadata.append({
                    "rollout_idx": ep.index,
                    "window_start": start_idx,
                    "window_end": end_idx,
                    "window_width": window_width,
                    "aggregation": aggregation_method,
                    "annotation_label": annotation_label,
                    "success": ep.success,
                    "num_samples": ep.num_samples,
                    "mean_influence": float(np.mean(window)),
                    "std_influence": float(np.std(window)),
                    "max_influence": float(np.max(window)),
                    "min_influence": float(np.min(window)),
                })

        if rollout_normalization != "none" and len(all_embeddings) > 0:
            embeddings_arr = np.array(all_embeddings)
            rollout_ids = np.array([m["rollout_idx"] for m in all_metadata])
            for r in np.unique(rollout_ids):
                mask = rollout_ids == r
                block = embeddings_arr[mask]
                block = block - block.mean(axis=0)
                if rollout_normalization == "normalize":
                    std = block.std(axis=0)
                    std[std < 1e-10] = 1.0
                    block = block / std
                embeddings_arr[mask] = block
            all_embeddings = list(embeddings_arr)

    else:  # level == "demo"
        _, demo_episodes, ep_idxs, _ = get_split_data(data, split)
        for demo_ep_idx, demo_ep in enumerate(demo_episodes):
            sl = slice(demo_ep.sample_start_idx, demo_ep.sample_end_idx)
            ep_emb = demo_emb[sl]  # (num_timesteps, D)
            num_timesteps = ep_emb.shape[0]
            if num_timesteps < window_width:
                continue

            quality_label = "unknown"
            if data.demo_quality_labels is not None:
                quality_label = data.demo_quality_labels.get(
                    demo_ep.index, "unknown"
                )

            annotation_labels_dict = {}
            if annotations:
                demo_split_type = (
                    "train" if split in ["train", "both"] else "holdout"
                )
                episode_annotations = get_episode_annotations(
                    annotations, str(demo_ep.index), split=demo_split_type
                )
                for t in range(num_timesteps):
                    annotation_labels_dict[t] = get_label_for_frame(
                        t, episode_annotations
                    )

            for start_idx in range(0, num_timesteps - window_width + 1, stride):
                end_idx = start_idx + window_width
                window = ep_emb[start_idx:end_idx, :]  # (window_width, D)
                embedding = _aggregate_window(
                    window, aggregation_method, n_components
                )
                all_embeddings.append(embedding)

                middle_timestep = start_idx + window_width // 2
                annotation_label = annotation_labels_dict.get(
                    middle_timestep, "no label"
                )
                all_metadata.append({
                    "demo_idx": demo_ep.index,
                    "window_start": start_idx,
                    "window_end": end_idx,
                    "window_width": window_width,
                    "aggregation": aggregation_method,
                    "annotation_label": annotation_label,
                    "quality_label": quality_label,
                    "num_samples": demo_ep.num_samples,
                    "raw_length": demo_ep.raw_length,
                    "mean_influence": float(np.mean(window)),
                    "std_influence": float(np.std(window)),
                    "max_influence": float(np.max(window)),
                    "min_influence": float(np.min(window)),
                })

    return np.array(all_embeddings), all_metadata


# ---------------------------------------------------------------------------
# Cached computation helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=20)
def _run_clustering(
    embeddings: np.ndarray,
    algorithm: str,
    scaling_method: str = "standard",
    # KMeans / MiniBatchKMeans / Agglomerative / Spectral / Birch
    n_clusters: int = 5,
    # KMeans
    kmeans_init: str = "k-means++",
    kmeans_n_init: int = 10,
    # MiniBatchKMeans
    mbk_batch_size: int = 1024,
    # DBSCAN
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    # OPTICS
    optics_min_samples: int = 5,
    optics_xi: float = 0.05,
    optics_min_cluster_size: float = 0.05,
    # HDBSCAN
    hdbscan_min_cluster_size: int = 5,
    hdbscan_min_samples: int = 5,
    hdbscan_cluster_selection_epsilon: float = 0.0,
    hdbscan_cluster_selection_method: str = "eom",
    hdbscan_alpha: float = 1.0,
    # Agglomerative
    agg_linkage: str = "ward",
    # Spectral
    spectral_affinity: str = "rbf",
    spectral_n_neighbors: int = 10,
    # GMM
    gmm_covariance_type: str = "full",
    # MeanShift
    meanshift_bandwidth: Optional[float] = None,
    # Birch
    birch_threshold: float = 0.5,
    birch_branching_factor: int = 50,
) -> np.ndarray:
    """Run a clustering algorithm on the embeddings. Returns cluster labels."""
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    # Scale features
    scaled = embeddings.copy()
    if scaling_method == "standard":
        scaled = StandardScaler().fit_transform(scaled)
    elif scaling_method == "robust":
        scaled = RobustScaler().fit_transform(scaled)
    elif scaling_method == "minmax":
        scaled = MinMaxScaler().fit_transform(scaled)

    if algorithm == "kmeans":
        from sklearn.cluster import KMeans

        model = KMeans(
            n_clusters=n_clusters,
            init=kmeans_init,
            n_init=kmeans_n_init,
            random_state=42,
        )
        return model.fit_predict(scaled)

    elif algorithm == "minibatch_kmeans":
        from sklearn.cluster import MiniBatchKMeans

        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=mbk_batch_size,
            random_state=42,
        )
        return model.fit_predict(scaled)

    elif algorithm == "dbscan":
        from sklearn.cluster import DBSCAN

        model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        return model.fit_predict(scaled)

    elif algorithm == "optics":
        from sklearn.cluster import OPTICS

        model = OPTICS(
            min_samples=optics_min_samples,
            xi=optics_xi,
            min_cluster_size=optics_min_cluster_size,
        )
        return model.fit_predict(scaled)

    elif algorithm == "hdbscan":
        from sklearn.cluster import HDBSCAN

        # Handle cluster_selection_epsilon (0.0 means None in sklearn's HDBSCAN)
        eps_param = (
            hdbscan_cluster_selection_epsilon
            if hdbscan_cluster_selection_epsilon > 0
            else 0.0
        )

        model = HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            cluster_selection_epsilon=eps_param,
            cluster_selection_method=hdbscan_cluster_selection_method,
            alpha=hdbscan_alpha,
        )
        return model.fit_predict(scaled)

    elif algorithm == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering

        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=agg_linkage)
        return model.fit_predict(scaled)

    elif algorithm == "spectral":
        from sklearn.cluster import SpectralClustering

        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity=spectral_affinity,
            n_neighbors=spectral_n_neighbors,
            random_state=42,
        )
        return model.fit_predict(scaled)

    elif algorithm == "gmm":
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type=gmm_covariance_type,
            random_state=42,
        )
        return model.fit_predict(scaled)

    elif algorithm == "meanshift":
        from sklearn.cluster import MeanShift

        kwargs = {}
        if meanshift_bandwidth is not None:
            kwargs["bandwidth"] = meanshift_bandwidth
        model = MeanShift(**kwargs)
        return model.fit_predict(scaled)

    elif algorithm == "birch":
        from sklearn.cluster import Birch

        model = Birch(
            n_clusters=n_clusters,
            threshold=birch_threshold,
            branching_factor=birch_branching_factor,
        )
        return model.fit_predict(scaled)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def _compute_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    scaling_method: str = "standard",
    annotation_labels: Optional[List[str]] = None,
) -> Dict:
    """Compute internal and external clustering evaluation metrics."""
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    scaled = embeddings.copy()
    if scaling_method == "standard":
        scaled = StandardScaler().fit_transform(scaled)
    elif scaling_method == "robust":
        scaled = RobustScaler().fit_transform(scaled)
    elif scaling_method == "minmax":
        scaled = MinMaxScaler().fit_transform(scaled)

    metrics = {}

    # Filter out noise for internal metrics
    valid_mask = labels != -1
    valid_labels = labels[valid_mask]
    valid_scaled = scaled[valid_mask]
    n_clusters = len(set(valid_labels))

    if n_clusters >= 2 and len(valid_scaled) > n_clusters:
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )

        metrics["silhouette"] = float(silhouette_score(valid_scaled, valid_labels))
        metrics["calinski_harabasz"] = float(
            calinski_harabasz_score(valid_scaled, valid_labels)
        )
        metrics["davies_bouldin"] = float(
            davies_bouldin_score(valid_scaled, valid_labels)
        )

    # External metrics (if annotations available)
    if annotation_labels is not None:
        valid_annotations = [
            annotation_labels[i] for i in range(len(labels)) if valid_mask[i]
        ]
        # Only compute if there are non-trivial annotations
        labeled_mask = [a != "no label" for a in valid_annotations]
        if sum(labeled_mask) > 0:
            from sklearn.metrics import (
                adjusted_rand_score,
                normalized_mutual_info_score,
            )

            filtered_labels = [
                valid_labels[i] for i in range(len(valid_labels)) if labeled_mask[i]
            ]
            filtered_annotations = [
                valid_annotations[i]
                for i in range(len(valid_annotations))
                if labeled_mask[i]
            ]
            if len(set(filtered_labels)) >= 2:
                metrics["adjusted_rand_index"] = float(
                    adjusted_rand_score(filtered_annotations, filtered_labels)
                )
                metrics["normalized_mutual_info"] = float(
                    normalized_mutual_info_score(filtered_annotations, filtered_labels)
                )

    # Count stats
    metrics["n_clusters"] = n_clusters
    metrics["n_noise"] = int(np.sum(~valid_mask))
    metrics["n_samples"] = len(labels)

    return metrics


def _compute_label_coherency(
    metadata: List[Dict],
    labels: np.ndarray,
) -> List[Dict]:
    """Compute label coherency statistics per cluster.

    Returns list of dicts with cluster_id, size, label_counts, purity,
    dominant_label, and entropy.
    """
    cluster_stats = []
    unique_clusters = sorted(set(labels))

    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        indices = np.where(mask)[0]
        size = len(indices)

        # Count annotation labels (excluding "no label")
        all_annotation_labels = [
            metadata[i].get("annotation_label", "no label") for i in indices
        ]
        labeled = [a for a in all_annotation_labels if a != "no label"]
        label_counts = dict(Counter(labeled))

        if labeled:
            most_common = Counter(labeled).most_common(1)[0]
            dominant_label = most_common[0]
            purity = most_common[1] / len(labeled)

            # Entropy
            probs = np.array(list(Counter(labeled).values())) / len(labeled)
            entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        else:
            dominant_label = "no label"
            purity = 0.0
            entropy = 0.0

        cluster_stats.append(
            {
                "cluster_id": cluster_id,
                "size": size,
                "label_counts": label_counts,
                "purity": purity,
                "dominant_label": dominant_label,
                "entropy": entropy,
                "n_labeled": len(labeled),
                "n_unlabeled": size - len(labeled),
            }
        )

    return cluster_stats


@st.cache_data(show_spinner=False, max_entries=10)
def _project_to_2d(
    embeddings: np.ndarray,
    method: str = "tsne",
    perplexity: int = 30,
    scaling_method: str = "standard",
) -> np.ndarray:
    """Project embeddings to 2D for visualization."""
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    scaled = embeddings.copy()
    if scaling_method == "standard":
        scaled = StandardScaler().fit_transform(scaled)
    elif scaling_method == "robust":
        scaled = RobustScaler().fit_transform(scaled)
    elif scaling_method == "minmax":
        scaled = MinMaxScaler().fit_transform(scaled)

    if method == "tsne":
        from sklearn.manifold import TSNE

        perp = min(perplexity, len(scaled) - 1)
        return TSNE(
            n_components=2, perplexity=perp, random_state=42, max_iter=1000
        ).fit_transform(scaled)
    elif method == "umap":
        try:
            import umap

            return umap.UMAP(n_components=2, n_jobs=32).fit_transform(scaled)
        except ImportError:
            st.warning("UMAP not installed, falling back to t-SNE")
            from sklearn.manifold import TSNE

            perp = min(perplexity, len(scaled) - 1)
            return TSNE(
                n_components=2, perplexity=perp, random_state=42, max_iter=1000
            ).fit_transform(scaled)
    else:
        raise ValueError(f"Unknown projection method: {method}")


# ---------------------------------------------------------------------------
# UI rendering
# ---------------------------------------------------------------------------


def _render_algorithm_params(
    demo_split: str, key_prefix_override: Optional[str] = None
) -> Dict:
    """Render algorithm selection and parameter widgets. Returns config dict."""
    key_prefix = (
        key_prefix_override if key_prefix_override is not None else f"clalg_{demo_split}"
    )

    algorithm = st.selectbox(
        "Clustering algorithm",
        options=[
            "kmeans",
            "minibatch_kmeans",
            "dbscan",
            "optics",
            "hdbscan",
            "agglomerative",
            "spectral",
            "gmm",
            "meanshift",
            "birch",
        ],
        format_func=lambda x: {
            "kmeans": "K-Means",
            "minibatch_kmeans": "Mini-Batch K-Means",
            "dbscan": "DBSCAN",
            "optics": "OPTICS",
            "hdbscan": "HDBSCAN",
            "agglomerative": "Agglomerative (Hierarchical)",
            "spectral": "Spectral Clustering",
            "gmm": "Gaussian Mixture Model",
            "meanshift": "Mean Shift",
            "birch": "BIRCH",
        }[x],
        key=f"{key_prefix}_algorithm",
    )

    params = {"algorithm": algorithm}

    # Algorithm descriptions and parameters
    if algorithm == "kmeans":
        st.caption(
            "Partitions data into K clusters by minimizing within-cluster variance. "
            "Fast and works well for spherical clusters."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            params["n_clusters"] = st.number_input(
                "Number of clusters (K)",
                min_value=2,
                max_value=50,
                value=5,
                key=f"{key_prefix}_km_k",
                help="Number of clusters to form.",
            )
        with col2:
            params["kmeans_init"] = st.selectbox(
                "Initialization",
                ["k-means++", "random"],
                key=f"{key_prefix}_km_init",
                help="k-means++: smart init that speeds convergence. random: random centroid selection.",
            )
        with col3:
            params["kmeans_n_init"] = st.number_input(
                "Number of initializations",
                min_value=1,
                max_value=20,
                value=10,
                key=f"{key_prefix}_km_ninit",
                help="Number of times to run with different seeds. Best result is kept.",
            )

    elif algorithm == "minibatch_kmeans":
        st.caption(
            "Faster variant of K-Means that uses mini-batches to reduce computation time. "
            "Good for large datasets (>10k samples)."
        )
        col1, col2 = st.columns(2)
        with col1:
            params["n_clusters"] = st.number_input(
                "Number of clusters (K)",
                min_value=2,
                max_value=50,
                value=5,
                key=f"{key_prefix}_mbk_k",
            )
        with col2:
            params["mbk_batch_size"] = st.number_input(
                "Batch size",
                min_value=100,
                max_value=10000,
                value=1024,
                key=f"{key_prefix}_mbk_batch",
                help="Size of mini-batches. Larger = more accurate but slower.",
            )

    elif algorithm == "dbscan":
        st.caption(
            "Density-based clustering that finds arbitrary-shaped clusters and detects noise/outliers. "
            "Does not require specifying number of clusters."
        )
        col1, col2 = st.columns(2)
        with col1:
            params["dbscan_eps"] = st.number_input(
                "Epsilon (eps)",
                min_value=0.01,
                max_value=50.0,
                value=0.5,
                step=0.1,
                format="%.2f",
                key=f"{key_prefix}_db_eps",
                help="Maximum distance between two samples in the same neighborhood. "
                "Smaller = tighter clusters, more noise points.",
            )
        with col2:
            params["dbscan_min_samples"] = st.number_input(
                "Min samples",
                min_value=2,
                max_value=50,
                value=5,
                key=f"{key_prefix}_db_min",
                help="Minimum number of samples in a neighborhood to form a core point. "
                "Higher = denser clusters required.",
            )

    elif algorithm == "optics":
        st.caption(
            "Improved DBSCAN that handles varying densities. "
            "Finds clusters of different densities without a fixed epsilon."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            params["optics_min_samples"] = st.number_input(
                "Min samples",
                min_value=2,
                max_value=50,
                value=5,
                key=f"{key_prefix}_op_min",
                help="Number of samples in a neighborhood for a core point.",
            )
        with col2:
            params["optics_xi"] = st.number_input(
                "Xi",
                min_value=0.01,
                max_value=0.5,
                value=0.05,
                step=0.01,
                format="%.2f",
                key=f"{key_prefix}_op_xi",
                help="Determines the minimum steepness on the reachability plot "
                "to define a cluster boundary. Smaller = more clusters.",
            )
        with col3:
            params["optics_min_cluster_size"] = st.number_input(
                "Min cluster size (fraction)",
                min_value=0.01,
                max_value=0.5,
                value=0.05,
                step=0.01,
                format="%.2f",
                key=f"{key_prefix}_op_mcs",
                help="Minimum number of samples in a cluster as fraction of total.",
            )

    elif algorithm == "hdbscan":
        st.caption(
            "Hierarchical density-based clustering that automatically determines the number of clusters. "
            "More robust than DBSCAN for varying densities and produces fewer noise points."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            params["hdbscan_min_cluster_size"] = st.number_input(
                "Min cluster size",
                min_value=2,
                max_value=500,
                value=5,
                key=f"{key_prefix}_hdb_mcs",
                help="The minimum number of samples in a group for that group to be considered a cluster.",
            )
        with col2:
            params["hdbscan_min_samples"] = st.number_input(
                "Min samples",
                min_value=1,
                max_value=100,
                value=5,
                key=f"{key_prefix}_hdb_min",
                help="Number of samples in a neighborhood for a point to be considered a core point. "
                "Higher = more conservative clustering. Use None to default to min_cluster_size.",
            )
        with col3:
            params["hdbscan_cluster_selection_epsilon"] = st.number_input(
                "Selection epsilon",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
                format="%.1f",
                key=f"{key_prefix}_hdb_eps",
                help="Distance threshold for cluster extraction. 0.0 = disabled (use cluster hierarchy). "
                "Increase to get more clusters at a specific distance scale.",
            )

        # Additional HDBSCAN parameters
        col4, col5 = st.columns(2)
        with col4:
            params["hdbscan_cluster_selection_method"] = st.selectbox(
                "Cluster selection method",
                ["eom", "leaf"],
                format_func=lambda x: {
                    "eom": "Excess of Mass (default)",
                    "leaf": "Leaf (more clusters)",
                }[x],
                key=f"{key_prefix}_hdb_method",
                help="eom: uses excess of mass (robust). leaf: selects leaf nodes (more clusters).",
            )
        with col5:
            params["hdbscan_alpha"] = st.number_input(
                "Alpha",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                format="%.1f",
                key=f"{key_prefix}_hdb_alpha",
                help="Distance scaling parameter. Larger = more focus on dense regions.",
            )

    elif algorithm == "agglomerative":
        st.caption(
            "Bottom-up hierarchical clustering that merges closest cluster pairs. "
            "Good for discovering hierarchical structure."
        )
        col1, col2 = st.columns(2)
        with col1:
            params["n_clusters"] = st.number_input(
                "Number of clusters",
                min_value=2,
                max_value=50,
                value=5,
                key=f"{key_prefix}_agg_k",
            )
        with col2:
            params["agg_linkage"] = st.selectbox(
                "Linkage",
                ["ward", "complete", "average", "single"],
                key=f"{key_prefix}_agg_link",
                help="ward: minimizes within-cluster variance (requires Euclidean). "
                "complete: max distance between clusters. "
                "average: mean distance. "
                "single: min distance (can produce elongated clusters).",
            )

    elif algorithm == "spectral":
        st.caption(
            "Uses graph Laplacian eigenvalues for clustering. "
            "Good for non-convex clusters and complex cluster shapes."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            params["n_clusters"] = st.number_input(
                "Number of clusters",
                min_value=2,
                max_value=50,
                value=5,
                key=f"{key_prefix}_spec_k",
            )
        with col2:
            params["spectral_affinity"] = st.selectbox(
                "Affinity",
                ["rbf", "nearest_neighbors"],
                key=f"{key_prefix}_spec_aff",
                help="rbf: Gaussian kernel (works well for most cases). "
                "nearest_neighbors: k-NN graph (better for non-uniformly distributed data).",
            )
        with col3:
            params["spectral_n_neighbors"] = st.number_input(
                "N neighbors",
                min_value=5,
                max_value=50,
                value=10,
                key=f"{key_prefix}_spec_nn",
                help="Number of neighbors for nearest_neighbors affinity.",
                disabled=params.get("spectral_affinity") != "nearest_neighbors",
            )

    elif algorithm == "gmm":
        st.caption(
            "Probabilistic model assuming data is generated from a mixture of Gaussians. "
            "Provides soft cluster assignments (probabilities)."
        )
        col1, col2 = st.columns(2)
        with col1:
            params["n_clusters"] = st.number_input(
                "Number of components",
                min_value=2,
                max_value=50,
                value=5,
                key=f"{key_prefix}_gmm_k",
            )
        with col2:
            params["gmm_covariance_type"] = st.selectbox(
                "Covariance type",
                ["full", "tied", "diag", "spherical"],
                key=f"{key_prefix}_gmm_cov",
                help="full: each component has its own covariance matrix. "
                "tied: all share one matrix. "
                "diag: diagonal covariance (axis-aligned ellipses). "
                "spherical: single variance per component.",
            )

    elif algorithm == "meanshift":
        st.caption(
            "Finds clusters by iterating towards density maxima. "
            "Automatically determines the number of clusters."
        )
        auto_bw = st.checkbox(
            "Auto-detect bandwidth",
            value=True,
            key=f"{key_prefix}_ms_auto",
            help="Automatically estimate bandwidth using sklearn's estimate_bandwidth.",
        )
        if not auto_bw:
            params["meanshift_bandwidth"] = st.number_input(
                "Bandwidth",
                min_value=0.1,
                max_value=50.0,
                value=1.0,
                step=0.1,
                format="%.1f",
                key=f"{key_prefix}_ms_bw",
                help="Radius of the kernel. Larger = fewer, broader clusters.",
            )
        else:
            params["meanshift_bandwidth"] = None

    elif algorithm == "birch":
        st.caption(
            "Scalable hierarchical clustering using a CF-tree. "
            "Efficient for very large datasets."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            params["n_clusters"] = st.number_input(
                "Number of clusters",
                min_value=2,
                max_value=50,
                value=5,
                key=f"{key_prefix}_birch_k",
            )
        with col2:
            params["birch_threshold"] = st.number_input(
                "Threshold",
                min_value=0.1,
                max_value=5.0,
                value=0.5,
                step=0.1,
                format="%.1f",
                key=f"{key_prefix}_birch_t",
                help="Radius of the subcluster in the CF tree. "
                "Smaller = more subclusters, finer granularity.",
            )
        with col3:
            params["birch_branching_factor"] = st.number_input(
                "Branching factor",
                min_value=10,
                max_value=200,
                value=50,
                key=f"{key_prefix}_birch_bf",
                help="Max number of subclusters per node in CF tree.",
            )

    return params


def _render_cluster_samples(
    cluster_id: int,
    cluster_indices: np.ndarray,
    metadata: List[Dict],
    representation: str,
    level: str,
    data: InfluenceData,
    demo_split: SplitType,
    cache_key: str,
    max_display: int = 6,
):
    """Render sample videos for a single cluster."""
    from influence_visualizer.render_frames import (
        frame_player,
        render_action_chunk,
        render_annotated_frame,
    )

    n_samples = len(cluster_indices)
    # Stable sampling
    sample_key = f"cluster_samples_{cache_key}_c{cluster_id}_{n_samples}_{max_display}"
    if sample_key not in st.session_state:
        if n_samples > max_display:
            st.session_state[sample_key] = list(
                np.random.choice(cluster_indices, size=max_display, replace=False)
            )
        else:
            st.session_state[sample_key] = list(cluster_indices)

    display_indices = st.session_state[sample_key]

    col_info, col_resample = st.columns([4, 1])
    with col_info:
        st.write(f"Showing {len(display_indices)} of {n_samples} samples")
    with col_resample:
        if n_samples > max_display:
            if st.button("Resample", key=f"resample_{cache_key}_c{cluster_id}"):
                st.session_state[sample_key] = list(
                    np.random.choice(cluster_indices, size=max_display, replace=False)
                )
                st.rerun()

    for i, point_idx in enumerate(display_indices):
        meta = metadata[point_idx]

        # Build title
        if level == "rollout":
            ep_idx = meta.get("rollout_idx", "?")
            success = meta.get("success", None)
            status = (
                "Success"
                if success
                else "Failure"
                if success is not None
                else "Unknown"
            )
        else:
            ep_idx = meta.get("demo_idx", "?")
            status = meta.get("quality_label", "unknown")

        if representation == "sliding_window":
            title = (
                f"#{i + 1}: {'Rollout' if level == 'rollout' else 'Demo'} {ep_idx} | "
                f"{status} | t={meta['window_start']}-{meta['window_end']}"
            )
        else:
            title = (
                f"#{i + 1}: {'Rollout' if level == 'rollout' else 'Demo'} {ep_idx} | "
                f"{status} | t={meta['timestep']}"
            )

        if meta.get("annotation_label", "no label") != "no label":
            title += f" | {meta['annotation_label']}"

        with st.expander(title, expanded=(i == 0)):
            st.caption(
                f"Mean influence: {meta.get('mean_influence', 0):.2f} | "
                f"Std: {meta.get('std_influence', 0):.2f}"
            )

            if level == "rollout":
                rollout_episodes = data.rollout_episodes
                ep_key = "rollout_idx"
                episode = next(
                    (ep for ep in rollout_episodes if ep.index == meta[ep_key]),
                    None,
                )
            else:
                # Get the appropriate demo episodes based on split
                if demo_split == "holdout":
                    demo_episodes = data.holdout_episodes
                else:
                    demo_episodes = data.demo_episodes
                ep_key = "demo_idx"
                episode = next(
                    (ep for ep in demo_episodes if ep.index == meta[ep_key]),
                    None,
                )
            if episode is None:
                st.warning(
                    f"Episode not found for {ep_key}={meta.get(ep_key)} "
                    f"(split={demo_split}). Skipping."
                )
                continue

            episode_length = episode.num_samples

            if representation == "sliding_window":
                window_start = meta["window_start"]
                window_end = meta["window_end"]
                default_t = window_start
            else:
                default_t = meta["timestep"]

            def _make_render_fn(
                _level, _episode, _meta, _representation, _cache_key, _point_idx
            ):
                """Create a render function with proper closure."""

                def _render_frame(t):
                    abs_idx = _episode.sample_start_idx + t
                    if _level == "rollout":
                        frame = data.get_rollout_frame(abs_idx)
                        action = data.get_rollout_action(abs_idx)
                    else:
                        frame = data.get_demo_frame(abs_idx)
                        action = data.get_demo_action(abs_idx)

                    # Highlight context
                    if _representation == "sliding_window":
                        ws = _meta["window_start"]
                        we = _meta["window_end"]
                        if ws <= t < we:
                            st.success(f"In window (t={ws}-{we - 1})")
                        else:
                            st.info(f"Outside window (t={ws}-{we - 1})")
                    else:
                        if t == _meta["timestep"]:
                            st.success(f"Selected timestep (t={_meta['timestep']})")

                    col_frame, col_action = st.columns([1, 1])
                    with col_frame:
                        ep_label = (
                            f"Rollout {_meta.get('rollout_idx', '?')}"
                            if _level == "rollout"
                            else f"Demo {_meta.get('demo_idx', '?')}"
                        )
                        render_annotated_frame(frame, f"t={t}", ep_label)
                    with col_action:
                        if action is not None:
                            render_action_chunk(
                                action[np.newaxis, :],
                                title=f"Action at t={t}",
                                unique_key=f"clalg_action_{_cache_key}_c{cluster_id}_{_point_idx}_{t}",
                            )

                return _render_frame

            render_fn = _make_render_fn(
                level, episode, meta, representation, cache_key, point_idx
            )

            frame_player(
                label="Timestep:",
                min_value=0,
                max_value=episode_length - 1,
                key=f"clalg_player_{cache_key}_c{cluster_id}_{point_idx}",
                default_value=default_t,
                default_fps=3.0,
                render_fn=render_fn,
                fragment_scope=True,
            )


@st.fragment
def _render_cluster_exploration_fragment(
    cluster_stats: List[Dict],
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    representation: str,
    level: str,
    data: InfluenceData,
    demo_split: SplitType,
    cache_key: str,
    key_prefix: str,
):
    """Isolated fragment for per-cluster video browsing.

    This is its own fragment so that frame_player reruns (fragment_scope=True)
    only re-execute this section, not the entire results/plots above.
    """
    st.subheader("Step 6: Explore Clusters")

    max_videos = st.slider(
        "Max videos per cluster",
        min_value=1,
        max_value=20,
        value=6,
        key=f"{key_prefix}_max_videos",
    )

    # Sort clusters by index (noise last)
    sorted_clusters = sorted(
        cluster_stats,
        key=lambda s: (s["cluster_id"] == -1, s["cluster_id"]),
    )

    for stat in sorted_clusters:
        cid = stat["cluster_id"]
        cluster_mask = cluster_labels == cid
        cluster_indices = np.where(cluster_mask)[0]

        label_str = (
            f"Noise ({stat['size']} samples)"
            if cid == -1
            else f"Cluster {cid} ({stat['size']} samples"
            + (
                f", purity={stat['purity']:.0%}, dominant={stat['dominant_label']}"
                if stat["n_labeled"] > 0
                else ""
            )
            + ")"
        )

        with st.expander(label_str, expanded=False):
            _render_cluster_samples(
                cluster_id=cid,
                cluster_indices=cluster_indices,
                metadata=metadata,
                representation=representation,
                level=level,
                data=data,
                demo_split=demo_split,
                cache_key=cache_key,
                max_display=max_videos,
            )


# ---------------------------------------------------------------------------
# Main fragment
# ---------------------------------------------------------------------------


@st.fragment
def _render_hierarchical_refinement_fragment(
    embeddings: np.ndarray,  # Embeddings used for main clustering (possibly already dim-reduced)
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    scaling_used: str,
    cache_key: str,
    key_prefix: str,
    demo_split: str = "train",
    embeddings_original: Optional[np.ndarray] = None,
):
    """Render hierarchical refinement section - re-cluster low-silhouette clusters.

    Identifies clusters with low average silhouette scores and re-clusters them
    separately. When embeddings_original is provided, refinement dimensionality
    reduction operates on the **original representation** (full influence vectors
    or pre-dim-red aggregated vectors), not on the already-reduced embeddings.
    Clustering is run only on the selected (low-silhouette) points.
    """
    st.subheader("Step 4: Hierarchical Refinement (Optional)")

    st.markdown("""
    **Problem**: When some clusters clearly separate but many points don't,
    they might operate at different feature scales.

    **Solution**: Re-cluster only the ambiguous points (low silhouette) separately.
    Dimensionality reduction (if enabled) is applied to the **original representation**
    of the selected points only—not to the already-reduced embedding space.
    You can choose any clustering algorithm (same options as the main menu).
    """)

    # Get silhouette score per sample (cached to avoid recomputing on every rerun)
    valid_mask = cluster_labels != -1
    if not np.any(valid_mask):
        st.warning("No valid clusters to refine (all noise).")
        return

    valid_labels = cluster_labels[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    if len(set(valid_labels)) < 2:
        st.warning("Need at least 2 clusters for refinement.")
        return

    refine_sil_key = f"{cache_key}_refine_sil_values"
    if refine_sil_key not in st.session_state:
        from sklearn.metrics import silhouette_samples
        from sklearn.preprocessing import (
            MinMaxScaler,
            RobustScaler,
            StandardScaler,
        )

        scaled = embeddings.copy()
        if scaling_used == "standard":
            scaled = StandardScaler().fit_transform(scaled)
        elif scaling_used == "robust":
            scaled = RobustScaler().fit_transform(scaled)
        elif scaling_used == "minmax":
            scaled = MinMaxScaler().fit_transform(scaled)

        valid_scaled = scaled[valid_mask]
        sil_values = silhouette_samples(valid_scaled, valid_labels)
        st.session_state[refine_sil_key] = sil_values
    else:
        sil_values = st.session_state[refine_sil_key]

    # Compute average silhouette per cluster
    unique_clusters = sorted(set(valid_labels))
    cluster_sil_scores = {}
    for cid in unique_clusters:
        cluster_mask = valid_labels == cid
        cluster_sil_scores[cid] = float(np.mean(sil_values[cluster_mask]))

    # UI for threshold selection
    col1, col2 = st.columns(2)
    with col1:
        sil_threshold = st.slider(
            "Silhouette threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key=f"{key_prefix}_sil_threshold",
            help="Clusters with average silhouette below this will be re-clustered.",
        )

    # Identify low-silhouette clusters
    low_sil_clusters = [
        cid for cid, score in cluster_sil_scores.items() if score < sil_threshold
    ]
    high_sil_clusters = [
        cid for cid, score in cluster_sil_scores.items() if score >= sil_threshold
    ]

    if not low_sil_clusters:
        st.info(
            f"✓ All clusters have silhouette ≥ {sil_threshold:.2f}. No refinement needed."
        )
        return

    n_low_samples = int(np.sum(np.isin(valid_labels, low_sil_clusters)))
    # Dimension of representation we'll use for refinement (original when available)
    source_embeddings_for_refine = (
        embeddings_original if embeddings_original is not None else embeddings
    )
    source_dim = source_embeddings_for_refine.shape[1]
    st.info(
        f"Found {len(low_sil_clusters)} cluster(s) with low silhouette, containing {n_low_samples} samples."
    )
    if embeddings_original is not None:
        st.caption(
            "Refinement will use the **original representation** "
            f"({source_dim} dimensions) for the selected points, not the already-reduced embeddings."
        )

    st.markdown("**Refinement options (applied only to selected points)**")

    # Dimensionality reduction on the subset only
    from influence_visualizer.render_clustering import (
        DIMRED_METHODS,
        _apply_dimensionality_reduction,
        _dimred_method_display_name,
    )

    try:
        import umap  # noqa: F401
        umap_available = True
    except ImportError:
        umap_available = False
    dimred_options_refine = (
        DIMRED_METHODS
        if umap_available
        else [m for m in DIMRED_METHODS if m != "umap"]
    )

    col_dimred1, col_dimred2, col_dimred3 = st.columns(3)
    with col_dimred1:
        apply_dimred_refine = st.checkbox(
            "Apply dimensionality reduction (on selected points only)",
            value=False,
            key=f"{key_prefix}_refine_apply_dimred",
            help="Run PCA, UMAP, etc. only on the low-silhouette subset before clustering.",
        )
    with col_dimred2:
        dimred_method_refine = st.selectbox(
            "Dim reduction method",
            dimred_options_refine,
            format_func=_dimred_method_display_name,
            key=f"{key_prefix}_refine_dimred_method",
            disabled=not apply_dimred_refine,
        )
    with col_dimred3:
        # Max components: at most source_dim, and for PCA/SVD at most n_samples-1
        max_comp_refine = min(source_dim, max(2, n_low_samples - 1))
        n_components_refine = st.number_input(
            "Reduced dimensions",
            min_value=2,
            max_value=max_comp_refine,
            value=min(20, max_comp_refine),
            key=f"{key_prefix}_refine_dimred_ncomp",
            disabled=not apply_dimred_refine,
        )

    if apply_dimred_refine and dimred_method_refine == "umap":
        col_umap1, col_umap2 = st.columns(2)
        with col_umap1:
            umap_n_neighbors_refine = st.number_input(
                "UMAP n_neighbors",
                min_value=2,
                max_value=min(200, n_low_samples - 1),
                value=min(15, n_low_samples - 1),
                key=f"{key_prefix}_refine_umap_nn",
            )
        with col_umap2:
            umap_min_dist_refine = st.slider(
                "UMAP min_dist",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                key=f"{key_prefix}_refine_umap_mindist",
            )
        umap_reproducible_refine = st.checkbox(
            "Reproducible UMAP (slower)",
            value=False,
            key=f"{key_prefix}_refine_umap_repro",
        )
        kernel_pca_kernel_refine = "rbf"
    elif apply_dimred_refine and dimred_method_refine == "kernel_pca":
        kernel_pca_kernel_refine = st.selectbox(
            "Kernel PCA kernel",
            ["rbf", "poly", "cosine", "sigmoid", "linear"],
            format_func=lambda x: {"rbf": "RBF", "poly": "Polynomial", "cosine": "Cosine", "sigmoid": "Sigmoid", "linear": "Linear"}[x],
            key=f"{key_prefix}_refine_kernel_pca_kernel",
            help="Kernel function for Kernel PCA.",
        )
        umap_n_neighbors_refine = 15
        umap_min_dist_refine = 0.1
        umap_reproducible_refine = False
    else:
        kernel_pca_kernel_refine = "rbf"
        umap_n_neighbors_refine = 15
        umap_min_dist_refine = 0.1
        umap_reproducible_refine = False

    # Full clustering algorithm options (same as main menu)
    st.markdown("**Clustering algorithm**")
    params_refine = _render_algorithm_params(
        demo_split, key_prefix_override=f"{key_prefix}_refine"
    )

    if st.button("Run Hierarchical Refinement", key=f"{key_prefix}_refine_btn"):
        # Extract samples from low-silhouette clusters.
        # Use original representation when available so refinement dim red operates on full vectors.
        low_sil_mask = np.isin(valid_labels, low_sil_clusters)
        refine_indices = valid_indices[low_sil_mask]
        source_embeddings = (
            embeddings_original if embeddings_original is not None else embeddings
        )
        refine_embeddings = source_embeddings[refine_indices].copy()
        refine_metadata = [metadata[i] for i in refine_indices]

        with st.spinner(f"Re-clustering {len(refine_indices)} samples..."):
            # Optional dimensionality reduction on the subset only
            if apply_dimred_refine:
                n_comp = min(
                    n_components_refine,
                    refine_embeddings.shape[0],
                    refine_embeddings.shape[1],
                )
                refine_embeddings = _apply_dimensionality_reduction(
                    refine_embeddings,
                    method=dimred_method_refine,
                    n_components=n_comp,
                    umap_n_neighbors=umap_n_neighbors_refine,
                    umap_min_dist=umap_min_dist_refine,
                    umap_reproducible=umap_reproducible_refine,
                    scaling_method="standard",
                    n_neighbors=min(15, refine_embeddings.shape[0] - 1),
                    kernel_pca_kernel=kernel_pca_kernel_refine,
                )

            # Run clustering (same API as main menu)
            refined_labels = _run_clustering(
                refine_embeddings,
                algorithm=params_refine["algorithm"],
                scaling_method=scaling_used,
                n_clusters=params_refine.get("n_clusters", 5),
                kmeans_init=params_refine.get("kmeans_init", "k-means++"),
                kmeans_n_init=params_refine.get("kmeans_n_init", 10),
                mbk_batch_size=params_refine.get("mbk_batch_size", 1024),
                dbscan_eps=params_refine.get("dbscan_eps", 0.5),
                dbscan_min_samples=params_refine.get("dbscan_min_samples", 5),
                optics_min_samples=params_refine.get("optics_min_samples", 5),
                optics_xi=params_refine.get("optics_xi", 0.05),
                optics_min_cluster_size=params_refine.get(
                    "optics_min_cluster_size", 0.05
                ),
                hdbscan_min_cluster_size=params_refine.get(
                    "hdbscan_min_cluster_size", 5
                ),
                hdbscan_min_samples=params_refine.get("hdbscan_min_samples", 5),
                hdbscan_cluster_selection_epsilon=params_refine.get(
                    "hdbscan_cluster_selection_epsilon", 0.0
                ),
                hdbscan_cluster_selection_method=params_refine.get(
                    "hdbscan_cluster_selection_method", "eom"
                ),
                hdbscan_alpha=params_refine.get("hdbscan_alpha", 1.0),
                agg_linkage=params_refine.get("agg_linkage", "ward"),
                spectral_affinity=params_refine.get("spectral_affinity", "rbf"),
                spectral_n_neighbors=params_refine.get(
                    "spectral_n_neighbors", 10
                ),
                gmm_covariance_type=params_refine.get(
                    "gmm_covariance_type", "full"
                ),
                meanshift_bandwidth=params_refine.get("meanshift_bandwidth"),
                birch_threshold=params_refine.get("birch_threshold", 0.5),
                birch_branching_factor=params_refine.get(
                    "birch_branching_factor", 50
                ),
            )

        n_refined_clusters = len(set(refined_labels) - {-1})
        refine_cache_key = f"{cache_key}_refined"
        # Clear 2D projection cache so it is recomputed for new refinement
        proj_key = f"{refine_cache_key}_proj"
        if proj_key in st.session_state:
            del st.session_state[proj_key]
        # Hide video exploration until user regenerates (so clusters are not outdated)
        st.session_state.pop(f"{cache_key}_video_exploration_shown", None)

        st.session_state[f"{refine_cache_key}_embeddings"] = refine_embeddings
        st.session_state[f"{refine_cache_key}_metadata"] = refine_metadata
        st.session_state[f"{refine_cache_key}_labels"] = refined_labels
        st.session_state[f"{refine_cache_key}_high_sil_clusters"] = high_sil_clusters
        st.session_state[f"{refine_cache_key}_low_sil_clusters"] = low_sil_clusters
        st.session_state[f"{refine_cache_key}_original_indices"] = refine_indices
        st.session_state[f"{refine_cache_key}_scaling"] = scaling_used
        st.session_state[f"{refine_cache_key}_dimred_used"] = apply_dimred_refine
        st.session_state[f"{refine_cache_key}_algorithm"] = params_refine["algorithm"]

        st.success(
            f"✓ Refinement complete: {n_refined_clusters} refined clusters from {len(low_sil_clusters)} original clusters"
        )

    # Display refinement results if available
    refine_cache_key = f"{cache_key}_refined"
    if f"{refine_cache_key}_labels" in st.session_state:
        st.markdown("---")
        st.markdown("### Refinement Results")

        refine_embeddings = st.session_state[f"{refine_cache_key}_embeddings"]
        refine_metadata = st.session_state[f"{refine_cache_key}_metadata"]
        refined_labels = st.session_state[f"{refine_cache_key}_labels"]
        high_sil_clusters = st.session_state[f"{refine_cache_key}_high_sil_clusters"]
        refine_scaling = st.session_state[f"{refine_cache_key}_scaling"]
        refine_algorithm = st.session_state.get(
            f"{refine_cache_key}_algorithm", "?"
        )
        refine_dimred_used = st.session_state.get(
            f"{refine_cache_key}_dimred_used", False
        )

        n_refined_clusters = len(set(refined_labels) - {-1})

        # Compute metrics for refined clustering
        if refine_scaling == "standard":
            refine_scaled = StandardScaler().fit_transform(refine_embeddings)
        elif refine_scaling == "robust":
            refine_scaled = RobustScaler().fit_transform(refine_embeddings)
        elif refine_scaling == "minmax":
            refine_scaled = MinMaxScaler().fit_transform(refine_embeddings)
        else:
            refine_scaled = refine_embeddings

        from sklearn.metrics import silhouette_score

        refined_sil_score = silhouette_score(refine_scaled, refined_labels)

        st.metric(
            "Refined Silhouette Score",
            f"{refined_sil_score:.3f}",
            help="Silhouette score of the refined clustering on the low-silhouette samples",
        )
        st.caption(
            f"Algorithm: {refine_algorithm} · "
            f"Dimensions: {refine_embeddings.shape[1]}"
            + (" (after dim red on subset)" if refine_dimred_used else "")
        )

        # Visualize refined clustering
        st.markdown("**Refined Clustering Visualization**")

        # Project to 2D (only on the refined subset)
        from sklearn.manifold import TSNE

        proj_key = f"{refine_cache_key}_proj"
        if proj_key not in st.session_state:
            with st.spinner("Computing 2D projection (selected points only)..."):
                perplexity = min(30, len(refine_scaled) // 3)
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                embeddings_2d = tsne.fit_transform(refine_scaled)
                st.session_state[proj_key] = embeddings_2d
        else:
            embeddings_2d = st.session_state[proj_key]

        color_by_refined = st.selectbox(
            "Color points by",
            ["cluster", "success", "timestep", "quality", "rollout_idx", "demo_idx"],
            format_func=lambda x: {
                "cluster": "Cluster",
                "success": "Success/Failure (rollout)",
                "timestep": "Timestep",
                "quality": "Demonstration quality",
                "rollout_idx": "Rollout index",
                "demo_idx": "Demonstration index",
            }[x],
            key=f"{key_prefix}_color_refined",
            help="Color the refined 2D projection by cluster or metadata.",
        )
        fig_refined = plotting.create_cluster_scatter_2d(
            embeddings_2d,
            refined_labels,
            refine_metadata,
            title=f"Refined Clustering ({n_refined_clusters} clusters, {refine_algorithm})",
            color_by=color_by_refined,
        )
        st.plotly_chart(fig_refined, width='stretch')

        # Silhouette plot for refined clustering
        refined_sil_values = silhouette_samples(refine_scaled, refined_labels)
        fig_refined_sil = plotting.create_silhouette_plot(
            refined_sil_values,
            refined_labels,
            title="Refined Clustering Silhouette Analysis",
        )
        st.plotly_chart(fig_refined_sil, width='stretch')

        # Original 2D plot with updated cluster assignments (high-sil kept, refined subset updated)
        main_2d_key = f"{cache_key}_embeddings_2d"
        if main_2d_key in st.session_state:
            st.markdown("**Original 2D projection with updated cluster assignments**")
            embeddings_2d_main = st.session_state[main_2d_key]
            refine_indices = st.session_state[f"{refine_cache_key}_original_indices"]
            # Merge: keep original labels for high-silhouette points; assign refined labels (offset) for refined subset
            merged_labels = np.array(cluster_labels, copy=True)
            offset = int(np.max(merged_labels)) + 1 if np.any(merged_labels >= 0) else 0
            for j, idx in enumerate(refine_indices):
                merged_labels[idx] = (
                    refined_labels[j] + offset if refined_labels[j] != -1 else -1
                )
            color_by_merged = st.selectbox(
                "Color points by",
                ["cluster", "success", "timestep", "quality", "rollout_idx", "demo_idx"],
                format_func=lambda x: {
                    "cluster": "Cluster",
                    "success": "Success/Failure (rollout)",
                    "timestep": "Timestep",
                    "quality": "Demonstration quality",
                    "rollout_idx": "Rollout index",
                    "demo_idx": "Demonstration index",
                }[x],
                key=f"{key_prefix}_color_merged",
                help="Color the original 2D projection by cluster or metadata.",
            )
            fig_merged = plotting.create_cluster_scatter_2d(
                embeddings_2d_main,
                merged_labels,
                metadata,
                title="Original projection with refined cluster assignments",
                color_by=color_by_merged,
            )
            st.plotly_chart(fig_merged, width='stretch')
            st.caption(
                "Same 2D coordinates as the main clustering plot; points from low-silhouette "
                "clusters now show their refined sub-cluster (offset labels)."
            )

        st.markdown("""
        **Next steps:**
        - Compare the refined silhouette scores with the original
        - Check if the refined clusters make more sense for the ambiguous samples
        - Consider the high-silhouette clusters as "well-separated phases"
        - Use the refined clusters to understand sub-behaviors within ambiguous regions
        """)


@st.fragment
def _render_temporal_coherence_fragment(
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    representation: str,
    level: str,
    data: InfluenceData,
    demo_split: SplitType,
    cache_key: str,
    key_prefix: str,
    annotations: Optional[Dict] = None,
):
    """Render a paginated episode browser showing temporal cluster assignments.

    This allows users to browse through episodes and see how cluster assignments
    change over time, helping identify temporal coherence patterns. Also shows
    a second bar with human annotations (or "Unassigned" where not annotated).
    """
    from influence_visualizer.render_annotation import get_episode_annotations

    st.subheader("Step 5: Browse Temporal Coherence")

    st.markdown(
        f"Browse through {'rollouts' if level == 'rollout' else 'demonstrations'} "
        "to see how cluster assignments evolve over time within each episode."
    )

    # Build mapping from episode to cluster assignments
    episode_cluster_map = {}

    if representation == "sliding_window":
        # For sliding windows, map each window to its episode and position
        for i, meta in enumerate(metadata):
            ep_key = f"{level}_idx"
            ep_idx = meta[ep_key]
            cluster_id = cluster_labels[i]
            window_start = meta["window_start"]
            window_end = meta["window_end"]

            if ep_idx not in episode_cluster_map:
                episode_cluster_map[ep_idx] = []

            episode_cluster_map[ep_idx].append(
                {
                    "window_start": window_start,
                    "window_end": window_end,
                    "cluster": cluster_id,
                }
            )
    else:  # individual timesteps
        # For timesteps, each timestep has a cluster assignment
        for i, meta in enumerate(metadata):
            ep_key = f"{level}_idx"
            ep_idx = meta[ep_key]
            cluster_id = cluster_labels[i]
            timestep = meta["timestep"]

            if ep_idx not in episode_cluster_map:
                episode_cluster_map[ep_idx] = []

            episode_cluster_map[ep_idx].append(
                {
                    "timestep": timestep,
                    "cluster": cluster_id,
                }
            )

    if not episode_cluster_map:
        st.warning("No episodes to display.")
        return

    # Get all unique cluster IDs globally (for consistent coloring across episodes)
    all_cluster_ids = sorted(set(cluster_labels))

    # Sort episodes by index
    sorted_episodes = sorted(episode_cluster_map.keys())

    # Pagination
    episodes_per_page = st.slider(
        "Episodes per page",
        min_value=1,
        max_value=20,
        value=5,
        key=f"{key_prefix}_episodes_per_page",
    )

    total_pages = (len(sorted_episodes) + episodes_per_page - 1) // episodes_per_page

    col_page1, col_page2, col_page3 = st.columns([1, 2, 1])
    with col_page2:
        current_page = st.number_input(
            f"Page (1-{total_pages})",
            min_value=1,
            max_value=total_pages,
            value=1,
            key=f"{key_prefix}_page_num",
        )

    # Get episodes for current page
    start_idx = (current_page - 1) * episodes_per_page
    end_idx = min(start_idx + episodes_per_page, len(sorted_episodes))
    page_episodes = sorted_episodes[start_idx:end_idx]

    st.markdown(
        f"**Showing episodes {start_idx + 1}-{end_idx} of {len(sorted_episodes)}**"
    )

    # Render each episode
    for ep_idx in page_episodes:
        # Get episode data
        if level == "rollout":
            episodes = data.rollout_episodes
            ep = episodes[ep_idx]
            ep_label = f"Rollout {ep_idx}"
            status = "Success" if ep.success else "Failure"
        else:
            if demo_split == "holdout":
                episodes = data.holdout_episodes
            else:
                episodes = data.demo_episodes
            ep = episodes[ep_idx]
            ep_label = f"Demo {ep_idx}"
            status = (
                data.demo_quality_labels.get(ep_idx, "unknown")
                if data.demo_quality_labels
                else "unknown"
            )

        num_frames = ep.num_samples

        # Build cluster assignment array for timeline
        # Initialize with -1 (unassigned) - these are frames not included in clustering
        cluster_timeline = [-1] * num_frames

        if representation == "sliding_window":
            # For sliding windows, assign cluster to all frames in the window
            for window_info in episode_cluster_map[ep_idx]:
                for t in range(window_info["window_start"], window_info["window_end"]):
                    if t < num_frames:
                        cluster_timeline[t] = window_info["cluster"]
        else:
            # For timesteps, direct assignment
            for ts_info in episode_cluster_map[ep_idx]:
                t = ts_info["timestep"]
                if t < num_frames:
                    cluster_timeline[t] = ts_info["cluster"]

        # Check if there are unassigned frames
        num_unassigned = sum(1 for c in cluster_timeline if c == -1)
        has_true_noise = any(
            c == -1 for c in cluster_labels
        )  # Check if algorithm produced noise

        # Human annotations for this episode (for second bar)
        episode_annotations = []
        if annotations:
            ann_split = "rollout" if level == "rollout" else demo_split
            episode_annotations = get_episode_annotations(
                annotations, str(ep_idx), split=ann_split
            )

        # Create expander for this episode
        with st.expander(
            f"{ep_label} | {status} | {num_frames} frames", expanded=False
        ):
            # Add info about unassigned frames if present
            if num_unassigned > 0 and not has_true_noise:
                st.caption(
                    f"ℹ️ {num_unassigned}/{num_frames} frames are unassigned "
                    "(not included in clustering sample). "
                    "This is normal for sliding windows with stride > 1 or sampled timesteps."
                )

            # Add frame player
            from influence_visualizer.render_frames import (
                frame_player,
                render_action_chunk,
                render_annotated_frame,
            )

            def _make_render_fn(
                _level,
                _episode,
                _cluster_timeline,
                _ep_idx,
                _cache_key,
                _has_noise,
                _num_frames,
                _all_cluster_ids,
                _episode_annotations,
            ):
                """Create a render function with proper closure."""

                def _render_frame(t):
                    # Automatic cluster assignments bar
                    st.caption("Automatic cluster assignments")
                    fig_timeline = plotting.create_cluster_timeline(
                        cluster_assignments=_cluster_timeline,
                        num_frames=_num_frames,
                        current_frame=t,
                        has_true_noise=_has_noise,
                        all_cluster_ids=_all_cluster_ids,
                    )
                    st.plotly_chart(
                        fig_timeline,
                        width='stretch',
                        key=f"timeline_{_cache_key}_{_ep_idx}_{t}",
                    )

                    # Human annotations bar
                    st.caption("Human annotations")
                    fig_annotations = plotting.create_label_timeline(
                        _episode_annotations,
                        _num_frames,
                        current_frame=t,
                        unlabeled_name="Unassigned",
                    )
                    st.plotly_chart(
                        fig_annotations,
                        width='stretch',
                        key=f"annotations_timeline_{_cache_key}_{_ep_idx}_{t}",
                    )

                    abs_idx = _episode.sample_start_idx + t
                    if _level == "rollout":
                        frame = data.get_rollout_frame(abs_idx)
                        action = data.get_rollout_action(abs_idx)
                    else:
                        frame = data.get_demo_frame(abs_idx)
                        action = data.get_demo_action(abs_idx)

                    # Show cluster assignment for this frame
                    cluster_id = _cluster_timeline[t]
                    if cluster_id == -1:
                        if _has_noise:
                            st.info(f"Frame t={t} | Cluster: Noise (from algorithm)")
                        else:
                            st.warning(
                                f"Frame t={t} | Unassigned (not in clustering sample)"
                            )
                    else:
                        st.success(f"Frame t={t} | Cluster: {cluster_id}")

                    col_frame, col_action = st.columns([1, 1])
                    with col_frame:
                        ep_label_str = (
                            f"{'Rollout' if _level == 'rollout' else 'Demo'} {_ep_idx}"
                        )
                        render_annotated_frame(frame, f"t={t}", ep_label_str)
                    with col_action:
                        if action is not None:
                            render_action_chunk(
                                action[np.newaxis, :],
                                title=f"Action at t={t}",
                                unique_key=f"temporal_action_{_cache_key}_{_ep_idx}_{t}",
                            )

                return _render_frame

            render_fn = _make_render_fn(
                level,
                ep,
                cluster_timeline,
                ep_idx,
                cache_key,
                has_true_noise,
                num_frames,
                all_cluster_ids,
                episode_annotations,
            )

            frame_player(
                label="Timestep:",
                min_value=0,
                max_value=num_frames - 1,
                key=f"temporal_player_{cache_key}_{ep_idx}",
                default_value=0,
                default_fps=3.0,
                render_fn=render_fn,
                fragment_scope=True,
            )


@st.fragment
def render_cluster_algorithms_fragment(
    data: InfluenceData,
    demo_split: SplitType = "train",
    annotations: Optional[Dict] = None,
    task_config: Optional[str] = None,
):
    """Main fragment for clustering algorithms section."""
    with st.expander("7. Clustering Algorithms", expanded=False):
        st.markdown("""
        Apply clustering algorithms to influence embeddings and evaluate cluster quality
        against human annotations. Unlike the t-SNE sections above which only project to 2D,
        this section clusters in the full embedding space and then projects for visualization.
        """)

        key_prefix = f"clalg_{demo_split}"

        # ---- Load saved result ----
        from influence_visualizer.clustering_results import (
            list_clustering_results,
            load_clustering_result,
        )

        if task_config:
            saved_names = list_clustering_results(task_config)
            if saved_names:
                load_col1, load_col2 = st.columns([2, 1])
                with load_col1:
                    selected_saved = st.selectbox(
                        "Saved result",
                        options=saved_names,
                        key=f"{key_prefix}_load_saved_select",
                        help="Load a previously saved clustering result to view it here.",
                    )
                with load_col2:
                    if st.button("Load", key=f"{key_prefix}_load_saved_btn"):
                        try:
                            cluster_labels, metadata, manifest = load_clustering_result(
                                task_config, selected_saved
                            )
                            st.session_state["clalg_loaded_labels"] = cluster_labels
                            st.session_state["clalg_loaded_metadata"] = metadata
                            st.session_state["clalg_loaded_manifest"] = manifest
                            st.session_state["clalg_loaded_name"] = selected_saved
                            st.success(f"Loaded: {selected_saved}")
                        except FileNotFoundError as e:
                            st.error(str(e))
                        except Exception as e:
                            st.error(f"Load failed: {e}")
                if "clalg_loaded_labels" in st.session_state:
                    if st.button("Clear loaded result", key=f"{key_prefix}_clear_loaded_btn"):
                        for key in (
                            "clalg_loaded_labels",
                            "clalg_loaded_metadata",
                            "clalg_loaded_manifest",
                            "clalg_loaded_name",
                        ):
                            st.session_state.pop(key, None)
                        st.rerun()
            else:
                st.caption(
                    "No saved results. Run clustering and save a result first."
                )
        else:
            st.caption(
                "Select a task config in the app to load saved clustering results."
            )

        st.divider()

        # ---- Step 1: Influence source and representation ----
        st.subheader("Step 1: Select Representation")

        influence_source = st.radio(
            "Influence source",
            ["trak", "infembed"],
            format_func=lambda x: {
                "trak": "TRAK influence explanations",
                "infembed": "InfEmbed influence embeddings",
            }[x],
            key=f"{key_prefix}_influence_source",
            help="TRAK: influence scores from the TRAK method. "
            "InfEmbed: precomputed low-dimensional influence embeddings (run script first).",
            horizontal=True,
        )

        col_repr, col_level = st.columns(2)
        with col_repr:
            representation = st.selectbox(
                "Representation",
                ["sliding_window", "timestep"],
                format_func=lambda x: {
                    "sliding_window": "Sliding Windows",
                    "timestep": "Individual Timesteps",
                }[x],
                key=f"{key_prefix}_repr",
                help="Sliding windows: aggregate over temporal slices. "
                "Timesteps: each timestep is a separate data point.",
            )
        with col_level:
            level = st.selectbox(
                "Cluster",
                ["rollout", "demo"],
                format_func=lambda x: "Rollouts"
                if x == "rollout"
                else "Demonstrations",
                key=f"{key_prefix}_level",
                help="Whether to cluster rollout or demonstration data.",
            )

        # Representation-specific parameters
        is_infembed = influence_source == "infembed"
        if is_infembed and representation == "timestep":
            st.info(
                "Use precomputed **InfEmbed** low-dimensional influence embeddings (per-timestep). "
                "Run the InfEmbed computation script first; see **INFEMBED_INTEGRATION_PLAN.md**."
            )
            # Optional dimensionality reduction (same as sliding window second layer)
            try:
                import umap as _umap_infembed

                _umap_available_infembed = True
            except ImportError:
                _umap_available_infembed = False

            st.markdown("**Optional dimensionality reduction**")
            from influence_visualizer.render_clustering import (
                DIMRED_METHODS as _DIMRED_METHODS_IF,
                _dimred_method_display_name as _dimred_display_if,
            )

            _dimred_options_if = (
                _DIMRED_METHODS_IF
                if _umap_available_infembed
                else [m for m in _DIMRED_METHODS_IF if m != "umap"]
            )
            _col1, _col2, _col3 = st.columns(3)
            with _col1:
                apply_dimred_infembed = st.checkbox(
                    "Apply dimensionality reduction",
                    value=False,
                    key=f"{key_prefix}_infembed_apply_dimred",
                    help="Apply PCA, UMAP, etc. to InfEmbed vectors before clustering.",
                )
            with _col2:
                dimred_method_infembed = st.selectbox(
                    "Dim reduction method",
                    _dimred_options_if,
                    format_func=_dimred_display_if,
                    key=f"{key_prefix}_infembed_dimred_method",
                    disabled=not apply_dimred_infembed,
                )
            with _col3:
                n_components_dimred_infembed = st.number_input(
                    "Reduced dimensions",
                    min_value=2,
                    max_value=100,
                    value=50,
                    key=f"{key_prefix}_infembed_dimred_ncomp",
                    disabled=not apply_dimred_infembed,
                )

            if apply_dimred_infembed and dimred_method_infembed == "umap":
                _cu1, _cu2, _cu3 = st.columns(3)
                with _cu1:
                    umap_n_neighbors_infembed = st.number_input(
                        "UMAP n_neighbors",
                        min_value=2,
                        max_value=100,
                        value=15,
                        key=f"{key_prefix}_infembed_umap_neighbors",
                    )
                with _cu2:
                    umap_min_dist_infembed = st.number_input(
                        "UMAP min_dist",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.1,
                        step=0.05,
                        format="%.2f",
                        key=f"{key_prefix}_infembed_umap_mindist",
                    )
                with _cu3:
                    umap_reproducible_infembed = st.checkbox(
                        "Reproducible",
                        value=False,
                        key=f"{key_prefix}_infembed_umap_repro",
                    )
                kernel_pca_kernel_infembed = "rbf"
            elif apply_dimred_infembed and dimred_method_infembed == "kernel_pca":
                kernel_pca_kernel_infembed = st.selectbox(
                    "Kernel PCA kernel",
                    ["rbf", "poly", "cosine", "sigmoid", "linear"],
                    format_func=lambda x: {
                        "rbf": "RBF",
                        "poly": "Polynomial",
                        "cosine": "Cosine",
                        "sigmoid": "Sigmoid",
                        "linear": "Linear",
                    }[x],
                    key=f"{key_prefix}_infembed_kernel_pca_kernel",
                )
                umap_n_neighbors_infembed = 15
                umap_min_dist_infembed = 0.1
                umap_reproducible_infembed = False
            else:
                kernel_pca_kernel_infembed = "rbf"
                umap_n_neighbors_infembed = 15
                umap_min_dist_infembed = 0.1
                umap_reproducible_infembed = False
        elif representation == "sliding_window":
            st.markdown("**First Layer: Sliding Window Aggregation**")

            # Add explanatory note based on level and influence source
            if is_infembed:
                st.info(
                    "**InfEmbed sliding windows:**\n"
                    "- Window slides along **timesteps** within each episode\n"
                    "- Each window: (window_width, D) embeddings → aggregate to (D,) or (n_components,) for SVD\n"
                    "- Same aggregation options as TRAK (mean, max, SVD, etc.)"
                )
            elif level == "rollout":
                st.info(
                    "**Window behavior for rollouts:**\n"
                    "- Window slides along **rollout timesteps**\n"
                    "- Each window shape: (window_width, num_demo_samples)\n"
                    "- Aggregation: along rollout timesteps → output vector of length num_demo_samples\n"
                    "- Result: Each window describes which demos influenced that rollout phase"
                )
            else:  # demo
                st.info(
                    "**Window behavior for demos:**\n"
                    "- Window slides along **demo timesteps**\n"
                    "- Each window shape: (num_rollout_samples, window_width)\n"
                    "- Aggregation: along demo timesteps → output vector of length num_rollout_samples\n"
                    "- Result: Each window describes how that demo phase influenced rollouts"
                )

            col1, col2, col3 = st.columns(3)
            with col1:
                window_label = f"Window width ({level} timesteps)"
                window_width = st.number_input(
                    window_label,
                    min_value=2,
                    max_value=50,
                    value=10,
                    key=f"{key_prefix}_sw_width",
                    help=f"Number of {level} timesteps in each sliding window.",
                )
            with col2:
                stride = st.number_input(
                    "Stride",
                    min_value=1,
                    max_value=20,
                    value=1,
                    key=f"{key_prefix}_sw_stride",
                    help=f"Step size for sliding the window along {level} timesteps.",
                )
            with col3:
                agg_help = (
                    f"How to aggregate the window along {level} timesteps. "
                    f"Result: one vector per window."
                )
                aggregation = st.selectbox(
                    "Aggregation method",
                    ["svd", "mean", "max", "min", "std", "sum", "median"],
                    key=f"{key_prefix}_sw_agg",
                    help=agg_help,
                )
            if aggregation == "svd":
                n_components_svd = st.number_input(
                    "Number of singular values",
                    min_value=2,
                    max_value=50,
                    value=10,
                    key=f"{key_prefix}_sw_ncomp",
                )
            else:
                n_components_svd = 10

            # Second layer: optional dimensionality reduction
            st.markdown("**Second Layer: Dimensionality Reduction (Optional)**")

            try:
                import umap

                umap_available = True
            except ImportError:
                umap_available = False

            col_dimred1, col_dimred2, col_dimred3 = st.columns(3)
            with col_dimred1:
                apply_dimred = st.checkbox(
                    "Apply dimensionality reduction",
                    value=False,
                    key=f"{key_prefix}_sw_apply_dimred",
                    help="Apply dimensionality reduction (PCA, UMAP, MDS, Isomap, etc.) to the aggregated 1D vectors.",
                )
            with col_dimred2:
                from influence_visualizer.render_clustering import (
                    DIMRED_METHODS,
                    _dimred_method_display_name,
                )

                dimred_options_sw = (
                    DIMRED_METHODS if umap_available else [m for m in DIMRED_METHODS if m != "umap"]
                )
                dimred_method_sw = st.selectbox(
                    "Dim reduction method",
                    dimred_options_sw,
                    format_func=_dimred_method_display_name,
                    key=f"{key_prefix}_sw_dimred_method",
                    help="Dimensionality reduction applied to aggregated vectors.",
                    disabled=not apply_dimred,
                )
            with col_dimred3:
                n_components_dimred_sw = st.number_input(
                    "Reduced dimensions",
                    min_value=2,
                    max_value=100,
                    value=50,
                    key=f"{key_prefix}_sw_dimred_ncomp",
                    help="Number of dimensions after reduction.",
                    disabled=not apply_dimred,
                )

            # UMAP-specific parameters for sliding window
            if apply_dimred and dimred_method_sw == "umap":
                col_umap1, col_umap2, col_umap3 = st.columns(3)
                with col_umap1:
                    umap_n_neighbors_sw = st.number_input(
                        "UMAP n_neighbors",
                        min_value=2,
                        max_value=100,
                        value=15,
                        key=f"{key_prefix}_sw_umap_neighbors",
                        help="Number of neighbors for UMAP graph construction.",
                    )
                with col_umap2:
                    umap_min_dist_sw = st.number_input(
                        "UMAP min_dist",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.1,
                        step=0.05,
                        format="%.2f",
                        key=f"{key_prefix}_sw_umap_mindist",
                        help="Minimum distance between points in embedding.",
                    )
                with col_umap3:
                    umap_reproducible_sw = st.checkbox(
                        "Reproducible",
                        value=False,
                        key=f"{key_prefix}_sw_umap_repro",
                        help="Use random_state for reproducibility (slower, single-threaded).",
                    )
                kernel_pca_kernel_sw = "rbf"
            elif apply_dimred and dimred_method_sw == "kernel_pca":
                kernel_pca_kernel_sw = st.selectbox(
                    "Kernel PCA kernel",
                    ["rbf", "poly", "cosine", "sigmoid", "linear"],
                    format_func=lambda x: {"rbf": "RBF", "poly": "Polynomial", "cosine": "Cosine", "sigmoid": "Sigmoid", "linear": "Linear"}[x],
                    key=f"{key_prefix}_sw_kernel_pca_kernel",
                    help="Kernel function for Kernel PCA.",
                )
                umap_n_neighbors_sw = 15
                umap_min_dist_sw = 0.1
                umap_reproducible_sw = False
            else:
                kernel_pca_kernel_sw = "rbf"
                umap_n_neighbors_sw = 15
                umap_min_dist_sw = 0.1
                umap_reproducible_sw = False
        elif representation == "timestep":
            try:
                import umap

                umap_available = True
            except ImportError:
                umap_available = False

            from influence_visualizer.render_clustering import (
                DIMRED_METHODS as _DIMRED_METHODS_TS,
                _dimred_method_display_name as _dimred_display_ts,
            )

            dimred_options_ts = (
                _DIMRED_METHODS_TS if umap_available else [m for m in _DIMRED_METHODS_TS if m != "umap"]
            )
            col1, col2 = st.columns(2)
            with col1:
                dimred_method = st.selectbox(
                    "Dim reduction",
                    dimred_options_ts,
                    format_func=_dimred_display_ts,
                    key=f"{key_prefix}_ts_method",
                    help="Dimensionality reduction applied before clustering.",
                )
            with col2:
                n_components_dimred = st.number_input(
                    "Reduced dimensions",
                    min_value=2,
                    max_value=100,
                    value=50,
                    key=f"{key_prefix}_ts_ncomp",
                    help="Number of dimensions after reduction.",
                )
            if dimred_method == "kernel_pca":
                kernel_pca_kernel_ts = st.selectbox(
                    "Kernel PCA kernel",
                    ["rbf", "poly", "cosine", "sigmoid", "linear"],
                    format_func=lambda x: {"rbf": "RBF", "poly": "Polynomial", "cosine": "Cosine", "sigmoid": "Sigmoid", "linear": "Linear"}[x],
                    key=f"{key_prefix}_ts_kernel_pca_kernel",
                    help="Kernel function for Kernel PCA.",
                )
            else:
                kernel_pca_kernel_ts = "rbf"

        # Sampling
        col_sample1, col_sample2 = st.columns(2)
        with col_sample1:
            use_max_samples = st.checkbox(
                "Limit maximum samples",
                value=False,
                key=f"{key_prefix}_limit",
            )
        with col_sample2:
            max_samples = st.slider(
                "Maximum samples",
                min_value=100,
                max_value=100000,
                value=5000,
                step=100,
                key=f"{key_prefix}_max",
                disabled=not use_max_samples,
            )

        # 2D Gaussian smoothing (TRAK only; InfEmbed uses precomputed embeddings)
        apply_gaussian = False
        gaussian_sigma = 1.0
        if not is_infembed:
            st.markdown("**Preprocessing: Gaussian smoothing**")
            col_smooth1, col_smooth2 = st.columns([1, 3])
            with col_smooth1:
                apply_gaussian = st.checkbox(
                    "Apply Gaussian smoothing",
                    value=False,
                    key=f"{key_prefix}_apply_gaussian",
                    help="2D smoothing on each rollout (or demo) influence matrix as a preprocessing step before extracting embeddings.",
                )
            with col_smooth2:
                if apply_gaussian:
                    gaussian_sigma = st.slider(
                        "Gaussian sigma",
                        min_value=0.5,
                        max_value=5.0,
                        value=1.0,
                        step=0.5,
                        key=f"{key_prefix}_gaussian_sigma",
                        help="Standard deviation for 2D Gaussian kernel (larger = more smoothing).",
                    )
                else:
                    gaussian_sigma = 1.0

        # Per-rollout normalization (timestep or sliding window, rollout level only)
        rollout_normalization = "none"
        if level == "rollout":
            rollout_normalization = st.selectbox(
                "Per-rollout normalization",
                options=["none", "center", "normalize"],
                format_func=lambda x: {
                    "none": "None",
                    "center": "Center only (subtract mean)",
                    "normalize": "Center + scale (full z-score per rollout)",
                }[x],
                index=0,
                key=f"{key_prefix}_rollout_normalization",
                help="Center: subtract each rollout's mean. Normalize: center + scale by std so "
                "only temporal shape matters.",
            )

        # Extract embeddings button
        if st.button("Extract Embeddings", key=f"{key_prefix}_extract"):
            try:
                with st.spinner("Extracting embeddings..."):
                    embeddings_original = None
                    if is_infembed and representation == "timestep":
                        try:
                            embeddings, metadata = load_infembed_embeddings(
                                data, demo_split, level
                            )
                        except FileNotFoundError as e:
                            st.error(str(e))
                            raise
                        if len(embeddings) == 0:
                            st.warning("No InfEmbed embeddings returned.")
                            raise ValueError("No InfEmbed embeddings")
                        embeddings_original = embeddings.copy()
                        if apply_dimred_infembed and len(embeddings) > 0:
                            from influence_visualizer.render_clustering import (
                                _apply_dimensionality_reduction,
                                _dimred_method_display_name,
                            )

                            with st.spinner(
                                f"Applying {_dimred_method_display_name(dimred_method_infembed)}..."
                            ):
                                embeddings = _apply_dimensionality_reduction(
                                    embeddings,
                                    method=dimred_method_infembed,
                                    n_components=n_components_dimred_infembed,
                                    umap_n_neighbors=umap_n_neighbors_infembed,
                                    umap_min_dist=umap_min_dist_infembed,
                                    umap_reproducible=umap_reproducible_infembed,
                                    scaling_method="standard",
                                    n_neighbors=15,
                                    kernel_pca_kernel=kernel_pca_kernel_infembed,
                                )
                    elif is_infembed and representation == "sliding_window":
                        from influence_visualizer.render_clustering import (
                            _apply_dimensionality_reduction,
                            _dimred_method_display_name,
                        )

                        embeddings, metadata = (
                            extract_infembed_sliding_window_embeddings(
                                data,
                                split=demo_split,
                                level=level,
                                window_width=window_width,
                                stride=stride,
                                aggregation_method=aggregation,
                                n_components=n_components_svd,
                                annotations=annotations,
                                rollout_normalization=rollout_normalization,
                            )
                        )
                        embeddings_original = embeddings.copy()
                        if apply_dimred and len(embeddings) > 0:
                            with st.spinner(
                                f"Applying {_dimred_method_display_name(dimred_method_sw)} dimensionality reduction..."
                            ):
                                embeddings = _apply_dimensionality_reduction(
                                    embeddings,
                                    method=dimred_method_sw,
                                    n_components=n_components_dimred_sw,
                                    umap_n_neighbors=umap_n_neighbors_sw,
                                    umap_min_dist=umap_min_dist_sw,
                                    umap_reproducible=umap_reproducible_sw,
                                    scaling_method="standard",
                                    n_neighbors=15,
                                    kernel_pca_kernel=kernel_pca_kernel_sw,
                                )
                    elif representation == "sliding_window":
                        from influence_visualizer.render_clustering import (
                            _apply_dimensionality_reduction,
                            extract_episode_sliding_window_embeddings,
                        )

                        embeddings, metadata = (
                            extract_episode_sliding_window_embeddings(
                                data,
                                split=demo_split,
                                level=level,
                                window_width=window_width,
                                stride=stride,
                                aggregation_method=aggregation,
                                n_components=n_components_svd,
                                use_parallel=False,
                                annotations=annotations,
                                apply_gaussian=apply_gaussian,
                                gaussian_sigma=gaussian_sigma,
                                rollout_normalization=rollout_normalization,
                            )
                        )

                        # Original = aggregated vectors before second-layer dim red
                        embeddings_original = embeddings.copy()

                        # Apply second layer dimensionality reduction if enabled
                        if apply_dimred and len(embeddings) > 0:
                            with st.spinner(
                                f"Applying {_dimred_method_display_name(dimred_method_sw)} dimensionality reduction..."
                            ):
                                embeddings = _apply_dimensionality_reduction(
                                    embeddings,
                                    method=dimred_method_sw,
                                    n_components=n_components_dimred_sw,
                                    umap_n_neighbors=umap_n_neighbors_sw,
                                    umap_min_dist=umap_min_dist_sw,
                                    umap_reproducible=umap_reproducible_sw,
                                    scaling_method="standard",  # Apply standard scaling for dimred
                                    n_neighbors=15,
                                    kernel_pca_kernel=kernel_pca_kernel_sw,
                                )
                    else:
                        from influence_visualizer.render_clustering import (
                            extract_dimred_timestep_embeddings,
                        )

                        embeddings, metadata, embeddings_original = (
                            extract_dimred_timestep_embeddings(
                                data,
                                split=demo_split,
                                level=level,
                                method=dimred_method,
                                n_components=n_components_dimred,
                                kernel_pca_kernel=kernel_pca_kernel_ts,
                                annotations=annotations,
                                apply_gaussian=apply_gaussian,
                                gaussian_sigma_matrix=gaussian_sigma,
                                rollout_normalization=rollout_normalization,
                                return_raw=True,
                            )
                        )

                if len(embeddings) == 0:
                    st.warning("No embeddings extracted. Check parameters.")
                    return

                # Apply sampling (same indices to original if stored)
                if use_max_samples and len(embeddings) > max_samples:
                    rng = np.random.RandomState(42)
                    sample_idx = rng.choice(
                        len(embeddings), size=max_samples, replace=False
                    )
                    embeddings = embeddings[sample_idx]
                    metadata = [metadata[i] for i in sample_idx]
                    if embeddings_original is not None:
                        embeddings_original = embeddings_original[sample_idx]

                cache_key = f"{key_prefix}_{influence_source}_{representation}_{level}"

                # Evict old caches before storing new ones
                _register_clustering_cache(key_prefix, cache_key)

                st.session_state[f"{cache_key}_embeddings"] = embeddings
                st.session_state[f"{cache_key}_metadata"] = metadata
                # Note: we intentionally do NOT store embeddings_original
                # in session state to avoid doubling memory usage.
                # Hierarchical refinement will use the (possibly dim-reduced)
                # embeddings instead.
                st.session_state.pop(
                    f"{cache_key}_embeddings_original", None
                )
                # Clear any previous clustering results for this key
                for k in list(st.session_state.keys()):
                    if k.startswith(f"{cache_key}_cluster"):
                        del st.session_state[k]

                st.success(
                    f"Extracted {len(embeddings)} embeddings "
                    f"(dimensionality: {embeddings.shape[1]})"
                )
            except Exception as e:
                st.error(f"Error extracting embeddings: {type(e).__name__}: {e}")

        # ---- Check if embeddings exist ----
        cache_key = f"{key_prefix}_{influence_source}_{representation}_{level}"
        emb_key = f"{cache_key}_embeddings"
        meta_key = f"{cache_key}_metadata"

        if emb_key not in st.session_state and "clalg_loaded_labels" not in st.session_state:
            st.info("Extract embeddings first, then configure and run clustering, or load a saved result above.")
            return

        if emb_key in st.session_state:
            embeddings = st.session_state[emb_key]
            metadata = st.session_state[meta_key]

            st.info(
                f"Embeddings: {len(embeddings)} samples, "
                f"dimensionality: {embeddings.shape[1]}"
            )

            # ---- Step 2: Clustering algorithm ----
            st.subheader("Step 2: Configure Clustering")

            params = _render_algorithm_params(demo_split)

            # Feature scaling
            scaling_method = st.selectbox(
                "Feature scaling",
                ["standard", "robust", "minmax", "none"],
                format_func=lambda x: {
                    "standard": "StandardScaler (zero mean, unit variance)",
                    "robust": "RobustScaler (outlier-resistant)",
                    "minmax": "MinMaxScaler (0-1 range)",
                    "none": "None",
                }[x],
                key=f"{key_prefix}_scaling",
                help="Applied before clustering and metrics computation.",
            )

            if st.button("Run Clustering", key=f"{key_prefix}_run", type="primary"):
                try:
                    with st.spinner("Running clustering algorithm..."):
                        cluster_labels = _run_clustering(
                            embeddings,
                            algorithm=params["algorithm"],
                            scaling_method=scaling_method,
                            n_clusters=params.get("n_clusters", 5),
                            kmeans_init=params.get("kmeans_init", "k-means++"),
                            kmeans_n_init=params.get("kmeans_n_init", 10),
                            mbk_batch_size=params.get("mbk_batch_size", 1024),
                            dbscan_eps=params.get("dbscan_eps", 0.5),
                            dbscan_min_samples=params.get("dbscan_min_samples", 5),
                            optics_min_samples=params.get("optics_min_samples", 5),
                            optics_xi=params.get("optics_xi", 0.05),
                            optics_min_cluster_size=params.get(
                                "optics_min_cluster_size", 0.05
                            ),
                            hdbscan_min_cluster_size=params.get(
                                "hdbscan_min_cluster_size", 5
                            ),
                            hdbscan_min_samples=params.get("hdbscan_min_samples", 5),
                            hdbscan_cluster_selection_epsilon=params.get(
                                "hdbscan_cluster_selection_epsilon", 0.0
                            ),
                            hdbscan_cluster_selection_method=params.get(
                                "hdbscan_cluster_selection_method", "eom"
                            ),
                            hdbscan_alpha=params.get("hdbscan_alpha", 1.0),
                            agg_linkage=params.get("agg_linkage", "ward"),
                            spectral_affinity=params.get("spectral_affinity", "rbf"),
                            spectral_n_neighbors=params.get("spectral_n_neighbors", 10),
                            gmm_covariance_type=params.get("gmm_covariance_type", "full"),
                            meanshift_bandwidth=params.get("meanshift_bandwidth"),
                            birch_threshold=params.get("birch_threshold", 0.5),
                            birch_branching_factor=params.get("birch_branching_factor", 50),
                        )

                    st.session_state[f"{cache_key}_cluster_labels"] = cluster_labels
                    st.session_state[f"{cache_key}_cluster_scaling"] = scaling_method
                    st.session_state[f"{cache_key}_cluster_algorithm"] = params["algorithm"]

                    n_clusters = len(set(cluster_labels) - {-1})
                    n_noise = int(np.sum(cluster_labels == -1))
                    msg = f"Clustering complete: {n_clusters} clusters found"
                    if n_noise > 0:
                        msg += f", {n_noise} noise points"
                    st.success(msg)
                except Exception as e:
                    st.error(f"Clustering failed: {type(e).__name__}: {e}")

        # ---- Step 3: Display results ----
        labels_key = f"{cache_key}_cluster_labels"
        has_loaded = "clalg_loaded_labels" in st.session_state
        if labels_key not in st.session_state and not has_loaded:
            return

        # Prefer normal path only when we have both labels and embeddings (so 2D/silhouette work)
        if (
            labels_key in st.session_state
            and emb_key in st.session_state
        ):
            # Normal path: use results from extract + run clustering
            cluster_labels = st.session_state[labels_key]
            metadata = st.session_state[meta_key]
            embeddings = st.session_state[emb_key]
            scaling_used = st.session_state.get(f"{cache_key}_cluster_scaling", "standard")
            algorithm_used = st.session_state.get(f"{cache_key}_cluster_algorithm", "?")
            is_loaded_result = False
        else:
            # Loaded path: use results from disk
            cluster_labels = st.session_state["clalg_loaded_labels"]
            metadata = st.session_state["clalg_loaded_metadata"]
            manifest = st.session_state.get("clalg_loaded_manifest", {})
            scaling_used = manifest.get("scaling", "standard")
            algorithm_used = manifest.get("algorithm", "?")
            is_loaded_result = True

        st.subheader("Step 3: Results")
        if is_loaded_result:
            loaded_name = st.session_state.get("clalg_loaded_name", "")
            st.caption(f"Loaded result: **{loaded_name}**")

        # Save clustering result (normal path only; requires task_config)
        if not is_loaded_result and task_config:
            from datetime import datetime
            from influence_visualizer.clustering_results import save_clustering_result
            n_clusters_val = len(set(cluster_labels) - {-1})
            default_save_name = (
                f"{representation}_{level}_{algorithm_used}_k{n_clusters_val}_"
                f"{datetime.now().strftime('%Y-%m-%d')}"
            )
            # Key includes params so when they change, the input resets to the new default name
            save_name_key = (
                f"{key_prefix}_save_result_name_{representation}_{level}_"
                f"{algorithm_used}_k{n_clusters_val}_{scaling_used}"
            )
            save_col1, save_col2 = st.columns([2, 1])
            with save_col1:
                save_name = st.text_input(
                    "Save clustering result",
                    value=default_save_name,
                    key=save_name_key,
                    help="Name for the saved result (used as directory name).",
                )
            with save_col2:
                if st.button("Save", key=f"{key_prefix}_save_result_btn"):
                    try:
                        path = save_clustering_result(
                            task_config=task_config,
                            name=save_name.strip() or default_save_name,
                            cluster_labels=cluster_labels,
                            metadata=metadata,
                            algorithm=algorithm_used,
                            scaling=scaling_used,
                            influence_source=influence_source,
                            representation=representation,
                            level=level,
                            n_clusters=n_clusters_val,
                            n_samples=len(cluster_labels),
                        )
                        st.success(f"Saved to {path}")
                    except Exception as e:
                        st.error(f"Save failed: {e}")
            st.divider()

        if is_loaded_result:
            # --- Loaded path: manifest, counts, cluster stats only ---
            manifest = st.session_state.get("clalg_loaded_manifest", {})
            st.markdown("**Manifest**")
            st.json({
                k: manifest.get(k)
                for k in (
                    "algorithm", "scaling", "influence_source", "representation",
                    "level", "n_clusters", "n_samples", "created", "task_config"
                )
                if manifest.get(k) is not None
            })
            n_clusters = len(set(cluster_labels) - {-1})
            n_noise = int(np.sum(cluster_labels == -1))
            st.metric("Clusters", n_clusters, help="Number of clusters (excluding noise).")
            st.metric("Noise points", n_noise)
            st.info(
                "2D projection and silhouette require embeddings. "
                "Re-run extraction and clustering to see them, or load a result that was saved with embeddings."
            )
            # --- Cluster statistics (same as normal path) ---
            cluster_stats = _compute_label_coherency(metadata, cluster_labels)
            st.markdown("**Cluster Statistics**")
            sorted_stats = sorted(
                cluster_stats,
                key=lambda s: (s["cluster_id"] == -1, s["cluster_id"]),
            )
            annotation_labels = [m.get("annotation_label", "no label") for m in metadata]
            has_annotations = any(a != "no label" for a in annotation_labels)
            if has_annotations:
                labeled_stats = [s for s in sorted_stats if s["n_labeled"] > 0]
                if labeled_stats:
                    st.markdown(
                        "| Cluster | Size | Labeled | Dominant Label | Purity | Entropy |\n"
                        "|---------|------|---------|----------------|--------|---------|\n"
                        + "\n".join(
                            f"| {'Noise' if s['cluster_id'] == -1 else s['cluster_id']} "
                            f"| {s['size']} | {s['n_labeled']} "
                            f"| {s['dominant_label']} | {s['purity']:.0%} "
                            f"| {s['entropy']:.2f} |"
                            for s in labeled_stats
                        )
                    )
                    fig_coherency = plotting.create_label_coherency_chart(
                        labeled_stats,
                        title="Annotation Label Distribution per Cluster",
                    )
                    st.plotly_chart(fig_coherency, width='stretch')
                else:
                    st.markdown(
                        "| Cluster | Size |\n|---------|------|\n"
                        + "\n".join(
                            f"| {'Noise' if s['cluster_id'] == -1 else s['cluster_id']} | {s['size']} |"
                            for s in sorted_stats
                        )
                    )
            else:
                st.markdown(
                    "| Cluster | Size |\n|---------|------|\n"
                    + "\n".join(
                        f"| {'Noise' if s['cluster_id'] == -1 else s['cluster_id']} | {s['size']} |"
                        for s in sorted_stats
                    )
                )
            fig_size = plotting.create_cluster_size_chart(
                cluster_stats, title="Cluster Size Distribution"
            )
            st.plotly_chart(fig_size, width='stretch')
            representation_loaded = manifest.get("representation", "sliding_window")
            fig_timestep_dist = plotting.create_cluster_timestep_distribution(
                metadata,
                cluster_labels,
                representation_loaded,
                title="Timestep distribution per cluster",
            )
            st.plotly_chart(fig_timestep_dist, width='stretch')
        else:
            # --- Normal path: full metrics, 2D, silhouette, etc. ---
            # --- Metrics ---
            annotation_labels = [m.get("annotation_label", "no label") for m in metadata]
            has_annotations = any(a != "no label" for a in annotation_labels)

            metrics = _compute_metrics(
                embeddings,
                cluster_labels,
                scaling_method=scaling_used,
                annotation_labels=annotation_labels if has_annotations else None,
            )

            # Display metrics
            st.markdown("**Evaluation Metrics**")
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric(
                    "Clusters",
                    metrics["n_clusters"],
                    help="Number of clusters found (excluding noise).",
                )
            with metric_cols[1]:
                if "silhouette" in metrics:
                    st.metric(
                        "Silhouette",
                        f"{metrics['silhouette']:.3f}",
                        help="How well-separated clusters are. Range: -1 to 1. Higher is better.",
                    )
                else:
                    st.metric("Silhouette", "N/A")

            # --- 2D Projection + Scatter ---
            st.markdown("**2D Visualization**")
            proj_col1, proj_col2 = st.columns(2)
            with proj_col1:
                proj_method = st.selectbox(
                    "Projection method",
                    ["tsne", "umap"],
                    format_func=lambda x: "t-SNE" if x == "tsne" else "UMAP",
                    key=f"{key_prefix}_proj_method",
                )
            with proj_col2:
                proj_perplexity = st.slider(
                    "Perplexity (t-SNE only)",
                    min_value=2,
                    max_value=50,
                    value=30,
                    key=f"{key_prefix}_proj_perp",
                    disabled=proj_method != "tsne",
                )

            proj_cache_key = (
                f"{cache_key}_proj_{proj_method}_{proj_perplexity}_{len(embeddings)}"
            )
            if proj_cache_key not in st.session_state:
                with st.spinner("Computing 2D projection..."):
                    embeddings_2d = _project_to_2d(
                        embeddings,
                        method=proj_method,
                        perplexity=proj_perplexity,
                        scaling_method=scaling_used,
                    )
                    st.session_state[proj_cache_key] = embeddings_2d
            else:
                embeddings_2d = st.session_state[proj_cache_key]
            # Store under fixed key so refinement fragment can show original plot with updated labels
            st.session_state[f"{cache_key}_embeddings_2d"] = embeddings_2d

            color_by_main = st.selectbox(
                "Color points by",
                ["cluster", "success", "timestep", "quality", "rollout_idx", "demo_idx"],
                format_func=lambda x: {
                    "cluster": "Cluster",
                    "success": "Success/Failure (rollout)",
                    "timestep": "Timestep",
                    "quality": "Demonstration quality",
                    "rollout_idx": "Rollout index",
                    "demo_idx": "Demonstration index",
                }[x],
                key=f"{key_prefix}_color_main",
                help="Color the 2D projection by cluster or metadata.",
            )
            fig_scatter = plotting.create_cluster_scatter_2d(
                embeddings_2d,
                cluster_labels,
                metadata,
                title=f"Cluster Assignments ({algorithm_used})",
                color_by=color_by_main,
            )
            st.plotly_chart(fig_scatter, width='stretch')

            # --- Silhouette plot ---
            if "silhouette" in metrics:
                sil_cache_key = f"{cache_key}_sil_values"
                if sil_cache_key not in st.session_state:
                    from sklearn.metrics import silhouette_samples
                    from sklearn.preprocessing import (
                        MinMaxScaler,
                        RobustScaler,
                        StandardScaler,
                    )

                    scaled = embeddings.copy()
                    if scaling_used == "standard":
                        scaled = StandardScaler().fit_transform(scaled)
                    elif scaling_used == "robust":
                        scaled = RobustScaler().fit_transform(scaled)
                    elif scaling_used == "minmax":
                        scaled = MinMaxScaler().fit_transform(scaled)

                    sil_values = silhouette_samples(scaled, cluster_labels)
                    st.session_state[sil_cache_key] = sil_values
                else:
                    sil_values = st.session_state[sil_cache_key]

                fig_sil = plotting.create_silhouette_plot(
                    sil_values, cluster_labels, title="Silhouette Analysis"
                )
                st.plotly_chart(fig_sil, width='stretch')

            # --- Cluster statistics ---
            cluster_stats = _compute_label_coherency(metadata, cluster_labels)

            st.markdown("**Cluster Statistics**")

            # Always show cluster size information (order by cluster index, noise last)
            sorted_stats = sorted(
                cluster_stats,
                key=lambda s: (s["cluster_id"] == -1, s["cluster_id"]),
            )

            if has_annotations:
                # Show full table with annotation metrics
                labeled_stats = [s for s in sorted_stats if s["n_labeled"] > 0]

                if labeled_stats:
                    st.markdown(
                        "| Cluster | Size | Labeled | Dominant Label | Purity | Entropy |\n"
                        "|---------|------|---------|----------------|--------|---------|\n"
                        + "\n".join(
                            f"| {'Noise' if s['cluster_id'] == -1 else s['cluster_id']} "
                            f"| {s['size']} | {s['n_labeled']} "
                            f"| {s['dominant_label']} | {s['purity']:.0%} "
                            f"| {s['entropy']:.2f} |"
                            for s in labeled_stats
                        )
                    )

                    # Show the coherency chart
                    fig_coherency = plotting.create_label_coherency_chart(
                        labeled_stats,
                        title="Annotation Label Distribution per Cluster",
                    )
                    st.plotly_chart(fig_coherency, width='stretch')
                else:
                    # Show simplified table without annotation metrics
                    st.markdown(
                        "| Cluster | Size |\n"
                        "|---------|------|\n"
                        + "\n".join(
                            f"| {'Noise' if s['cluster_id'] == -1 else s['cluster_id']} "
                            f"| {s['size']} |"
                            for s in sorted_stats
                        )
                    )
                    st.info("No annotated samples found in clustering results.")
            else:
                # Show simplified table without annotation metrics
                st.markdown(
                    "| Cluster | Size |\n"
                    "|---------|------|\n"
                    + "\n".join(
                        f"| {'Noise' if s['cluster_id'] == -1 else s['cluster_id']} "
                        f"| {s['size']} |"
                        for s in sorted_stats
                    )
                )

            # --- Cluster size chart ---
            fig_size = plotting.create_cluster_size_chart(
                cluster_stats, title="Cluster Size Distribution"
            )
            st.plotly_chart(fig_size, width='stretch')

            # --- Timestep distribution per cluster ---
            st.markdown("**Timestep distribution per cluster**")
            fig_timestep_dist = plotting.create_cluster_timestep_distribution(
                metadata,
                cluster_labels,
                representation,
                title="Timestep distribution per cluster",
            )
            st.plotly_chart(fig_timestep_dist, width='stretch')

            # ---- Step 4: Hierarchical refinement ----
            _render_hierarchical_refinement_fragment(
                embeddings=embeddings,
                cluster_labels=cluster_labels,
                metadata=metadata,
                scaling_used=scaling_used,
                cache_key=cache_key,
                key_prefix=key_prefix,
                demo_split=demo_split,
                embeddings_original=None,
            )

            # ---- Video exploration (below clustering; use refined clusters if refinement was run) ----
            st.markdown("---")
            st.markdown("### Video exploration (browse clusters)")

            video_exploration_key = f"{cache_key}_video_exploration_shown"
            if st.button(
                "Generate previews / exploration",
                key=f"{key_prefix}_gen_video_exploration",
                help="Show temporal coherence browsing and per-cluster sample videos. "
                "If you run hierarchical refinement again, this section will hide until you click again to refresh with updated clusters.",
            ):
                st.session_state[video_exploration_key] = True

            if st.session_state.get(video_exploration_key, False):
                refine_cache_key = f"{cache_key}_refined"
                if f"{refine_cache_key}_labels" in st.session_state:
                    refined_labels = st.session_state[f"{refine_cache_key}_labels"]
                    refine_indices = st.session_state[
                        f"{refine_cache_key}_original_indices"
                    ]
                    labels_for_videos = np.array(cluster_labels, copy=True)
                    offset = (
                        int(np.max(labels_for_videos)) + 1
                        if np.any(labels_for_videos >= 0)
                        else 0
                    )
                    for j, idx in enumerate(refine_indices):
                        labels_for_videos[idx] = (
                            refined_labels[j] + offset
                            if refined_labels[j] != -1
                            else -1
                        )
                    cluster_stats_for_videos = _compute_label_coherency(
                        metadata, labels_for_videos
                    )
                else:
                    labels_for_videos = cluster_labels
                    cluster_stats_for_videos = cluster_stats

                _render_temporal_coherence_fragment(
                    cluster_labels=labels_for_videos,
                    metadata=metadata,
                    representation=representation,
                    level=level,
                    data=data,
                    demo_split=demo_split,
                    cache_key=cache_key,
                    key_prefix=key_prefix,
                    annotations=annotations,
                )

                _render_cluster_exploration_fragment(
                    cluster_stats=cluster_stats_for_videos,
                    cluster_labels=labels_for_videos,
                    metadata=metadata,
                    representation=representation,
                    level=level,
                    data=data,
                    demo_split=demo_split,
                    cache_key=cache_key,
                    key_prefix=key_prefix,
                )
