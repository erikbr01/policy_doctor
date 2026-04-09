"""Clustering analysis functions for exploring influence embedding patterns.

This module provides tools for:
1. Extracting influence embeddings at different aggregation levels
2. Visualizing embeddings using t-SNE for cluster discovery
3. Suggesting methods for embedding full influence matrix pairs
"""

import os
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import streamlit as st
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.decomposition import (
    FactorAnalysis,
    FastICA,
    PCA,
    TruncatedSVD,
)
from sklearn.manifold import (
    Isomap,
    MDS,
    SpectralEmbedding,
    TSNE,
)
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# Configure Numba for Streamlit compatibility
# Per Streamlit docs: https://docs.streamlit.io/develop/concepts/design/multithreading
# Streamlit runs scripts in separate threads, so we need thread-safe execution
if "NUMBA_THREADING_LAYER" not in os.environ:
    # Use 'omp' or 'tbb' for thread-safe parallel execution
    # 'workqueue' is NOT thread-safe and will crash with Streamlit
    os.environ["NUMBA_THREADING_LAYER"] = "omp"  # OpenMP - widely available

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from influence_visualizer import plotting
from influence_visualizer.data_loader import InfluenceData

# ---------------------------------------------------------------------------
# Session state memory management
# ---------------------------------------------------------------------------

MAX_EMBEDDING_CACHES_PER_SECTION = 3


def _register_embedding_cache(section_prefix: str, cache_key: str) -> None:
    """Track an embedding cache key and evict the oldest when over the limit.

    Each clustering section (sw_embeddings, episode_sw, mp_embeddings, etc.)
    independently tracks its most recent cache keys.  When the limit is
    exceeded the oldest entry and all its associated keys are removed.
    """
    tracker_key = f"_cache_order_{section_prefix}"
    order: list = st.session_state.get(tracker_key, [])

    if cache_key in order:
        order.remove(cache_key)
    order.append(cache_key)

    while len(order) > MAX_EMBEDDING_CACHES_PER_SECTION:
        old_key = order.pop(0)
        # Remove the primary tuple and any derived keys (tsne caches, etc.)
        for k in list(st.session_state.keys()):
            if isinstance(k, str) and (k == old_key or k.startswith(f"{old_key}_")):
                del st.session_state[k]

    st.session_state[tracker_key] = order
from influence_visualizer.render_annotation import (
    get_episode_annotations,
    get_label_for_frame,
)
from influence_visualizer.render_heatmaps import SplitType, get_split_data
from influence_visualizer.render_local_behaviors import (
    compute_hog_features,
    extract_sliding_window_features,
)


def extract_demo_embeddings(
    data: InfluenceData,
    split: SplitType = "train",
) -> Tuple[np.ndarray, List[Dict]]:
    """Extract influence embeddings for training demonstrations.

    Each demonstration gets an embedding vector representing its influence
    across all rollout samples (aggregate over rollout length).

    Args:
        data: InfluenceData object
        split: Which demo split to use ("train", "holdout", or "both")

    Returns:
        Tuple of (embeddings, metadata) where:
        - embeddings: shape (num_demos, num_rollout_samples)
        - metadata: list of dicts with demo_idx, quality_label, num_samples
    """
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)

    # Safety check: prevent creating arrays that are too large
    num_demos = len(demo_episodes)
    num_rollout_samples = influence_matrix.shape[0]
    num_elements = num_demos * num_rollout_samples
    MAX_EMBEDDING_ELEMENTS = 100_000_000  # 100M elements max (~400MB for float32)

    if num_elements > MAX_EMBEDDING_ELEMENTS:
        raise ValueError(
            f"Demo embedding matrix too large: {num_demos:,} demos × "
            f"{num_rollout_samples:,} rollout samples = {num_elements:,} elements "
            f"(max: {MAX_EMBEDDING_ELEMENTS:,}).\n\n"
            f"Try using 'train' or 'holdout' split instead of 'both' to reduce demo count."
        )

    # For each demo episode, extract its influence across all rollout samples
    embeddings = []
    metadata = []

    for demo_ep_idx, demo_ep in enumerate(demo_episodes):
        demo_sample_idxs = ep_idxs[demo_ep_idx]

        # Get influence matrix for this demo: shape (num_rollout_samples, num_demo_timesteps)
        demo_influence = influence_matrix[:, demo_sample_idxs]

        # Aggregate over demo timesteps to get one vector per demo
        # Shape: (num_rollout_samples,)
        demo_embedding = np.mean(demo_influence, axis=1)

        embeddings.append(demo_embedding)

        # Get quality label if available
        quality_label = "unknown"
        if data.demo_quality_labels is not None:
            quality_label = data.demo_quality_labels.get(demo_ep.index, "unknown")

        metadata.append(
            {
                "demo_idx": demo_ep.index,
                "quality_label": quality_label,
                "num_samples": demo_ep.num_samples,
                "raw_length": demo_ep.raw_length,
            }
        )

    return np.array(embeddings), metadata


def extract_rollout_embeddings(
    data: InfluenceData,
    split: SplitType = "train",
) -> Tuple[np.ndarray, List[Dict]]:
    """Extract influence embeddings for rollout episodes.

    Each rollout gets an embedding vector representing influences from
    all demonstration samples (aggregate over training demonstration length).

    Args:
        data: InfluenceData object
        split: Which demo split to use ("train", "holdout", or "both")

    Returns:
        Tuple of (embeddings, metadata) where:
        - embeddings: shape (num_rollouts, num_demo_samples)
        - metadata: list of dicts with rollout_idx, success, num_samples
    """
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)

    # Safety check: prevent creating arrays that are too large
    num_rollouts = len(data.rollout_episodes)
    num_demo_samples = influence_matrix.shape[1]
    num_elements = num_rollouts * num_demo_samples
    MAX_EMBEDDING_ELEMENTS = 100_000_000  # 100M elements max (~400MB for float32)

    if num_elements > MAX_EMBEDDING_ELEMENTS:
        raise ValueError(
            f"Rollout embedding matrix too large: {num_rollouts:,} rollouts × "
            f"{num_demo_samples:,} demo samples = {num_elements:,} elements "
            f"(max: {MAX_EMBEDDING_ELEMENTS:,}).\n\n"
            f"Try using 'train' or 'holdout' split instead of 'both' to reduce demo sample count."
        )

    # For each rollout episode, extract its influence from all demo samples
    embeddings = []
    metadata = []

    for rollout_ep in data.rollout_episodes:
        rollout_sample_indices = np.arange(
            rollout_ep.sample_start_idx, rollout_ep.sample_end_idx
        )

        # Get influence matrix for this rollout: shape (num_rollout_timesteps, num_demo_samples)
        rollout_influence = influence_matrix[rollout_sample_indices, :]

        # Aggregate over rollout timesteps to get one vector per rollout
        # Shape: (num_demo_samples,)
        rollout_embedding = np.mean(rollout_influence, axis=0)

        embeddings.append(rollout_embedding)

        metadata.append(
            {
                "rollout_idx": rollout_ep.index,
                "success": rollout_ep.success,
                "num_samples": rollout_ep.num_samples,
            }
        )

    return np.array(embeddings), metadata


def _dimred_method_display_name(method: str) -> str:
    """Return a display name for a dimensionality reduction method."""
    names = {
        "pca": "PCA",
        "kernel_pca": "Kernel PCA",
        "umap": "UMAP",
        "umap_direct": "UMAP",
        "truncated_svd": "Truncated SVD",
        "factor_analysis": "Factor Analysis",
        "ica": "ICA",
        "mds": "MDS",
        "isomap": "Isomap",
        "spectral": "Spectral Embedding",
    }
    return names.get(method, method.upper())


DIMRED_METHODS = [
    "pca",
    "kernel_pca",
    "umap",
    "truncated_svd",
    "factor_analysis",
    "ica",
    "mds",
    "isomap",
    "spectral",
]


@st.cache_data(show_spinner=False, max_entries=10)
def _apply_dimensionality_reduction(
    vectors_matrix: np.ndarray,
    method: str,
    n_components: int,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_reproducible: bool = False,
    scaling_method: str = "standard",
    n_neighbors: int = 15,
    kernel_pca_kernel: str = "rbf",
) -> np.ndarray:
    """Apply dimensionality reduction (cached for performance).

    This function is cached to avoid recomputing expensive operations.
    Per Streamlit docs, expensive computations should be cached to work
    properly with Streamlit's threading model.

    Args:
        scaling_method: Feature scaling method ("standard", "robust", "minmax", "none")
        umap_reproducible: If True, use random_state=42 (slower, single-threaded).
                          If False, no random_state (faster, uses all cores, non-deterministic).
        n_neighbors: Used by Isomap and Spectral (nearest-neighbor graph).
    """
    n_components = min(n_components, vectors_matrix.shape[0], vectors_matrix.shape[1])
    n_neighbors = min(n_neighbors, max(2, vectors_matrix.shape[0] - 1))

    # Apply feature scaling based on selected method
    if scaling_method == "standard":
        # StandardScaler: zero mean, unit variance (best for most cases)
        scaler = StandardScaler()
        vectors_matrix = scaler.fit_transform(vectors_matrix)
    elif scaling_method == "robust":
        # RobustScaler: robust to outliers (uses median and IQR)
        scaler = RobustScaler()
        vectors_matrix = scaler.fit_transform(vectors_matrix)
    elif scaling_method == "minmax":
        # MinMaxScaler: scales to [0, 1] range
        scaler = MinMaxScaler()
        vectors_matrix = scaler.fit_transform(vectors_matrix)
    elif scaling_method == "none":
        # No scaling (not recommended for distance-based methods)
        pass
    else:
        raise ValueError(f"Unknown scaling method: {scaling_method}")

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
        return reducer.fit_transform(vectors_matrix)
    elif method == "kernel_pca":
        from sklearn.decomposition import KernelPCA

        reducer = KernelPCA(
            n_components=n_components,
            kernel=kernel_pca_kernel,
            gamma=None,  # 1 / n_features for rbf/poly/sigmoid
            random_state=42,
        )
        return reducer.fit_transform(vectors_matrix)
    elif method == "truncated_svd":
        reducer = TruncatedSVD(n_components=n_components, random_state=42)
        return reducer.fit_transform(vectors_matrix)
    elif method == "factor_analysis":
        reducer = FactorAnalysis(
            n_components=n_components, max_iter=500, random_state=42
        )
        return reducer.fit_transform(vectors_matrix)
    elif method == "ica":
        reducer = FastICA(n_components=n_components, random_state=42, max_iter=500)
        return reducer.fit_transform(vectors_matrix)
    elif method == "mds":
        reducer = MDS(
            n_components=n_components,
            metric=True,
            random_state=42,
            n_init=4,
            max_iter=300,
            dissimilarity="euclidean",
        )
        return reducer.fit_transform(vectors_matrix)
    elif method == "isomap":
        n_n = max(2, min(n_neighbors, vectors_matrix.shape[0] - 1))
        reducer = Isomap(n_components=n_components, n_neighbors=n_n)
        return reducer.fit_transform(vectors_matrix)
    elif method == "spectral":
        n_n = max(2, min(n_neighbors, vectors_matrix.shape[0] - 1))
        reducer = SpectralEmbedding(
            n_components=n_components,
            n_neighbors=n_n,
            random_state=42,
        )
        return reducer.fit_transform(vectors_matrix)
    elif method in ["umap", "umap_direct"]:
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available")

        # IMPORTANT: UMAP forces n_jobs=1 when random_state is set
        # See: https://umap-learn.readthedocs.io/en/latest/reproducibility.html
        # Trade-off: reproducibility (slow) vs speed (non-deterministic)

        if umap_reproducible:
            # Reproducible mode: single-threaded
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                metric="euclidean",
                n_jobs=32,
                low_memory=False,
                verbose=False,
            )
        else:
            # Fast mode: multi-threaded (use all cores)
            n_jobs = -1 if len(vectors_matrix) > 1000 else 1
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                metric="euclidean",
                n_jobs=32,
                low_memory=False,
                verbose=False,
            )

        return reducer.fit_transform(vectors_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")


def extract_dimred_timestep_embeddings(
    data: InfluenceData,
    split: SplitType = "train",
    level: Literal["rollout", "demo"] = "rollout",
    method: Literal[
        "pca",
        "kernel_pca",
        "umap",
        "truncated_svd",
        "factor_analysis",
        "ica",
        "mds",
        "isomap",
        "spectral",
    ] = "umap",
    n_components: int = 50,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_reproducible: bool = False,
    kernel_pca_kernel: str = "rbf",
    gaussian_sigma: float = 0.0,
    scaling_method: str = "standard",
    annotations: Optional[Dict] = None,
    apply_gaussian: bool = False,
    gaussian_sigma_matrix: float = 1.0,
    rollout_normalization: Literal["none", "center", "normalize"] = "none",
    return_raw: bool = False,
) -> Tuple[np.ndarray, List[Dict], Optional[np.ndarray]]:
    """Extract dimensionality-reduced embeddings for individual timesteps of rollouts or demos.

    For each timestep in each rollout/demo, apply PCA or UMAP to reduce the influence vector
    dimensionality, then visualize with t-SNE.

    Args:
        data: InfluenceData object
        split: Which demo split to use ("train", "holdout", or "both")
        level: Whether to extract timesteps for "rollout" or "demo" episodes
        method: Dimensionality reduction method (pca, umap, truncated_svd, factor_analysis, ica, mds, isomap, spectral)
        n_components: Number of components to extract
        umap_n_neighbors: UMAP n_neighbors parameter (only used if method="umap")
        umap_min_dist: UMAP min_dist parameter (only used if method="umap")
        gaussian_sigma: Standard deviation for 1D Gaussian smoothing of vectors (0 = no smoothing)
        apply_gaussian: Whether to apply 2D Gaussian smoothing to influence matrix beforehand
        gaussian_sigma_matrix: Sigma for 2D matrix smoothing (only if apply_gaussian is True)
        rollout_normalization: When level is "rollout", how to normalize per rollout:
            "none": no per-rollout transform.
            "center": subtract each rollout's mean vector (emphasizes temporal shape;
            preserves amplitude of variation across rollouts).
            "normalize": center then divide by each rollout's std (full per-rollout
            z-score). Emphasizes shape only; makes all rollouts equally variable in
            magnitude.
        return_raw: If True, also return raw vectors (before dim red) as third element.

    Returns:
        Tuple of (embeddings, metadata, raw_vectors_or_None) where:
        - embeddings: shape (total_timesteps, n_components)
        - metadata: list of dicts with episode_idx, timestep, and stats
        - raw_vectors_or_None: if return_raw=True, shape (total_timesteps, original_dim); else None
    """
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)

    # Apply 2D Gaussian smoothing to influence matrix if requested (same as local behaviors tab)
    if apply_gaussian and gaussian_sigma_matrix > 0:
        influence_matrix = gaussian_filter(
            influence_matrix, sigma=gaussian_sigma_matrix
        )

    all_vectors = []
    all_metadata = []

    if level == "rollout":
        # For each rollout timestep, get the influence vector across all demo samples
        for rollout_ep in data.rollout_episodes:
            # Get annotations for this rollout episode if available
            episode_annotations = []
            if annotations:
                episode_id_str = str(rollout_ep.index)
                episode_annotations = get_episode_annotations(
                    annotations, episode_id_str, split="rollout"
                )

            for t in range(rollout_ep.num_samples):
                abs_idx = rollout_ep.sample_start_idx + t
                # Get influence vector: shape (num_demo_samples,)
                influence_vector = influence_matrix[abs_idx, :]

                # Apply Gaussian smoothing if requested
                if gaussian_sigma > 0:
                    influence_vector = gaussian_filter1d(
                        influence_vector, sigma=gaussian_sigma, mode="nearest"
                    )

                # Get annotation label for this timestep
                annotation_label = "no label"
                if episode_annotations:
                    annotation_label = get_label_for_frame(t, episode_annotations)

                all_vectors.append(influence_vector)
                all_metadata.append(
                    {
                        "rollout_idx": rollout_ep.index,
                        "timestep": t,
                        "success": rollout_ep.success,
                        "num_samples": rollout_ep.num_samples,
                        "annotation_label": annotation_label,
                        "mean_influence": float(np.mean(influence_vector)),
                        "std_influence": float(np.std(influence_vector)),
                        "max_influence": float(np.max(influence_vector)),
                        "min_influence": float(np.min(influence_vector)),
                    }
                )
    else:  # level == "demo"
        # For each demo timestep, get the influence vector across all rollout samples
        for demo_ep_idx, demo_ep in enumerate(demo_episodes):
            demo_sample_idxs = ep_idxs[demo_ep_idx]

            # Get annotations for this demo episode if available
            episode_annotations = []
            if annotations:
                episode_id_str = str(demo_ep.index)
                # Determine which split this demo belongs to
                demo_split_type = "train" if split in ["train", "both"] else "holdout"
                episode_annotations = get_episode_annotations(
                    annotations, episode_id_str, split=demo_split_type
                )

            for t_idx, abs_idx in enumerate(demo_sample_idxs):
                # Get influence vector: shape (num_rollout_samples,)
                influence_vector = influence_matrix[:, abs_idx]

                # Apply Gaussian smoothing if requested
                if gaussian_sigma > 0:
                    influence_vector = gaussian_filter1d(
                        influence_vector, sigma=gaussian_sigma, mode="nearest"
                    )

                # Get quality label if available
                quality_label = "unknown"
                if data.demo_quality_labels is not None:
                    quality_label = data.demo_quality_labels.get(
                        demo_ep.index, "unknown"
                    )

                # Get annotation label for this timestep
                annotation_label = "no label"
                if episode_annotations:
                    annotation_label = get_label_for_frame(t_idx, episode_annotations)

                all_vectors.append(influence_vector)
                all_metadata.append(
                    {
                        "demo_idx": demo_ep.index,
                        "timestep": t_idx,
                        "quality_label": quality_label,
                        "annotation_label": annotation_label,
                        "num_samples": demo_ep.num_samples,
                        "raw_length": demo_ep.raw_length,
                        "mean_influence": float(np.mean(influence_vector)),
                        "std_influence": float(np.std(influence_vector)),
                        "max_influence": float(np.max(influence_vector)),
                        "min_influence": float(np.min(influence_vector)),
                    }
                )

    # Stack all vectors into a matrix (original representation before dim red)
    vectors_matrix = np.array(all_vectors)

    # Optional: per-rollout normalization so clustering reflects temporal shape.
    if rollout_normalization != "none" and level == "rollout" and len(all_metadata) > 0:
        rollout_ids = np.array([m["rollout_idx"] for m in all_metadata])
        for r in np.unique(rollout_ids):
            mask = rollout_ids == r
            block = vectors_matrix[mask]
            block = block - block.mean(axis=0)
            if rollout_normalization == "normalize":
                std = block.std(axis=0)
                std[std < 1e-10] = 1.0  # avoid division by zero
                block = block / std
            vectors_matrix[mask] = block

    # Apply dimensionality reduction using cached function
    # This follows Streamlit best practices for expensive computations
    method_display = _dimred_method_display_name(method)
    with st.spinner(
        f"Applying {method_display} (n_components={n_components})..."
    ):
        embeddings = _apply_dimensionality_reduction(
            vectors_matrix,
            method=method,
            n_components=n_components,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            umap_reproducible=umap_reproducible,
            scaling_method=scaling_method,
            kernel_pca_kernel=kernel_pca_kernel,
        )

    if return_raw:
        return embeddings, all_metadata, vectors_matrix
    return embeddings, all_metadata, None


def extract_sliding_window_embeddings_all_pairs(
    data: InfluenceData,
    split: SplitType = "train",
    window_height: int = 5,
    window_width: int = 5,
    stride: int = 2,
    method: Literal["flatten", "hog"] = "flatten",
    hog_bins: int = 8,
) -> Tuple[np.ndarray, List[Dict]]:
    """Extract sliding window embeddings from all rollout-demo influence matrix pairs.

    For each (rollout, demo) pair, extracts sliding windows from the influence matrix
    and embeds them using the specified method.

    Args:
        data: InfluenceData object
        split: Which demo split to use ("train", "holdout", or "both")
        window_height: Height of sliding window (rollout dimension)
        window_width: Width of sliding window (demo dimension)
        stride: Stride for sliding window
        method: Embedding method ("flatten" or "hog")
        hog_bins: Number of bins for HOG (only used if method="hog")

    Returns:
        Tuple of (embeddings, metadata) where:
        - embeddings: shape (total_windows_across_all_pairs, feature_dim)
        - metadata: list of dicts with rollout_idx, demo_idx, window position, success, quality
    """
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)

    all_embeddings = []
    all_metadata = []

    # For each rollout-demo pair
    for rollout_ep in data.rollout_episodes:
        rollout_sample_indices = np.arange(
            rollout_ep.sample_start_idx, rollout_ep.sample_end_idx
        )

        for demo_ep_idx, demo_ep in enumerate(demo_episodes):
            demo_sample_idxs = ep_idxs[demo_ep_idx]

            # Extract the local transition matrix for this pair
            # Shape: (num_rollout_timesteps, num_demo_timesteps)
            transition_matrix = influence_matrix[
                np.ix_(rollout_sample_indices, demo_sample_idxs)
            ]

            # Check if matrix is large enough for sliding window
            if (
                transition_matrix.shape[0] < window_height
                or transition_matrix.shape[1] < window_width
            ):
                continue

            # Extract sliding window features
            window_features, window_metadata = extract_sliding_window_features(
                transition_matrix,
                window_height=window_height,
                window_width=window_width,
                stride=stride,
                method=method,
                hog_bins=hog_bins,
            )

            # Add rollout/demo context to metadata
            quality_label = "unknown"
            if data.demo_quality_labels is not None:
                quality_label = data.demo_quality_labels.get(demo_ep.index, "unknown")

            for i, (feature, win_meta) in enumerate(
                zip(window_features, window_metadata)
            ):
                all_embeddings.append(feature)
                all_metadata.append(
                    {
                        "rollout_idx": rollout_ep.index,
                        "demo_idx": demo_ep.index,
                        "success": rollout_ep.success,
                        "quality_label": quality_label,
                        "rollout_start": win_meta["rollout_start"],
                        "rollout_end": win_meta["rollout_end"],
                        "demo_start": win_meta["demo_start"],
                        "demo_end": win_meta["demo_end"],
                        "mean_influence": win_meta["mean_influence"],
                        "std_influence": win_meta["std_influence"],
                        "max_influence": win_meta["max_influence"],
                        "min_influence": win_meta["min_influence"],
                    }
                )

    return np.array(all_embeddings), all_metadata


def render_tsne_plot(
    embeddings: np.ndarray,
    metadata: List[Dict],
    title: str,
    color_by: str = "success",
    key_suffix: str = "",
    perplexity: int = 30,
    tsne_cache: Optional[Dict] = None,
    enable_selection: bool = False,
) -> tuple[Optional[np.ndarray], Optional[dict]]:
    """Render an interactive t-SNE scatter plot for influence embeddings.

    Returns:
        Tuple of (embeddings_2d, selection):
        - embeddings_2d: The 2D t-SNE embeddings (N, 2) array, or None if plotting failed
        - selection: Selection dict from st.plotly_chart if enable_selection=True, else None

    Args:
        embeddings: Embedding vectors, shape (num_samples, embedding_dim)
        metadata: List of metadata dicts for each sample
        title: Plot title
        color_by: Metadata field to color by ("success", "quality_label", "mean_influence", "std_influence")
        key_suffix: Unique suffix for streamlit component key
        perplexity: Perplexity parameter for t-SNE (default: 30)
        tsne_cache: Optional dict to cache t-SNE results by perplexity
        enable_selection: Whether to enable interactive selection (default: False)
    """
    if len(embeddings) < 2:
        st.warning("Need at least 2 samples for visualization.")
        return None, None

    # Check if embeddings are already 2D (e.g., from UMAP direct)
    if embeddings.shape[1] == 2:
        embeddings_2d = embeddings
    else:
        # Adjust perplexity based on sample size if needed
        perplexity = min(perplexity, max(5, len(embeddings) // 3))

        # Check cache for this perplexity
        if tsne_cache is not None and perplexity in tsne_cache:
            embeddings_2d = tsne_cache[perplexity]
        else:
            # Perform t-SNE
            with st.spinner(
                f"Computing t-SNE for {len(embeddings)} samples (perplexity={perplexity})..."
            ):
                tsne = TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    random_state=42,
                    max_iter=1000,
                )
                embeddings_2d = tsne.fit_transform(embeddings)

                # Cache the result
                if tsne_cache is not None:
                    tsne_cache[perplexity] = embeddings_2d

    # Check if continuous coloring is requested
    continuous_color_modes = [
        "mean_influence",
        "std_influence",
        "max_influence",
        "min_influence",
        "window_start",
    ]

    if color_by in continuous_color_modes:
        # Use continuous color scale
        color_values = np.array([m.get(color_by, 0) for m in metadata])

        # Build hover text
        hover_texts = []
        for m in metadata:
            if "rollout_start" in m and "demo_start" in m:
                # Sliding window metadata (matrix pairs)
                hover_texts.append(
                    f"Rollout {m['rollout_idx']} × Demo {m['demo_idx']}<br>"
                    f"Window: R[{m['rollout_start']}:{m['rollout_end']}] × D[{m['demo_start']}:{m['demo_end']}]<br>"
                    f"Success: {m.get('success', 'Unknown')}<br>"
                    f"Quality: {m.get('quality_label', 'unknown')}<br>"
                    f"Mean: {m.get('mean_influence', 0):.4f}<br>"
                    f"Std: {m.get('std_influence', 0):.4f}"
                )
            elif "window_start" in m and "window_end" in m:
                # Episode sliding window metadata (rollout or demo)
                if "rollout_idx" in m:
                    hover_texts.append(
                        f"Rollout {m['rollout_idx']}<br>"
                        f"Window: [{m['window_start']}:{m['window_end']}] (width={m.get('window_width', 'N/A')})<br>"
                        f"Aggregation: {m.get('aggregation', 'N/A')}<br>"
                        f"Success: {m.get('success', 'Unknown')}<br>"
                        f"Mean: {m.get('mean_influence', 0):.4f}<br>"
                        f"Std: {m.get('std_influence', 0):.4f}<br>"
                        f"Max: {m.get('max_influence', 0):.4f}"
                    )
                else:  # demo_idx
                    hover_texts.append(
                        f"Demo {m['demo_idx']}<br>"
                        f"Window: [{m['window_start']}:{m['window_end']}] (width={m.get('window_width', 'N/A')})<br>"
                        f"Aggregation: {m.get('aggregation', 'N/A')}<br>"
                        f"Quality: {m.get('quality_label', 'unknown')}<br>"
                        f"Mean: {m.get('mean_influence', 0):.4f}<br>"
                        f"Std: {m.get('std_influence', 0):.4f}<br>"
                        f"Max: {m.get('max_influence', 0):.4f}"
                    )
            elif "rollout_idx" in m and "demo_idx" in m:
                # Matrix pair metadata (without sliding window)
                hover_texts.append(
                    f"Rollout {m['rollout_idx']} × Demo {m['demo_idx']}<br>"
                    f"Rollout samples: {m.get('rollout_samples', 'N/A')}<br>"
                    f"Demo samples: {m.get('demo_samples', 'N/A')}<br>"
                    f"Success: {m.get('success', 'Unknown')}<br>"
                    f"Mean influence: {m.get('mean_influence', 0):.4f}<br>"
                    f"Std influence: {m.get('std_influence', 0):.4f}"
                )
            elif "rollout_idx" in m:
                # Rollout metadata
                hover_texts.append(
                    f"Rollout {m['rollout_idx']}<br>"
                    f"Samples: {m.get('num_samples', 'N/A')}<br>"
                    f"Success: {m.get('success', 'Unknown')}"
                )
            else:
                # Demo metadata
                hover_texts.append(
                    f"Demo {m.get('demo_idx', 'N/A')}<br>"
                    f"Quality: {m.get('quality_label', 'unknown')}<br>"
                    f"Samples: {m.get('num_samples', 'N/A')}"
                )

        # Create figure with continuous color scale
        import plotly.graph_objects as go

        color_label = color_by.replace("_", " ").title()
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=embeddings_2d[:, 0],
                y=embeddings_2d[:, 1],
                mode="markers",
                marker=dict(
                    size=10,
                    color=color_values,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title=color_label),
                    line=dict(width=1, color="white"),
                    opacity=0.7,
                ),
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            height=600,
            hovermode="closest",
            dragmode="lasso" if enable_selection else "zoom",
        )

        if enable_selection:
            selection = st.plotly_chart(
                fig,
                width="stretch",
                key=f"tsne_{key_suffix}",
                on_select="rerun",
                selection_mode=["points", "box", "lasso"],
            )
            return embeddings_2d, selection
        else:
            st.plotly_chart(fig, width="stretch", key=f"tsne_{key_suffix}")
            return embeddings_2d, None

    # Create groups for the plotting function (categorical coloring)
    groups = []

    if color_by == "success":
        # Color by success/failure
        success_mask = np.array([m.get("success", None) for m in metadata])

        for success_val, color, label in [
            (True, "green", "Success"),
            (False, "red", "Failure"),
            (None, "gray", "Unknown"),
        ]:
            mask = success_mask == success_val
            if not np.any(mask):
                continue

            indices = np.where(mask)[0]
            hover_text = []
            for j in indices:
                m = metadata[j]
                if "rollout_start" in m and "demo_start" in m:
                    # Sliding window metadata (matrix pairs)
                    hover_text.append(
                        f"Rollout {m['rollout_idx']} × Demo {m['demo_idx']}<br>"
                        f"Window: R[{m['rollout_start']}:{m['rollout_end']}] × D[{m['demo_start']}:{m['demo_end']}]<br>"
                        f"Success: {m.get('success', 'Unknown')}<br>"
                        f"Quality: {m.get('quality_label', 'unknown')}<br>"
                        f"Mean: {m.get('mean_influence', 0):.4f}"
                    )
                elif "window_start" in m and "window_end" in m:
                    # Episode sliding window metadata (rollout or demo)
                    if "rollout_idx" in m:
                        hover_text.append(
                            f"Rollout {m['rollout_idx']}<br>"
                            f"Window: [{m['window_start']}:{m['window_end']}] (width={m.get('window_width', 'N/A')})<br>"
                            f"Aggregation: {m.get('aggregation', 'N/A')}<br>"
                            f"Success: {m.get('success', 'Unknown')}<br>"
                            f"Mean: {m.get('mean_influence', 0):.4f}"
                        )
                    else:  # demo_idx
                        hover_text.append(
                            f"Demo {m['demo_idx']}<br>"
                            f"Window: [{m['window_start']}:{m['window_end']}] (width={m.get('window_width', 'N/A')})<br>"
                            f"Aggregation: {m.get('aggregation', 'N/A')}<br>"
                            f"Quality: {m.get('quality_label', 'unknown')}<br>"
                            f"Mean: {m.get('mean_influence', 0):.4f}"
                        )
                elif "rollout_idx" in m and "demo_idx" in m:
                    # Matrix pair metadata (without sliding window)
                    hover_text.append(
                        f"Rollout {m['rollout_idx']} × Demo {m['demo_idx']}<br>"
                        f"Rollout samples: {m.get('rollout_samples', 'N/A')}<br>"
                        f"Demo samples: {m.get('demo_samples', 'N/A')}<br>"
                        f"Success: {m.get('success', 'Unknown')}"
                    )
                elif "rollout_idx" in m:
                    # Rollout metadata
                    hover_text.append(
                        f"Rollout {m['rollout_idx']}<br>"
                        f"Samples: {m.get('num_samples', 'N/A')}<br>"
                        f"Success: {m.get('success', 'Unknown')}"
                    )
                else:
                    # Demo metadata
                    hover_text.append(
                        f"Demo {m.get('demo_idx', 'N/A')}<br>"
                        f"Quality: {m.get('quality_label', 'unknown')}<br>"
                        f"Samples: {m.get('num_samples', 'N/A')}"
                    )

            groups.append(
                {
                    "name": label,
                    "indices": indices,
                    "color": color,
                    "hover_texts": hover_text,
                }
            )

    elif color_by == "quality_label":
        # Color by quality label
        quality_labels = [m.get("quality_label", "unknown") for m in metadata]
        unique_qualities = sorted(set(quality_labels))

        color_map = {
            "worse": "red",
            "okay": "orange",
            "better": "green",
            "unknown": "gray",
        }

        for quality in unique_qualities:
            mask = np.array(quality_labels) == quality
            indices = np.where(mask)[0]

            hover_text = []
            for j in indices:
                m = metadata[j]
                if "rollout_start" in m and "demo_start" in m:
                    # Sliding window metadata (matrix pairs)
                    hover_text.append(
                        f"Rollout {m['rollout_idx']} × Demo {m['demo_idx']}<br>"
                        f"Window: R[{m['rollout_start']}:{m['rollout_end']}] × D[{m['demo_start']}:{m['demo_end']}]<br>"
                        f"Success: {m.get('success', 'Unknown')}<br>"
                        f"Quality: {m.get('quality_label', 'unknown')}<br>"
                        f"Mean: {m.get('mean_influence', 0):.4f}<br>"
                        f"Std: {m.get('std_influence', 0):.4f}"
                    )
                elif "window_start" in m and "window_end" in m:
                    # Episode sliding window metadata (rollout or demo)
                    if "rollout_idx" in m:
                        hover_text.append(
                            f"Rollout {m['rollout_idx']}<br>"
                            f"Window: [{m['window_start']}:{m['window_end']}] (width={m.get('window_width', 'N/A')})<br>"
                            f"Aggregation: {m.get('aggregation', 'N/A')}<br>"
                            f"Success: {m.get('success', 'Unknown')}<br>"
                            f"Mean: {m.get('mean_influence', 0):.4f}<br>"
                            f"Std: {m.get('std_influence', 0):.4f}"
                        )
                    else:  # demo_idx
                        hover_text.append(
                            f"Demo {m['demo_idx']}<br>"
                            f"Window: [{m['window_start']}:{m['window_end']}] (width={m.get('window_width', 'N/A')})<br>"
                            f"Aggregation: {m.get('aggregation', 'N/A')}<br>"
                            f"Quality: {m.get('quality_label', 'unknown')}<br>"
                            f"Mean: {m.get('mean_influence', 0):.4f}<br>"
                            f"Std: {m.get('std_influence', 0):.4f}"
                        )
                elif "rollout_idx" in m and "demo_idx" in m:
                    # Matrix pair metadata
                    hover_text.append(
                        f"Rollout {m['rollout_idx']} × Demo {m['demo_idx']}<br>"
                        f"Rollout samples: {m.get('rollout_samples', 'N/A')}<br>"
                        f"Demo samples: {m.get('demo_samples', 'N/A')}<br>"
                        f"Success: {m.get('success', 'Unknown')}<br>"
                        f"Quality: {m.get('quality_label', 'unknown')}<br>"
                        f"Mean influence: {m.get('mean_influence', 0):.4f}<br>"
                        f"Std influence: {m.get('std_influence', 0):.4f}"
                    )
                elif "demo_idx" in m:
                    # Demo metadata
                    hover_text.append(
                        f"Demo {m['demo_idx']}<br>"
                        f"Quality: {m.get('quality_label', 'unknown')}<br>"
                        f"Samples: {m.get('num_samples', 'N/A')}<br>"
                        f"Raw Length: {m.get('raw_length', 'N/A')}"
                    )
                else:
                    # Rollout metadata (shouldn't happen for quality_label coloring but just in case)
                    hover_text.append(
                        f"Rollout {m.get('rollout_idx', 'N/A')}<br>"
                        f"Samples: {m.get('num_samples', 'N/A')}<br>"
                        f"Success: {m.get('success', 'Unknown')}"
                    )

            groups.append(
                {
                    "name": quality.title(),
                    "indices": indices,
                    "color": color_map.get(quality, "gray"),
                    "hover_texts": hover_text,
                }
            )

    elif color_by == "annotation_label":
        # Color by annotation label
        annotation_labels = [m.get("annotation_label", "no label") for m in metadata]
        unique_labels = sorted(set(annotation_labels))

        # Use a colorful palette for annotation labels
        import plotly.express as px

        colors = px.colors.qualitative.Plotly
        if len(unique_labels) > len(colors):
            # If we have more labels than colors, cycle through them
            colors = colors * (len(unique_labels) // len(colors) + 1)

        for idx, label in enumerate(unique_labels):
            mask = np.array(annotation_labels) == label
            indices = np.where(mask)[0]

            hover_text = []
            for j in indices:
                m = metadata[j]
                if "rollout_start" in m and "demo_start" in m:
                    # Sliding window metadata (matrix pairs)
                    hover_text.append(
                        f"Rollout {m['rollout_idx']} × Demo {m['demo_idx']}<br>"
                        f"Window: R[{m['rollout_start']}:{m['rollout_end']}] × D[{m['demo_start']}:{m['demo_end']}]<br>"
                        f"Annotation: {m.get('annotation_label', 'no label')}<br>"
                        f"Success: {m.get('success', 'Unknown')}<br>"
                        f"Quality: {m.get('quality_label', 'unknown')}<br>"
                        f"Mean: {m.get('mean_influence', 0):.4f}<br>"
                        f"Std: {m.get('std_influence', 0):.4f}"
                    )
                elif "window_start" in m and "window_end" in m:
                    # Episode sliding window metadata (rollout or demo)
                    if "rollout_idx" in m:
                        hover_text.append(
                            f"Rollout {m['rollout_idx']}<br>"
                            f"Window: [{m['window_start']}:{m['window_end']}] (width={m.get('window_width', 'N/A')})<br>"
                            f"Annotation: {m.get('annotation_label', 'no label')}<br>"
                            f"Aggregation: {m.get('aggregation', 'N/A')}<br>"
                            f"Success: {m.get('success', 'Unknown')}<br>"
                            f"Mean: {m.get('mean_influence', 0):.4f}<br>"
                            f"Std: {m.get('std_influence', 0):.4f}"
                        )
                    else:  # demo_idx
                        hover_text.append(
                            f"Demo {m['demo_idx']}<br>"
                            f"Window: [{m['window_start']}:{m['window_end']}] (width={m.get('window_width', 'N/A')})<br>"
                            f"Annotation: {m.get('annotation_label', 'no label')}<br>"
                            f"Aggregation: {m.get('aggregation', 'N/A')}<br>"
                            f"Quality: {m.get('quality_label', 'unknown')}<br>"
                            f"Mean: {m.get('mean_influence', 0):.4f}<br>"
                            f"Std: {m.get('std_influence', 0):.4f}"
                        )
                elif "rollout_idx" in m and "demo_idx" in m:
                    # Matrix pair metadata
                    hover_text.append(
                        f"Rollout {m['rollout_idx']} × Demo {m['demo_idx']}<br>"
                        f"Annotation: {m.get('annotation_label', 'no label')}<br>"
                        f"Rollout samples: {m.get('rollout_samples', 'N/A')}<br>"
                        f"Demo samples: {m.get('demo_samples', 'N/A')}<br>"
                        f"Success: {m.get('success', 'Unknown')}<br>"
                        f"Quality: {m.get('quality_label', 'unknown')}<br>"
                        f"Mean influence: {m.get('mean_influence', 0):.4f}<br>"
                        f"Std influence: {m.get('std_influence', 0):.4f}"
                    )
                elif "rollout_idx" in m:
                    # Rollout timestep metadata
                    hover_text.append(
                        f"Rollout {m['rollout_idx']}<br>"
                        f"Timestep: {m.get('timestep', 'N/A')}<br>"
                        f"Annotation: {m.get('annotation_label', 'no label')}<br>"
                        f"Success: {m.get('success', 'Unknown')}<br>"
                        f"Samples: {m.get('num_samples', 'N/A')}"
                    )
                else:
                    # Demo timestep metadata
                    hover_text.append(
                        f"Demo {m.get('demo_idx', 'N/A')}<br>"
                        f"Timestep: {m.get('timestep', 'N/A')}<br>"
                        f"Annotation: {m.get('annotation_label', 'no label')}<br>"
                        f"Quality: {m.get('quality_label', 'unknown')}<br>"
                        f"Samples: {m.get('num_samples', 'N/A')}"
                    )

            groups.append(
                {
                    "name": label.replace("_", " ").title()
                    if label != "no label"
                    else "No Label",
                    "indices": indices,
                    "color": colors[idx % len(colors)],
                    "hover_texts": hover_text,
                }
            )

    # Use the pure plotting function to create the figure
    fig = plotting.create_embedding_plot(
        embeddings_2d=embeddings_2d,
        groups=groups,
        title=title,
    )

    # Update layout for selection if enabled
    if enable_selection:
        fig.update_layout(dragmode="lasso")
        selection = st.plotly_chart(
            fig,
            width="stretch",
            key=f"tsne_{key_suffix}",
            on_select="rerun",
            selection_mode=["points", "box", "lasso"],
        )
        return embeddings_2d, selection
    else:
        st.plotly_chart(fig, width="stretch", key=f"tsne_{key_suffix}")
        return embeddings_2d, None


def extract_matrix_pair_embeddings(
    data: InfluenceData,
    split: SplitType = "train",
    n_components: int = 10,
    method: str = "singular_values",
) -> Tuple[np.ndarray, List[Dict]]:
    """Extract SVD-based embeddings for all rollout-demo matrix pairs.

    For each (rollout, demo) pair, extracts the local influence matrix and
    embeds it using SVD (singular value decomposition).

    Args:
        data: InfluenceData object
        split: Which demo split to use ("train", "holdout", or "both")
        n_components: Number of singular values to use
        method: Embedding method to use:
            - "singular_values": Use top-k singular values as embedding (k-dimensional)
            - "low_rank_flatten": Reconstruct low-rank matrix using top-k components,
              then flatten (rollout_len × demo_len dimensional)

    Returns:
        Tuple of (embeddings, metadata) where:
        - embeddings: shape (num_rollout_eps * num_demo_eps, embedding_dim)
        - metadata: list of dicts with rollout_idx, demo_idx, success, quality_label
    """
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)

    # Safety check: prevent processing too many pairs
    num_rollouts = len(data.rollout_episodes)
    num_demos = len(demo_episodes)
    num_pairs = num_rollouts * num_demos
    MAX_MATRIX_PAIRS = 50_000  # 50K pairs max (e.g., 250 rollouts × 200 demos)

    if num_pairs > MAX_MATRIX_PAIRS:
        raise ValueError(
            f"Too many rollout-demo pairs to process: {num_rollouts:,} rollouts × "
            f"{num_demos:,} demos = {num_pairs:,} pairs "
            f"(max: {MAX_MATRIX_PAIRS:,}).\n\n"
            f"This would require extracting {num_pairs:,} transition matrices and computing SVD for each. "
            f"Try:\n"
            f"- Use a smaller split (train vs holdout vs both)\n"
            f"- This visualization is not suitable for very large datasets\n"
            f"- Use the other clustering views (demo or rollout clustering) instead"
        )

    embeddings = []
    metadata = []

    # For low_rank_flatten method, determine target shape upfront
    target_rollout_len = None
    target_demo_len = None
    if method == "low_rank_flatten":
        # Use median episode lengths as target to avoid extreme padding
        rollout_lengths = [ep.num_samples for ep in data.rollout_episodes]
        demo_lengths = [ep.num_samples for ep in demo_episodes]
        target_rollout_len = int(np.median(rollout_lengths))
        target_demo_len = int(np.median(demo_lengths))

    # For each rollout-demo pair
    for rollout_ep in data.rollout_episodes:
        rollout_sample_indices = np.arange(
            rollout_ep.sample_start_idx, rollout_ep.sample_end_idx
        )

        for demo_ep_idx, demo_ep in enumerate(demo_episodes):
            demo_sample_idxs = ep_idxs[demo_ep_idx]

            # Extract the local transition matrix for this pair
            # Shape: (num_rollout_timesteps, num_demo_timesteps)
            transition_matrix = influence_matrix[
                np.ix_(rollout_sample_indices, demo_sample_idxs)
            ]

            # Apply SVD
            try:
                U, s, Vt = np.linalg.svd(transition_matrix, full_matrices=False)

                if method == "singular_values":
                    # Method 1: Use top-k singular values as embedding
                    if len(s) < n_components:
                        embedding = np.zeros(n_components)
                        embedding[: len(s)] = s
                    else:
                        embedding = s[:n_components]

                elif method == "low_rank_flatten":
                    # Method 2: Reconstruct low-rank matrix to fixed shape and flatten
                    k = min(n_components, len(s))

                    # Get top-k components
                    U_k = U[:, :k]
                    S_k = np.diag(s[:k])
                    Vt_k = Vt[:k, :]

                    # Adjust U_k to target_rollout_len
                    if U_k.shape[0] < target_rollout_len:
                        # Pad with zeros
                        U_k_padded = np.zeros((target_rollout_len, k))
                        U_k_padded[: U_k.shape[0], :] = U_k
                        U_k = U_k_padded
                    elif U_k.shape[0] > target_rollout_len:
                        # Truncate
                        U_k = U_k[:target_rollout_len, :]

                    # Adjust Vt_k to target_demo_len
                    if Vt_k.shape[1] < target_demo_len:
                        # Pad with zeros
                        Vt_k_padded = np.zeros((k, target_demo_len))
                        Vt_k_padded[:, : Vt_k.shape[1]] = Vt_k
                        Vt_k = Vt_k_padded
                    elif Vt_k.shape[1] > target_demo_len:
                        # Truncate
                        Vt_k = Vt_k[:, :target_demo_len]

                    # Reconstruct to fixed shape: (target_rollout_len, target_demo_len)
                    reconstructed = U_k @ S_k @ Vt_k

                    # Flatten the reconstructed matrix
                    embedding = reconstructed.flatten()

                else:
                    raise ValueError(f"Unknown method: {method}")

                embeddings.append(embedding)

                # Get quality label if available
                quality_label = "unknown"
                if data.demo_quality_labels is not None:
                    quality_label = data.demo_quality_labels.get(
                        demo_ep.index, "unknown"
                    )

                # Compute statistics of the transition matrix for coloring
                mean_influence = float(np.mean(transition_matrix))
                std_influence = float(np.std(transition_matrix))
                max_influence = float(np.max(transition_matrix))
                min_influence = float(np.min(transition_matrix))

                metadata.append(
                    {
                        "rollout_idx": rollout_ep.index,
                        "demo_idx": demo_ep.index,
                        "success": rollout_ep.success,
                        "quality_label": quality_label,
                        "rollout_samples": len(rollout_sample_indices),
                        "demo_samples": len(demo_sample_idxs),
                        "mean_influence": mean_influence,
                        "std_influence": std_influence,
                        "max_influence": max_influence,
                        "min_influence": min_influence,
                    }
                )

            except np.linalg.LinAlgError:
                # Skip if SVD fails
                continue

    # For low_rank_flatten, padding was needed but we'll use a different approach
    # All embeddings should already have the same length if we use fixed target shape
    return np.array(embeddings), metadata


def _extract_episode_sliding_windows(args):
    """Helper function for parallel extraction of sliding windows from a single episode.

    Args:
        args: Tuple of (episode_data, window_width, stride, aggregation_method, n_components, episode_idx, episode_type, metadata, annotation_labels_dict)

    Returns:
        Tuple of (embeddings_list, metadata_list)
    """
    (
        episode_data,
        window_width,
        stride,
        aggregation_method,
        n_components,
        episode_idx,
        episode_type,
        extra_metadata,
        annotation_labels_dict,
    ) = args

    # episode_data is 2D: either (rollout_timesteps, demo_samples) for rollout
    # or (rollout_samples, demo_timesteps) for demo
    num_timesteps = (
        episode_data.shape[0] if episode_type == "rollout" else episode_data.shape[1]
    )

    embeddings = []
    metadata = []

    # Extract sliding windows along the timestep dimension
    for start_idx in range(0, num_timesteps - window_width + 1, stride):
        end_idx = start_idx + window_width

        if episode_type == "rollout":
            # Window over rollout timesteps: shape (window_width, demo_samples)
            window = episode_data[start_idx:end_idx, :]
            # Aggregate over window/time dimension (axis=0) to get influence pattern across demos
            agg_axis = 0
        else:  # demo
            # Window over demo timesteps: shape (rollout_samples, window_width)
            window = episode_data[:, start_idx:end_idx]
            # Aggregate over window/time dimension (axis=1) to get influence pattern across rollouts
            agg_axis = 1

        # Aggregate the window according to the specified method
        if aggregation_method == "svd":
            # Use top-k singular values as embedding
            try:
                U, s, Vt = np.linalg.svd(window, full_matrices=False)
                if len(s) < n_components:
                    embedding = np.zeros(n_components)
                    embedding[: len(s)] = s
                else:
                    embedding = s[:n_components]
            except np.linalg.LinAlgError:
                # If SVD fails, use zeros
                embedding = np.zeros(n_components)
        elif aggregation_method == "mean":
            embedding = np.mean(window, axis=agg_axis)
        elif aggregation_method == "max":
            embedding = np.max(window, axis=agg_axis)
        elif aggregation_method == "min":
            embedding = np.min(window, axis=agg_axis)
        elif aggregation_method == "std":
            embedding = np.std(window, axis=agg_axis)
        elif aggregation_method == "sum":
            embedding = np.sum(window, axis=agg_axis)
        elif aggregation_method == "median":
            embedding = np.median(window, axis=agg_axis)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        embeddings.append(embedding)

        # Get annotation label for the window (use middle timestep as representative)
        middle_timestep = start_idx + window_width // 2
        annotation_label = annotation_labels_dict.get(middle_timestep, "no label")

        # Store metadata
        window_meta = {
            f"{episode_type}_idx": episode_idx,
            "window_start": start_idx,
            "window_end": end_idx,
            "window_width": window_width,
            "aggregation": aggregation_method,
            "annotation_label": annotation_label,
            "mean_influence": float(np.mean(window)),
            "std_influence": float(np.std(window)),
            "max_influence": float(np.max(window)),
            "min_influence": float(np.min(window)),
        }
        window_meta.update(extra_metadata)
        metadata.append(window_meta)

    return embeddings, metadata


def extract_episode_sliding_window_embeddings(
    data: InfluenceData,
    split: SplitType = "train",
    level: Literal["rollout", "demo"] = "rollout",
    window_width: int = 10,
    stride: int = 1,
    aggregation_method: Literal[
        "mean", "max", "min", "std", "sum", "median", "svd"
    ] = "mean",
    n_components: int = 10,
    use_parallel: bool = True,
    annotations: Optional[Dict] = None,
    apply_gaussian: bool = False,
    gaussian_sigma: float = 1.0,
    rollout_normalization: Literal["none", "center", "normalize"] = "none",
) -> Tuple[np.ndarray, List[Dict]]:
    """Extract sliding window embeddings for individual rollouts or demos.

    For each rollout/demo, compute sliding windows along the timestep dimension
    and aggregate each window into a vector using the specified method.

    Args:
        data: InfluenceData object
        split: Which demo split to use ("train", "holdout", or "both")
        level: Whether to extract windows for "rollout" or "demo" episodes
        window_width: Width of sliding window in timesteps
        stride: Stride for sliding window
        aggregation_method: How to aggregate the window into a vector
        n_components: Number of singular values to use (only for svd method)
        use_parallel: Whether to use multiprocessing for parallel extraction
        apply_gaussian: Whether to apply 2D Gaussian smoothing to influence matrices
        gaussian_sigma: Sigma for Gaussian smoothing (only if apply_gaussian is True)
        rollout_normalization: When level is "rollout", "none" | "center" | "normalize"
            (same semantics as in extract_dimred_timestep_embeddings).

    Returns:
        Tuple of (embeddings, metadata) where:
        - embeddings: shape (total_windows, embedding_dim)
        - metadata: list of dicts with episode_idx, window position, and stats
    """
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)

    all_embeddings = []
    all_metadata = []

    if level == "rollout":
        # For each rollout, extract sliding windows over rollout timesteps
        # Each window aggregates to a vector of length num_demo_samples

        # Prepare arguments for parallel processing
        args_list = []
        for rollout_ep in data.rollout_episodes:
            rollout_sample_indices = np.arange(
                rollout_ep.sample_start_idx, rollout_ep.sample_end_idx
            )

            # Get the rollout's influence data: shape (num_rollout_timesteps, num_demo_samples)
            rollout_data = influence_matrix[rollout_sample_indices, :].copy()
            if apply_gaussian and gaussian_sigma > 0:
                rollout_data = gaussian_filter(rollout_data, sigma=gaussian_sigma)

            # Skip if rollout is too short for the window
            if rollout_data.shape[0] < window_width:
                continue

            # Pre-compute annotation labels for all timesteps in this episode
            annotation_labels_dict = {}
            if annotations:
                episode_id_str = str(rollout_ep.index)
                episode_annotations = get_episode_annotations(
                    annotations, episode_id_str, split="rollout"
                )
                # Build a dict mapping timestep -> label for fast lookup
                for t in range(rollout_data.shape[0]):
                    annotation_labels_dict[t] = get_label_for_frame(
                        t, episode_annotations
                    )

            extra_metadata = {
                "success": rollout_ep.success,
                "num_samples": rollout_ep.num_samples,
            }

            args_list.append(
                (
                    rollout_data,
                    window_width,
                    stride,
                    aggregation_method,
                    n_components,
                    rollout_ep.index,
                    "rollout",
                    extra_metadata,
                    annotation_labels_dict,
                )
            )

        # Process in parallel or sequentially
        if use_parallel and len(args_list) > 1:
            num_processes = min(cpu_count(), len(args_list))
            with Pool(processes=num_processes) as pool:
                results = pool.map(_extract_episode_sliding_windows, args_list)
        else:
            results = [_extract_episode_sliding_windows(args) for args in args_list]

        # Collect results
        for emb, meta in results:
            all_embeddings.extend(emb)
            all_metadata.extend(meta)

        # Per-rollout normalization (same as timestep representation)
        if rollout_normalization != "none" and len(all_embeddings) > 0:
            embeddings_arr = np.array(all_embeddings)
            rollout_ids = np.array(
                [m["rollout_idx"] for m in all_metadata]
            )
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
        # For each demo, extract sliding windows over demo timesteps
        # Each window aggregates to a vector of length num_rollout_samples

        # Prepare arguments for parallel processing
        args_list = []
        for demo_ep_idx, demo_ep in enumerate(demo_episodes):
            demo_sample_idxs = ep_idxs[demo_ep_idx]

            # Get the demo's influence data: shape (num_rollout_samples, num_demo_timesteps)
            demo_data = influence_matrix[:, demo_sample_idxs].copy()
            if apply_gaussian and gaussian_sigma > 0:
                demo_data = gaussian_filter(demo_data, sigma=gaussian_sigma)

            # Skip if demo is too short for the window
            if demo_data.shape[1] < window_width:
                continue

            # Get quality label if available
            quality_label = "unknown"
            if data.demo_quality_labels is not None:
                quality_label = data.demo_quality_labels.get(demo_ep.index, "unknown")

            # Pre-compute annotation labels for all timesteps in this episode
            annotation_labels_dict = {}
            if annotations:
                episode_id_str = str(demo_ep.index)
                # Determine which split this demo belongs to
                demo_split_type = "train" if split in ["train", "both"] else "holdout"
                episode_annotations = get_episode_annotations(
                    annotations, episode_id_str, split=demo_split_type
                )
                # Build a dict mapping timestep -> label for fast lookup
                for t in range(demo_data.shape[1]):
                    annotation_labels_dict[t] = get_label_for_frame(
                        t, episode_annotations
                    )

            extra_metadata = {
                "quality_label": quality_label,
                "num_samples": demo_ep.num_samples,
                "raw_length": demo_ep.raw_length,
            }

            args_list.append(
                (
                    demo_data,
                    window_width,
                    stride,
                    aggregation_method,
                    n_components,
                    demo_ep.index,
                    "demo",
                    extra_metadata,
                    annotation_labels_dict,
                )
            )

        # Process in parallel or sequentially
        if use_parallel and len(args_list) > 1:
            num_processes = min(cpu_count(), len(args_list))
            with Pool(processes=num_processes) as pool:
                results = pool.map(_extract_episode_sliding_windows, args_list)
        else:
            results = [_extract_episode_sliding_windows(args) for args in args_list]

        # Collect results
        for embeddings, metadata in results:
            all_embeddings.extend(embeddings)
            all_metadata.extend(metadata)

    return np.array(all_embeddings), all_metadata


def get_influence_window_image(
    data: InfluenceData,
    meta: dict,
    split: SplitType = "train",
) -> np.ndarray:
    """Extract the influence window matrix for visualization.

    Args:
        data: InfluenceData object
        meta: Metadata dict containing window information
        split: Which demo split to use

    Returns:
        2D numpy array representing the influence window
    """
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)

    if "rollout_idx" in meta and "demo_idx" not in meta:
        # Rollout window - extract window over rollout timesteps
        rollout_ep = data.rollout_episodes[meta["rollout_idx"]]

        # Get absolute sample indices for this window
        window_start_abs = rollout_ep.sample_start_idx + meta["window_start"]
        window_end_abs = rollout_ep.sample_start_idx + meta["window_end"]

        # Extract window from influence matrix
        window = influence_matrix[window_start_abs:window_end_abs, :]

    elif "demo_idx" in meta and "rollout_idx" not in meta:
        # Demo window - extract window over demo timesteps
        demo_ep = demo_episodes[meta["demo_idx"]]
        demo_sample_idxs = ep_idxs[meta["demo_idx"]]

        # Get indices for this window within the demo
        window_start = meta["window_start"]
        window_end = meta["window_end"]

        # Extract window from influence matrix
        window_demo_idxs = demo_sample_idxs[window_start:window_end]
        window = influence_matrix[:, window_demo_idxs]

    else:
        # Fallback: return empty array
        window = np.zeros((10, 10))

    return window


def render_matrix_embedding_suggestions():
    """Render suggestions for embedding full influence matrix pairs."""
    st.header("Matrix Embedding Methods")

    st.markdown("""
    For clustering based on local rollout-demonstration influence matrix pairs,
    you need to embed each matrix into a single vector. Here are several approaches:
    """)

    methods = [
        {
            "Method": "Flatten",
            "Description": "Vectorize the entire matrix",
            "Dimensionality": "rollout_len × demo_len",
            "Pros": "Preserves all information",
            "Cons": "Very high dimensional, may be sparse",
            "Use Case": "When you have many matrices but each is small",
        },
        {
            "Method": "Statistics",
            "Description": "Compute summary statistics (mean, std, min, max, percentiles)",
            "Dimensionality": "~5-10 features",
            "Pros": "Low dimensional, interpretable",
            "Cons": "Loses spatial structure",
            "Use Case": "Quick exploration, interpretable clusters",
        },
        {
            "Method": "SVD/PCA",
            "Description": "Low-rank approximation using top-k singular values/components",
            "Dimensionality": "User-defined k (e.g., 10-50)",
            "Pros": "Captures main variance, reduces noise",
            "Cons": "Requires choosing k, less interpretable",
            "Use Case": "Large matrices, noise reduction",
        },
        {
            "Method": "Row/Col Aggregation",
            "Description": "Concatenate mean/max along each axis",
            "Dimensionality": "rollout_len + demo_len",
            "Pros": "Preserves temporal structure",
            "Cons": "Loses correlation between axes",
            "Use Case": "When temporal patterns matter more than interactions",
        },
        {
            "Method": "Structural Features",
            "Description": "Extract features like diagonal strength, symmetry, peak counts",
            "Dimensionality": "~10-20 features",
            "Pros": "Domain-specific, interpretable",
            "Cons": "Requires domain knowledge to design",
            "Use Case": "When you have specific hypotheses about matrix structure",
        },
    ]

    import pandas as pd

    df = pd.DataFrame(methods)
    st.table(df)

    st.divider()

    st.subheader("Example Code Snippets")

    with st.expander("1. Flatten Method"):
        st.code(
            """
# Extract a single transition matrix
rollout_ep = data.rollout_episodes[rollout_idx]
demo_ep = demo_episodes[demo_idx]

rollout_samples = np.arange(rollout_ep.sample_start_idx, rollout_ep.sample_end_idx)
demo_samples = ep_idxs[demo_idx]

transition_matrix = influence_matrix[np.ix_(rollout_samples, demo_samples)]

# Flatten to vector
embedding = transition_matrix.flatten()
""",
            language="python",
        )

    with st.expander("2. Statistics Method"):
        st.code(
            """
# Compute summary statistics
embedding = np.array([
    np.mean(transition_matrix),
    np.std(transition_matrix),
    np.min(transition_matrix),
    np.max(transition_matrix),
    np.percentile(transition_matrix, 25),
    np.percentile(transition_matrix, 50),
    np.percentile(transition_matrix, 75),
])
""",
            language="python",
        )

    with st.expander("3. SVD/PCA Method"):
        st.code(
            """
from sklearn.decomposition import TruncatedSVD

# Apply SVD to get low-rank approximation
k = 10  # Number of components
svd = TruncatedSVD(n_components=k)

# Flatten and fit (or use directly if matrix is small)
flattened = transition_matrix.flatten().reshape(1, -1)
embedding = svd.fit_transform(flattened).flatten()

# Alternative: use singular values directly
U, s, Vt = np.linalg.svd(transition_matrix, full_matrices=False)
embedding = s[:k]  # Top k singular values
""",
            language="python",
        )

    with st.expander("4. Row/Column Aggregation Method"):
        st.code(
            """
# Aggregate along each axis
row_means = np.mean(transition_matrix, axis=1)  # Mean across demo timesteps
col_means = np.mean(transition_matrix, axis=0)  # Mean across rollout timesteps

# Concatenate
embedding = np.concatenate([row_means, col_means])

# Can also use max, std, etc.
row_max = np.max(transition_matrix, axis=1)
col_max = np.max(transition_matrix, axis=0)
embedding_max = np.concatenate([row_max, col_max])
""",
            language="python",
        )

    with st.expander("5. Structural Features Method"):
        st.code(
            """
# Extract structural features
features = []

# Diagonal strength
min_dim = min(transition_matrix.shape)
diagonal = np.array([transition_matrix[i, i] for i in range(min_dim)])
features.extend([
    np.mean(diagonal),
    np.std(diagonal),
])

# Symmetry (if matrix is square or can be made square)
if transition_matrix.shape[0] == transition_matrix.shape[1]:
    symmetry = np.mean(np.abs(transition_matrix - transition_matrix.T))
    features.append(symmetry)

# Peak counts
from scipy.ndimage import maximum_filter, minimum_filter
local_max = transition_matrix == maximum_filter(transition_matrix, size=3)
local_min = transition_matrix == minimum_filter(transition_matrix, size=3)
features.extend([
    np.sum(local_max),
    np.sum(local_min),
])

# Gradient magnitude
grad_x = np.gradient(transition_matrix, axis=0)
grad_y = np.gradient(transition_matrix, axis=1)
grad_mag = np.sqrt(grad_x**2 + grad_y**2)
features.extend([
    np.mean(grad_mag),
    np.std(grad_mag),
])

embedding = np.array(features)
""",
            language="python",
        )

    st.divider()

    st.subheader("Recommendations")

    st.markdown("""
    **For initial exploration:**
    - Start with **Statistics** method for quick insights
    - Use **t-SNE** or **UMAP** on the embeddings to visualize clusters

    **For detailed analysis:**
    - Try **SVD/PCA** to capture main patterns while reducing noise
    - Use **Structural Features** if you have specific hypotheses

    **For comparison:**
    - Implement multiple methods and compare clustering results
    - Use silhouette scores or other metrics to evaluate cluster quality
    """)


@st.fragment
def _render_demo_clustering_fragment(data, demo_split):
    with st.expander("1. Cluster Training Demonstrations", expanded=False):
        st.markdown("""
        Each demonstration gets an embedding vector representing its influence across all rollout samples.
        Goal: Identify separable clusters of demonstrations based on influence patterns.

        **Note:** Each demo's embedding is computed by averaging the influence matrix over demo timesteps,
        resulting in a single influence value per rollout sample.
        """)
        show_demo_tsne_key = f"show_demo_clustering_tsne_{demo_split}"
        if st.button(
            "Generate Demo Clustering t-SNE", key=f"gen_demo_clustering_{demo_split}"
        ):
            st.session_state[show_demo_tsne_key] = True

        if st.session_state.get(show_demo_tsne_key, False):
            try:
                with st.spinner("Extracting demonstration embeddings..."):
                    demo_embeddings, demo_metadata = extract_demo_embeddings(
                        data, demo_split
                    )
                st.caption(
                    f"Extracted {len(demo_embeddings)} embeddings, dim {demo_embeddings.shape[1]}"
                )
                render_tsne_plot(
                    demo_embeddings,
                    demo_metadata,
                    title="t-SNE: Demo Influence Embeddings",
                    color_by="quality_label",
                    key_suffix=f"demos_{demo_split}",
                )
            except ValueError as e:
                st.error(f"⚠️ Cannot generate demo clustering: {str(e)}")


@st.fragment
def _render_rollout_clustering_fragment(data, demo_split):
    with st.expander("2. Cluster Rollouts", expanded=False):
        st.markdown("""
        Each rollout gets an embedding vector representing influences from all demonstration samples.
        Goal: Identify whether successes and failures form distinct clusters based on influence patterns.
        """)
        show_rollout_tsne_key = f"show_rollout_clustering_tsne_{demo_split}"
        if st.button(
            "Generate Rollout Clustering t-SNE",
            key=f"gen_rollout_clustering_{demo_split}",
        ):
            st.session_state[show_rollout_tsne_key] = True

        if st.session_state.get(show_rollout_tsne_key, False):
            try:
                with st.spinner("Extracting rollout embeddings..."):
                    rollout_embeddings, rollout_metadata = extract_rollout_embeddings(
                        data, demo_split
                    )
                st.caption(
                    f"Extracted {len(rollout_embeddings)} embeddings, dim {rollout_embeddings.shape[1]}"
                )
                render_tsne_plot(
                    rollout_embeddings,
                    rollout_metadata,
                    title="t-SNE: Rollout Influence Embeddings",
                    color_by="success",
                    key_suffix=f"rollouts_{demo_split}",
                )
            except ValueError as e:
                st.error(f"⚠️ Cannot generate rollout clustering: {str(e)}")


@st.fragment
def _render_sliding_window_clustering_fragment(data, demo_split):
    with st.expander("4. Cluster Sliding Windows (All Pairs)", expanded=False):
        st.markdown("""
        Extract sliding windows from all rollout-demo influence matrices and cluster them.
        Each point represents a local window from a specific rollout-demo pair.

        **Use case:** Identify recurring local patterns across different rollout-demo interactions.
        """)

        # Sliding window parameters
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            window_height = st.number_input(
                "Window height (rollout)",
                min_value=2,
                max_value=20,
                value=5,
                key=f"sw_window_height_{demo_split}",
                help="Height of sliding window in rollout dimension",
            )
        with col2:
            window_width = st.number_input(
                "Window width (demo)",
                min_value=2,
                max_value=20,
                value=5,
                key=f"sw_window_width_{demo_split}",
                help="Width of sliding window in demo dimension",
            )
        with col3:
            stride = st.number_input(
                "Stride",
                min_value=1,
                max_value=10,
                value=2,
                key=f"sw_stride_{demo_split}",
                help="Stride for sliding window",
            )
        with col4:
            embedding_method = st.selectbox(
                "Embedding method",
                options=["flatten", "hog"],
                index=0,
                key=f"sw_method_{demo_split}",
                help="flatten: raw values | hog: gradient histogram",
            )

        # t-SNE visualization controls
        st.divider()
        st.markdown("**Preprocessing Options**")

        scaling_method = st.selectbox(
            "Feature scaling",
            options=["standard", "robust", "minmax", "none"],
            format_func=lambda x: {
                "standard": "StandardScaler ",
                "robust": "RobustScaler (outlier-resistant)",
                "minmax": "MinMaxScaler (0-1 range)",
                "none": "None ",
            }[x],
            index=0,
            key=f"episode_sw_scaling_{demo_split}",
            help="Feature scaling method applied before t-SNE. StandardScaler (zero mean, unit variance) is recommended for most cases. RobustScaler is better if you have outliers.",
        )

        st.markdown("**t-SNE Visualization Controls**")

        col6, col7, col8, col9 = st.columns(4)
        with col6:
            color_by_sw = st.selectbox(
                "Color by",
                options=["success", "quality_label", "mean_influence", "std_influence"],
                format_func=lambda x: {
                    "success": "Success/Failure",
                    "quality_label": "Demo Quality",
                    "mean_influence": "Mean Influence",
                    "std_influence": "Std Influence",
                }[x],
                key=f"color_by_sw_{demo_split}",
                help="Choose how to color the t-SNE plot points",
            )
        with col7:
            perplexity_sw = st.slider(
                "Perplexity",
                min_value=2,
                max_value=50,
                value=30,
                step=1,
                key=f"perplexity_sw_{demo_split}",
                help="t-SNE perplexity parameter (balance between local and global structure)",
            )
        with col8:
            use_max_samples_sw = st.checkbox(
                "Limit maximum samples",
                value=False,
                key=f"use_max_samples_sw_{demo_split}",
                help="Enable to subsample embeddings for faster t-SNE computation",
            )
        with col9:
            max_samples_sw = st.slider(
                "Maximum samples",
                min_value=100,
                max_value=100000,
                value=1000,
                step=100,
                key=f"max_samples_sw_{demo_split}",
                disabled=not use_max_samples_sw,
                help="Maximum number of samples to use for t-SNE visualization",
            )

        if st.button(
            "Generate Sliding Window t-SNE",
            key=f"gen_sliding_window_clustering_{demo_split}",
        ):
            try:
                with st.spinner(
                    "Extracting sliding window embeddings from all pairs..."
                ):
                    sw_embeddings, sw_metadata = (
                        extract_sliding_window_embeddings_all_pairs(
                            data,
                            demo_split,
                            window_height=window_height,
                            window_width=window_width,
                            stride=stride,
                            method=embedding_method,
                            hog_bins=8,
                        )
                    )

                # Apply sampling if enabled
                if use_max_samples_sw and len(sw_embeddings) > max_samples_sw:
                    st.info(
                        f"Sampling {max_samples_sw} out of {len(sw_embeddings)} embeddings for t-SNE visualization"
                    )
                    # Random sampling
                    rng = np.random.RandomState(42)
                    sample_indices = rng.choice(
                        len(sw_embeddings), size=max_samples_sw, replace=False
                    )
                    sw_embeddings = sw_embeddings[sample_indices]
                    sw_metadata = [sw_metadata[i] for i in sample_indices]

                # Store in session state (evict old caches first)
                cache_key = f"sw_embeddings_{demo_split}_{window_height}_{window_width}_{stride}_{embedding_method}_{max_samples_sw if use_max_samples_sw else 'all'}"
                _register_embedding_cache("sw_embeddings", cache_key)
                st.session_state[cache_key] = (sw_embeddings, sw_metadata)
                st.session_state[f"{cache_key}_tsne_cache"] = {}

                st.success(
                    f"Extracted {len(sw_embeddings)} embeddings, dim {sw_embeddings.shape[1]}"
                )
            except ValueError as e:
                st.error(f"⚠️ Cannot generate sliding window clustering: {str(e)}")

        # Display the plot if embeddings exist
        cache_key = f"sw_embeddings_{demo_split}_{window_height}_{window_width}_{stride}_{embedding_method}_{max_samples_sw if use_max_samples_sw else 'all'}"
        if cache_key in st.session_state:
            sw_embeddings, sw_metadata = st.session_state[cache_key]

            # Get or create t-SNE cache for this configuration (include scaling method)
            tsne_cache_key = f"{cache_key}_tsne_cache_{scaling_method}"
            if tsne_cache_key not in st.session_state:
                st.session_state[tsne_cache_key] = {}

            render_tsne_plot(
                sw_embeddings,
                sw_metadata,
                title=f"t-SNE: Sliding Window Embeddings ({window_height}×{window_width}, stride={stride})",
                color_by=color_by_sw,
                perplexity=perplexity_sw,
                key_suffix=f"sliding_windows_{demo_split}_{window_height}_{window_width}_{stride}_{embedding_method}_{color_by_sw}_{perplexity_sw}",
                tsne_cache=st.session_state[tsne_cache_key],
            )


@st.fragment
def _render_episode_sliding_window_clustering_fragment(
    data, demo_split, annotations=None
):
    with st.expander("3. Cluster Rollout/Demo Sliding Windows", expanded=False):
        st.markdown("""
        Extract sliding windows from individual rollouts or demos (not pairs).
        Each window is aggregated along the timestep dimension into a single vector.

        **Use case:** Identify temporal patterns within individual episodes and see how they cluster.
        """)

        # Level selector
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            level = st.selectbox(
                "Cluster",
                options=["rollout", "demo"],
                format_func=lambda x: "Rollouts" if x == "rollout" else "Demos",
                key=f"episode_sw_level_{demo_split}",
                help="Choose whether to cluster rollout or demo windows",
            )

        with col2:
            window_width = st.number_input(
                "Window width (timesteps)",
                min_value=2,
                max_value=50,
                value=10,
                key=f"episode_sw_width_{demo_split}",
                help="Width of sliding window in timesteps",
            )

        with col3:
            stride = st.number_input(
                "Stride",
                min_value=1,
                max_value=20,
                value=1,
                key=f"episode_sw_stride_{demo_split}",
                help="Stride for sliding window",
            )

        with col4:
            aggregation = st.selectbox(
                "Aggregation method",
                options=["svd", "mean", "max", "min", "std", "sum", "median"],
                index=0,
                key=f"episode_sw_agg_{demo_split}",
                help="How to aggregate the window into a vector",
            )

        with col5:
            use_parallel = st.checkbox(
                "Use parallel processing",
                value=False,
                key=f"episode_sw_parallel_{demo_split}",
                help="Use multiprocessing for faster extraction (may cause pickling errors with Streamlit fragments)",
            )

        # SVD components parameter (only shown if svd is selected)
        if aggregation == "svd":
            n_components = st.number_input(
                "Number of singular values",
                min_value=2,
                max_value=50,
                value=10,
                key=f"episode_sw_n_components_{demo_split}",
                help="Number of top singular values to use as features",
            )
        else:
            n_components = 10  # Default value when not using SVD

        # Feature scaling selector
        scaling_method = st.selectbox(
            "Feature scaling",
            options=["standard", "robust", "minmax", "none"],
            format_func=lambda x: {
                "standard": "StandardScaler ",
                "robust": "RobustScaler (outlier-resistant)",
                "minmax": "MinMaxScaler (0-1 range)",
                "none": "None ",
            }[x],
            index=0,
            key=f"episode_sw_scaling_section3_{demo_split}",
            help="Feature scaling method applied before t-SNE. StandardScaler (zero mean, unit variance) is recommended for most cases. RobustScaler is better if you have outliers.",
        )

        # t-SNE visualization controls
        st.divider()
        st.markdown("**t-SNE Visualization Controls**")

        col6, col7, col8, col9 = st.columns(4)
        with col6:
            # Determine available color options based on level
            if level == "rollout":
                color_options = [
                    "success",
                    "annotation_label",
                    "window_start",
                    "mean_influence",
                    "std_influence",
                    "max_influence",
                ]
                default_color = "success"
            else:  # demo
                color_options = [
                    "quality_label",
                    "annotation_label",
                    "window_start",
                    "mean_influence",
                    "std_influence",
                    "max_influence",
                ]
                default_color = "quality_label"

            color_by = st.selectbox(
                "Color by",
                options=color_options,
                index=0,
                format_func=lambda x: {
                    "success": "Success/Failure",
                    "quality_label": "Demo Quality",
                    "annotation_label": "Annotation Label",
                    "window_start": "Window Start Position",
                    "mean_influence": "Mean Influence",
                    "std_influence": "Std Influence",
                    "max_influence": "Max Influence",
                }.get(x, x),
                key=f"episode_sw_color_{demo_split}",
                help="Choose how to color the t-SNE plot points",
            )

        with col7:
            perplexity = st.slider(
                "Perplexity",
                min_value=2,
                max_value=50,
                value=30,
                step=1,
                key=f"episode_sw_perplexity_{demo_split}",
                help="t-SNE perplexity parameter",
            )

        with col8:
            use_max_samples = st.checkbox(
                "Limit maximum samples",
                value=False,
                key=f"episode_sw_limit_{demo_split}",
                help="Enable to subsample embeddings for faster t-SNE",
            )

        with col9:
            max_samples = st.slider(
                "Maximum samples",
                min_value=100,
                max_value=100000,
                value=5000,
                step=100,
                key=f"episode_sw_max_{demo_split}",
                disabled=not use_max_samples,
                help="Maximum number of samples to use for t-SNE",
            )

        if st.button(
            f"Generate {level.title()} Sliding Window t-SNE",
            key=f"gen_episode_sw_{demo_split}",
        ):
            try:
                with st.spinner(
                    f"Extracting sliding window embeddings from {level}s..."
                ):
                    embeddings, metadata = extract_episode_sliding_window_embeddings(
                        data,
                        split=demo_split,
                        level=level,
                        window_width=window_width,
                        stride=stride,
                        aggregation_method=aggregation,
                        n_components=n_components,
                        use_parallel=use_parallel,
                        annotations=annotations,
                    )

                if len(embeddings) == 0:
                    st.warning(
                        f"No windows extracted. Try reducing window width or check that {level}s are long enough."
                    )
                    return

                # Apply sampling if enabled
                if use_max_samples and len(embeddings) > max_samples:
                    st.info(
                        f"Sampling {max_samples} out of {len(embeddings)} embeddings for t-SNE visualization"
                    )
                    rng = np.random.RandomState(42)
                    sample_indices = rng.choice(
                        len(embeddings), size=max_samples, replace=False
                    )
                    embeddings = embeddings[sample_indices]
                    metadata = [metadata[i] for i in sample_indices]

                # Store in session state (evict old caches first)
                cache_key_suffix = f"_{n_components}" if aggregation == "svd" else ""
                cache_key = f"episode_sw_{demo_split}_{level}_{window_width}_{stride}_{aggregation}{cache_key_suffix}_{max_samples if use_max_samples else 'all'}"
                _register_embedding_cache("episode_sw", cache_key)
                st.session_state[cache_key] = (embeddings, metadata)
                st.session_state[f"{cache_key}_tsne_cache"] = {}
                st.session_state[f"{cache_key}_embeddings_2d"] = (
                    None  # Will be populated after t-SNE
                )

                # Display info
                st.success(
                    f"**Extracted {len(embeddings)} sliding windows, "
                    f"dimensionality: {embeddings.shape[1]}**"
                )
            except ValueError as e:
                st.error(
                    f"⚠️ Cannot generate {level} sliding window clustering: {str(e)}"
                )

        # Display the plot if embeddings exist
        cache_key_suffix = f"_{n_components}" if aggregation == "svd" else ""
        cache_key = f"episode_sw_{demo_split}_{level}_{window_width}_{stride}_{aggregation}{cache_key_suffix}_{max_samples if use_max_samples else 'all'}"
        if cache_key in st.session_state:
            embeddings, metadata = st.session_state[cache_key]

            # Display count and dimensionality
            agg_display = (
                f"{aggregation} (k={n_components})"
                if aggregation == "svd"
                else aggregation
            )
            st.info(
                f"📊 **{len(embeddings)} sliding windows** | "
                f"**Dimensionality: {embeddings.shape[1]}** | "
                f"**Window width: {window_width} timesteps** | "
                f"**Stride: {stride}** | "
                f"**Aggregation: {agg_display}**"
            )

            # Get or create t-SNE cache for this configuration (include scaling method)
            tsne_cache_key = f"{cache_key}_tsne_cache_{scaling_method}"
            if tsne_cache_key not in st.session_state:
                st.session_state[tsne_cache_key] = {}

            # Apply feature scaling based on selected method
            embeddings_scaled = embeddings.copy()
            if scaling_method == "standard":
                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler()
                embeddings_scaled = scaler.fit_transform(embeddings_scaled)
            elif scaling_method == "robust":
                from sklearn.preprocessing import RobustScaler

                scaler = RobustScaler()
                embeddings_scaled = scaler.fit_transform(embeddings_scaled)
            elif scaling_method == "minmax":
                from sklearn.preprocessing import MinMaxScaler

                scaler = MinMaxScaler()
                embeddings_scaled = scaler.fit_transform(embeddings_scaled)
            elif scaling_method == "none":
                pass  # No scaling

            title_suffix = f", k={n_components}" if aggregation == "svd" else ""
            embeddings_2d, selection = render_tsne_plot(
                embeddings_scaled,
                metadata,
                title=f"t-SNE: {level.title()} Sliding Window Embeddings (width={window_width}, stride={stride}, agg={aggregation}{title_suffix}, scaling={scaling_method})",
                color_by=color_by,
                perplexity=perplexity,
                key_suffix=f"episode_sw_{demo_split}_{level}_{window_width}_{stride}_{aggregation}{cache_key_suffix}_{color_by}_{perplexity}_{scaling_method}",
                tsne_cache=st.session_state[tsne_cache_key],
                enable_selection=True,
            )

            # Handle selection
            selected_indices = []
            if selection and "selection" in selection:
                # When using categorical coloring, the plot has multiple traces.
                # We need to map trace-relative indices to absolute metadata indices.
                if "points" in selection["selection"]:
                    # Use points array which contains curve_number and point_index
                    selected_indices = _convert_plotly_selection_to_absolute_indices(
                        selection["selection"]["points"],
                        metadata,
                        color_by,
                    )
                elif "point_indices" in selection["selection"]:
                    # Fallback for continuous coloring (single trace)
                    selected_indices = selection["selection"]["point_indices"]

            # Display selected windows in a fragment
            _render_selected_windows(
                selected_indices,
                metadata,
                level,
                data,
                demo_split,
                cache_key,
            )


def _convert_plotly_selection_to_absolute_indices(
    points: List[Dict],
    metadata: List[Dict],
    color_by: str,
) -> List[int]:
    """Convert Plotly selection points to absolute metadata indices.

    When using categorical coloring, Plotly creates multiple traces (one per category).
    The selection returns points with trace-relative indices. This function converts
    them to absolute indices into the metadata list.

    Args:
        points: List of point dicts from Plotly selection, each with 'curve_number' and 'point_index'
        metadata: Full metadata list
        color_by: The coloring mode used (e.g., 'success', 'quality_label')

    Returns:
        List of absolute indices into the metadata list
    """
    if not points:
        return []

    # Build groups matching the exact logic in render_tsne_plot
    # This ensures trace numbers align correctly even when some categories are empty
    trace_to_indices = {}
    trace_num = 0

    if color_by == "success":
        # Reproduce the exact logic from render_tsne_plot for success coloring
        success_mask = np.array([m.get("success", None) for m in metadata])

        for success_val, color, label in [
            (True, "green", "Success"),
            (False, "red", "Failure"),
            (None, "gray", "Unknown"),
        ]:
            mask = success_mask == success_val
            if not np.any(mask):
                continue  # Skip empty categories - this is why we can't use enumerate

            indices = np.where(mask)[0]
            trace_to_indices[trace_num] = indices.tolist()
            trace_num += 1

    elif color_by == "quality_label":
        # Reproduce the exact logic from render_tsne_plot for quality_label coloring
        quality_labels = [m.get("quality_label", "unknown") for m in metadata]
        unique_qualities = sorted(set(quality_labels))

        for quality in unique_qualities:
            mask = np.array(quality_labels) == quality
            indices = np.where(mask)[0]
            if len(indices) > 0:
                trace_to_indices[trace_num] = indices.tolist()
                trace_num += 1

    elif color_by == "annotation_label":
        # Reproduce the exact logic from render_tsne_plot for annotation_label coloring
        annotation_labels = [m.get("annotation_label", "no label") for m in metadata]
        unique_labels = sorted(set(annotation_labels))

        for label in unique_labels:
            mask = np.array(annotation_labels) == label
            indices = np.where(mask)[0]
            if len(indices) > 0:
                trace_to_indices[trace_num] = indices.tolist()
                trace_num += 1

    else:
        # For continuous coloring or other modes, assume single trace
        return [p.get("point_index", 0) for p in points]

    # Convert selection points to absolute indices
    absolute_indices = []
    for point in points:
        curve_num = point.get("curve_number", 0)
        point_idx = point.get("point_index", 0)

        if curve_num in trace_to_indices:
            trace_indices = trace_to_indices[curve_num]
            if point_idx < len(trace_indices):
                absolute_indices.append(trace_indices[point_idx])

    return absolute_indices


def _render_selected_windows(
    selected_indices,
    metadata,
    level,
    data,
    demo_split,
    cache_key,
):
    """Render selected windows with video players.

    Called from within a @st.fragment-decorated parent function, so video playback
    with fragment_scope=True will only rerun that parent fragment.
    """
    if selected_indices and len(selected_indices) > 0:
        st.divider()
        st.subheader("🔍 Selected Windows")
        st.success(f"✅ Selected {len(selected_indices)} windows")

        # Filters
        st.markdown("**Filters**")
        col1, col2, col3 = st.columns(3)

        with col1:
            if level == "rollout":
                filter_outcome = st.selectbox(
                    "Filter by outcome",
                    ["All", "Success", "Failure"],
                    key=f"filter_{cache_key}",
                )
            else:
                filter_quality = st.selectbox(
                    "Filter by quality",
                    ["All", "Better", "Okay", "Worse"],
                    key=f"filter_{cache_key}",
                )

        with col2:
            min_std = st.slider(
                "Min std_influence",
                0.0,
                100.0,
                0.0,
                key=f"min_std_{cache_key}",
            )

        with col3:
            max_display = st.slider(
                "Max windows",
                1,
                50,
                12,
                key=f"max_disp_{cache_key}",
            )

        # Apply filters
        filtered_indices = selected_indices.copy()

        if level == "rollout" and filter_outcome != "All":
            success_val = filter_outcome == "Success"
            filtered_indices = [
                i for i in filtered_indices if metadata[i].get("success") == success_val
            ]
        elif level == "demo" and filter_quality != "All":
            quality_map = {
                "Better": "better",
                "Okay": "okay",
                "Worse": "worse",
            }
            filtered_indices = [
                i
                for i in filtered_indices
                if metadata[i].get("quality_label")
                == quality_map.get(filter_quality, filter_quality.lower())
            ]

        filtered_indices = [
            i
            for i in filtered_indices
            if metadata[i].get("std_influence", 0) >= min_std
        ]

        if len(filtered_indices) > 0:
            # Use stable sampling: store sampled indices in session state
            # so they don't change on every rerun during video playback
            sample_key = (
                f"sampled_display_{cache_key}_{len(filtered_indices)}_{max_display}"
            )
            if sample_key not in st.session_state:
                if len(filtered_indices) > max_display:
                    st.session_state[sample_key] = list(
                        np.random.choice(
                            filtered_indices,
                            size=max_display,
                            replace=False,
                        )
                    )
                else:
                    st.session_state[sample_key] = list(filtered_indices)

            display_indices = st.session_state[sample_key]

            # Resample button and info
            col_info, col_resample = st.columns([4, 1])
            with col_info:
                st.write(
                    f"Showing {len(display_indices)} of {len(filtered_indices)} windows"
                )
            with col_resample:
                if len(filtered_indices) > max_display:
                    if st.button("🔀 Resample", key=f"resample_{cache_key}"):
                        st.session_state[sample_key] = list(
                            np.random.choice(
                                filtered_indices,
                                size=max_display,
                                replace=False,
                            )
                        )
                        st.rerun()

            # Render each window in an expander with its own fragment
            for i, point_idx in enumerate(display_indices):
                meta = metadata[point_idx]
                if level == "rollout":
                    rollout_idx = meta["rollout_idx"]
                    success = meta.get("success", False)
                    status = "✅" if success else "❌"
                    title = f"#{i + 1}: Rollout {rollout_idx} | {status} | t={meta['window_start']}-{meta['window_end']} | mean={meta.get('mean_influence', 0):.1f}"
                else:
                    demo_idx = meta["demo_idx"]
                    quality = meta.get("quality_label", "unknown")
                    title = f"#{i + 1}: Demo {demo_idx} | {quality} | t={meta['window_start']}-{meta['window_end']} | mean={meta.get('mean_influence', 0):.1f}"

                # Each window gets its own fragment to isolate video playback
                _render_window_card(
                    title=title,
                    point_idx=point_idx,
                    metadata=metadata,
                    level=level,
                    data=data,
                    demo_split=demo_split,
                    cache_key=cache_key,
                    card_idx=i,
                )
        else:
            st.warning("No windows match filters")


def _render_window_card(
    title,
    point_idx,
    metadata,
    level,
    data,
    demo_split,
    cache_key,
    card_idx,
):
    """Render a single window card with expander and video player.

    Called from within a @st.fragment parent, so we use fragment_scope=True
    to rerun only the parent fragment during video playback.
    """
    from influence_visualizer.render_frames import (
        frame_player,
        render_action_chunk,
        render_annotated_frame,
    )

    meta = metadata[point_idx]
    window_start = meta["window_start"]
    window_end = meta["window_end"]

    with st.expander(title, expanded=(card_idx == 0)):
        if level == "rollout":
            rollout_idx = meta["rollout_idx"]
            rollout_episodes = data.rollout_episodes
            episode = rollout_episodes[rollout_idx]
            episode_length = episode.num_samples

            st.caption(
                f"Std influence: {meta.get('std_influence', 0):.1f} | "
                f"Max influence: {meta.get('max_influence', 0):.1f} | "
                f"Window: t={window_start}-{window_end - 1} (episode length: {episode_length})"
            )

            def _render_rollout_frame(timestep):
                abs_idx = episode.sample_start_idx + timestep
                frame = data.get_rollout_frame(abs_idx)
                action = data.get_rollout_action(abs_idx)

                # Check if timestep is within the window
                in_window = window_start <= timestep < window_end
                if in_window:
                    st.success(f"✓ In window (t={window_start}-{window_end - 1})")
                else:
                    st.info(f"Outside window (t={window_start}-{window_end - 1})")

                col_frame, col_action = st.columns([1, 1])
                with col_frame:
                    render_annotated_frame(
                        frame,
                        f"t={timestep}",
                        f"Rollout {rollout_idx}",
                    )
                with col_action:
                    if action is not None:
                        render_action_chunk(
                            action[np.newaxis, :],
                            title=f"Action at t={timestep}",
                            unique_key=f"win_action_{cache_key}_{point_idx}_{timestep}",
                        )

            frame_player(
                label="Timestep:",
                min_value=0,
                max_value=episode_length - 1,
                key=f"window_player_{cache_key}_{point_idx}",
                default_value=window_start,
                default_fps=3.0,
                render_fn=_render_rollout_frame,
                fragment_scope=True,
            )
        else:
            demo_idx = meta["demo_idx"]
            demo_episodes = data.get_demo_episodes(demo_split)
            episode = demo_episodes[demo_idx]
            episode_length = episode.num_samples

            st.caption(
                f"Std influence: {meta.get('std_influence', 0):.1f} | "
                f"Max influence: {meta.get('max_influence', 0):.1f} | "
                f"Window: t={window_start}-{window_end - 1} (episode length: {episode_length})"
            )

            def _render_demo_frame(timestep):
                abs_idx = episode.sample_start_idx + timestep
                frame = data.get_demo_frame(abs_idx)
                action = data.get_demo_action(abs_idx)

                # Check if timestep is within the window
                in_window = window_start <= timestep < window_end
                if in_window:
                    st.success(f"✓ In window (t={window_start}-{window_end - 1})")
                else:
                    st.info(f"Outside window (t={window_start}-{window_end - 1})")

                col_frame, col_action = st.columns([1, 1])
                with col_frame:
                    render_annotated_frame(
                        frame,
                        f"t={timestep}",
                        f"Demo {demo_idx}",
                    )
                with col_action:
                    if action is not None:
                        render_action_chunk(
                            action[np.newaxis, :],
                            title=f"Action at t={timestep}",
                            unique_key=f"win_action_{cache_key}_{point_idx}_{timestep}",
                        )

            frame_player(
                label="Timestep:",
                min_value=0,
                max_value=episode_length - 1,
                key=f"window_player_{cache_key}_{point_idx}",
                default_value=window_start,
                default_fps=3.0,
                render_fn=_render_demo_frame,
                fragment_scope=True,
            )


@st.fragment
def _render_pca_timestep_clustering_fragment(data, demo_split, annotations=None):
    """Render dimensionality reduction + t-SNE clustering for individual timesteps with lasso selection and video playback."""
    with st.expander(
        "6. Cluster Rollout/Demo Timesteps (PCA/UMAP + t-SNE)", expanded=False
    ):
        st.markdown("""
        Extract influence vectors for individual timesteps from rollouts or demos,
        apply dimensionality reduction (PCA or UMAP), then visualize with t-SNE.

        **Difference from sliding windows:** This analyzes single timesteps rather than windows,
        showing the influence pattern at each individual moment.

        **PCA vs UMAP:**
        - **PCA + t-SNE**: Fast, linear reduction to N dims, then t-SNE to 2D. Good baseline.
        - **UMAP + t-SNE**: Non-linear reduction to N dims, then t-SNE to 2D. Better intermediate features.
        - **UMAP direct to 2D**: Single-step non-linear reduction. Often best for visualization!
        """)

        # Show warning if UMAP not available
        if not UMAP_AVAILABLE:
            st.warning(
                "⚠️ UMAP not installed. Install with `pip install umap-learn` to enable UMAP method."
            )

        # Method and level selectors
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            dimred_method = st.selectbox(
                "Dim reduction method",
                options=["umap_direct", "umap", "pca"] if UMAP_AVAILABLE else ["pca"],
                format_func=lambda x: {
                    "umap_direct": "UMAP → 2D (direct)",
                    "umap": "UMAP → t-SNE",
                    "pca": "PCA → t-SNE",
                }[x],
                key=f"dimred_method_{demo_split}",
                help="Dimensionality reduction method",
            )

        with col2:
            level = st.selectbox(
                "Cluster",
                options=["rollout", "demo"],
                format_func=lambda x: "Rollout Timesteps"
                if x == "rollout"
                else "Demo Timesteps",
                key=f"pca_level_{demo_split}",
                help="Choose whether to cluster rollout or demo timesteps",
            )

        # Feature scaling selector (new row for better visibility)
        st.markdown("**Preprocessing Options**")
        col_scale1, col_scale2, col_scale3, col_scale4, col_scale5 = st.columns(5)
        with col_scale1:
            scaling_method = st.selectbox(
                "Feature scaling",
                options=["standard", "robust", "minmax", "none"],
                format_func=lambda x: {
                    "standard": "StandardScaler ",
                    "robust": "RobustScaler (outlier-resistant)",
                    "minmax": "MinMaxScaler (0-1 range)",
                    "none": "None ",
                }[x],
                index=0,
                key=f"scaling_method_{demo_split}",
                help="Feature scaling method. StandardScaler (zero mean, unit variance) is recommended for most cases. RobustScaler is better if you have outliers.",
            )

        with col3:
            if dimred_method == "umap_direct":
                n_components = 2  # Fixed at 2 for direct visualization
                st.info("UMAP → 2D (no t-SNE)")
            else:
                n_components = st.number_input(
                    f"{dimred_method.upper()} components",
                    min_value=2,
                    max_value=200,
                    value=50,
                    key=f"pca_n_components_{demo_split}",
                    help=f"Number of {dimred_method.upper()} components to extract before t-SNE",
                )

        with col4:
            gaussian_sigma = st.number_input(
                "Gaussian σ",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.5,
                key=f"gaussian_sigma_{demo_split}",
                help="Apply Gaussian smoothing to influence vectors (0 = no smoothing). Helps reduce noise.",
            )

        with col5:
            subsample_episodes = st.checkbox(
                "Subsample episodes",
                value=False,
                key=f"pca_subsample_{demo_split}",
                help="Enable to only analyze a subset of episodes for faster computation",
            )

        # UMAP-specific parameters
        if dimred_method in ["umap", "umap_direct"]:
            col_umap1, col_umap2, col_umap3 = st.columns(3)
            with col_umap1:
                umap_n_neighbors = st.slider(
                    "UMAP n_neighbors",
                    min_value=2,
                    max_value=200,
                    value=15,
                    key=f"umap_n_neighbors_{demo_split}",
                    help="Controls local vs global structure (higher = more global)",
                )
            with col_umap2:
                umap_min_dist = st.slider(
                    "UMAP min_dist",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.05,
                    key=f"umap_min_dist_{demo_split}",
                    help="Minimum distance between points in embedding (higher = more spread)",
                )
            with col_umap3:
                umap_reproducible = st.checkbox(
                    "Reproducible",
                    value=False,
                    key=f"umap_reproducible_{demo_split}",
                    help="Enable for reproducible results (slower, single-threaded). Disable for faster parallel execution.",
                )

            # Show performance warning
            if umap_reproducible:
                st.info(
                    "⚠️ Reproducible mode: UMAP will use single-threaded execution (slower)"
                )
            else:
                st.success(
                    "⚡ Fast mode: UMAP will use all CPU cores (non-deterministic results)"
                )
        else:
            umap_n_neighbors = 15
            umap_min_dist = 0.1
            umap_reproducible = False

        # Subsampling parameters
        if subsample_episodes:
            max_episodes = st.slider(
                "Max episodes to analyze",
                min_value=1,
                max_value=100,
                value=20,
                key=f"pca_max_episodes_{demo_split}",
                help="Maximum number of episodes to include",
            )
        else:
            max_episodes = None

        # t-SNE visualization controls
        st.divider()
        st.markdown("**t-SNE Visualization Controls**")

        col4, col5, col6, col7 = st.columns(4)
        with col4:
            # Determine available color options based on level
            if level == "rollout":
                color_options = [
                    "success",
                    "annotation_label",
                    "timestep",
                    "mean_influence",
                    "std_influence",
                    "max_influence",
                ]
                default_color = "success"
            else:  # demo
                color_options = [
                    "quality_label",
                    "annotation_label",
                    "timestep",
                    "mean_influence",
                    "std_influence",
                    "max_influence",
                ]
                default_color = "quality_label"

            color_by = st.selectbox(
                "Color by",
                options=color_options,
                index=0,
                format_func=lambda x: {
                    "success": "Success/Failure",
                    "quality_label": "Demo Quality",
                    "annotation_label": "Annotation Label",
                    "timestep": "Timestep Position",
                    "mean_influence": "Mean Influence",
                    "std_influence": "Std Influence",
                    "max_influence": "Max Influence",
                }.get(x, x),
                key=f"pca_color_{demo_split}",
                help="Choose how to color the t-SNE plot points",
            )

        with col5:
            perplexity = st.slider(
                "Perplexity",
                min_value=2,
                max_value=50,
                value=30,
                step=1,
                key=f"pca_perplexity_{demo_split}",
                help="t-SNE perplexity parameter",
            )

        with col6:
            use_max_samples = st.checkbox(
                "Limit maximum samples",
                value=False,
                key=f"pca_limit_{demo_split}",
                help="Enable to subsample timesteps for faster t-SNE",
            )

        with col7:
            max_samples = st.slider(
                "Maximum samples",
                min_value=100,
                max_value=100000,
                value=5000,
                step=100,
                key=f"pca_max_{demo_split}",
                disabled=not use_max_samples,
                help="Maximum number of timesteps to use for t-SNE",
            )

        if st.button(
            f"Generate {level.title()} Timestep t-SNE",
            key=f"gen_pca_timestep_{demo_split}",
        ):
            try:
                # Extract embeddings
                embeddings, metadata, _ = extract_dimred_timestep_embeddings(
                    data,
                    split=demo_split,
                    level=level,
                    method=dimred_method,
                    n_components=n_components,
                    umap_n_neighbors=umap_n_neighbors,
                    umap_min_dist=umap_min_dist,
                    umap_reproducible=umap_reproducible,
                    gaussian_sigma=gaussian_sigma,
                    scaling_method=scaling_method,
                    annotations=annotations,
                )

                if len(embeddings) == 0:
                    st.warning(f"No timesteps extracted. Check that {level}s exist.")
                    return

                # Apply episode subsampling if enabled
                if subsample_episodes and max_episodes:
                    # Group by episode
                    if level == "rollout":
                        episode_indices = {}
                        for i, m in enumerate(metadata):
                            ep_idx = m["rollout_idx"]
                            if ep_idx not in episode_indices:
                                episode_indices[ep_idx] = []
                            episode_indices[ep_idx].append(i)
                    else:
                        episode_indices = {}
                        for i, m in enumerate(metadata):
                            ep_idx = m["demo_idx"]
                            if ep_idx not in episode_indices:
                                episode_indices[ep_idx] = []
                            episode_indices[ep_idx].append(i)

                    # Sample episodes
                    if len(episode_indices) > max_episodes:
                        rng = np.random.RandomState(42)
                        selected_episodes = rng.choice(
                            list(episode_indices.keys()),
                            size=max_episodes,
                            replace=False,
                        )

                        # Get all timesteps from selected episodes
                        keep_indices = []
                        for ep_idx in selected_episodes:
                            keep_indices.extend(episode_indices[ep_idx])

                        embeddings = embeddings[keep_indices]
                        metadata = [metadata[i] for i in keep_indices]

                        st.info(
                            f"Subsampled to {max_episodes} episodes "
                            f"({len(embeddings)} timesteps total)"
                        )

                # Apply timestep sampling if enabled
                if use_max_samples and len(embeddings) > max_samples:
                    st.info(
                        f"Sampling {max_samples} out of {len(embeddings)} timesteps for t-SNE visualization"
                    )
                    rng = np.random.RandomState(42)
                    sample_indices = rng.choice(
                        len(embeddings), size=max_samples, replace=False
                    )
                    embeddings = embeddings[sample_indices]
                    metadata = [metadata[i] for i in sample_indices]

                # Store in session state (evict old caches first)
                cache_key = f"{dimred_method}_timestep_{demo_split}_{level}_{n_components}_{umap_n_neighbors if dimred_method in ['umap', 'umap_direct'] else 'na'}_{max_episodes if subsample_episodes else 'all'}_{max_samples if use_max_samples else 'all'}"
                _register_embedding_cache("dimred_timestep", cache_key)
                st.session_state[cache_key] = (embeddings, metadata)
                st.session_state[f"{cache_key}_tsne_cache"] = {}

                # Display info
                st.success(
                    f"**Extracted {len(embeddings)} timesteps, "
                    f"{dimred_method.upper()} dimensionality: {embeddings.shape[1]}**"
                )
            except ValueError as e:
                st.error(f"⚠️ Cannot generate {level} timestep clustering: {str(e)}")

        # Display the plot if embeddings exist
        cache_key = f"{dimred_method}_timestep_{demo_split}_{level}_{n_components}_{umap_n_neighbors if dimred_method in ['umap', 'umap_direct'] else 'na'}_{max_episodes if subsample_episodes else 'all'}_{max_samples if use_max_samples else 'all'}"
        if cache_key in st.session_state:
            embeddings, metadata = st.session_state[cache_key]

            # Display count and dimensionality
            if dimred_method == "umap_direct":
                method_params = f"UMAP direct to 2D, n_neighbors={umap_n_neighbors}"
            elif dimred_method == "umap":
                method_params = f"{dimred_method.upper()} components={n_components}, n_neighbors={umap_n_neighbors}"
            else:
                method_params = f"{dimred_method.upper()} components={n_components}"
            st.info(
                f"📊 **{len(embeddings)} timesteps** | "
                f"**{dimred_method.upper()} dimensionality: {embeddings.shape[1]}** | "
                f"**{method_params}**"
            )

            # Get or create t-SNE cache for this configuration
            tsne_cache_key = f"{cache_key}_tsne_cache"
            if tsne_cache_key not in st.session_state:
                st.session_state[tsne_cache_key] = {}

            # Render plot with lasso selection enabled
            if dimred_method == "umap_direct":
                # UMAP already produced 2D embeddings, skip t-SNE
                embeddings_2d = embeddings
                title_suffix = (
                    f"UMAP n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist}"
                )

                # Use render_tsne_plot but with pre-computed 2D embeddings
                # We'll pass perplexity but it won't be used since embeddings are already 2D
                title = f"UMAP 2D: {level.title()} Timestep Embeddings ({title_suffix})"
            else:
                # Apply t-SNE on top of PCA/UMAP embeddings
                title_suffix = (
                    f"{dimred_method.upper()} n_components={n_components}, n_neighbors={umap_n_neighbors}"
                    if dimred_method == "umap"
                    else f"{dimred_method.upper()} n_components={n_components}"
                )
                title = f"t-SNE: {level.title()} Timestep Embeddings ({title_suffix})"

            embeddings_2d, selection = render_tsne_plot(
                embeddings,
                metadata,
                title=title,
                color_by=color_by,
                perplexity=perplexity if dimred_method != "umap_direct" else 30,
                key_suffix=f"{dimred_method}_timestep_{demo_split}_{level}_{n_components}_{color_by}_{perplexity if dimred_method != 'umap_direct' else 'na'}",
                tsne_cache=st.session_state[tsne_cache_key]
                if dimred_method != "umap_direct"
                else None,
                enable_selection=True,
            )

            # Handle selection
            selected_indices = []
            if selection and "selection" in selection:
                if "points" in selection["selection"]:
                    # Use points array which contains curve_number and point_index
                    selected_indices = _convert_plotly_selection_to_absolute_indices(
                        selection["selection"]["points"],
                        metadata,
                        color_by,
                    )
                elif "point_indices" in selection["selection"]:
                    # Fallback for continuous coloring (single trace)
                    selected_indices = selection["selection"]["point_indices"]

            # Display selected timesteps
            _render_selected_timesteps(
                selected_indices,
                metadata,
                level,
                data,
                demo_split,
                cache_key,
            )


def _render_selected_timesteps(
    selected_indices,
    metadata,
    level,
    data,
    demo_split,
    cache_key,
):
    """Render selected timesteps with video players."""
    if selected_indices and len(selected_indices) > 0:
        st.divider()
        st.subheader("🔍 Selected Timesteps")
        st.success(f"✅ Selected {len(selected_indices)} timesteps")

        # Filters
        st.markdown("**Filters**")
        col1, col2, col3 = st.columns(3)

        with col1:
            if level == "rollout":
                filter_outcome = st.selectbox(
                    "Filter by outcome",
                    ["All", "Success", "Failure"],
                    key=f"filter_timestep_{cache_key}",
                )
            else:
                filter_quality = st.selectbox(
                    "Filter by quality",
                    ["All", "Better", "Okay", "Worse"],
                    key=f"filter_timestep_{cache_key}",
                )

        with col2:
            min_std = st.slider(
                "Min std_influence",
                0.0,
                100.0,
                0.0,
                key=f"min_std_timestep_{cache_key}",
            )

        with col3:
            max_display = st.slider(
                "Max timesteps",
                1,
                50,
                12,
                key=f"max_disp_timestep_{cache_key}",
            )

        # Apply filters
        filtered_indices = selected_indices.copy()

        if level == "rollout" and filter_outcome != "All":
            success_val = filter_outcome == "Success"
            filtered_indices = [
                i for i in filtered_indices if metadata[i].get("success") == success_val
            ]
        elif level == "demo" and filter_quality != "All":
            quality_map = {
                "Better": "better",
                "Okay": "okay",
                "Worse": "worse",
            }
            filtered_indices = [
                i
                for i in filtered_indices
                if metadata[i].get("quality_label")
                == quality_map.get(filter_quality, filter_quality.lower())
            ]

        filtered_indices = [
            i
            for i in filtered_indices
            if metadata[i].get("std_influence", 0) >= min_std
        ]

        if len(filtered_indices) > 0:
            # Use stable sampling
            sample_key = (
                f"sampled_timestep_{cache_key}_{len(filtered_indices)}_{max_display}"
            )
            if sample_key not in st.session_state:
                if len(filtered_indices) > max_display:
                    st.session_state[sample_key] = list(
                        np.random.choice(
                            filtered_indices,
                            size=max_display,
                            replace=False,
                        )
                    )
                else:
                    st.session_state[sample_key] = list(filtered_indices)

            display_indices = st.session_state[sample_key]

            # Resample button and info
            col_info, col_resample = st.columns([4, 1])
            with col_info:
                st.write(
                    f"Showing {len(display_indices)} of {len(filtered_indices)} timesteps"
                )
            with col_resample:
                if len(filtered_indices) > max_display:
                    if st.button("🔀 Resample", key=f"resample_timestep_{cache_key}"):
                        st.session_state[sample_key] = list(
                            np.random.choice(
                                filtered_indices,
                                size=max_display,
                                replace=False,
                            )
                        )
                        st.rerun()

            # Render each timestep
            for i, point_idx in enumerate(display_indices):
                meta = metadata[point_idx]
                if level == "rollout":
                    rollout_idx = meta["rollout_idx"]
                    timestep = meta["timestep"]
                    success = meta.get("success", False)
                    status = "✅" if success else "❌"
                    title = f"#{i + 1}: Rollout {rollout_idx} | {status} | t={timestep} | mean={meta.get('mean_influence', 0):.1f}"
                else:
                    demo_idx = meta["demo_idx"]
                    timestep = meta["timestep"]
                    quality = meta.get("quality_label", "unknown")
                    title = f"#{i + 1}: Demo {demo_idx} | {quality} | t={timestep} | mean={meta.get('mean_influence', 0):.1f}"

                _render_timestep_card(
                    title=title,
                    point_idx=point_idx,
                    metadata=metadata,
                    level=level,
                    data=data,
                    demo_split=demo_split,
                    cache_key=cache_key,
                    card_idx=i,
                )
        else:
            st.warning("No timesteps match filters")


def _render_timestep_card(
    title,
    point_idx,
    metadata,
    level,
    data,
    demo_split,
    cache_key,
    card_idx,
):
    """Render a single timestep card with expander and video player."""
    from influence_visualizer.render_frames import (
        frame_player,
        render_action_chunk,
        render_annotated_frame,
    )

    meta = metadata[point_idx]
    timestep = meta["timestep"]

    with st.expander(title, expanded=(card_idx == 0)):
        if level == "rollout":
            rollout_idx = meta["rollout_idx"]
            rollout_episodes = data.rollout_episodes
            episode = rollout_episodes[rollout_idx]
            episode_length = episode.num_samples

            st.caption(
                f"Std influence: {meta.get('std_influence', 0):.1f} | "
                f"Max influence: {meta.get('max_influence', 0):.1f} | "
                f"Timestep: t={timestep} (episode length: {episode_length})"
            )

            def _render_rollout_frame(t):
                abs_idx = episode.sample_start_idx + t
                frame = data.get_rollout_frame(abs_idx)
                action = data.get_rollout_action(abs_idx)

                # Highlight if this is the selected timestep
                if t == timestep:
                    st.success(f"✓ Selected timestep (t={timestep})")
                else:
                    st.info(f"Timestep t={t}")

                col_frame, col_action = st.columns([1, 1])
                with col_frame:
                    render_annotated_frame(
                        frame,
                        f"t={t}",
                        f"Rollout {rollout_idx}",
                    )
                with col_action:
                    if action is not None:
                        render_action_chunk(
                            action[np.newaxis, :],
                            title=f"Action at t={t}",
                            unique_key=f"timestep_action_{cache_key}_{point_idx}_{t}",
                        )

            frame_player(
                label="Timestep:",
                min_value=0,
                max_value=episode_length - 1,
                key=f"timestep_player_{cache_key}_{point_idx}",
                default_value=timestep,
                default_fps=3.0,
                render_fn=_render_rollout_frame,
                fragment_scope=True,
            )
        else:
            demo_idx = meta["demo_idx"]
            demo_episodes = data.get_demo_episodes(demo_split)
            episode = demo_episodes[demo_idx]
            episode_length = episode.num_samples

            st.caption(
                f"Std influence: {meta.get('std_influence', 0):.1f} | "
                f"Max influence: {meta.get('max_influence', 0):.1f} | "
                f"Timestep: t={timestep} (episode length: {episode_length})"
            )

            def _render_demo_frame(t):
                abs_idx = episode.sample_start_idx + t
                frame = data.get_demo_frame(abs_idx)
                action = data.get_demo_action(abs_idx)

                # Highlight if this is the selected timestep
                if t == timestep:
                    st.success(f"✓ Selected timestep (t={timestep})")
                else:
                    st.info(f"Timestep t={t}")

                col_frame, col_action = st.columns([1, 1])
                with col_frame:
                    render_annotated_frame(
                        frame,
                        f"t={t}",
                        f"Demo {demo_idx}",
                    )
                with col_action:
                    if action is not None:
                        render_action_chunk(
                            action[np.newaxis, :],
                            title=f"Action at t={t}",
                            unique_key=f"timestep_action_{cache_key}_{point_idx}_{t}",
                        )

            frame_player(
                label="Timestep:",
                min_value=0,
                max_value=episode_length - 1,
                key=f"timestep_player_{cache_key}_{point_idx}",
                default_value=timestep,
                default_fps=3.0,
                render_fn=_render_demo_frame,
                fragment_scope=True,
            )


@st.fragment
def _render_matrix_pair_clustering_fragment(data, demo_split):
    with st.expander("5. Cluster Rollout-Demo Matrix Pairs (SVD)", expanded=False):
        st.markdown("""
        Each rollout-demo pair generates a local influence matrix. Using SVD, we extract an embedding vector for each matrix pair.
        """)
        _, demo_episodes, _, _ = get_split_data(data, demo_split)
        num_pairs = len(data.rollout_episodes) * len(demo_episodes)
        if num_pairs > 50_000:
            st.error(
                f"⚠️ Dataset too large for matrix pair clustering: {num_pairs:,} pairs (max: 50,000)."
            )
            return

        col1, col2 = st.columns(2)
        with col1:
            embedding_method = st.radio(
                "SVD Embedding Method",
                options=["singular_values", "low_rank_flatten"],
                format_func=lambda x: "Singular Values"
                if x == "singular_values"
                else "Low-Rank Flatten",
                key=f"svd_method_{demo_split}",
            )
        with col2:
            n_components = st.slider(
                "Number of SVD components",
                min_value=5,
                max_value=50,
                value=10,
                key=f"svd_components_{demo_split}",
            )

        # t-SNE visualization controls
        st.divider()
        st.markdown("**t-SNE Visualization Controls**")

        col3, col4, col5, col6 = st.columns(4)
        with col3:
            color_by = st.selectbox(
                "Color by",
                options=["success", "quality_label", "mean_influence", "std_influence"],
                format_func=lambda x: {
                    "success": "Success/Failure",
                    "quality_label": "Demo Quality",
                    "mean_influence": "Mean Influence",
                    "std_influence": "Std Influence",
                }[x],
                key=f"color_by_{demo_split}",
                help="Choose how to color the t-SNE plot points",
            )
        with col4:
            perplexity_mp = st.slider(
                "Perplexity",
                min_value=2,
                max_value=50,
                value=30,
                step=1,
                key=f"perplexity_mp_{demo_split}",
                help="t-SNE perplexity parameter (balance between local and global structure)",
            )
        with col5:
            use_max_samples = st.checkbox(
                "Limit maximum samples",
                value=False,
                key=f"use_max_samples_{demo_split}",
                help="Enable to subsample embeddings for faster t-SNE computation",
            )
        with col6:
            max_samples = st.slider(
                "Maximum samples",
                min_value=100,
                max_value=100000,
                value=1000,
                step=100,
                key=f"max_samples_{demo_split}",
                disabled=not use_max_samples,
                help="Maximum number of samples to use for t-SNE visualization",
            )

        if st.button(
            "Generate Matrix Pair t-SNE", key=f"gen_matrix_pair_clustering_{demo_split}"
        ):
            with st.spinner("Extracting SVD embeddings..."):
                matrix_embeddings, matrix_metadata = extract_matrix_pair_embeddings(
                    data, demo_split, n_components=n_components, method=embedding_method
                )

            # Apply sampling if enabled
            if use_max_samples and len(matrix_embeddings) > max_samples:
                st.info(
                    f"Sampling {max_samples} out of {len(matrix_embeddings)} embeddings for t-SNE visualization"
                )
                # Random sampling
                rng = np.random.RandomState(42)
                sample_indices = rng.choice(
                    len(matrix_embeddings), size=max_samples, replace=False
                )
                matrix_embeddings = matrix_embeddings[sample_indices]
                matrix_metadata = [matrix_metadata[i] for i in sample_indices]

            # Store in session state (evict old caches first)
            cache_key = f"mp_embeddings_{demo_split}_{n_components}_{embedding_method}_{max_samples if use_max_samples else 'all'}"
            _register_embedding_cache("mp_embeddings", cache_key)
            st.session_state[cache_key] = (matrix_embeddings, matrix_metadata)
            st.session_state[f"{cache_key}_tsne_cache"] = {}

            st.success(
                f"Extracted {len(matrix_embeddings)} embeddings, dim {matrix_embeddings.shape[1]}"
            )

        # Display the plot if embeddings exist
        cache_key = f"mp_embeddings_{demo_split}_{n_components}_{embedding_method}_{max_samples if use_max_samples else 'all'}"
        if cache_key in st.session_state:
            matrix_embeddings, matrix_metadata = st.session_state[cache_key]

            # Get or create t-SNE cache for this configuration
            tsne_cache_key = f"{cache_key}_tsne_cache"
            if tsne_cache_key not in st.session_state:
                st.session_state[tsne_cache_key] = {}

            render_tsne_plot(
                matrix_embeddings,
                matrix_metadata,
                title=f"t-SNE: Matrix Pair Embeddings (SVD - {embedding_method})",
                color_by=color_by,
                perplexity=perplexity_mp,
                key_suffix=f"matrix_pairs_{demo_split}_{n_components}_{embedding_method}_{color_by}_{perplexity_mp}",
                tsne_cache=st.session_state[tsne_cache_key],
            )


def render_clustering_tab(
    data: InfluenceData,
    demo_split: SplitType = "train",
    annotation_file: Optional[str] = None,
    task_config: Optional[str] = None,
):
    """Render the main clustering analysis tab using localized fragments."""
    try:
        st.header("Clustering Analysis")

        # Check dataset size FIRST before doing anything else
        num_rollouts = len(data.rollout_episodes)
        num_demos = len(data.demo_episodes)

        if num_rollouts > 1000 or num_demos > 500:
            st.error(
                f"⚠️ **Dataset too large**: {num_rollouts:,} rollouts, {num_demos:,} demos."
            )
            return

        # Load annotations if available
        from influence_visualizer.render_annotation import load_annotations

        annotations = None
        if annotation_file and task_config:
            try:
                annotations = load_annotations(annotation_file, task_config=task_config)
            except (FileNotFoundError, ValueError) as e:
                st.warning(f"Could not load annotations: {e}")

        # Subtabs
        tab_cluster_gen, tab_behavior_graph = st.tabs(
            ["Cluster Generation", "Behavior Graph"]
        )

        with tab_cluster_gen:
            st.markdown(
                "Explore clustering patterns in influence embeddings "
                "using t-SNE visualization."
            )

            st.divider()
            _render_demo_clustering_fragment(data, demo_split)

            st.divider()
            _render_rollout_clustering_fragment(data, demo_split)

            st.divider()
            _render_episode_sliding_window_clustering_fragment(
                data, demo_split, annotations
            )

            st.divider()
            _render_pca_timestep_clustering_fragment(
                data, demo_split, annotations
            )

            st.divider()
            st.subheader("Matrix-Based Clustering")
            st.markdown("""
            The following methods analyze patterns in rollout-demo influence
            matrix pairs. Choose based on what local patterns you want to
            discover.
            """)

            _render_sliding_window_clustering_fragment(data, demo_split)

            st.divider()
            _render_matrix_pair_clustering_fragment(data, demo_split)

            st.divider()
            from influence_visualizer.render_cluster_algorithms import (
                render_cluster_algorithms_fragment,
            )

            render_cluster_algorithms_fragment(
                data, demo_split, annotations, task_config=task_config
            )

        with tab_behavior_graph:
            from influence_visualizer.render_behavior_graph import (
                render_behavior_graph_tab,
            )

            render_behavior_graph_tab(data, demo_split, annotations, task_config=task_config)

    except Exception as e:
        st.error(f"⚠️ Error loading clustering tab: {type(e).__name__}: {str(e)}")
