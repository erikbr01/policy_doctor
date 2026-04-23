"""Clustering: KNN, HDBSCAN, Gaussian Mixture; UMAP, PCA; normalization (default: none)."""

import os
from typing import List, Dict, Any, Tuple, Literal

if "NUMBA_THREADING_LAYER" not in os.environ:
    os.environ["NUMBA_THREADING_LAYER"] = "omp"

import numpy as np

def _check_umap():
    try:
        import umap
        return umap
    except (ImportError, RuntimeError):
        return None

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


def reduce_dimensions(
    embeddings: np.ndarray,
    method: Literal["umap", "pca"] = "umap",
    n_components: int = 2,
    **kwargs: Any,
) -> np.ndarray:
    """Reduce embedding dimension. method: 'umap' (default) or 'pca'."""
    if method == "pca":
        reducer = PCA(n_components=min(n_components, embeddings.shape[1], embeddings.shape[0]))
        return reducer.fit_transform(embeddings).astype(np.float32)
    if method == "umap":
        umap = _check_umap()
        if umap is None:
            raise ImportError("umap-learn is required for UMAP. Install with: pip install umap-learn")
        n_jobs = kwargs.pop("n_jobs", 32)
        print(f"  [UMAP] Starting: {embeddings.shape} -> {n_components}d, n_jobs={n_jobs}")
        reducer = umap.UMAP(n_components=n_components, n_jobs=n_jobs, low_memory=False, **kwargs)
        result = reducer.fit_transform(embeddings).astype(np.float32)
        print(f"  [UMAP] Done: {result.shape}")
        return result
    raise ValueError(f"method must be 'umap' or 'pca', got {method!r}")


def normalize_embeddings(
    embeddings: np.ndarray,
    method: Literal["none", "standard", "minmax", "robust", "l2"] = "none",
) -> np.ndarray:
    """Normalize embeddings. Default: none."""
    if method == "none":
        return np.asarray(embeddings, dtype=np.float32)
    if method == "standard":
        return StandardScaler().fit_transform(embeddings).astype(np.float32)
    if method == "minmax":
        return MinMaxScaler().fit_transform(embeddings).astype(np.float32)
    if method == "robust":
        return RobustScaler().fit_transform(embeddings).astype(np.float32)
    if method == "l2":
        from sklearn.preprocessing import normalize as sk_normalize

        return sk_normalize(embeddings, norm="l2", axis=1).astype(np.float32)
    raise ValueError(f"method must be none/standard/minmax/robust/l2, got {method!r}")


def cluster_knn(
    embeddings_2d: np.ndarray,
    n_neighbors: int = 5,
) -> np.ndarray:
    """Label each point by the majority cluster of its k nearest neighbors (for existing labels). Not used for raw clustering; use for propagation."""
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings_2d)
    # Return noise for all (no pre-existing labels); callers can use this for KNN propagation.
    return np.full(embeddings_2d.shape[0], -1, dtype=np.int32)


def cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> np.ndarray:
    """Cluster with HDBSCAN. Returns labels (-1 = noise)."""
    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan is required. Install with: pip install hdbscan")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    return clusterer.fit_predict(embeddings).astype(np.int32)


def cluster_gaussian_mixture(
    embeddings: np.ndarray,
    n_components: int = 5,
    random_state: int = 0,
) -> np.ndarray:
    """Cluster with Gaussian Mixture. Returns labels 0..n_components-1."""
    gm = GaussianMixture(n_components=n_components, random_state=random_state)
    return gm.fit_predict(embeddings).astype(np.int32)


def cluster_kmeans(
    embeddings: np.ndarray,
    n_clusters: int = 5,
    init: str = "k-means++",
    n_init: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """Cluster with K-Means. Returns labels 0..n_clusters-1."""
    from sklearn.cluster import KMeans
    model = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        random_state=random_state,
    )
    return model.fit_predict(embeddings).astype(np.int32)


# ---------------------------------------------------------------------------
# Fit-and-return variants (return fitted model alongside transformed data)
# ---------------------------------------------------------------------------

def fit_normalize_embeddings(
    embeddings: np.ndarray,
    method: Literal["none", "standard", "minmax", "robust", "l2"] = "none",
) -> Tuple[np.ndarray, Any]:
    """Like normalize_embeddings but returns (transformed, fitted_scaler_or_None).

    The returned scaler can be used to transform new samples via
    ``scaler.transform(new_sample)``.  Returns ``None`` for methods that
    don't have a stateful sklearn model ("none", "l2").
    """
    if method == "none":
        return np.asarray(embeddings, dtype=np.float32), None
    if method == "l2":
        from sklearn.preprocessing import normalize as sk_normalize
        return sk_normalize(embeddings, norm="l2", axis=1).astype(np.float32), None
    scaler_cls = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}[method]
    scaler = scaler_cls()
    transformed = scaler.fit_transform(embeddings).astype(np.float32)
    return transformed, scaler


def fit_reduce_dimensions(
    embeddings: np.ndarray,
    method: Literal["umap", "pca"] = "umap",
    n_components: int = 2,
    **kwargs: Any,
) -> Tuple[np.ndarray, Any]:
    """Like reduce_dimensions but returns (transformed, fitted_reducer).

    The returned reducer supports ``.transform(new_sample)`` for new points.
    """
    if method == "pca":
        reducer = PCA(n_components=min(n_components, embeddings.shape[1], embeddings.shape[0]))
        transformed = reducer.fit_transform(embeddings).astype(np.float32)
        return transformed, reducer
    if method == "umap":
        umap = _check_umap()
        if umap is None:
            raise ImportError("umap-learn is required for UMAP. Install with: pip install umap-learn")
        n_jobs = kwargs.pop("n_jobs", 32)
        print(f"  [UMAP] Starting: {embeddings.shape} -> {n_components}d, n_jobs={n_jobs}")
        reducer = umap.UMAP(n_components=n_components, n_jobs=n_jobs, low_memory=False, **kwargs)
        transformed = reducer.fit_transform(embeddings).astype(np.float32)
        print(f"  [UMAP] Done: {transformed.shape}")
        return transformed, reducer
    raise ValueError(f"method must be 'umap' or 'pca', got {method!r}")


def fit_cluster_kmeans(
    embeddings: np.ndarray,
    n_clusters: int = 5,
    init: str = "k-means++",
    n_init: int = 10,
    random_state: int = 42,
) -> Tuple[np.ndarray, Any]:
    """Like cluster_kmeans but returns (labels, fitted_kmeans_model).

    The returned model supports ``.predict(new_sample)`` for new points.
    """
    from sklearn.cluster import KMeans
    model = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        random_state=random_state,
    )
    labels = model.fit_predict(embeddings).astype(np.int32)
    return labels, model


def run_clustering(
    embeddings: np.ndarray,
    method: Literal["knn", "hdbscan", "gaussian_mixture", "kmeans"] = "hdbscan",
    dim_reduce: Literal["umap", "pca"] = "umap",
    n_components_2d: int = 2,
    normalize: Literal["none", "standard", "minmax", "robust"] = "none",
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Run clustering pipeline: optional normalize -> dim reduce -> cluster. Returns (labels, coords_2d, metrics)."""
    emb = normalize_embeddings(embeddings, method=normalize)
    coords_2d = reduce_dimensions(emb, method=dim_reduce, n_components=n_components_2d)
    if method == "hdbscan":
        labels = cluster_hdbscan(emb, **{k: v for k, v in kwargs.items() if k in ("min_cluster_size", "min_samples")})
    elif method == "gaussian_mixture":
        labels = cluster_gaussian_mixture(emb, **{k: v for k, v in kwargs.items() if k in ("n_components", "random_state")})
    elif method == "kmeans":
        labels = cluster_kmeans(emb, **{k: v for k, v in kwargs.items() if k in ("n_clusters", "init", "n_init", "random_state")})
    else:
        labels = cluster_knn(coords_2d, **{k: v for k, v in kwargs.items() if k == "n_neighbors"})
    n_clusters = len(set(labels) - {-1})
    metrics = {"n_clusters": n_clusters, "n_noise": int((labels == -1).sum())}
    return labels, coords_2d, metrics
