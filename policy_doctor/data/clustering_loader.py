"""Load clustering results from disk. No dependency on influence_visualizer.

Results are stored under <config_root>/<task_config>/clustering/<name>/
with manifest.yaml, cluster_labels.npy, and metadata.json.

Fitted pipeline models (normalizer, prescaler, UMAP reducer, KMeans) are
optionally saved alongside as clustering_models.pkl (joblib format) so that
new data points can be projected through the exact same pipeline and assigned
to the correct cluster.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

_MODELS_FILENAME = "clustering_models.pkl"


@dataclass
class ClusteringModels:
    """Container for the fitted sklearn/UMAP objects from one clustering run."""
    normalizer: Optional[Any]          # fitted scaler or None (for 'none'/'l2')
    normalizer_method: str             # e.g. "none", "standard"
    prescaler: Optional[Any]           # fitted scaler or None
    prescaler_method: str              # e.g. "standard"
    reducer: Optional[Any]             # fitted UMAP / PCA model
    reducer_method: str                # e.g. "umap", "pca"
    kmeans: Optional[Any]              # fitted KMeans model (or None if not kmeans)


def save_clustering_models(
    result_dir: Path,
    normalizer: Optional[Any],
    normalizer_method: str,
    prescaler: Optional[Any],
    prescaler_method: str,
    reducer: Optional[Any],
    reducer_method: str,
    kmeans: Optional[Any],
) -> Path:
    """Serialize fitted pipeline models to ``<result_dir>/clustering_models.pkl``.

    Returns the path to the written file.
    """
    import joblib

    result_dir = Path(result_dir)
    models = ClusteringModels(
        normalizer=normalizer,
        normalizer_method=normalizer_method,
        prescaler=prescaler,
        prescaler_method=prescaler_method,
        reducer=reducer,
        reducer_method=reducer_method,
        kmeans=kmeans,
    )
    out_path = result_dir / _MODELS_FILENAME
    joblib.dump(models, out_path)
    return out_path


def load_clustering_models(result_dir: Path) -> ClusteringModels:
    """Load fitted pipeline models from ``<result_dir>/clustering_models.pkl``.

    Raises ``FileNotFoundError`` if the file does not exist (e.g., the
    clustering run predates model saving).
    """
    import joblib

    path = Path(result_dir) / _MODELS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"clustering_models.pkl not found in {result_dir}. "
            "Re-run clustering with a version of RunClusteringStep that saves models."
        )
    return joblib.load(path)


def load_clustering_result_from_path(
    result_dir: Path,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Load a clustering result from an explicit directory path.

    The directory must contain manifest.yaml, cluster_labels.npy, and metadata.json.

    Args:
        result_dir: Path to the clustering result directory.

    Returns:
        (cluster_labels, metadata, manifest).
    """
    result_dir = Path(result_dir)
    if not result_dir.exists() or not result_dir.is_dir():
        raise FileNotFoundError(f"Clustering result not found: {result_dir}")
    manifest_path = result_dir / "manifest.yaml"
    labels_path = result_dir / "cluster_labels.npy"
    metadata_path = result_dir / "metadata.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest missing: {manifest_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Cluster labels missing: {labels_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata missing: {metadata_path}")
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f) or {}
    cluster_labels = np.load(labels_path)
    with open(metadata_path) as f:
        metadata = json.load(f)
    return cluster_labels, metadata, manifest


def get_clustering_dir(task_config: str, config_root: Path) -> Path:
    """Return the clustering results directory for a task config under config_root."""
    return config_root / task_config / "clustering"


def load_clustering_result(
    task_config: str,
    name: str,
    config_root: Path,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Load a clustering result from disk under config_root.

    Args:
        task_config: Task config name (e.g. 'transport_mh_jan28').
        name: Saved result name (directory name under clustering/).
        config_root: Root containing task config dirs (e.g. .../configs).

    Returns:
        (cluster_labels, metadata, manifest).
    """
    result_dir = get_clustering_dir(task_config, config_root) / name
    return load_clustering_result_from_path(result_dir)
