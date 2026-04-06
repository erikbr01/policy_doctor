"""Load clustering results from disk. No dependency on influence_visualizer.

Results are stored under <config_root>/<task_config>/clustering/<name>/
with manifest.yaml, cluster_labels.npy, and metadata.json.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml


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
