"""Clustering result persistence: save/load/list clustering results to disk.

Results are stored under influence_visualizer/configs/<task_config>/clustering/<name>/
with manifest.yaml, cluster_labels.npy, and metadata.json.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


def get_clustering_dir(task_config: str) -> Path:
    """Return the clustering results directory for a task config."""
    return Path(__file__).parent / "configs" / task_config / "clustering"


def _slugify(name: str) -> str:
    """Convert a display name to a safe directory name."""
    s = re.sub(r"[^\w\s-]", "", name)
    s = re.sub(r"[-\s]+", "_", s).strip("_")
    return s or "unnamed"


def list_clustering_results(task_config: str) -> List[str]:
    """List saved clustering result names for a task config.

    Returns subdirs that contain both manifest.yaml and cluster_labels.npy,
    sorted alphabetically.
    """
    clustering_dir = get_clustering_dir(task_config)
    if not clustering_dir.exists():
        return []
    names = []
    for path in clustering_dir.iterdir():
        if path.is_dir():
            if (path / "manifest.yaml").exists() and (path / "cluster_labels.npy").exists():
                names.append(path.name)
    return sorted(names)


def load_clustering_result_from_path(
    result_dir: Path,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Load a clustering result from an explicit directory path.

    The directory must contain manifest.yaml, cluster_labels.npy, and metadata.json.

    Args:
        result_dir: Path to the clustering result directory (e.g.
            influence_visualizer/configs/transport_mh_jan28/clustering/sliding_window_rollout_kmeans_k15_2026_03_05/).

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


def load_clustering_result(
    task_config: str,
    name: str,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Load a clustering result from disk.

    Args:
        task_config: Task config name (e.g. 'lift_mh').
        name: Saved result name (directory name under clustering/).

    Returns:
        (cluster_labels, metadata, manifest).

    Raises:
        FileNotFoundError: If the result directory or required files are missing.
    """
    result_dir = get_clustering_dir(task_config) / name

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


def save_clustering_result(
    name: str,
    cluster_labels: np.ndarray,
    metadata: List[Dict[str, Any]],
    *,
    algorithm: str,
    scaling: str,
    influence_source: str,
    representation: str,
    level: str,
    n_clusters: int,
    n_samples: int,
    output_dir: "Path | None" = None,
    task_config: "str | None" = None,
) -> Path:
    """Save a clustering result to disk.

    Args:
        name: Display name; will be slugified for the directory name.
        cluster_labels: 1D array of cluster labels.
        metadata: List of per-slice metadata dicts (JSON-serializable).
        algorithm: Clustering algorithm used (e.g. 'kmeans').
        scaling: Feature scaling method (e.g. 'standard').
        influence_source: e.g. 'trak', 'infembed'.
        representation: e.g. 'sliding_window', 'timestep'.
        level: e.g. 'rollout', 'demo'.
        n_clusters: Number of clusters (excluding noise).
        n_samples: Number of samples (len(cluster_labels)).
        output_dir: Explicit directory to write into.  When provided,
            results are saved to ``output_dir / slug`` and ``task_config``
            is ignored.  Preferred for pipeline use.
        task_config: Legacy task config name; results saved to
            ``iv/configs/<task_config>/clustering/<slug>``.  Used by the
            Streamlit app.  Ignored when ``output_dir`` is set.

    Returns:
        Path to the result directory.
    """
    if output_dir is None and task_config is None:
        raise ValueError(
            "Either output_dir or task_config must be provided to save_clustering_result."
        )
    slug = _slugify(name)
    if output_dir is not None:
        result_dir = Path(output_dir) / slug
    else:
        result_dir = get_clustering_dir(task_config) / slug
    result_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "algorithm": algorithm,
        "scaling": scaling,
        "influence_source": influence_source,
        "representation": representation,
        "level": level,
        "n_clusters": n_clusters,
        "n_samples": n_samples,
        "created": datetime.now().isoformat(),
        "task_config": task_config,
    }

    with open(result_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f, default_flow_style=False, sort_keys=False)

    np.save(result_dir / "cluster_labels.npy", cluster_labels)

    def _json_serial(obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if np.isfinite(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(result_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=_json_serial)

    return result_dir
