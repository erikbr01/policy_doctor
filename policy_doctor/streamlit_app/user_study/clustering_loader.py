"""Resolve study clusterings from a sweep directory (best w/s per K by silhouette)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import streamlit as st
import yaml

from policy_doctor.behaviors.behavior_graph import BehaviorGraph

_DIR_RE = re.compile(
    r"^(?P<rep>.+)_w(?P<w>\d+)_s(?P<s>\d+)_seed\d+_kmeans_k(?P<k>\d+)$"
)


def _read_metrics(path: Path) -> dict[str, Any]:
    try:
        with open(path / "metrics.json") as f:
            return json.load(f)
    except Exception:
        return {}


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        with open(path / "manifest.yaml") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _ordering_from_path(sweep_root: Path, clust_dir: Path) -> str:
    return sweep_root.name


@st.cache_data(show_spinner=False)
def build_sweep_index(sweep_root_str: str) -> list[dict[str, Any]]:
    """Index every clustering directly under ``sweep_root``."""
    sweep_root = Path(sweep_root_str)
    if not sweep_root.is_dir():
        return []

    ordering = sweep_root.name
    index: list[dict[str, Any]] = []
    for clust_dir in sorted(sweep_root.iterdir()):
        if not clust_dir.is_dir():
            continue
        if not (clust_dir / "cluster_labels.npy").exists():
            continue
        m = _DIR_RE.match(clust_dir.name)
        if not m:
            continue
        index.append({
            "path": clust_dir,
            "rep": m.group("rep"),
            "ordering": ordering,
            "w": int(m.group("w")),
            "s": int(m.group("s")),
            "k": int(m.group("k")),
            "metrics": _read_metrics(clust_dir),
            "manifest": _read_manifest(clust_dir),
        })
    return index


def best_entry_for_k(
    index: list[dict[str, Any]],
    *,
    rep: str,
    ordering: str,
    k: int,
) -> Optional[dict[str, Any]]:
    """Return the sweep row for ``k`` with the highest ``silhouette_mean`` w/s."""
    cands = [
        e for e in index
        if e["rep"] == rep and e["ordering"] == ordering and e["k"] == k
    ]
    if not cands:
        return None

    def _sil(entry: dict[str, Any]) -> float:
        return float((entry.get("metrics") or {}).get("silhouette_mean", float("-inf")))

    return max(cands, key=_sil)


def resolve_k_options(clust_cfg: dict[str, Any], root: Path) -> list[int]:
    sweep_root = root / clust_cfg["sweep_root"]
    index = build_sweep_index(str(sweep_root))
    rep = clust_cfg["rep"]
    ordering = clust_cfg.get("ordering", sweep_root.name)
    available = sorted({
        e["k"] for e in index if e["rep"] == rep and e["ordering"] == ordering
    })
    requested = clust_cfg.get("k_options") or available
    if isinstance(requested, int):
        requested = [requested]
    filtered = [int(k) for k in requested if int(k) in available]
    return filtered or available


def apply_clustering_for_k(
    pfx: str,
    k: int,
    clust_cfg: dict[str, Any],
    root: Path,
) -> list[str]:
    """Load clustering ``k`` into ``st.session_state``; return errors (empty on success)."""
    sweep_root = root / clust_cfg["sweep_root"]
    index = build_sweep_index(str(sweep_root))
    rep = clust_cfg["rep"]
    ordering = clust_cfg.get("ordering", sweep_root.name)
    entry = best_entry_for_k(index, rep=rep, ordering=ordering, k=int(k))
    if entry is None:
        return [f"No clustering found for K={k} (rep={rep!r}, ordering={ordering!r})."]

    clust_dir: Path = entry["path"]
    labels_path = clust_dir / "cluster_labels.npy"
    meta_path = clust_dir / "metadata.json"
    if not labels_path.exists() or not meta_path.exists():
        return [f"Missing cluster_labels.npy or metadata.json in {clust_dir}"]

    labels = np.load(str(labels_path))
    with open(meta_path) as f:
        metadata = json.load(f)

    st.session_state[f"{pfx}_labels"] = labels
    st.session_state[f"{pfx}_metadata"] = metadata
    st.session_state[f"{pfx}_clustering_dir"] = str(clust_dir)
    st.session_state[f"{pfx}_clustering_w"] = entry["w"]
    st.session_state[f"{pfx}_clustering_s"] = entry["s"]

    coords_path = clust_dir / "embeddings_reduced.npy"
    st.session_state[f"{pfx}_coords"] = (
        np.load(str(coords_path)) if coords_path.exists() else None
    )

    graph = BehaviorGraph.from_cluster_assignments(
        labels,
        metadata,
        level="rollout" if any("rollout_idx" in m for m in metadata) else "demo",
    )
    st.session_state[f"{pfx}_graph"] = graph
    st.session_state[f"{pfx}_loaded_k"] = int(k)
    return []
