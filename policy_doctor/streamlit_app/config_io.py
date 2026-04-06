"""Config export/import helpers for the Streamlit app.

Provides YAML export (download button) and import (file uploader) widgets
that tabs use to persist and reload pipeline step configurations.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import yaml

from policy_doctor.paths import REPO_ROOT, iv_task_configs_base

_REPO_ROOT = REPO_ROOT


def resolve_task_config_stem(config: object, stem_override: Optional[str] = None) -> str:
    """Task config key: YAML filename stem from the sidebar (e.g. ``transport_mh_jan28``).

    Used for saved clustering, pipeline ``task_config=``, etc. ``stem_override`` should be
    the sidebar selection when available.
    """
    if stem_override:
        return stem_override
    stem = getattr(config, "config_stem", "") or ""
    if stem:
        return stem
    return getattr(config, "name", "") or "unknown"


# Backward-compatible alias
task_config_stem_for_iv_paths = resolve_task_config_stem


def merge_clustering_params_for_display(
    manifest: Optional[Dict[str, Any]],
    session_params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Combine saved manifest fields with ``st.session_state['clustering_params']`` for display/export."""
    out: Dict[str, Any] = dict(session_params or {})
    if manifest:
        if manifest.get("scaling") is not None:
            out["normalize"] = manifest["scaling"]
        if manifest.get("n_clusters") is not None:
            out["n_clusters"] = int(manifest["n_clusters"])
        if manifest.get("influence_source"):
            out["clustering_influence_source"] = str(manifest["influence_source"]).lower()
        if manifest.get("level"):
            out["clustering_level"] = manifest["level"]
        for k in ("window_width", "stride", "aggregation", "umap_n_components", "umap_random_state"):
            if k in manifest and manifest[k] is not None:
                out[k] = manifest[k]
        if manifest.get("demo_split") is not None:
            out["clustering_demo_split"] = manifest["demo_split"]
    return out


def sync_clustering_session_from_manifest(manifest: Dict[str, Any]) -> None:
    """Update ``clustering_params`` and ``clustering_influence_source`` after loading a saved run."""
    inf = str(manifest.get("influence_source", "infembed")).lower()
    st.session_state["clustering_influence_source"] = inf
    cur: Dict[str, Any] = dict(st.session_state.get("clustering_params") or {})
    cur["normalize"] = manifest.get("scaling", cur.get("normalize", "none"))
    cur["n_clusters"] = int(manifest.get("n_clusters", cur.get("n_clusters", 20)))
    cur["clustering_influence_source"] = inf
    cur["clustering_level"] = manifest.get("level", cur.get("clustering_level", "rollout"))
    for k in ("window_width", "stride", "aggregation", "umap_n_components", "umap_random_state"):
        if k in manifest and manifest[k] is not None:
            cur[k] = manifest[k]
    if manifest.get("demo_split") is not None:
        cur["clustering_demo_split"] = manifest["demo_split"]
    st.session_state["clustering_params"] = cur


def render_config_export(
    config_dict: Dict[str, Any],
    default_filename: str = "config.yaml",
    label: str = "Export config",
    key: str = "export",
) -> None:
    """Render a download button for a YAML config dict."""
    yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
    st.download_button(
        label=label,
        data=yaml_str,
        file_name=default_filename,
        mime="text/yaml",
        key=key,
    )


def render_config_import(
    label: str = "Import config (YAML)",
    key: str = "import",
) -> Optional[Dict[str, Any]]:
    """Render a sidebar file uploader for YAML configs. Returns parsed dict or None."""
    uploaded = st.sidebar.file_uploader(label, type=["yaml", "yml"], key=key)
    if uploaded is not None:
        try:
            return yaml.safe_load(uploaded.read())
        except Exception as e:
            st.sidebar.error(f"Failed to parse YAML: {e}")
    return None


def render_comparison_json_import(key: str = "cmp_upload") -> None:
    """Sidebar uploader for comparison tab JSON; writes cmp_baseline, cmp_curated, cmp_seeds."""
    uploaded = st.sidebar.file_uploader(
        "Comparison results (JSON)",
        type=["json"],
        key=key,
    )
    if uploaded is None:
        return
    try:
        payload = json.loads(uploaded.read())
        baseline = payload.get("baseline", {})
        curated = payload.get("curated", {})
        seeds = list(set(list(baseline.keys()) + list(curated.keys())))
        st.session_state["cmp_baseline"] = baseline
        st.session_state["cmp_curated"] = curated
        st.session_state["cmp_seeds"] = seeds
        st.sidebar.success("Comparison JSON loaded — open the Comparison tab to view")
    except Exception as e:
        st.sidebar.error(f"Failed to parse comparison JSON: {e}")


def render_save_to_disk(
    config_dict: Dict[str, Any],
    default_path: str,
    label: str = "Save config to disk",
    key: str = "save_disk",
) -> Optional[pathlib.Path]:
    """Render a text input + button to save a config YAML to disk."""
    col1, col2 = st.columns([3, 1])
    with col1:
        path_str = st.text_input("Save path", value=default_path, key=f"{key}_path")
    with col2:
        st.write("")  # spacer
        st.write("")
        do_save = st.button(label, key=f"{key}_btn")
    if do_save and path_str:
        out_path = pathlib.Path(path_str)
        if not out_path.is_absolute():
            out_path = _REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        st.success(f"Saved to {out_path}")
        return out_path
    return None


def _dedupe_resolved_paths(paths: list[pathlib.Path]) -> list[pathlib.Path]:
    seen: set[pathlib.Path] = set()
    out: list[pathlib.Path] = []
    for p in paths:
        try:
            key = p.resolve()
        except OSError:
            key = p
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _iv_configs_root() -> pathlib.Path:
    """``influence_visualizer/configs`` next to the installed ``clustering_results`` module."""
    import influence_visualizer.clustering_results as _cr

    return pathlib.Path(_cr.__file__).resolve().parent / "configs"


def iv_configs_roots() -> list[pathlib.Path]:
    """All IV task-config roots to scan (package install + vendored / monorepo tree when different)."""
    roots = [
        _iv_configs_root(),
        iv_task_configs_base(_REPO_ROOT),
    ]
    return [p for p in _dedupe_resolved_paths(roots) if p.is_dir()]


def clustering_roots_for_task(task_config: str) -> list[pathlib.Path]:
    """Per-task ``clustering`` directories (union of package path and repo ``influence_visualizer/configs``)."""
    from influence_visualizer.clustering_results import get_clustering_dir as iv_get_clustering_dir

    candidates = [
        iv_get_clustering_dir(task_config),
        iv_task_configs_base(_REPO_ROOT) / task_config / "clustering",
    ]
    return [p for p in _dedupe_resolved_paths(candidates) if p.is_dir()]


def _is_valid_clustering_result_dir(path: pathlib.Path) -> bool:
    return path.is_dir() and (path / "manifest.yaml").exists() and (path / "cluster_labels.npy").exists()


def list_clustering_results(task_config: str) -> list[str]:
    """List clustering run directory names for a task key, merging all known storage roots."""
    names: set[str] = set()
    for croot in clustering_roots_for_task(task_config):
        for path in croot.iterdir():
            if _is_valid_clustering_result_dir(path):
                names.add(path.name)
    return sorted(names)


def load_task_clustering_result(
    task_config: str,
    name: str,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Load a saved clustering run, trying each clustering root (package vs monorepo)."""
    from policy_doctor.data.clustering_loader import load_clustering_result_from_path

    tried: list[pathlib.Path] = []
    for root in clustering_roots_for_task(task_config):
        p = root / name
        tried.append(p)
        if _is_valid_clustering_result_dir(p):
            return load_clustering_result_from_path(p)
    raise FileNotFoundError(
        f"No clustering result {name!r} for task {task_config!r}. Tried: {tried}"
    )


def discover_clustering_task_keys() -> list[str]:
    """Task keys that have at least one valid saved clustering run under any IV configs root."""
    keys: set[str] = set()
    for cfg_root in iv_configs_roots():
        for task_dir in cfg_root.iterdir():
            if not task_dir.is_dir():
                continue
            cdir = task_dir / "clustering"
            if not cdir.is_dir():
                continue
            if any(_is_valid_clustering_result_dir(p) for p in cdir.iterdir() if p.is_dir()):
                keys.add(task_dir.name)
    return sorted(keys)


def clustering_results_dir_for_task(task_config: str) -> pathlib.Path:
    """Primary hint path for clustering (first existing root, else canonical IV path)."""
    from influence_visualizer.clustering_results import get_clustering_dir

    roots = clustering_roots_for_task(task_config)
    if roots:
        return roots[0]
    return get_clustering_dir(task_config)


expected_iv_clustering_dir = clustering_results_dir_for_task


def get_clustering_dir(task_config: str, name: str) -> pathlib.Path:
    """Get the path to a clustering result directory."""
    from influence_visualizer.clustering_results import get_clustering_dir as _get_iv

    return _get_iv(task_config) / name
