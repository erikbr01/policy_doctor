"""Study config (study.yaml) and anonymous participant IDs for the user study."""

from __future__ import annotations

import json
import secrets
from pathlib import Path

import numpy as np
import streamlit as st
import yaml

from policy_doctor.behaviors.behavior_graph import BehaviorGraph
from policy_doctor.streamlit_app.user_study.clustering_loader import (
    apply_clustering_for_k,
    resolve_k_options,
)
from policy_doctor.streamlit_app.user_study.strategies import load_study_config

PARTICIPANT_ID_KEY = "study_participant_id"


def repo_root() -> Path:
    return Path(__file__).parents[3]


def study_config_path(root: Path | None = None) -> Path:
    root = root or repo_root()
    return root / "policy_doctor" / "configs" / "user_study" / "study.yaml"


def tasks_dir(root: Path | None = None) -> Path:
    root = root or repo_root()
    return root / "policy_doctor" / "configs" / "user_study" / "tasks"


def _resolve(p: str, root: Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else root / path


def get_study_task_name(root: Path | None = None) -> str:
    root = root or repo_root()
    cfg_path = study_config_path(root)
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Study config not found: {cfg_path}. "
            "Create policy_doctor/configs/user_study/study.yaml with a `task` key."
        )
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    task = cfg.get("task")
    if not task or not isinstance(task, str):
        raise ValueError(
            f"{cfg_path} must set `task` to a tasks/*.yaml stem "
            "(e.g. kendama_may22)."
        )
    return task


def load_study_yaml(root: Path | None = None) -> dict:
    root = root or repo_root()
    cfg_path = study_config_path(root)
    if not cfg_path.is_file():
        return {}
    return yaml.safe_load(cfg_path.read_text()) or {}


def ensure_participant_id() -> str:
    """Stable random ID for this browser session; never shown to the participant."""
    if PARTICIPANT_ID_KEY not in st.session_state:
        st.session_state[PARTICIPANT_ID_KEY] = secrets.token_hex(8)
    return st.session_state[PARTICIPANT_ID_KEY]


def load_task(
    pfx: str,
    *,
    needs_graph: bool,
    root: Path | None = None,
) -> tuple[str, str, list[str]]:
    """Load the configured study task into ``st.session_state``.

    Returns ``(participant_id, task_name, errors)``. ``errors`` is empty on success.
    """
    root = root or repo_root()
    participant_id = ensure_participant_id()

    try:
        task_name = get_study_task_name(root)
    except (FileNotFoundError, ValueError) as exc:
        return participant_id, "", [str(exc)]

    if st.session_state.get(f"{pfx}_loaded_task") == task_name:
        return participant_id, task_name, []

    task_path = tasks_dir(root) / f"{task_name}.yaml"
    if not task_path.is_file():
        return participant_id, task_name, [
            f"Task config not found: {task_path}. "
            f"Check `task` in {study_config_path(root)}."
        ]

    task_cfg = yaml.safe_load(task_path.read_text()) or {}
    study_cfg = load_study_yaml(root)
    mp4_dir = _resolve(task_cfg["mp4_dir"], root)
    config_path = _resolve(task_cfg["study_config"], root)

    errors: list[str] = []
    index_path = mp4_dir / "index.json"
    if not mp4_dir.is_dir():
        errors.append(f"MP4 directory not found: {mp4_dir}")
    elif not index_path.exists():
        errors.append("index.json not found in MP4 directory")
    if not config_path.exists():
        errors.append(f"Study config not found: {config_path}")

    graph_cfg = study_cfg.get("graph") or {}
    clust_cfg = graph_cfg.get("clustering") if needs_graph else None
    clust_dir = None
    if needs_graph:
        if clust_cfg:
            k_options = resolve_k_options(clust_cfg, root)
            if not k_options:
                errors.append(
                    f"No clusterings found under {clust_cfg.get('sweep_root')} "
                    f"for rep={clust_cfg.get('rep')!r}."
                )
            else:
                graph_cfg = {**graph_cfg, "k_options": k_options}
                default_k = int(clust_cfg.get("default_k", k_options[0]))
                if default_k not in k_options:
                    default_k = k_options[0]
                graph_cfg = {**graph_cfg, "default_k": default_k}
        else:
            clust_dir = _resolve(task_cfg["clustering_dir"], root)
            labels_path = clust_dir / "cluster_labels.npy"
            meta_path = clust_dir / "metadata.json"
            if not clust_dir.is_dir():
                errors.append(f"Clustering directory not found: {clust_dir}")
            elif not labels_path.exists() or not meta_path.exists():
                errors.append("cluster_labels.npy or metadata.json missing")

    if errors:
        return participant_id, task_name, errors

    with open(index_path) as f:
        st.session_state[f"{pfx}_index"] = json.load(f)

    cfg = load_study_config(config_path)
    st.session_state[f"{pfx}_strategies"] = cfg["strategies"]
    st.session_state[f"{pfx}_budget"] = cfg.get("budget", {}).get("total_demos", 500)
    st.session_state[f"{pfx}_alloc_step"] = cfg.get("budget", {}).get("allocation_step", 25)
    st.session_state[f"{pfx}_mp4_dir"] = str(mp4_dir)
    rollout_limit = study_cfg.get("rollout_time_limit_seconds")
    if rollout_limit is None:
        rollout_limit = task_cfg.get("rollout_time_limit_seconds", 600)
    st.session_state[f"{pfx}_rollout_limit"] = rollout_limit
    _dvd = task_cfg.get("demo_videos_dir")
    st.session_state[f"{pfx}_demo_videos_dir"] = (
        str(_resolve(_dvd, root)) if _dvd else str(mp4_dir / "demo_videos")
    )
    st.session_state[f"{pfx}_study_graph"] = graph_cfg

    if needs_graph:
        if clust_cfg:
            k_load = int(graph_cfg["default_k"])
            load_errs = apply_clustering_for_k(pfx, k_load, clust_cfg, root)
            if load_errs:
                return participant_id, task_name, load_errs
            try:
                st.session_state[f"{pfx}_graph_k_idx"] = graph_cfg["k_options"].index(k_load)
            except ValueError:
                st.session_state[f"{pfx}_graph_k_idx"] = 0
        else:
            assert clust_dir is not None
            labels = np.load(str(clust_dir / "cluster_labels.npy"))
            with open(clust_dir / "metadata.json") as f:
                metadata = json.load(f)
            st.session_state[f"{pfx}_labels"] = labels
            st.session_state[f"{pfx}_metadata"] = metadata
            st.session_state[f"{pfx}_clustering_dir"] = str(clust_dir)

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

    st.session_state[f"{pfx}_loaded_task"] = task_name
    return participant_id, task_name, []
