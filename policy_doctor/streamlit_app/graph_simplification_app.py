"""Graph Simplification Comparison.

Side-by-side comparison of methods to clean up the behavior graph.

Run with:
    conda activate policy_doctor && streamlit run \\
        policy_doctor/streamlit_app/graph_simplification_app.py --server.port 8530
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure this worktree's policy_doctor is imported (not the editable install
# pointing at the main repo).
_WORKTREE_ROOT = Path(__file__).resolve().parents[2]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))
# Drop any cached main-repo policy_doctor modules
for _mod in [k for k in list(sys.modules.keys()) if k.startswith("policy_doctor")]:
    if hasattr(sys.modules[_mod], "__file__") and sys.modules[_mod].__file__:
        if str(_WORKTREE_ROOT) not in sys.modules[_mod].__file__:
            del sys.modules[_mod]

import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from policy_doctor.behaviors.behavior_graph import BehaviorGraph
from policy_doctor.behaviors import graph_simplification as gs
from policy_doctor.behaviors.clustering_temporal import (
    build_episode_cluster_map,
    build_cluster_timeline,
)
from policy_doctor.plotting.plotly.behavior_graph import create_behavior_graph_plot
from policy_doctor.plotting.plotly.clusters import CLUSTER_COLORS
from policy_doctor.streamlit_app.components.mp4_player import mp4_player
from policy_doctor.streamlit_app.user_study.graph_explorer import render_graph_full_width

st.set_page_config(page_title="Graph Simplification Comparison", layout="wide")

# ── Sidebar: pick a clustering ───────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Some clustering data lives in the main repo (not the worktree). Look in both.
_MAIN_REPO = Path("/Users/erik/stanford/asl_rotation/policy_doctor")
_IV_CONFIGS = _REPO_ROOT / "third_party" / "influence_visualizer" / "configs"
_IV_CONFIGS_MAIN = _MAIN_REPO / "third_party" / "influence_visualizer" / "configs"


def _list_clusterings(root: Path) -> List[Path]:
    """Find all clustering dirs (each has cluster_labels.npy)."""
    out = []
    for d in sorted(root.rglob("cluster_labels.npy")):
        out.append(d.parent)
    return out


@st.cache_data
def _load_clustering(path: str) -> Tuple[np.ndarray, List[Dict], Optional[np.ndarray], Dict]:
    """Returns (labels, metadata, embeddings_reduced or None, manifest)."""
    p = Path(path)
    labels = np.load(p / "cluster_labels.npy")
    with open(p / "metadata.json") as f:
        meta = json.load(f)
    emb_path = p / "embeddings_reduced.npy"
    emb = np.load(emb_path) if emb_path.exists() else None
    import yaml
    with open(p / "manifest.yaml") as f:
        manifest = yaml.safe_load(f) or {}
    return labels, meta, emb, manifest


st.sidebar.header("Source clustering")

available: List[Path] = []
for root in [_IV_CONFIGS, _IV_CONFIGS_MAIN, _MAIN_REPO / "e1_experiments"]:
    if root.exists():
        available.extend(_list_clusterings(root))
# Dedupe (same dir may appear twice if worktree mirrors a tracked file)
seen = set()
unique: List[Path] = []
for p in available:
    key = str(p.resolve())
    if key not in seen:
        seen.add(key)
        unique.append(p)
available = unique

if not available:
    st.error(f"No clusterings found under {_IV_CONFIGS} or {_E1}.")
    st.stop()


def _has_emb(p: Path) -> bool:
    return (p / "embeddings_reduced.npy").exists()


@st.cache_data(show_spinner=False)
def _short_manifest(path_str: str) -> Dict:
    """Cheap manifest read for the dropdown labels."""
    import yaml as _y
    try:
        with open(Path(path_str) / "manifest.yaml") as f:
            return _y.safe_load(f) or {}
    except Exception:
        return {}


def _pretty_label(p: Path) -> str:
    """Compact label: 'policy_emb/bottleneck_plan_t0 · k=20 · jan28 ·emb'."""
    m = _short_manifest(str(p))
    rep = m.get("influence_source") or m.get("slice_representation") or "?"
    k = m.get("n_clusters", "?")
    layer = (m.get("rep_kwargs") or {}).get("layer", "") if isinstance(m.get("rep_kwargs"), dict) else ""
    task_tag = "?"
    for i, part in enumerate(p.parts):
        if part == "configs" and i + 1 < len(p.parts):
            task_tag = p.parts[i + 1]; break
        if part == "e1_experiments" and i + 1 < len(p.parts):
            task_tag = "e1/" + p.parts[i + 1]; break
    rep_full = f"{rep}/{layer}" if layer else rep
    emb_tag = " ·emb" if _has_emb(p) else ""
    return f"{rep_full} · k={k} · {task_tag}{emb_tag}"


# Group by representation, then sort by k within each group
def _rep_key(p: Path) -> str:
    m = _short_manifest(str(p))
    rep = m.get("influence_source") or m.get("slice_representation") or "?"
    layer = (m.get("rep_kwargs") or {}).get("layer", "") if isinstance(m.get("rep_kwargs"), dict) else ""
    return f"{rep}/{layer}" if layer else rep


available_sorted = sorted(
    available,
    key=lambda p: (_rep_key(p), int(_short_manifest(str(p)).get("n_clusters", 0) or 0), str(p)),
)

# Default: prefer the InfEmbed k=20 (the noisy one the user originally complained about)
default_idx = 0
for i, p in enumerate(available_sorted):
    if "auto_pipeline_test_mar13" in str(p) and "k20" in str(p):
        default_idx = i
        break

st.sidebar.markdown("**Filter clusterings:**")
all_reps = sorted({_rep_key(p) for p in available_sorted})
chosen_reps = st.sidebar.multiselect(
    "Representations",
    options=all_reps,
    default=all_reps,
    help="Only show clusterings using one of these feature spaces.",
)
filtered = [p for p in available_sorted if _rep_key(p) in chosen_reps] or available_sorted
# Re-map default if filtered out
default_idx = min(default_idx, len(filtered) - 1)
if available_sorted[default_idx] not in filtered:
    default_idx = 0

choice = st.sidebar.selectbox(
    "Clustering",
    options=range(len(filtered)),
    format_func=lambda i: _pretty_label(filtered[i]),
    index=default_idx,
)
clustering_path = filtered[choice]
labels0, meta, emb, manifest = _load_clustering(str(clustering_path))
labels0 = labels0.astype(np.int64)

rep_kwargs = manifest.get("rep_kwargs") or {}
layer = rep_kwargs.get("layer", "") if isinstance(rep_kwargs, dict) else ""
st.sidebar.markdown(
    f"**source:** `{manifest.get('influence_source', '?')}`"
    + (f" / `{layer}`" if layer else "")
    + f"  \n**algo:** `{manifest.get('algorithm', '?')}`  k=`{manifest.get('n_clusters', '?')}`  \n"
    f"**windowing:** W=`{manifest.get('window_width', '?')}` "
    f"S=`{manifest.get('stride', '?')}` agg=`{manifest.get('aggregation', '?')}`  \n"
    f"**n_samples:** {len(labels0):,}  \n"
    f"**emb shape:** {emb.shape if emb is not None else 'N/A'}"
)

# Episodes count
n_eps = len(set(m.get("rollout_idx", m.get("demo_idx", 0)) for m in meta))
st.sidebar.markdown(f"**episodes:** {n_eps}")

# Level: rollout vs demo
level = manifest.get("level", "rollout")

# ── Settings shared across views ─────────────────────────────────────────────

st.sidebar.header("Display")
height = st.sidebar.slider("Plot height (px)", 400, 1200, 600, step=50)
min_prob_display = st.sidebar.slider(
    "Display min-prob (post-method)", 0.0, 0.5, 0.0, step=0.01,
    help="Hides edges below this probability AFTER the method runs. Display-only.",
)
use_temporal_layout = st.sidebar.checkbox(
    "Use temporal (mean-timestep) layout instead of BFS-layered",
    value=True,
)
use_native_renderer = st.sidebar.checkbox(
    "Use native SVG renderer (clickable nodes/edges, video clips)",
    value=True,
    help="Native interactive graph component. Click any node or edge to see its details.",
)

# MP4 dir auto-discovery (so the native renderer can show video clips)
_DEFAULT_MP4_DIR = Path("/tmp/study_mp4s")
mp4_root_str = st.sidebar.text_input(
    "MP4 root (optional)",
    value=str(_DEFAULT_MP4_DIR),
    help="Directory containing <task>/index.json + ep*.mp4 files. Used by the native renderer.",
)
mp4_root = Path(mp4_root_str)


# ── Cached graph builders ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _build_graph(labels: np.ndarray, _meta_id: int) -> BehaviorGraph:
    return BehaviorGraph.from_cluster_assignments(labels, meta, level=level)


def _layout_for(graph: BehaviorGraph, labels: np.ndarray) -> Optional[Dict[int, Tuple[float, float]]]:
    if use_temporal_layout:
        return gs.temporal_layout(graph, labels, meta, level=level)
    return None


def _graph_stats(graph: BehaviorGraph) -> Tuple[int, int]:
    n_nodes = len([n for n in graph.nodes if n >= 0])
    n_edges = sum(len(t) for t in graph.transition_counts.values())
    return n_nodes, n_edges


def _plot(graph: BehaviorGraph, labels: np.ndarray, title: str) -> go.Figure:
    pos = _layout_for(graph, labels)
    return create_behavior_graph_plot(
        graph,
        min_probability=min_prob_display,
        height=height,
        title=title,
        pos=pos,
    )


# Auto-discover MP4s for the source clustering's task. Only accept MP4s
# whose episode indices and frame counts are consistent with the clustering's
# metadata — otherwise we'd display wrong videos for the wrong task.
def _task_tag_from_path(p: Path) -> Optional[str]:
    parts = p.parts
    for i, part in enumerate(parts):
        if part == "configs" and i + 1 < len(parts):
            return parts[i + 1]
        if part == "e1_experiments" and i + 1 < len(parts):
            return None  # e1 experiments are on different tasks; can't match
    return None


def _mp4_matches_clustering(mp4_index: Dict, meta_list: List[Dict]) -> bool:
    """Verify the MP4 index's episodes cover the clustering's windows.

    Each (rollout_idx, window_end) in meta must be ≤ the corresponding
    episode's frame_count in the MP4 index. Otherwise the MP4s are for a
    different task / shorter rollouts.
    """
    ep_frames = {ep["index"]: int(ep.get("frame_count", 0)) for ep in mp4_index.get("episodes", [])}
    if not ep_frames:
        return False
    ep_key = "rollout_idx" if any("rollout_idx" in m for m in meta_list) else "demo_idx"
    sample = meta_list[::max(1, len(meta_list) // 200)]  # 200-pt sanity sample
    for m in sample:
        ep = m.get(ep_key)
        if ep is None or ep not in ep_frames:
            return False
        we = int(m.get("window_end", m.get("timestep", 0) + 1))
        if we > ep_frames[ep]:
            return False
    return True


def _resolve_mp4_dir_and_index() -> Tuple[Optional[Path], Dict, Optional[str]]:
    """Returns (path, index, mismatch_reason)."""
    if not mp4_root.exists():
        return None, {"episodes": []}, f"MP4 root `{mp4_root}` does not exist."
    task_tag = _task_tag_from_path(clustering_path)
    candidates: List[Path] = []
    if task_tag:
        candidates.append(mp4_root / task_tag)
    if mp4_root.is_dir():
        for c in mp4_root.iterdir():
            if (c / "index.json").exists() and c not in candidates:
                candidates.append(c)
    for c in candidates:
        idx_path = c / "index.json"
        if not idx_path.exists():
            continue
        try:
            mp4_idx = json.load(open(idx_path))
        except Exception:
            continue
        if _mp4_matches_clustering(mp4_idx, meta):
            return c, mp4_idx, None
    if task_tag:
        return None, {"episodes": []}, (
            f"No MP4 directory in `{mp4_root}` matches the source clustering's task "
            f"`{task_tag}` (episode indices / frame counts disagree). "
            f"Cluster inspector and episode browser are disabled."
        )
    return None, {"episodes": []}, (
        f"This clustering is from an e1_experiments sweep — no MP4s available. "
        f"Cluster inspector and episode browser are disabled."
    )


_MP4_DIR, _MP4_INDEX, _MP4_MISMATCH = _resolve_mp4_dir_and_index()
if _MP4_DIR is not None:
    st.sidebar.success(f"MP4s: {_MP4_DIR.name} ({len(_MP4_INDEX.get('episodes', []))} eps)")
elif _MP4_MISMATCH:
    st.sidebar.warning(_MP4_MISMATCH)


# ── Cluster names: per-clustering YAML persistence ───────────────────────────

_NAMES_DIR = _REPO_ROOT / "data" / "cluster_names"
_NAMES_DIR.mkdir(parents=True, exist_ok=True)


def _names_key() -> str:
    """Stable identifier for the current clustering's name YAML file."""
    # Use the clustering directory name (already encodes task + algo + k + seed)
    return clustering_path.name


def _load_names() -> Dict[int, str]:
    p = _NAMES_DIR / f"{_names_key()}.yaml"
    if not p.exists():
        return {}
    try:
        import yaml as _y
        data = _y.safe_load(p.read_text()) or {}
        return {int(k): str(v) for k, v in data.items()}
    except Exception:
        return {}


def _save_names(names: Dict[int, str]) -> Path:
    import yaml as _y
    p = _NAMES_DIR / f"{_names_key()}.yaml"
    p.write_text(_y.safe_dump({int(k): str(v) for k, v in names.items()}, sort_keys=True))
    return p


# Load on session start
_NAMES_STATE_KEY = f"cluster_names::{_names_key()}"
if _NAMES_STATE_KEY not in st.session_state:
    st.session_state[_NAMES_STATE_KEY] = _load_names()


def _name_for(cluster_id: int) -> str:
    """Resolve a display name for a cluster, falling back to 'Behavior <id>'."""
    custom = st.session_state[_NAMES_STATE_KEY].get(int(cluster_id))
    return custom if custom else f"Behavior {cluster_id}"


def _apply_names_to_graph(graph: BehaviorGraph) -> BehaviorGraph:
    """Return a shallow copy of the graph with cluster_node.name replaced by user names."""
    names_map = st.session_state[_NAMES_STATE_KEY]
    if not names_map:
        return graph
    new_nodes = {}
    from dataclasses import replace
    for nid, node in graph.nodes.items():
        if node.is_special:
            new_nodes[nid] = node
        elif int(nid) in names_map and names_map[int(nid)]:
            new_nodes[nid] = replace(node, name=names_map[int(nid)])
        else:
            new_nodes[nid] = node
    return BehaviorGraph(
        nodes=new_nodes,
        transition_counts=graph.transition_counts,
        transition_probs=graph.transition_probs,
        num_episodes=graph.num_episodes,
        level=graph.level,
    )


def _show_graph(
    graph: BehaviorGraph,
    labels: np.ndarray,
    title: str,
    key_prefix: str,
) -> None:
    """Render the graph. Uses native SVG component when enabled. Applies user names."""
    graph = _apply_names_to_graph(graph)
    if not use_native_renderer:
        st.plotly_chart(_plot(graph, labels, title), use_container_width=True, key=f"plotly_{key_prefix}_{title}")
        return
    pos = _layout_for(graph, labels)
    render_graph_full_width(
        graph=graph,
        labels=labels,
        metadata=meta,
        mp4_dir=_MP4_DIR if _MP4_DIR is not None else Path("/tmp/_nonexistent_mp4_root"),
        mp4_index=_MP4_INDEX,
        key_prefix=f"native_{key_prefix}_{abs(hash(title))}",
        min_edge_prob=min_prob_display,
        pos=pos,
    )


def _render_episode_browser(
    labels: np.ndarray,
    title: str,
    key_prefix: str,
    episodes_per_page: int = 6,
) -> None:
    """Paginated episodes with MP4 player + per-frame cluster-assignment timeline.

    Reflects the *current* labels — re-renders cleanly when a method changes them.
    """
    if _MP4_DIR is None:
        st.info(_MP4_MISMATCH or f"No MP4 index at `{mp4_root}`; episode browser disabled.")
        return
    with st.expander(f"📼 Episode browser — cluster timeline under each video ({title})", expanded=False):
        rep = "sliding_window"
        ep_map = build_episode_cluster_map(labels, meta, rep, level)

        idx_eps = [ep["index"] for ep in _MP4_INDEX.get("episodes", [])]
        avail = sorted(set(ep_map.keys()) & set(idx_eps))
        if not avail:
            st.warning("No episodes available in both clustering and MP4 index.")
            return

        col_f, col_p = st.columns([2, 1])
        with col_f:
            filt = st.radio(
                "Filter", ["All", "Success only", "Failure only"],
                horizontal=True, key=f"{key_prefix}_filt",
            )
        success_by_ep = {ep["index"]: ep.get("success") for ep in _MP4_INDEX["episodes"]}
        if filt == "Success only":
            avail = [e for e in avail if success_by_ep.get(e) is True]
        elif filt == "Failure only":
            avail = [e for e in avail if success_by_ep.get(e) is False]
        if not avail:
            st.info(f"No episodes match filter '{filt}'.")
            return

        total_pages = max(1, (len(avail) + episodes_per_page - 1) // episodes_per_page)
        page = st.number_input(
            f"Page (1–{total_pages})", min_value=1, max_value=total_pages,
            value=1, key=f"{key_prefix}_pg",
        )
        with col_p:
            st.caption(f"{len(avail)} episodes · {episodes_per_page} per page · page {page}/{total_pages}")
        lo = (int(page) - 1) * episodes_per_page
        hi = min(lo + episodes_per_page, len(avail))
        fps = int(_MP4_INDEX.get("fps") or 10)
        cluster_names = {nid: f"Cluster {nid}" for nid in sorted(set(int(c) for c in labels) - {-1})}

        # Lay out as a 2-column grid for compactness.
        cols_per_row = 2
        page_eps = avail[lo:hi]
        for row_i in range(0, len(page_eps), cols_per_row):
            row_eps = page_eps[row_i:row_i + cols_per_row]
            row_cols = st.columns(len(row_eps))
            for col, ep_id in zip(row_cols, row_eps):
                ep_entry = next((e for e in _MP4_INDEX["episodes"] if e["index"] == ep_id), None)
                if ep_entry is None:
                    continue
                num_frames = int(ep_entry.get("frame_count") or 0)
                timeline = build_cluster_timeline(num_frames, ep_id, ep_map, rep)
                status = "✓" if ep_entry.get("success") is True else \
                         "✗" if ep_entry.get("success") is False else ""
                with col:
                    st.caption(f"Episode {ep_id} — {status} · {num_frames} frames")
                    mp4_player(
                        _MP4_DIR / ep_entry["path"],
                        key=f"{key_prefix}_vid_{ep_id}",
                        max_height_px=240,
                        total_frames=num_frames,
                        fps=fps,
                        per_frame_labels=np.asarray(timeline, dtype=np.int64),
                        cluster_colors=list(CLUSTER_COLORS),
                        cluster_names=cluster_names,
                    )


# ── Tabs: one per method family ──────────────────────────────────────────────

st.title("Behavior Graph Simplification — Method Comparison")
st.caption(
    "Pick a source clustering on the left. Each tab applies one (or more) simplification "
    "methods and renders the resulting graph next to the baseline. Edge probabilities are "
    "computed AFTER the method runs and are unaffected by display-time filtering except for "
    "the 'Display min-prob' slider."
)

baseline_graph = _build_graph(labels0, id(meta))
n_nodes0, n_edges0 = _graph_stats(baseline_graph)
st.markdown(f"**Baseline:** {n_nodes0} cluster nodes, {n_edges0} transitions")

tab_inspect, tab_smooth, tab_prune, tab_merge, tab_recluster, tab_layout, tab_combo, tab_reps = st.tabs([
    "🔍 Cluster inspector",
    "1. Temporal smoothing",
    "2. Edge pruning",
    "3. Node merging",
    "4. Re-clustering",
    "5. Layout only",
    "6. Combined pipeline",
    "7. Compare representations",
])


# ── Tab 0: Cluster inspector ─────────────────────────────────────────────────

with tab_inspect:
    # ── What you're looking at ───────────────────────────────────────────────
    _ww = manifest.get("window_width", 5)
    _ss = manifest.get("stride", 2)
    _agg = manifest.get("aggregation", "sum")
    _layer = (manifest.get("rep_kwargs") or {}).get("layer", "") if isinstance(manifest.get("rep_kwargs"), dict) else ""
    _src = manifest.get("influence_source") or manifest.get("slice_representation", "?")
    _rep_pretty = f"{_src}/{_layer}" if _layer else _src
    with st.container(border=True):
        st.markdown(f"### 📋 What you're looking at")
        st.markdown(
            f"**Source clustering:** `{clustering_path.name}` "
            f"(switch in the sidebar — try `policy_emb/bottleneck_plan_t0` for a cleaner baseline)\n\n"
            f"**Pipeline that produced these clusters:**\n"
            f"1. Take each rollout episode's per-timestep features in the **{_rep_pretty}** space.\n"
            f"2. Slide a **W={_ww}** window with **stride={_ss}**, aggregate by **{_agg}** → "
            f"one feature vector per window ({len(labels0):,} windows total across {n_eps} episodes).\n"
            f"3. Pre-scale → UMAP to {emb.shape[1] if emb is not None else '?'} dims (saved to `embeddings_reduced.npy`).\n"
            f"4. Run **{manifest.get('algorithm', 'kmeans')}** with K={manifest.get('n_clusters', '?')} → "
            f"every window gets a single integer cluster label, in [0, K-1].\n\n"
            f"**A 'cluster' is a set of windows the algorithm thinks are similar in feature space.** "
            f"Each clip below is one such window — a {_ww}-frame slice of one rollout episode, "
            f"played back at native fps with the orange bar marking the slice. "
            f"If the clips in a cluster look like the *same kind of behavior* to you, the cluster "
            f"is coherent; if not, the feature space and/or K are wrong for this task."
        )
        st.caption(
            "All other tabs (smoothing / pruning / re-clustering / etc.) "
            "operate as transformations *on top of* this baseline assignment."
        )

    st.markdown(
        "**Is this cluster actually one behavior?** Pick a cluster, page through its clips, "
        "watch each. Rename clusters as you go — names persist to "
        f"`{_NAMES_DIR}` and propagate to every other tab in the app."
    )

    if _MP4_DIR is None:
        st.warning(_MP4_MISMATCH or f"No MP4 index at `{mp4_root}`. Cluster inspector needs videos.")
    else:
        cluster_ids = sorted(set(int(c) for c in labels0) - {-1})

        col_pick, col_n = st.columns([3, 1])
        with col_pick:
            chosen = st.multiselect(
                "Clusters to inspect (showing all by default)",
                options=cluster_ids,
                default=cluster_ids[:6] if len(cluster_ids) > 6 else cluster_ids,
                format_func=lambda c: f"{c}: {_name_for(c)}",
                key="inspect_clusters",
            )
        with col_n:
            n_clips = st.number_input(
                "Clips per cluster", min_value=2, max_value=12, value=6, step=2,
                key="inspect_n_clips",
            )

        # ── Bulk rename panel ────────────────────────────────────────────────
        with st.expander("✏️ Edit names (saves to YAML)", expanded=False):
            names_state = st.session_state[_NAMES_STATE_KEY]
            cols_per_row = 3
            edited = {}
            for row_i in range(0, len(cluster_ids), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for col, cid in zip(row_cols, cluster_ids[row_i:row_i + cols_per_row]):
                    with col:
                        edited[cid] = st.text_input(
                            f"Cluster {cid}",
                            value=names_state.get(cid, ""),
                            placeholder=f"Behavior {cid}",
                            key=f"name_input_{cid}",
                        )
            save_col, clear_col, info_col = st.columns([1, 1, 4])
            with save_col:
                if st.button("💾 Save", key="save_names"):
                    # Drop empty entries
                    new_names = {cid: name.strip() for cid, name in edited.items() if name.strip()}
                    st.session_state[_NAMES_STATE_KEY] = new_names
                    path = _save_names(new_names)
                    st.success(f"Saved {len(new_names)} names → {path.name}")
            with clear_col:
                if st.button("🗑 Clear all", key="clear_names"):
                    st.session_state[_NAMES_STATE_KEY] = {}
                    _save_names({})
                    st.rerun()
            with info_col:
                st.caption(f"YAML: `{_NAMES_DIR / (_names_key() + '.yaml')}`")

        # ── Cluster card grid ────────────────────────────────────────────────
        if not chosen:
            st.info("Pick at least one cluster above.")
        else:
            # Pre-compute window indices and episode-success per cluster
            ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
            ep_success = {ep["index"]: ep.get("success") for ep in _MP4_INDEX.get("episodes", [])}
            mp4_by_ep = {ep["index"]: ep for ep in _MP4_INDEX.get("episodes", [])}
            fps = int(_MP4_INDEX.get("fps") or 10)
            rng = np.random.default_rng(42)

            # Map cluster_id -> list of (sample_idx, ep_idx, window_start, window_end, success)
            cluster_windows: Dict[int, List[Tuple[int, int, int, int, Optional[bool]]]] = {}
            for cid in chosen:
                mask = (labels0 == cid)
                idxs = np.where(mask)[0]
                tmp = []
                for i in idxs:
                    m = meta[i]
                    ep_idx = m[ep_key]
                    ws = int(m.get("window_start", m.get("timestep", 0)))
                    we = int(m.get("window_end", ws + 1))
                    tmp.append((int(i), int(ep_idx), ws, we, ep_success.get(int(ep_idx))))
                cluster_windows[cid] = tmp

            for cid in chosen:
                wins = cluster_windows[cid]
                n_total = len(wins)
                if n_total == 0:
                    st.warning(f"Cluster {cid} has 0 windows.")
                    continue
                # Stats
                eps_in = {w[1] for w in wins}
                eps_success_in = sum(1 for e in eps_in if ep_success.get(e) is True)
                success_rate = eps_success_in / len(eps_in) if eps_in else 0.0
                pos_fracs = []
                for _, ep_idx, ws, we, _ in wins:
                    total_f = mp4_by_ep.get(ep_idx, {}).get("frame_count") or 0
                    if total_f <= 0:
                        continue  # skip windows for which we don't know episode length
                    frac = ((ws + we) / 2) / total_f
                    if 0.0 <= frac <= 1.5:  # tolerate small overshoot from window padding
                        pos_fracs.append(min(1.0, frac))
                mean_pos = float(np.mean(pos_fracs)) if pos_fracs else 0.0

                # ── Build a deterministic ordering of windows for paging.    ──
                # Strategy "diverse first": iterate episodes round-robin so each
                # page samples from many episodes before repeating; then fall
                # back to remaining windows in order.
                cluster_rng = np.random.default_rng(int(cid) * 9967 + 42)
                wins_by_ep_local: Dict[int, List] = {}
                for w in wins:
                    wins_by_ep_local.setdefault(w[1], []).append(w)
                # Shuffle within each episode (random clip pick), then shuffle episode order
                for ep, lst in wins_by_ep_local.items():
                    cluster_rng.shuffle(lst)
                ep_order_local = list(wins_by_ep_local.keys())
                cluster_rng.shuffle(ep_order_local)
                # Round-robin: take 1 from each episode, then 2 from each, ...
                ordered: List = []
                round_idx = 0
                while sum(len(v) for v in wins_by_ep_local.values()) > 0:
                    for ep in ep_order_local:
                        bucket = wins_by_ep_local[ep]
                        if round_idx < len(bucket):
                            ordered.append(bucket[round_idx])
                    round_idx += 1
                    if all(round_idx >= len(b) for b in wins_by_ep_local.values()):
                        break

                # Pagination state
                page_key = f"inspect_page_{_names_key()}_{cid}"
                if page_key not in st.session_state:
                    st.session_state[page_key] = 0
                total_pages = max(1, (n_total + int(n_clips) - 1) // int(n_clips))
                page = max(0, min(st.session_state[page_key], total_pages - 1))

                # Header + pager controls
                h_col, prev_col, page_col, next_col = st.columns([8, 1, 2, 1])
                with h_col:
                    st.markdown(
                        f"### {cid}: **{_name_for(cid)}**  ·  "
                        f"{n_total} windows · {len(eps_in)} episodes · "
                        f"{success_rate:.0%} success · mean position {mean_pos:.0%} through episode"
                    )
                with prev_col:
                    if st.button("←", key=f"{page_key}_prev", disabled=(page == 0)):
                        st.session_state[page_key] = max(0, page - 1)
                        st.rerun()
                with page_col:
                    st.markdown(
                        f"<div style='text-align:center;padding-top:8px;color:#888;font-size:0.85em;'>"
                        f"Page {page + 1}/{total_pages} · clips {page * int(n_clips) + 1}–"
                        f"{min((page + 1) * int(n_clips), n_total)} of {n_total}"
                        f"</div>", unsafe_allow_html=True,
                    )
                with next_col:
                    if st.button("→", key=f"{page_key}_next", disabled=(page >= total_pages - 1)):
                        st.session_state[page_key] = min(total_pages - 1, page + 1)
                        st.rerun()

                lo = page * int(n_clips)
                hi = min(lo + int(n_clips), n_total)
                samples = ordered[lo:hi]

                cpr = 3
                for row_i in range(0, len(samples), cpr):
                    row = samples[row_i:row_i + cpr]
                    row_cols = st.columns(cpr)
                    for col, (_, ep_idx, ws, we, succ) in zip(row_cols, row):
                        ep_entry = mp4_by_ep.get(ep_idx)
                        if ep_entry is None:
                            continue
                        total_f = ep_entry.get("frame_count") or 0
                        status = "✓" if succ is True else "✗" if succ is False else ""
                        with col:
                            st.caption(f"Ep {ep_idx} {status} · frames {ws}–{we}")
                            mp4_player(
                                _MP4_DIR / ep_entry["path"],
                                key=f"inspect_{cid}_{ep_idx}_{ws}_p{page}",
                                max_height_px=180,
                                total_frames=total_f,
                                fps=fps,
                                slice_start=ws,
                                slice_end=we,
                            )
                st.markdown("---")


# ── Tab 1: Temporal smoothing ────────────────────────────────────────────────

with tab_smooth:
    st.markdown(
        "Label-sequence smoothing **before** building the graph. Reduces "
        "A→B→A→B flicker that survives run-length collapse."
    )
    smooth_method = st.radio(
        "Method",
        ["Median filter (C1)", "Sticky DP decoder (C3)", "Gaussian HMM Viterbi (C2)"],
        horizontal=True,
    )
    new_labels: np.ndarray
    if smooth_method.startswith("Median"):
        w = st.slider("Median window", 1, 21, 5, step=2, key="median_w")
        new_labels = gs.median_filter_labels(labels0, meta, window=w, level=level)
        title = f"Median filter (w={w})"
    elif smooth_method.startswith("Sticky"):
        if emb is None:
            st.warning("This clustering has no embeddings_reduced.npy; sticky decoder unavailable.")
            st.stop()
        lam = st.slider("Stickiness λ (transition penalty)", 0.0, 30.0, 5.0, step=0.5, key="sticky_lam")
        new_labels = gs.sticky_decoder(labels0, emb, meta, lambda_stick=lam, level=level)
        title = f"Sticky DP (λ={lam})"
    else:
        if emb is None:
            st.warning("This clustering has no embeddings_reduced.npy; HMM unavailable.")
            st.stop()
        n_states = st.slider("HMM states", 2, 20, 8, key="hmm_k")
        new_labels = gs.hmm_smooth(emb, meta, n_states=n_states, level=level)
        title = f"HMM (n_states={n_states})"

    g_new = _build_graph(new_labels, id(meta) + hash(title))
    nn, ne = _graph_stats(g_new)
    st.markdown(f"**{title}** → {nn} nodes, {ne} edges (vs baseline {n_nodes0}/{n_edges0})")
    _show_graph(g_new, new_labels, title, key_prefix=f"smooth_main_{title}")
    _render_episode_browser(new_labels, title, key_prefix=f"smooth_eps_{abs(hash(title))}")
    with st.expander("Baseline (for comparison)"):
        _show_graph(baseline_graph, labels0, "Baseline", key_prefix=f"smooth_base_{title}")


# ── Tab 2: Edge pruning ──────────────────────────────────────────────────────

with tab_prune:
    st.markdown(
        "Hard-prune edges in the graph (not just hide). Removed edges' counts are dropped "
        "and probabilities are renormalized — downstream value / path computations see "
        "the pruned graph."
    )
    mode = st.radio("Pruning mode", ["By count (D1)", "By probability"], horizontal=True)
    if mode.startswith("By count"):
        min_count = st.slider("Min edge count", 1, 200, 5, key="prune_count")
        g_new = gs.prune_edges_by_count(baseline_graph, min_count=min_count)
        title = f"Prune by count (min={min_count})"
    else:
        min_p = st.slider("Min edge probability", 0.0, 0.5, 0.05, step=0.01, key="prune_prob")
        g_new = gs.prune_edges_by_prob(baseline_graph, min_prob=min_p)
        title = f"Prune by prob (min={min_p:.2f})"
    nn, ne = _graph_stats(g_new)
    st.markdown(f"**{title}** → {nn} nodes, {ne} edges (vs baseline {n_nodes0}/{n_edges0})")
    _show_graph(g_new, labels0, title, key_prefix=f"prune_main_{title}")
    _render_episode_browser(labels0, title, key_prefix=f"prune_eps_{abs(hash(title))}")
    with st.expander("Baseline (for comparison)"):
        _show_graph(baseline_graph, labels0, "Baseline", key_prefix=f"prune_base_{title}")


# ── Tab 3: Node merging ──────────────────────────────────────────────────────

with tab_merge:
    st.markdown(
        "Merge cluster nodes that are similar in embedding space (D4) or that look like "
        "stable continuations of another node (D8, ENAP-style)."
    )
    method = st.radio("Method", ["Cosine-similar centroids (D4)", "Stable-phase prune (D8)"], horizontal=True)
    if method.startswith("Cosine"):
        if emb is None:
            st.warning("This clustering has no embeddings_reduced.npy; centroid merging unavailable.")
            st.stop()
        thresh = st.slider("Similarity threshold", 0.0, 0.999, 0.9, step=0.005, key="merge_thresh")
        new_labels = gs.merge_similar_centroids(labels0, emb, meta, sim_threshold=thresh, level=level)
        g_new = _build_graph(new_labels, id(meta) + hash(f"merge_{thresh}"))
        title = f"Merge similar centroids (≥{thresh:.2f})"
    else:
        g_new, new_labels = gs.stable_phase_prune(baseline_graph, labels0, meta)
        title = "Stable-phase prune (D8)"
    nn, ne = _graph_stats(g_new)
    st.markdown(f"**{title}** → {nn} nodes, {ne} edges (vs baseline {n_nodes0}/{n_edges0})")
    _show_graph(g_new, new_labels, title, key_prefix=f"merge_main_{title}")
    _render_episode_browser(new_labels, title, key_prefix=f"merge_eps_{abs(hash(title))}")
    with st.expander("Baseline (for comparison)"):
        _show_graph(baseline_graph, labels0, "Baseline", key_prefix=f"merge_base_{title}")


# ── Tab 4: Re-clustering ─────────────────────────────────────────────────────

with tab_recluster:
    st.markdown(
        "Replace the cluster assignment entirely. Auto-K picks the K with the best "
        "silhouette score. Spectral builds the transition affinity between micro-clusters "
        "and groups them into macro-clusters. Change-point detects boundaries per episode "
        "first, then clusters segment means."
    )
    method = st.radio(
        "Method",
        ["Auto-K KMeans (B1)", "Spectral on transition graph (B4)", "Change-point + KMeans (D6+C5)"],
        horizontal=True,
    )
    if method.startswith("Auto-K"):
        if emb is None:
            st.warning("This clustering has no embeddings_reduced.npy; auto-K unavailable.")
            st.stop()
        k_lo, k_hi = st.slider("K range", 2, 25, (4, 15), key="autok_range")
        new_labels, best_k, scores = gs.auto_k_kmeans(emb, k_range=(k_lo, k_hi))
        title = f"Auto-K (best k={best_k})"
        with st.expander("Silhouette scores by K"):
            st.json({str(k): round(v, 3) for k, v in scores.items()})
    elif method.startswith("Spectral"):
        n_macro = st.slider("Macro-cluster count", 2, 15, 8, key="spec_k")
        new_labels = gs.spectral_transition_clustering(labels0, meta, n_macro=n_macro, level=level)
        title = f"Spectral on transition graph (k_macro={n_macro})"
    else:
        if emb is None:
            st.warning("This clustering has no embeddings_reduced.npy; change-point unavailable.")
            st.stop()
        cp_k = st.slider("Segment cluster count", 2, 15, 8, key="cp_k")
        cp_pen = st.slider("Change-point penalty", 1.0, 50.0, 10.0, step=1.0, key="cp_pen")
        new_labels = gs.change_point_segmentation(emb, meta, n_macro=cp_k, penalty=cp_pen, level=level)
        title = f"Change-point (k={cp_k}, pen={cp_pen})"
    g_new = _build_graph(new_labels, id(meta) + hash(title))
    nn, ne = _graph_stats(g_new)
    st.markdown(f"**{title}** → {nn} nodes, {ne} edges (vs baseline {n_nodes0}/{n_edges0})")
    _show_graph(g_new, new_labels, title, key_prefix=f"recluster_main_{title}")
    _render_episode_browser(new_labels, title, key_prefix=f"recluster_eps_{abs(hash(title))}")
    with st.expander("Baseline (for comparison)"):
        _show_graph(baseline_graph, labels0, "Baseline", key_prefix=f"recluster_base_{title}")


# ── Tab 5: Layout only ───────────────────────────────────────────────────────

with tab_layout:
    st.markdown(
        "**Layout only.** No change to the graph structure; just plot with a different "
        "layout to test whether the noise is structural or visual."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Default BFS-layered**")
        fig1 = create_behavior_graph_plot(
            baseline_graph, min_probability=min_prob_display, height=height,
            title="BFS-layered (default)",
        )
        st.plotly_chart(fig1, use_container_width=True, key="layout_bfs")
    with c2:
        st.markdown("**Temporal (mean-timestep x)**")
        pos = gs.temporal_layout(baseline_graph, labels0, meta, level=level)
        fig2 = create_behavior_graph_plot(
            baseline_graph, min_probability=min_prob_display, height=height,
            title="Temporal layout (E1)", pos=pos,
        )
        st.plotly_chart(fig2, use_container_width=True, key="layout_temporal")


# ── Tab 6: Combined pipeline ─────────────────────────────────────────────────

with tab_combo:
    st.markdown(
        "**Stack methods.** Order: smooth → merge → stable-phase prune → edge prune."
    )

    # Controls in a single row above the graph.
    with st.container(border=True):
        st.markdown("##### Pipeline configuration")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            use_smooth = st.checkbox("Smooth labels", value=True, key="combo_use_smooth")
            smooth_mode = st.selectbox(
                "Smoother", ["median", "sticky"],
                disabled=not use_smooth, key="combo_smooth_mode",
            )
            smooth_strength = st.slider(
                "Smoother strength", 1.0, 20.0, 5.0,
                disabled=not use_smooth, key="combo_smooth",
            )
        with c2:
            use_merge = st.checkbox("Merge similar centroids", value=False, key="combo_use_merge")
            merge_thresh = st.slider(
                "Centroid sim threshold", 0.0, 0.999, 0.92,
                disabled=not use_merge, key="combo_merge",
            )
        with c3:
            use_stable = st.checkbox("ENAP stable-phase prune", value=False, key="combo_use_stable")
        with c4:
            use_prune = st.checkbox("Prune low-count edges", value=True, key="combo_use_prune")
            prune_count = st.slider(
                "Min edge count", 1, 100, 5,
                disabled=not use_prune, key="combo_prune",
            )

    new_labels = labels0.copy()
    if use_smooth:
        if smooth_mode == "median":
            new_labels = gs.median_filter_labels(new_labels, meta, window=int(smooth_strength), level=level)
        else:
            if emb is None:
                st.warning("This clustering has no embeddings_reduced.npy; sticky unavailable. Skipping.")
            else:
                new_labels = gs.sticky_decoder(new_labels, emb, meta, lambda_stick=smooth_strength, level=level)
    if use_merge and emb is not None:
        new_labels = gs.merge_similar_centroids(new_labels, emb, meta, sim_threshold=merge_thresh, level=level)
    g = _build_graph(new_labels, id(meta) + hash(("combo", use_smooth, smooth_mode, smooth_strength, use_merge, merge_thresh, use_prune, prune_count, use_stable)))
    if use_stable:
        g, new_labels = gs.stable_phase_prune(g, new_labels, meta)
    if use_prune:
        g = gs.prune_edges_by_count(g, min_count=prune_count)
    nn, ne = _graph_stats(g)

    st.markdown(
        f"**Combined pipeline** → {nn} nodes, {ne} edges  (baseline: {n_nodes0}/{n_edges0})"
    )
    _show_graph(g, new_labels, f"Combined: {nn} nodes / {ne} edges", key_prefix="combined_main")
    _render_episode_browser(new_labels, "combined", key_prefix="combined_eps")


# ── Tab 7: Compare representations ───────────────────────────────────────────

with tab_reps:
    st.markdown(
        "Same simplification settings, different **feature representations**. "
        "If the cleaned-up graph still looks bad in one representation but clean in another, "
        "the feature space is the limiting factor — not the simplification method."
    )
    # Group by (task, representation, k) so the user can compare apples to apples.
    @st.cache_data(show_spinner=False)
    def _build_repr_index() -> List[Dict]:
        import yaml as _y
        entries: List[Dict] = []
        for p in available:
            try:
                with open(p / "manifest.yaml") as f:
                    mm = _y.safe_load(f) or {}
            except Exception:
                continue
            rep = mm.get("influence_source") or mm.get("slice_representation") or "?"
            # Try to infer a task tag from the path
            ps = p.parts
            task_tag = "?"
            for i, part in enumerate(ps):
                if part == "configs" and i + 1 < len(ps):
                    task_tag = ps[i + 1]
                    break
                if part == "e1_experiments" and i + 1 < len(ps):
                    task_tag = "e1/" + ps[i + 1]
                    break
            entries.append({
                "path": p,
                "rep": rep,
                "k": mm.get("n_clusters", "?"),
                "task": task_tag,
                "has_emb": (p / "embeddings_reduced.npy").exists(),
                "level": mm.get("level", "rollout"),
            })
        return entries

    entries = _build_repr_index()
    tasks = sorted({e["task"] for e in entries})
    if not tasks:
        st.info("No clusterings found.")
    else:
        task_pick = st.selectbox(
            "Task / experiment family",
            tasks,
            index=tasks.index("transport_mh_jan28") if "transport_mh_jan28" in tasks else 0,
        )
        cands = [e for e in entries if e["task"] == task_pick]
        if not cands:
            st.info(f"No clusterings under task {task_pick}.")
        else:
            # ── Raw-quality table (no smoothing, no pruning) ─────────────────
            @st.cache_data(show_spinner=False)
            def _raw_stats_for(path_str: str) -> Dict:
                from collections import defaultdict
                pth = Path(path_str)
                L = np.load(pth / "cluster_labels.npy").astype(np.int64)
                with open(pth / "metadata.json") as f:
                    M = json.load(f)
                g = BehaviorGraph.from_cluster_assignments(L, M, level="rollout")
                n_nodes = len([n for n in g.nodes if n >= 0])
                n_edges = sum(len(t) for t in g.transition_counts.values())
                eps = defaultdict(list)
                for i, mm in enumerate(M):
                    eps[mm.get("rollout_idx", mm.get("demo_idx", 0))].append(
                        (mm.get("window_start", mm.get("timestep", 0)), int(L[i]))
                    )
                runs: List[int] = []; swaps = 0; total = 0
                for _, seq in eps.items():
                    seq.sort()
                    labs = [v for _, v in seq]
                    i = 0
                    while i < len(labs):
                        j = i + 1
                        while j < len(labs) and labs[j] == labs[i]:
                            j += 1
                        runs.append(j - i); i = j
                    for k in range(1, len(labs)):
                        total += 1
                        if labs[k] != labs[k-1]:
                            swaps += 1
                return {
                    "nodes": n_nodes, "edges": n_edges,
                    "avg_run": float(np.mean(runs)) if runs else 0.0,
                    "swap_rate": swaps / max(1, total),
                }

            st.markdown(
                "**Raw graph quality across feature spaces** "
                "(no smoothing, no pruning). Lower swap rate / longer run length → "
                "cleaner *intrinsic* structure → feature space tracks behaviour, not noise."
            )
            import pandas as pd
            rows = []
            for e in cands:
                stats = _raw_stats_for(str(e["path"]))
                rows.append({
                    "representation": e["rep"],
                    "K": e["k"],
                    "nodes": stats["nodes"],
                    "edges": stats["edges"],
                    "avg run length": round(stats["avg_run"], 2),
                    "swap rate": f"{stats['swap_rate']:.1%}",
                    "name": e["path"].name,
                })
            df = pd.DataFrame(rows).sort_values(["swap rate", "edges"])
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown("---")
            # Group equivalent clusterings (same rep + k, different seeds) and
            # use a single representative — the user just wants to compare
            # algorithms, not enumerate seed variants.
            import yaml as _yy
            def _alg_label(e: Dict) -> str:
                try:
                    with open(e["path"] / "manifest.yaml") as _f:
                        mm = _yy.safe_load(_f) or {}
                except Exception:
                    mm = {}
                rep = mm.get("influence_source") or mm.get("slice_representation") or e["rep"]
                layer = (mm.get("rep_kwargs") or {}).get("layer") if isinstance(mm.get("rep_kwargs"), dict) else None
                rep_full = f"{rep}/{layer}" if layer else rep
                return f"{rep_full} · k={e['k']}"
            grouped: Dict[str, Path] = {}
            for e in cands:
                lab = _alg_label(e)
                if lab not in grouped:
                    grouped[lab] = e["path"]
                    e["label"] = lab
            options_sorted = sorted(grouped.keys())
            # Sensible defaults: prefer policy_emb if available, then state, then trak, then infembed
            def _priority(lbl: str) -> int:
                if lbl.startswith("policy_emb"): return 0
                if lbl.startswith("state"): return 1
                if lbl.startswith("trak"): return 2
                if lbl.startswith("infembed"): return 3
                return 4
            default_choices = sorted(options_sorted, key=lambda l: (_priority(l), l))[:4]
            choices = st.multiselect(
                "Algorithm × K",
                options=options_sorted,
                default=default_choices,
                help="Each option is one (representation, K) pair. Seeds are collapsed.",
            )
            ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([2, 2, 3])
            with ctrl_c1:
                cols_per_row = st.radio(
                    "Plots per row", [1, 2, 3], horizontal=True, index=1,
                    help="1 = full-width, 2 = side-by-side (recommended), 3 = thumbnails",
                )
            with ctrl_c2:
                rep_height = st.slider("Plot height", 300, 900, 550, step=50, key="rep_height")
            with ctrl_c3:
                apply_pipe = st.checkbox(
                    "Apply simplification (sticky λ=5 + prune count≥5)", value=True
                )
            # Map each chosen label back to a single representative path.
            chosen_paths = [grouped[lab] for lab in choices if lab in grouped]
            chosen_labels = [lab for lab in choices if lab in grouped]
            for i in range(0, len(chosen_paths), int(cols_per_row)):
                cols = st.columns(int(cols_per_row))
                for j, (lab, pp) in enumerate(zip(
                    chosen_labels[i : i + int(cols_per_row)],
                    chosen_paths[i : i + int(cols_per_row)],
                )):
                    ent = {"rep": lab.split(" · ")[0], "k": lab.split("k=")[1] if "k=" in lab else "?"}
                    with cols[j]:
                        try:
                            L, M, E, mm = _load_clustering(str(pp))
                            L = L.astype(np.int64)
                            n_eps_i = len(set(m.get("rollout_idx", m.get("demo_idx", 0)) for m in M))
                            lvl = mm.get("level", "rollout")
                            new = L.copy()
                            if apply_pipe and E is not None:
                                try:
                                    new = gs.sticky_decoder(new, E, M, lambda_stick=5.0, level=lvl)
                                except Exception:
                                    pass
                            g = BehaviorGraph.from_cluster_assignments(new, M, level=lvl)
                            n_raw_edges = sum(len(t) for t in g.transition_counts.values())
                            if apply_pipe:
                                g = gs.prune_edges_by_count(g, min_count=5)
                            nn = len([n for n in g.nodes if n >= 0])
                            ne = sum(len(t) for t in g.transition_counts.values())
                            pos = gs.temporal_layout(g, new, M, level=lvl) if E is not None else None
                            title_pretty = f"{ent['rep']} k={ent['k']}" + (" (smoothed)" if apply_pipe else " (raw)")
                            st.markdown(
                                f"##### {title_pretty}  ·  **{nn} nodes / {ne} edges**"
                                + (f"  _(raw: {n_raw_edges})_" if apply_pipe else "")
                            )
                            fig = create_behavior_graph_plot(
                                g, min_probability=min_prob_display, height=int(rep_height),
                                title="", pos=pos,
                            )
                            st.plotly_chart(fig, use_container_width=True, key=f"rep_{pp.name}")
                            st.caption(
                                f"{n_eps_i} eps · emb={'yes' if E is not None else 'no'} · `{pp.name}`"
                            )
                        except Exception as e:
                            st.error(f"{ent['label']}: {e}")
