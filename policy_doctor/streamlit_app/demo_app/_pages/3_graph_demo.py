"""Interactive graph-testing playground.

Lets the user pick a task, drill into a clustering (representation × K ×
window × stride × aggregation), choose a visualization (native-SVG tree
default, plus sunburst / icicle / treemap / BFS Markov / temporal Markov),
and interactively explore nodes and edges with the existing click-to-
explore panel.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

# Re-root sys.path so policy_doctor resolves to this bundle, not a stale
# editable install.
_WORKTREE = Path(__file__).resolve().parents[4]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))
for _m in [k for k in list(sys.modules.keys()) if k.startswith("policy_doctor")]:
    _file = getattr(sys.modules.get(_m), "__file__", None) or ""
    if _file and str(_WORKTREE) not in _file:
        del sys.modules[_m]

from policy_doctor.behaviors.behavior_graph import (
    BehaviorGraph,
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
)
from policy_doctor.behaviors import graph_simplification as gs
from policy_doctor.streamlit_app.components.trajectory_tree_view import (
    render_trajectory_tree,
)
from policy_doctor.streamlit_app.user_study.graph_explorer import (
    render_graph_full_width,
)
from policy_doctor.streamlit_app.user_study.graph_plot import (
    compute_pruned_graph_nodes,
    render_graph_component,
)

st.set_page_config(page_title="Graph Demo — Policy Doctor", layout="wide")
st.title("Graph Demo")
st.caption(
    "Pick a task, choose a clustering, and explore the behavior graph or "
    "trajectory tree. All nodes and edges are clickable."
)


# ── Locate clusterings ────────────────────────────────────────────────────────

_REPO = _WORKTREE
_IV_CFG = _REPO / "third_party" / "influence_visualizer" / "configs"


@st.cache_data(show_spinner=False)
def _list_tasks() -> List[str]:
    tasks: set = set()
    if _IV_CFG.is_dir():
        for d in _IV_CFG.iterdir():
            if (d / "clustering").is_dir() and any((d / "clustering").iterdir()):
                tasks.add(d.name)
    return sorted(tasks)


@st.cache_data(show_spinner=False)
def _clusterings_for_task(task: str) -> List[Path]:
    out: List[Path] = []
    clu_dir = _IV_CFG / task / "clustering"
    if clu_dir.is_dir():
        for d in sorted(clu_dir.iterdir()):
            if (d / "cluster_labels.npy").exists():
                out.append(d)
    return out


@st.cache_data(show_spinner=False)
def _read_manifest(path_str: str) -> Dict:
    import yaml
    try:
        with open(Path(path_str) / "manifest.yaml") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def _load_clustering(path_str: str) -> Tuple[np.ndarray, List[Dict], Optional[np.ndarray], Dict]:
    p = Path(path_str)
    labels = np.load(p / "cluster_labels.npy").astype(np.int64)
    with open(p / "metadata.json") as f:
        meta = json.load(f)
    emb_path = p / "embeddings_reduced.npy"
    emb = np.load(emb_path) if emb_path.exists() else None
    return labels, meta, emb, _read_manifest(path_str)


# ── Sidebar: appearance + task + cascading clustering picker ────────────────

st.sidebar.header("Appearance")
light_mode = st.sidebar.toggle(
    "Light mode",
    value=st.session_state.get("light_mode", False),
    key="light_mode",
)
colorblind_mode = st.sidebar.toggle(
    "Colorblind",
    value=st.session_state.get("colorblind_mode", False),
    key="colorblind_mode",
)
if light_mode:
    # Streamlit's runtime theme is set in config.toml; we override the
    # surfaces that need to look right on white. The palette is
    # deliberately muted (warm grays) so input chrome doesn't punch out.
    #
    # Variables (kept here so tuning is in one place):
    #   --text:    body / heading copy
    #   --muted:   captions, helper text
    #   --surface: input/button background
    #   --border:  input/button border
    st.markdown(
        """
        <style>
          :root {
            --pd-text:    #2b2b2b;
            --pd-muted:   #6b6b6b;
            --pd-surface: #f4f4f5;
            --pd-border:  #d4d4d8;
          }
          /* Main canvas + sidebar */
          [data-testid="stAppViewContainer"], [data-testid="stHeader"],
          [data-testid="stSidebar"], section.main, .stApp, .block-container,
          body { background: #ffffff !important; color: var(--pd-text) !important; }
          /* Text + headers in the content area */
          .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
          .stApp p, .stApp label, .stApp .stMarkdown,
          [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
            color: var(--pd-text) !important;
          }
          /* Captions sit slightly dimmer */
          .stApp [data-testid="stCaptionContainer"], .stApp small,
          .stApp [class*="caption"] { color: var(--pd-muted) !important; }
          /* Dividers + chip-style buttons need light borders, not dark */
          hr, [data-testid="stDivider"] { border-color: var(--pd-border) !important; }
          /* Sidebar nav: page links are very faint on white by default. */
          [data-testid="stSidebarNav"] a, [data-testid="stSidebarNav"] span,
          [data-testid="stSidebarNavLink"], [data-testid="stSidebarNavLink"] * {
            color: var(--pd-text) !important;
          }

          /* ── Inputs ───────────────────────────────────────────────────
             By default Streamlit ships its dark theme through to widgets
             even after we flip the page bg, so dropdowns/buttons end up
             nearly black on the new white canvas. Force them onto the
             muted surface palette. */
          /* Selectbox (BaseWeb Select) */
          .stApp [data-baseweb="select"] > div {
            background-color: var(--pd-surface) !important;
            border-color: var(--pd-border) !important;
            color: var(--pd-text) !important;
          }
          .stApp [data-baseweb="select"] svg { fill: var(--pd-muted) !important; }
          /* Native popover for selectbox options */
          [data-baseweb="popover"] [role="listbox"],
          [data-baseweb="popover"] [role="option"] {
            background-color: #ffffff !important;
            color: var(--pd-text) !important;
          }
          [data-baseweb="popover"] [role="option"]:hover {
            background-color: var(--pd-surface) !important;
          }
          /* Text + number inputs */
          .stApp input[type="text"], .stApp input[type="number"],
          .stApp textarea {
            background-color: var(--pd-surface) !important;
            border-color: var(--pd-border) !important;
            color: var(--pd-text) !important;
          }
          /* Buttons — secondary uses the muted surface, primary keeps
             Streamlit's red accent so the active "highlight path"
             button is visibly differentiated from the inactive ones.
             stDownloadButton needs the same treatment (it renders as a
             separate testid and the default rule above misses it). */
          .stApp .stButton > button[kind="secondary"],
          .stApp .stButton > button:not([kind]),
          .stApp .stDownloadButton > button,
          .stApp [data-testid="stDownloadButton"] button {
            background-color: var(--pd-surface) !important;
            border: 1px solid var(--pd-border) !important;
            color: var(--pd-text) !important;
          }
          .stApp .stButton > button[kind="secondary"]:hover,
          .stApp .stButton > button:not([kind]):hover,
          .stApp .stDownloadButton > button:hover,
          .stApp [data-testid="stDownloadButton"] button:hover {
            background-color: #ebebed !important;
            border-color: #b4b4b8 !important;
          }
          /* Disabled state: lighter surface, dimmer text. */
          .stApp .stDownloadButton > button:disabled,
          .stApp [data-testid="stDownloadButton"] button:disabled {
            background-color: #fafafa !important;
            color: var(--pd-muted) !important;
            border-color: #e4e4e7 !important;
          }
          .stApp .stButton > button[kind="primary"] {
            background-color: #ff4b4b !important;
            border: 1px solid #ff4b4b !important;
            color: #ffffff !important;
          }
          .stApp .stButton > button[kind="primary"]:hover {
            background-color: #ff2b2b !important;
            border-color: #ff2b2b !important;
          }
          /* Sliders: leave the red track, tone down the thumb halo and the
             min/max labels. */
          .stApp [data-baseweb="slider"] [role="slider"] {
            border-color: var(--pd-border) !important;
          }
          /* Expander frame + body. Streamlit's stExpanderDetails container
             keeps the default dark fill even after our top-level bg flip,
             so we override every surface it draws on. */
          .stApp [data-testid="stExpander"],
          .stApp [data-testid="stExpander"] details,
          .stApp [data-testid="stExpander"] details > summary,
          .stApp [data-testid="stExpanderDetails"],
          .stApp [data-testid="stExpanderContent"] {
            background-color: #ffffff !important;
            color: var(--pd-text) !important;
            border-color: var(--pd-border) !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
_theme = "light" if light_mode else "dark"

st.sidebar.header("Task")
_tasks = _list_tasks()
if not _tasks:
    st.error("No clusterings found.")
    st.stop()
default_task_idx = _tasks.index("transport_mh_jan28") if "transport_mh_jan28" in _tasks else 0
task = st.sidebar.selectbox("Task", _tasks, index=default_task_idx)

# When the task changes, clear stale per-component state (selected node,
# selected edge, last-click dedupe, render token) so the next click in the
# new graph isn't silently deduped against the previous task's last click.
_prev_task = st.session_state.get("_demo_prev_task")
if _prev_task != task:
    st.session_state["_demo_prev_task"] = task
    for prefix in ("demo_markov_graph", "demo_tree"):
        for suffix in ("_selected", "_selected_edge", "_last_click", "_render_token"):
            st.session_state.pop(prefix + suffix, None)

st.sidebar.header("Clustering")
_cands = _clusterings_for_task(task)
if not _cands:
    st.error(f"No clusterings under {task}.")
    st.stop()

# Only show representations from the official sweep. The directory name
# already encodes the rep + obs/action strategy unambiguously, so derive
# the dropdown label from the slug (strip the trailing _w{W}_s{S} and
# _seed0_kmeans_k{K} suffixes).
import re as _re
_OFFICIAL_REPS = {
    "infembed",
    "policy_emb",
    "trak",
    "policy_emb_bottleneck_plan_t0",
    "state_full_history",
    "state_action_full_history_full_plan",
}

def _rep_from_slug(slug: str) -> str:
    s = _re.sub(r"_seed\d+_kmeans_k\d+$", "", slug)
    s = _re.sub(r"_w\d+_s\d+$", "", s)
    return s

# Index by (rep, k, w, s, agg)
_INDEX = []
for p in _cands:
    rep_full = _rep_from_slug(p.name)
    if rep_full not in _OFFICIAL_REPS:
        continue
    m = _read_manifest(str(p))
    _INDEX.append({
        "path": p,
        "rep": rep_full,
        "k": int(m.get("n_clusters", 0) or 0),
        "w": int(m.get("window_width", 5) or 5),
        "s": int(m.get("stride", 2) or 2),
        "agg": str(m.get("aggregation", "sum") or "sum"),
    })

def _filter(rep=None, k=None, w=None, s=None, agg=None):
    out = _INDEX
    if rep is not None: out = [e for e in out if e["rep"] == rep]
    if k is not None: out = [e for e in out if e["k"] == k]
    if w is not None: out = [e for e in out if e["w"] == w]
    if s is not None: out = [e for e in out if e["s"] == s]
    if agg is not None: out = [e for e in out if e["agg"] == agg]
    return out


_DEFAULT = {"rep": "policy_emb_bottleneck_plan_t0", "k": 8}

def _pick(label, options, key, default=None):
    if not options:
        st.sidebar.warning(f"No {label} available.")
        return None
    prev = st.session_state.get(key)
    if prev in options:
        idx = options.index(prev)
    elif default in options:
        idx = options.index(default)
    else:
        idx = 0
    return st.sidebar.selectbox(label, options=options, index=idx, key=key)


_REP_DESCRIPTIONS = {
    "infembed":
        "**InfEmbed** — influence-based representation computed via a "
        "low-rank factorization of the Hessian.",
    "policy_emb":
        "**Policy embedding (bottleneck)** — diffusion U-Net bottleneck "
        "(mid-block) activations extracted at the final denoising step, "
        "aggregated over a sliding window and UMAP-reduced to 50D.",
    "trak":
        "**TRAK** — per-timestep features from the TRAK influence-score "
        "matrix (rollout × training demos), reduced to 200D with truncated "
        "SVD before windowing.",
    "policy_emb_bottleneck_plan_t0":
        "**Policy embedding (bottleneck, plan t=0)** — diffusion U-Net "
        "bottleneck (mid-block) activations at the final denoising step.",
    "state_full_history":
        "**State (full history)** — full observation history, flattened and "
        "UMAP-reduced.",
    "state_action_full_history_full_plan":
        "**State_action** — concatenation of the full observation history "
        "and the full action chunk, flattened and UMAP-reduced.",
}

reps = sorted({e["rep"] for e in _INDEX})
rep_pick = _pick("Embedding", reps, "demo_rep", default=_DEFAULT["rep"])
if rep_pick in _REP_DESCRIPTIONS:
    st.sidebar.caption(_REP_DESCRIPTIONS[rep_pick])
filt = _filter(rep=rep_pick)
ks = sorted({e["k"] for e in filt})
k_pick = _pick("K (clusters)", ks, "demo_k", default=_DEFAULT["k"])
filt = _filter(rep=rep_pick, k=k_pick)
ws = sorted({e["w"] for e in filt})
w_pick = _pick("W (window width)", ws, "demo_w")
filt = _filter(rep=rep_pick, k=k_pick, w=w_pick)
ss_ = sorted({e["s"] for e in filt})
s_pick = _pick("S (stride)", ss_, "demo_s")
filt = _filter(rep=rep_pick, k=k_pick, w=w_pick, s=s_pick)
# Aggregation is always "mean" in the bundled clusterings; the dropdown
# would be a single-option no-op. If you generate variants with other
# aggregations, re-expose the picker here.
aggs = sorted({e["agg"] for e in filt})
agg_pick = aggs[0] if aggs else None
filt = _filter(rep=rep_pick, k=k_pick, w=w_pick, s=s_pick, agg=agg_pick)
if not filt:
    st.error("No clustering matches the chosen combination.")
    st.stop()
clu_path = filt[0]["path"]

labels, meta, emb, manifest = _load_clustering(str(clu_path))
level = manifest.get("level", "rollout")

# When the clustering changes, clear stale node/edge selection and bump the
# render token so the Markov graph iframe fully re-renders with new data.
_prev_clu = st.session_state.get("_demo_prev_clu")
if _prev_clu != str(clu_path):
    st.session_state["_demo_prev_clu"] = str(clu_path)
    for _sfx in ("_selected", "_selected_edge", "_last_click"):
        st.session_state.pop(f"demo_markov_graph{_sfx}", None)
    _rt = "demo_markov_graph_render_token"
    st.session_state[_rt] = st.session_state.get(_rt, 0) + 1

# MP4 resolution is hard-coded — the bundle puts videos in a known
# location. The metadata / debug block below the dropdowns was dev
# scaffolding, not useful for the demo.
_MP4_ROOT = Path("/tmp/study_mp4s")
_MP4_DIR = _MP4_ROOT / task if (_MP4_ROOT / task / "index.json").exists() else None
_MP4_INDEX = json.load(open(_MP4_DIR / "index.json")) if _MP4_DIR else {"episodes": []}


# ── Header info row ──────────────────────────────────────────────────────────
# Demos actually used to train the policy. The robomimic mh datasets
# ship with 300 demos each, but cupid's training config sets
# dataset_mask_kwargs.train_ratio=0.64 (uniform_quality=True), so 192
# demos go into training and 108 are held out for attribution / OOD.
_TASK_DEMOS = {
    "transport_mh_jan28":   192,
    "square_mh_feb5":       192,
    "lift_mh_jan26":        192,
    # pi0.5-LIBERO tasks — demos come from the full LIBERO benchmark (10 tasks × 50 demos each)
    "pi05_libero_spatial":  500,
    "pi05_libero_object":   500,
    "pi05_libero_goal":     500,
}
_task_pretty = task.split("_")
_task_display = " ".join(w.capitalize() if w not in ("mh",) else w.upper() for w in _task_pretty[:-1])
_n_rollouts_meta = len(set(m.get("rollout_idx", m.get("demo_idx", 0)) for m in meta))
_n_rollouts = len(_MP4_INDEX.get("episodes", [])) or _n_rollouts_meta
_n_success = sum(1 for ep in _MP4_INDEX.get("episodes", []) if ep.get("success") is True)
if _n_rollouts and not _n_success:
    # Fall back to success info embedded in clustering metadata (works for all tasks,
    # including pi05 which doesn't have an MP4 index).
    _succ_rollouts = {m.get("rollout_idx") for m in meta if m.get("success") is True}
    _all_rollouts  = {m.get("rollout_idx") for m in meta if m.get("rollout_idx") is not None}
    _n_success = len(_succ_rollouts)
    if _n_rollouts == 0:
        _n_rollouts = len(_all_rollouts)
_success_rate = (_n_success / _n_rollouts) if _n_rollouts else 0.0

_m1, _m2, _m3, _m4 = st.columns(4)
_m1.metric("Task", _task_display or task)
_m2.metric("Demos used", _TASK_DEMOS.get(task, "—"))
_m3.metric("Rollouts", _n_rollouts)
_m4.metric("Success rate", f"{_success_rate:.0%}" if _n_rollouts else "—")
st.divider()


# ── Main column: viz controls + render ───────────────────────────────────────

VIZ_OPTIONS = [
    "tree_native_svg",
    "tree_sunburst",
    "tree_icicle",
    "markov_svg_bfs",
    "markov_svg_temporal",
]
VIZ_LABELS = {
    "tree_native_svg":     "Trajectory tree",
    "tree_sunburst":       "Sunburst",
    "tree_icicle":         "Icicle",
    "markov_svg_bfs":      "Markov graph — BFS-layered",
    "markov_svg_temporal": "Markov graph — temporal mean",
}

c_viz, c_color = st.columns([2, 1])
with c_viz:
    viz_type = st.selectbox(
        "Visualization",
        options=VIZ_OPTIONS,
        format_func=lambda v: VIZ_LABELS[v],
        index=0,
    )
is_tree = viz_type.startswith("tree_")
with c_color:
    if is_tree:
        color_opts = ["outcome", "id", "value"]
        color_labels = {
            "outcome": "Outcome (success rate)",
            "id": "Cluster ID (palette)",
            "value": "Value V(s) (Bellman)",
        }
    else:
        color_opts = ["id", "value", "timesteps"]
        color_labels = {
            "id": "Cluster ID (palette)",
            "value": "Value V(s) (Bellman)",
            "timesteps": "Timestep count (viridis)",
        }
    # For the Markov views, the value-based divergent red→green colouring
    # is the most informative default (and what the paper figures use).
    # For trees we keep the cluster-id palette as the entry point.
    _default_color_idx = 0 if is_tree else color_opts.index("value")
    color_by = st.selectbox(
        "Color nodes by",
        options=color_opts,
        format_func=lambda v: color_labels[v],
        index=_default_color_idx,
    )

n_total_eps = len(set(m.get("rollout_idx", m.get("demo_idx", 0)) for m in meta))
_default_min_branch = max(2, int(n_total_eps * 0.02))  # ~2% of rollouts
min_branch = st.slider(
    "Hide transitions where count(s, s′) < N",
    1, max(50, _default_min_branch + 10), _default_min_branch,
    help=(
        "count(s, s′) is the number of rollouts in which the transition s → s′ "
        "was observed. Edges with fewer than N observations are hidden, and "
        "any nodes that become unreachable from START are pruned as a "
        "consequence."
    ),
)
max_depth = 500

# ── Advanced viz settings ────────────────────────────────────────────────────
# Hidden by default — these knobs tune the encoding (width ↔ probability,
# radius ↔ visit count) and let users export the graph as paper-ready SVG.
with st.expander("Advanced viz settings", expanded=False):
    _c_style, _c_w, _c_r = st.columns(3)
    with _c_style:
        edge_style = st.radio(
            "Edges",
            options=["arrows", "lines"],
            index=1,
            horizontal=True,
            help=(
                "Default is ‘lines’: width + grey level already encode "
                "transition probability cleanly. Switch to ‘arrows’ if "
                "you need explicit direction-of-motion."
            ),
        )
    with _c_w:
        edge_width_slope = st.slider(
            "Edge width per probability (px)",
            min_value=0.0, max_value=12.0, value=5.0, step=0.5,
            help=(
                "Linear: width = 0.6 + slope × p. The slope is the extra "
                "pixels added to the stroke at p = 1. Grey opacity also "
                "tracks p linearly from 0.2 to 1.0."
            ),
        )
    with _c_r:
        node_size_slope = st.slider(
            "Node radius per visit fraction (px)",
            min_value=0.0, max_value=36.0, value=24.0, step=1.0,
            help=(
                "Linear: radius = 6 + slope × (visits / max-visits). The "
                "slope is the extra pixels added to the radius at the "
                "most-visited node."
            ),
        )

    # SVG export. The iframe auto-publishes a fresh snapshot to Python on
    # every render (deduped by content hash), so this download_button is
    # always armed with the latest SVG — one click, native browser
    # download. (anchor.click() driven from inside the iframe isn't a real
    # user gesture, so Chrome drops it; the parent-window button works.)
    _captured = st.session_state.get("captured_svg", "")
    st.download_button(
        "⬇ Export SVG",
        data=(_captured or "").encode("utf-8"),
        file_name="behavior_graph.svg",
        mime="image/svg+xml",
        disabled=not _captured,
        help="Saves the current graph state (no background) as behavior_graph.svg.",
    )

# Build the underlying Markov graph (used for value computation + Markov views).
@st.cache_data(show_spinner=False)
def _graph(_labels_bytes: bytes, _meta_id: int, level: str) -> BehaviorGraph:
    arr = np.frombuffer(_labels_bytes, dtype=np.int64)
    return BehaviorGraph.from_cluster_assignments(arr, meta, level=level)

bg = _graph(labels.tobytes(), id(meta), level)
node_values: Dict[int, float] = {}
if color_by == "value":
    try:
        node_values = bg.compute_values()
    except Exception as e:
        st.warning(f"compute_values() failed ({e}); falling back to ID coloring.")
        color_by = "id"

# ── Filter summary: how many nodes / edges does the threshold drop ───────────
_n_nodes_total = len(bg.nodes)
_n_edges_total = sum(len(t) for t in bg.transition_counts.values())
_excluded = compute_pruned_graph_nodes(
    bg, min_visit_prob=0.0, n_total=bg.num_episodes,
    min_edge_count=int(min_branch),
)
_n_edges_hidden = sum(
    1
    for src, targets in bg.transition_counts.items()
    for tgt, cnt in targets.items()
    if src in _excluded or tgt in _excluded or cnt < int(min_branch)
)
_n_nodes_pruned = len(_excluded)
# Edge-count threshold is meaningful for the Markov views; the tree view
# uses min_branch to filter nodes directly and renders its own pruning
# summary inside render_trajectory_tree.
if not is_tree:
    st.caption(f"{_n_edges_hidden} edges hidden, {_n_nodes_pruned} nodes pruned.")


# ── Dispatch ─────────────────────────────────────────────────────────────────

if is_tree:
    render_trajectory_tree(
        labels=labels,
        metadata=meta,
        view_mode=viz_type.replace("tree_", ""),
        min_branch=int(min_branch),
        max_depth_cap=int(max_depth),
        color_mode=color_by,
        node_values=node_values,
        cluster_names=None,
        mp4_dir=_MP4_DIR,
        mp4_index=_MP4_INDEX,
        height=600,
        level=level,
        key_prefix="demo_tree",
        theme=_theme,
        edge_style=edge_style,
        edge_width_slope=float(edge_width_slope),
        node_size_slope=float(node_size_slope),
    )
else:
    # Markov SVG (BFS or temporal layout)
    excluded = compute_pruned_graph_nodes(
        bg, min_visit_prob=0.0, n_total=bg.num_episodes,
        min_edge_count=int(min_branch),
    )
    pos = None
    if viz_type == "markov_svg_temporal":
        try:
            pos = gs.temporal_layout(bg, labels, meta, level=level)
        except Exception as e:
            st.warning(f"Temporal layout failed ({e}); falling back to BFS-layered.")
    color_override: Optional[Dict[int, str]] = None
    _SPECIAL = {SUCCESS_NODE_ID, FAILURE_NODE_ID, START_NODE_ID, END_NODE_ID}
    _OUTCOME_BINS = (
        # Okabe-Ito diverging: vermilion → orange → yellow → teal → blue
        ["#D55E00", "#E69F00", "#F0E442", "#009E73", "#0072B2"]
        if colorblind_mode else
        ["#d62728", "#ff7f0e", "#e8c32a", "#9dc95d", "#2ca02c"]
    )
    def _div(t: float) -> str:
        return _OUTCOME_BINS[min(4, int(max(0.0, min(1.0, t)) * 5))]
    _SUCCESS_COL = "#0072B2" if colorblind_mode else "#2ca02c"
    _FAILURE_COL = "#D55E00" if colorblind_mode else "#d62728"
    # Always override terminal node colors so colorblind mode applies everywhere
    color_override = {
        nid: (_SUCCESS_COL if node.cluster_id == SUCCESS_NODE_ID else _FAILURE_COL)
        for nid, node in bg.nodes.items()
        if node.cluster_id in (SUCCESS_NODE_ID, FAILURE_NODE_ID)
    }
    if color_by == "value" and node_values:
        non_term = [v for cid, v in node_values.items() if cid not in _SPECIAL]
        vr = max((abs(v) for v in non_term), default=1.0) or 1.0
        color_override = {
            nid: _div(0.5 + node_values.get(nid, 0.0) / (2 * vr))
            for nid in bg.nodes if nid not in _SPECIAL
        }
    elif color_by == "timesteps":
        try:
            import plotly.express as _px
            viridis = _px.colors.sequential.Viridis
        except Exception:
            viridis = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]
        ts_max = max(
            (np.log1p(bg.nodes[nid].num_timesteps) for nid in bg.nodes if nid not in _SPECIAL),
            default=1.0,
        ) or 1.0
        color_override = {}
        for nid in bg.nodes:
            if nid in _SPECIAL: continue
            t = np.log1p(bg.nodes[nid].num_timesteps) / ts_max
            idx = int(min(len(viridis) - 1, max(0, t * (len(viridis) - 1))))
            color_override[nid] = viridis[idx]

    render_graph_full_width(
        graph=bg,
        labels=labels,
        metadata=meta,
        mp4_dir=_MP4_DIR if _MP4_DIR is not None else Path("/tmp/_nonexistent"),
        mp4_index=_MP4_INDEX,
        key_prefix="demo_markov",
        min_edge_count=int(min_branch),
        pos=pos,
        excluded_node_ids=excluded,
        color_override=color_override,
        theme=_theme,
        edge_style=edge_style,
        edge_width_slope=float(edge_width_slope),
        node_size_slope=float(node_size_slope),
    )
