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
# Allow falling back to the main repo too (for development outside the bundle)
_IV_CFG_FALLBACK = Path("/Users/erik/stanford/asl_rotation/policy_doctor/third_party/influence_visualizer/configs")


@st.cache_data(show_spinner=False)
def _list_tasks() -> List[str]:
    tasks: set = set()
    for root in (_IV_CFG, _IV_CFG_FALLBACK):
        if root.is_dir():
            for d in root.iterdir():
                if (d / "clustering").is_dir() and any((d / "clustering").iterdir()):
                    tasks.add(d.name)
    return sorted(tasks)


@st.cache_data(show_spinner=False)
def _clusterings_for_task(task: str) -> List[Path]:
    out: List[Path] = []
    for root in (_IV_CFG, _IV_CFG_FALLBACK):
        clu_dir = root / task / "clustering"
        if clu_dir.is_dir():
            for d in sorted(clu_dir.iterdir()):
                if (d / "cluster_labels.npy").exists():
                    out.append(d)
    # Dedupe by resolved absolute path
    seen, unique = set(), []
    for p in out:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


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


# ── Sidebar: task + cascading clustering picker ──────────────────────────────

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

# Index by (rep, k, w, s, agg)
_INDEX = []
for p in _cands:
    m = _read_manifest(str(p))
    rep = m.get("influence_source") or m.get("slice_representation") or "?"
    layer = (m.get("rep_kwargs") or {}).get("layer", "") if isinstance(m.get("rep_kwargs"), dict) else ""
    rep_full = f"{rep}/{layer}" if layer else rep
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


_DEFAULT = {"rep": "policy_emb/bottleneck_plan_t0", "k": 5}

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


reps = sorted({e["rep"] for e in _INDEX})
rep_pick = _pick("Embedding", reps, "demo_rep", default=_DEFAULT["rep"])
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
    "transport_mh_jan28": 192,
    "square_mh_feb5":     192,
    "lift_mh_jan26":      192,
}
_task_pretty = task.split("_")
_task_display = " ".join(w.capitalize() if w not in ("mh",) else w.upper() for w in _task_pretty[:-1])
_n_rollouts_meta = len(set(m.get("rollout_idx", m.get("demo_idx", 0)) for m in meta))
_n_rollouts = len(_MP4_INDEX.get("episodes", [])) or _n_rollouts_meta
_n_success = sum(1 for ep in _MP4_INDEX.get("episodes", []) if ep.get("success") is True)
_success_rate = (_n_success / _n_rollouts) if _n_rollouts else 0.0

_m1, _m2, _m3, _m4 = st.columns(4)
_m1.metric("Task", _task_display or task)
_m2.metric("Demos used", _TASK_DEMOS.get(task, "—"),
           help="Demos in the policy's training set (64% of the 300-demo "
                "MH dataset; the remaining 36% is held out for attribution).")
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
    color_by = st.selectbox(
        "Color nodes by",
        options=color_opts,
        format_func=lambda v: color_labels[v],
        index=0,
    )

n_total_eps = len(set(m.get("rollout_idx", m.get("demo_idx", 0)) for m in meta))
min_branch = st.slider(
    "Hide transitions where count(s, s′) < N",
    1, 50, 2,
    help=(
        "count(s, s′) is the number of rollouts in which the transition s → s′ "
        "was observed. Edges with fewer than N observations are hidden, and "
        "any nodes that become unreachable from START are pruned as a "
        "consequence."
    ),
)
max_depth = 500

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
st.caption(
    f"Hidden {_n_edges_hidden} / {_n_edges_total} edges  ·  "
    f"{_n_nodes_pruned} / {_n_nodes_total} nodes pruned (unreachable from START)"
)


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
    if color_by == "value" and node_values:
        def _div(t: float) -> str:
            t = max(0.0, min(1.0, t))
            return f"rgb({int(214+(44-214)*t)},{int(39+(160-39)*t)},{int(40+(44-40)*t)})"
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
    )
