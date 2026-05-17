"""Standalone Streamlit demo for behavior-graph simplification methods.

Uses the canonical graph viz from the policy_doctor graph demo
(``render_graph_full_width`` / ``render_trajectory_tree``) with the same
viz-type / color-mode / edge-style controls.

Run from the worktree root:
    streamlit run policy_doctor/streamlit_app/simplification_demo/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

_WORKTREE = Path(__file__).resolve().parents[3]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from policy_doctor.behaviors.behavior_graph import (  # noqa: E402
    BehaviorGraph,
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
)
from policy_doctor.behaviors import graph_simplification as gs  # noqa: E402
from policy_doctor.behaviors.simplification import (  # noqa: E402
    LEVER_GRIDS,
    METHOD_DESCRIPTIONS,
    METHOD_LEVER_LABELS,
    METHODS,
    run_method,
)
from policy_doctor.streamlit_app.components.trajectory_tree_view import (  # noqa: E402
    render_trajectory_tree,
)
from policy_doctor.streamlit_app.user_study.graph_explorer import (  # noqa: E402
    render_graph_full_width,
)
from policy_doctor.streamlit_app.user_study.graph_plot import (  # noqa: E402
    compute_pruned_graph_nodes,
)

_DATA_ROOTS = [
    _WORKTREE / "third_party" / "influence_visualizer" / "configs",
    Path(
        "/Users/erik/stanford/asl_rotation/policy_doctor/.claude/worktrees/"
        "graph-simplification/third_party/influence_visualizer/configs"
    ),
    Path(
        "/Users/erik/stanford/asl_rotation/policy_doctor/third_party/"
        "influence_visualizer/configs"
    ),
]
_RESULTS_DIR = _WORKTREE / "docs" / "simplification_results"
_MP4_ROOT = Path("/tmp/study_mp4s")


@st.cache_data(show_spinner=False)
def _load_mp4_index(task: str) -> Tuple[Path, Dict]:
    """Return (mp4_dir, mp4_index) for the canonical /tmp/study_mp4s bundle.

    Falls back to ``(/tmp/_nonexistent, {"episodes": []})`` if no index is
    found — in that case the graph_explorer renders its standard "no videos
    found" notice when a node is clicked.
    """
    cand = _MP4_ROOT / task
    idx_path = cand / "index.json"
    if idx_path.exists():
        try:
            with idx_path.open() as f:
                return cand, json.load(f)
        except Exception:
            pass
    return Path("/tmp/_simplification_demo_no_mp4"), {"episodes": []}


# ---------------------------------------------------------------------------
# Data discovery (mirrors graph_demo)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def list_tasks() -> List[str]:
    tasks: set = set()
    for root in _DATA_ROOTS:
        if root.is_dir():
            for d in root.iterdir():
                clu = d / "clustering"
                if clu.is_dir() and any(clu.iterdir()):
                    tasks.add(d.name)
    return sorted(tasks)


@st.cache_data(show_spinner=False)
def list_clusterings(task: str) -> List[Path]:
    out: List[Path] = []
    for root in _DATA_ROOTS:
        clu = root / task / "clustering"
        if clu.is_dir():
            for d in sorted(clu.iterdir()):
                if (d / "cluster_labels.npy").exists():
                    out.append(d)
    seen, unique = set(), []
    for p in out:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


@st.cache_data(show_spinner=False)
def load_clustering(path_str: str) -> Tuple[np.ndarray, List[Dict], Dict]:
    p = Path(path_str)
    labels = np.load(p / "cluster_labels.npy").astype(np.int64)
    with (p / "metadata.json").open() as f:
        meta = json.load(f)
    manifest = {}
    mf = p / "manifest.yaml"
    if mf.exists():
        try:
            import yaml
            with mf.open() as f:
                manifest = yaml.safe_load(f) or {}
        except Exception:
            manifest = {}
    return labels, meta, manifest


@st.cache_data(show_spinner=False)
def k_sweep_baseline(task: str, family_prefix: str) -> List[Dict]:
    """Compute passthrough metrics for every clustering matching family_prefix.

    family_prefix is the slug stripped of `_k{K}` (so e.g.
    "policy_emb_bottleneck_plan_t0_w5_s1_seed0_kmeans" matches all K).
    Returns a list of dicts with the same shape as a FrontierPoint dict.
    """
    import re
    from policy_doctor.behaviors.simplification.metrics import compute_metrics

    out: List[Dict] = []
    for c in list_clusterings(task):
        name = c.name
        m = re.match(rf"^{re.escape(family_prefix)}_k(\d+)$", name)
        if not m:
            continue
        k = int(m.group(1))
        try:
            labels, meta, manifest = load_clustering(str(c))
        except Exception:
            continue
        level = manifest.get("level", "rollout")
        g = BehaviorGraph.from_cluster_assignments(labels, meta, level=level)
        metrics = compute_metrics(
            g, labels, meta,
            original_labels=labels, node_mapping={},
        )
        d = metrics.as_dict()
        d.update({
            "method": "passthrough_k_sweep",
            "lever": float(k),
            "k": k,
            "heldout_nll_per_original_bits": None,
        })
        out.append(d)
    out.sort(key=lambda p: p["n_nodes"])
    return out


@st.cache_data(show_spinner=False)
def read_benchmark(task: str, clustering_name: str) -> Optional[Dict]:
    path = _RESULTS_DIR / f"{task}__{clustering_name}.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _run_method_cached(
    method: str, clust_dir_str: str, lever: float,
) -> Tuple[np.ndarray, Dict, Dict[int, int]]:
    """Run a method, return (new_labels, metrics_dict, mapping)."""
    labels, meta, manifest = load_clustering(clust_dir_str)
    level = manifest.get("level", "rollout")
    g = BehaviorGraph.from_cluster_assignments(labels, meta, level=level)
    res = run_method(method, g, labels, meta, lever=lever)
    return res.new_labels, res.metrics.as_dict(), res.node_mapping


# ---------------------------------------------------------------------------
# Pareto plot (Plotly, dark theme)
# ---------------------------------------------------------------------------

def render_pareto_figure(
    points_by_method: Dict[str, List[Dict]],
    selected_method: Optional[str] = None,
    selected_lever: Optional[float] = None,
    x_field: str = "n_nodes",
    y_field: str = "markov_violation_bits",
    y_label: str = "Markov violation (bits)",
    k_sweep_points: Optional[List[Dict]] = None,
):
    import plotly.graph_objects as go

    traces = []
    palette = [
        "#60a5fa", "#ef4444", "#10b981", "#f59e0b",
        "#a855f7", "#06b6d4", "#84cc16", "#f97316",
        "#ec4899", "#14b8a6", "#6366f1",
    ]

    # Capture the passthrough point for the baseline reference line.
    baseline_y: Optional[float] = None
    baseline_x: Optional[float] = None
    pp = points_by_method.get("passthrough", [])
    for p in pp:
        if "error" in p: continue
        if p.get(x_field) is not None and p.get(y_field) is not None:
            baseline_y = float(p[y_field])
            baseline_x = float(p[x_field])
            break

    # Map y_field → bootstrap CI field names (if any).
    _CI_FIELDS = {
        "markov_violation_bits": ("markov_ci_lo", "markov_ci_hi"),
        "markov_violation_2nd_bits": ("markov_2nd_ci_lo", "markov_2nd_ci_hi"),
        "heldout_nll_per_original_bits": ("heldout_nll_ci_lo", "heldout_nll_ci_hi"),
    }
    ci_lo_field, ci_hi_field = _CI_FIELDS.get(y_field, (None, None))

    for i, (method, pts) in enumerate(points_by_method.items()):
        pts = [p for p in pts if "error" not in p and p.get(x_field) is not None and p.get(y_field) is not None]
        if not pts:
            continue
        pts_sorted = sorted(pts, key=lambda p: (p[x_field], p[y_field]))
        x = [p[x_field] for p in pts_sorted]
        y = [p[y_field] for p in pts_sorted]
        # Highlight passthrough specifically — it's only one point and easy to miss.
        if method == "passthrough":
            traces.append(go.Scatter(
                x=x, y=y, mode="markers", name="passthrough (baseline)",
                marker=dict(size=18, color="#ffffff", symbol="diamond",
                            line=dict(color="#0f172a", width=2)),
                hovertemplate=f"passthrough (raw)<br>{x_field}=%{{x}}<br>{y_field}=%{{y:.3f}}<extra></extra>",
            ))
            continue
        color = palette[i % len(palette)]
        # Render bootstrap CI band if available and we're on a method we care about
        if ci_lo_field and method == selected_method:
            lo_vals = [p.get(ci_lo_field) for p in pts_sorted]
            hi_vals = [p.get(ci_hi_field) for p in pts_sorted]
            if all(v is not None for v in lo_vals + hi_vals):
                traces.append(go.Scatter(
                    x=x + x[::-1],
                    y=lo_vals + hi_vals[::-1],
                    fill="toself",
                    fillcolor=color,
                    opacity=0.18,
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{method} 95% CI",
                ))
        traces.append(go.Scatter(
            x=x, y=y, mode="lines+markers", name=method,
            line=dict(color=color, width=2 if method == selected_method else 1.2),
            marker=dict(size=8 if method == selected_method else 6, color=color),
            opacity=1.0 if method == selected_method else 0.55,
            hovertemplate=f"{method}<br>{x_field}=%{{x}}<br>{y_field}=%{{y:.3f}}<extra></extra>",
        ))
    if selected_method and selected_lever is not None and selected_method in points_by_method:
        pts = points_by_method[selected_method]
        match = min(pts, key=lambda p: abs(p.get("lever", 0) - selected_lever)) if pts else None
        if match and match.get(x_field) is not None and match.get(y_field) is not None:
            traces.append(go.Scatter(
                x=[match[x_field]], y=[match[y_field]],
                mode="markers", name=f"★ current operating point",
                marker=dict(size=18, color="#fbbf24", symbol="star",
                            line=dict(color="#92400e", width=2)),
                showlegend=True,
            ))
    # K-sweep: passthrough metrics across multiple clustering K values
    # (controlling node count by RE-CLUSTERING rather than by simplification).
    if k_sweep_points:
        ks = [p for p in k_sweep_points
              if p.get(x_field) is not None and p.get(y_field) is not None]
        if ks:
            ks_sorted = sorted(ks, key=lambda p: p[x_field])
            traces.append(go.Scatter(
                x=[p[x_field] for p in ks_sorted],
                y=[p[y_field] for p in ks_sorted],
                mode="lines+markers",
                name="K-sweep (re-cluster baseline)",
                line=dict(color="#ffffff", width=2.5, dash="dot"),
                marker=dict(size=14, color="#ffffff", symbol="diamond-open",
                            line=dict(color="#ffffff", width=2)),
                hovertemplate=(
                    "K-sweep baseline<br>K=%{customdata}<br>"
                    f"{x_field}=%{{x}}<br>{y_field}=%{{y:.3f}}<extra></extra>"
                ),
                customdata=[p.get("k", "") for p in ks_sorted],
            ))

    fig = go.Figure(data=traces)
    if baseline_y is not None:
        fig.add_hline(
            y=baseline_y,
            line=dict(color="#fbbf24", width=1, dash="dash"),
            annotation_text=f"current K baseline = {baseline_y:.3f}",
            annotation_position="top right",
            annotation_font_color="#fbbf24",
        )
    fig.update_layout(
        xaxis_title=x_field,
        yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=10, t=40, b=40),
        height=420,
        template="plotly_dark",
    )
    return fig


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Behavior Graph Simplification — Policy Doctor",
    layout="wide",
)
st.title("Behavior Graph Simplification")
st.caption(
    "Same graph viz as the canonical Graph Demo (trajectory tree / sunburst / "
    "icicle / Markov-BFS / Markov-temporal). Pick a method on the left, slide "
    "its lever, and watch the topology change. See "
    "`docs/simplification_findings.md` for the data-grounded analysis."
)

# ---------------------------------------------------------------------------
# Sidebar: theme + task + clustering + method
# ---------------------------------------------------------------------------

st.sidebar.header("Appearance")
light_mode = st.sidebar.toggle("Light mode", value=False)
theme = "light" if light_mode else "dark"

st.sidebar.header("Data")
tasks = list_tasks()
if not tasks:
    st.error("No clusterings found. See _DATA_ROOTS in this file.")
    st.stop()
bench_default = "transport_mh_jan28" if "transport_mh_jan28" in tasks else tasks[0]
task = st.sidebar.selectbox(
    "Task", tasks, index=tasks.index(bench_default) if bench_default in tasks else 0,
)
clusterings = list_clusterings(task)
if not clusterings:
    st.error(f"No clusterings for {task}.")
    st.stop()

clust_names = [c.name for c in clusterings]
preferred = next(
    (n for n in clust_names if "policy_emb_bottleneck_plan_t0_w5_s1_seed0_kmeans_k15" in n),
    None,
) or next(
    (n for n in clust_names if "policy_emb_bottleneck_plan_t0_w5_s1_seed0_kmeans_k5" in n),
    None,
) or clust_names[0]
clust_name = st.sidebar.selectbox(
    "Clustering", clust_names,
    index=clust_names.index(preferred) if preferred in clust_names else 0,
)
clust_dir = next(c for c in clusterings if c.name == clust_name)

labels_raw, metadata, manifest = load_clustering(str(clust_dir))
level = manifest.get("level", "rollout")
graph_raw = BehaviorGraph.from_cluster_assignments(labels_raw, metadata, level=level)
ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
n_eps = len({m[ep_key] for m in metadata})

_mp4_dir, _mp4_index = _load_mp4_index(task)

st.sidebar.metric("Samples", len(labels_raw))
st.sidebar.metric("Episodes", n_eps)
st.sidebar.metric("Raw cluster nodes", len(graph_raw.cluster_nodes))
if _mp4_index.get("episodes"):
    st.sidebar.caption(f"MP4 index: {len(_mp4_index['episodes'])} videos available")
else:
    st.sidebar.caption(f"MP4 index: none at {_MP4_ROOT}/{task}/index.json")

st.sidebar.divider()
st.sidebar.header("Method")
method_names = sorted(METHODS.keys())
default_method = "hoeffding_merge" if "hoeffding_merge" in method_names else method_names[0]
method = st.sidebar.selectbox(
    "Simplification method", method_names,
    index=method_names.index(default_method),
    help="`passthrough` shows the raw graph. Any other method simplifies it according to its single lever.",
)
st.sidebar.caption(METHOD_DESCRIPTIONS.get(method, ""))

grid = LEVER_GRIDS.get(method, np.array([0.0]))
if method == "pcca_plus":
    grid = np.arange(2, max(3, len(graph_raw.cluster_nodes))).astype(float)
lever_label = METHOD_LEVER_LABELS.get(method, "lever")

if len(grid) > 1:
    lever_value = st.sidebar.select_slider(
        lever_label,
        options=[float(v) for v in grid],
        value=float(grid[len(grid) // 2]),
        format_func=lambda x: f"{x:.4g}",
    )
else:
    lever_value = float(grid[0])
    st.sidebar.caption(f"{lever_label} = {lever_value:.4g}")

# ---------------------------------------------------------------------------
# Run the method
# ---------------------------------------------------------------------------

with st.spinner(f"Running {method}..."):
    new_labels, metrics, mapping = _run_method_cached(
        method, str(clust_dir), float(lever_value),
    )
graph_simp = BehaviorGraph.from_cluster_assignments(new_labels, metadata, level=level)

# ---------------------------------------------------------------------------
# Top metric row (Task / Episodes / Raw nodes / After-method nodes)
# ---------------------------------------------------------------------------

m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
m1.metric("Task", task.split("_")[0].capitalize())
m2.metric("Episodes", n_eps)
m3.metric("Raw nodes", len(graph_raw.cluster_nodes),
          help="↕ Tradeoff axis — no 'better' direction; smaller is simpler, larger is more expressive.")
m4.metric("After-method nodes", metrics["n_nodes"],
          delta=metrics["n_nodes"] - len(graph_raw.cluster_nodes),
          help="↕ Tradeoff axis — pick where on the Pareto frontier you want to land.")
mv1 = metrics["markov_violation_bits"]
mv2 = metrics.get("markov_violation_2nd_bits", 0.0) or 0.0
m5.metric("MV 1st-order (bits) ↓", f"{mv1:.3f}",
          help="↓ Lower is better. I(orig_prev_{t-1}; orig_next_t | merged_curr_t) — "
               "1-step memory thrown away by the abstraction. 0 bits = perfectly Markov.")
m6.metric("MV 2nd-order (bits) ↓", f"{mv2:.3f}",
          delta=f"{(mv2 - mv1):+.3f} vs 1st" if mv2 > 0 else None,
          delta_color="inverse",
          help=(
              "↓ Lower is better. I((prev_{t-1}, prev_{t-2}); next | merged_curr). "
              "When this is noticeably larger than the 1st-order value, the "
              "abstraction is hiding length-2 memory that the 1st-order metric "
              "and the current vomm_split_merge cannot fix. Diagnostic only."
          ))
m7.metric("MDL score (bits) ↓*", f"{metrics['mdl_score']:.1f}",
          help="↓ Lower is *nominally* better. predictive NLL + (k_params / 2) · log₂(N_original). "
               "*BUT* this metric is biased toward aggressive merging (a 1-node "
               "graph trivially wins). Use Markov violation as the primary axis; "
               "MDL is a secondary diagnostic. See findings doc.")

st.divider()

# ---------------------------------------------------------------------------
# Visualization controls (mirror the canonical graph demo)
# ---------------------------------------------------------------------------

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
c_viz, c_color, c_show = st.columns([2, 1, 1])
with c_viz:
    viz_type = st.selectbox(
        "Visualization", options=VIZ_OPTIONS,
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
    _default_color_idx = 0 if is_tree else color_opts.index("value")
    color_by = st.selectbox(
        "Color nodes by", options=color_opts,
        format_func=lambda v: color_labels[v],
        index=_default_color_idx,
    )
with c_show:
    show_raw_too = st.toggle(
        "Show raw graph too", value=False,
        help="Render the raw (unsimplified) graph above the simplified one for side-by-side comparison.",
    )

min_branch = st.slider(
    "Hide transitions where count(s, s′) < N",
    1, 50, 2,
    help=(
        "count(s, s′) is the number of rollouts in which the transition s → s′ "
        "was observed. Edges with fewer than N observations are hidden, and "
        "any nodes that become unreachable from START are pruned as a consequence."
    ),
)

with st.expander("Advanced viz settings", expanded=False):
    _c_style, _c_w, _c_r = st.columns(3)
    with _c_style:
        edge_style = st.radio(
            "Edges", options=["arrows", "lines"], index=1, horizontal=True,
        )
    with _c_w:
        edge_width_slope = st.slider(
            "Edge width per probability (px)",
            min_value=0.0, max_value=12.0, value=5.0, step=0.5,
        )
    with _c_r:
        node_size_slope = st.slider(
            "Node radius per visit fraction (px)",
            min_value=0.0, max_value=36.0, value=24.0, step=1.0,
        )

# ---------------------------------------------------------------------------
# Graph rendering helpers (mirror the dispatch in 3_graph_demo.py)
# ---------------------------------------------------------------------------

def _build_color_override(
    graph: BehaviorGraph, color_by: str,
) -> Optional[Dict[int, str]]:
    if color_by not in ("value", "timesteps"):
        return None
    _SPECIAL = {SUCCESS_NODE_ID, FAILURE_NODE_ID, START_NODE_ID, END_NODE_ID}
    if color_by == "value":
        try:
            node_values = graph.compute_values()
        except Exception:
            return None

        def _div(t: float) -> str:
            t = max(0.0, min(1.0, t))
            return f"rgb({int(214+(44-214)*t)},{int(39+(160-39)*t)},{int(40+(44-40)*t)})"

        non_term = [v for cid, v in node_values.items() if cid not in _SPECIAL]
        vr = max((abs(v) for v in non_term), default=1.0) or 1.0
        return {
            nid: _div(0.5 + node_values.get(nid, 0.0) / (2 * vr))
            for nid in graph.nodes if nid not in _SPECIAL
        }
    # timesteps
    try:
        import plotly.express as _px
        viridis = _px.colors.sequential.Viridis
    except Exception:
        viridis = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]
    ts_max = max(
        (np.log1p(graph.nodes[nid].num_timesteps) for nid in graph.nodes if nid not in _SPECIAL),
        default=1.0,
    ) or 1.0
    override: Dict[int, str] = {}
    for nid in graph.nodes:
        if nid in _SPECIAL:
            continue
        t = np.log1p(graph.nodes[nid].num_timesteps) / ts_max
        idx = int(min(len(viridis) - 1, max(0, t * (len(viridis) - 1))))
        override[nid] = viridis[idx]
    return override




def render_graph_panel(
    graph: BehaviorGraph,
    labels: np.ndarray,
    viz_type: str,
    color_by: str,
    min_branch: int,
    edge_style: str,
    edge_width_slope: float,
    node_size_slope: float,
    key_prefix: str,
    theme: str,
    _mp4_dir: Path,
    _mp4_index: Dict,
) -> None:
    is_tree = viz_type.startswith("tree_")
    node_values: Dict[int, float] = {}
    if color_by == "value" and is_tree:
        try:
            node_values = graph.compute_values()
        except Exception as e:
            st.warning(f"compute_values() failed ({e}); falling back to ID coloring.")
            color_by = "id"

    if is_tree:
        render_trajectory_tree(
            labels=labels,
            metadata=metadata,
            view_mode=viz_type.replace("tree_", ""),
            min_branch=int(min_branch),
            max_depth_cap=500,
            color_mode=color_by,
            node_values=node_values,
            cluster_names=None,
            mp4_dir=_mp4_dir,
            mp4_index=_mp4_index,
            height=600,
            level=level,
            key_prefix=key_prefix,
            theme=theme,
            edge_style=edge_style,
            edge_width_slope=float(edge_width_slope),
            node_size_slope=float(node_size_slope),
        )
        return

    excluded = compute_pruned_graph_nodes(
        graph, min_visit_prob=0.0, n_total=graph.num_episodes,
        min_edge_count=int(min_branch),
    )
    pos = None
    if viz_type == "markov_svg_temporal":
        try:
            pos = gs.temporal_layout(graph, labels, metadata, level=level)
        except Exception as e:
            st.warning(f"Temporal layout failed ({e}); falling back to BFS-layered.")
    color_override = _build_color_override(graph, color_by)
    render_graph_full_width(
        graph=graph,
        labels=labels,
        metadata=metadata,
        mp4_dir=_mp4_dir,
        mp4_index=_mp4_index,
        key_prefix=key_prefix,
        min_edge_count=int(min_branch),
        pos=pos,
        excluded_node_ids=excluded,
        color_override=color_override,
        theme=theme,
        edge_style=edge_style,
        edge_width_slope=float(edge_width_slope),
        node_size_slope=float(node_size_slope),
    )


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

if show_raw_too:
    st.markdown(f"### Raw graph — {len(graph_raw.cluster_nodes)} cluster nodes")
    render_graph_panel(
        graph=graph_raw,
        labels=labels_raw,
        viz_type=viz_type,
        color_by=color_by,
        min_branch=min_branch,
        edge_style=edge_style,
        edge_width_slope=edge_width_slope,
        node_size_slope=node_size_slope,
        key_prefix="raw",
        theme=theme,
        _mp4_dir=_mp4_dir,
        _mp4_index=_mp4_index,
    )
    st.divider()

simp_header = (
    f"### Simplified — `{method}` (lever = {lever_value:.4g}) → "
    f"{metrics['n_nodes']} cluster nodes"
    if method != "passthrough"
    else f"### Raw graph — {len(graph_raw.cluster_nodes)} cluster nodes"
)
st.markdown(simp_header)
render_graph_panel(
    graph=graph_simp,
    labels=new_labels,
    viz_type=viz_type,
    color_by=color_by,
    min_branch=min_branch,
    edge_style=edge_style,
    edge_width_slope=edge_width_slope,
    node_size_slope=node_size_slope,
    key_prefix="simp",
    theme=theme,
    _mp4_dir=_mp4_dir,
    _mp4_index=_mp4_index,
)

# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Pareto frontier across methods (pre-computed)")
bench = read_benchmark(task, clust_name)
if bench is None:
    st.info(
        f"No benchmark JSON for `{task}` / `{clust_name}`. "
        "Run `python scripts/benchmark_simplification.py` to generate them. "
        f"Looking under `{_RESULTS_DIR}`."
    )
else:
    st.caption(
        f"Pre-computed sweep — n_eps={bench['n_episodes']}, "
        f"raw_nodes={bench['n_cluster_nodes_raw']}, 5-fold CV held-out NLL. "
        "Held-out values are noisy at low episode count — see "
        "`docs/simplification_findings.md` for the analysis."
    )
    pareto_y_choice = st.radio(
        "Y axis (↓ = lower is better on all of these)", [
            ("↓ Markov violation 1st-order (bits) — recommended", "markov_violation_bits"),
            ("↓ Markov violation 2nd-order (bits) — diagnostic", "markov_violation_2nd_bits"),
            ("↓* Held-out NLL / original transition (bits) — biased toward merging", "heldout_nll_per_original_bits"),
            ("↓* MDL score (bits) — biased toward merging", "mdl_score"),
        ],
        format_func=lambda x: x[0],
        horizontal=False,
    )
    pareto_x_choice = st.radio(
        "X axis (↕ tradeoff axis — no 'better' direction)", [
            ("↕ Cluster nodes (n_nodes)", "n_nodes"),
            ("↕ Free transition params", "n_free_params"),
        ],
        format_func=lambda x: x[0],
        horizontal=True,
    )
    # Derive the family prefix (strip "_k{K}" from the selected clustering name)
    # to find sibling clusterings at other K values.
    import re as _re
    family_prefix = _re.sub(r"_k\d+$", "", clust_name)
    k_sweep = k_sweep_baseline(task, family_prefix)

    fig = render_pareto_figure(
        bench["results"],
        selected_method=method,
        selected_lever=float(lever_value),
        x_field=pareto_x_choice[1],
        y_field=pareto_y_choice[1],
        y_label=pareto_y_choice[0],
        k_sweep_points=k_sweep,
    )
    st.plotly_chart(fig, use_container_width=True, key="pareto")
    if k_sweep:
        st.caption(
            f"**K-sweep baseline** (white dashed) = passthrough metrics for the "
            f"`{family_prefix}` family at K ∈ "
            f"{{{', '.join(str(p['k']) for p in k_sweep)}}}. "
            f"This is what you get by changing the clustering instead of "
            f"simplifying — a non-simplification way to control node count."
        )

with st.expander("Method registry", expanded=False):
    for name in sorted(METHODS.keys()):
        st.markdown(f"**{name}** — lever: *{METHOD_LEVER_LABELS.get(name, '')}*")
        st.caption(METHOD_DESCRIPTIONS.get(name, ""))
