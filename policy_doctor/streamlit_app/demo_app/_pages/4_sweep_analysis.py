"""Sweep Analysis — per-task winners + Pareto frontiers.

For every clustering written to ``data/demo_sweep/<task>/run_clustering/
clustering/<ordering>/<slug>/``, ``metrics.json`` records several scores; this
page compares clusterings on the ones that are unbiased across runs:

  - silhouette_mean    (higher = better) — geometric cluster cohesion.
  - mi_success         (higher = better) — mutual information between cluster
        label and episode outcome (nats). See docs/graph_evaluation.md §2.2.3.

Plus two diagnostic-only axes (no clear direction):

  - swap_rate_per_frame   stride-fair label-change density (§2.2.2)
  - distinct_per_episode  mean # behavioral phases visited per episode (§2.2.1)

Markov-based metrics are *not* compared here because they aggregate over a
different subset of testable cluster nodes per clustering — cross-clustering
comparison would be biased. See the Markov detail section for per-clustering
drill-down.

This page answers two questions:

  1. **Which (rep, K, W, S, ordering, agg) wins each metric?** — one row
     per metric, per task.
  2. **What does the metric vs. each parameter axis look like?** — per
     parameter, a scatter of all points overlaid with the Pareto-best
     envelope (max or min depending on the metric direction).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Re-root sys.path so policy_doctor resolves to this bundle, not a stale
# editable install. (Mirrors 3_graph_demo.py.)
_WORKTREE = Path(__file__).resolve().parents[4]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))
for _m in [k for k in list(sys.modules.keys()) if k.startswith("policy_doctor")]:
    _file = getattr(sys.modules.get(_m), "__file__", None) or ""
    if _file and str(_WORKTREE) not in _file:
        del sys.modules[_m]

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

st.header("Sweep Analysis")

# ── Paths ─────────────────────────────────────────────────────────────────────

# parents[4] = repo root (demo_app/_pages → demo_app → streamlit_app →
# policy_doctor → repo). Same as 3_graph_demo.py.
_REPO_ROOT = _WORKTREE
_DEMO_SWEEP = _REPO_ROOT / "data" / "demo_sweep"
_MP4_BASE = Path("/tmp/study_mp4s")


# ── Graph rendering helpers (used by the Markov detail section) ──────────────

@st.cache_resource(show_spinner="Building behavior graph…")
def _graph_for_clustering(path_str: str):
    """Return (graph, labels, metadata, level) for a given clustering dir."""
    from policy_doctor.behaviors.behavior_graph import BehaviorGraph
    p = Path(path_str)
    labels = np.load(p / "cluster_labels.npy").astype(np.int64)
    with open(p / "metadata.json") as f:
        metadata = json.load(f)
    with open(p / "manifest.yaml") as f:
        manifest_local = yaml.safe_load(f) or {}
    level = manifest_local.get("level", "rollout")
    graph = BehaviorGraph.from_cluster_assignments(labels, metadata, level=level)
    return graph, labels, metadata, level


def _resolve_mp4_for_task(task: str) -> Tuple[Path, Dict]:
    """Find /tmp/study_mp4s/<task>/index.json if present."""
    candidate = _MP4_BASE / task
    idx_path = candidate / "index.json"
    if candidate.is_dir() and idx_path.exists():
        try:
            return candidate, json.loads(idx_path.read_text())
        except Exception:
            pass
    return Path("/tmp/_nonexistent_mp4_root"), {"episodes": []}


# ── Metric directions ────────────────────────────────────────────────────────

# direction:
#   "higher" — more is unambiguously better
#   "lower"  — less is unambiguously better
#   None     — informative but ambiguous; do not pick winners or draw a frontier
#
# Markov-based metrics (violation, testable_fraction) are omitted from the
# comparison axes — they aggregate over a different node subset per clustering,
# so cross-clustering comparison is biased. See the Markov detail section below
# for per-clustering drill-down.
METRICS: Dict[str, Dict] = {
    "silhouette_mean":       {"label": "Silhouette",            "direction": "higher", "fmt": ".4f"},
    "mi_success":            {"label": "MI(label; success)",    "direction": "higher", "fmt": ".4f"},
    "swap_rate_per_frame":   {"label": "Swap rate per frame",   "direction": None,     "fmt": ".4f"},
    "distinct_per_episode":  {"label": "Distinct clusters/ep",  "direction": None,     "fmt": ".2f"},
}

PARAM_AXES = [
    ("k",        "K (clusters)",   "numeric"),
    ("w",        "W (window)",     "numeric"),
    ("s",        "S (stride)",     "numeric"),
    ("ordering", "UMAP ordering",  "categorical"),
    ("rep",      "Representation", "categorical"),
]


# ── Index loader ─────────────────────────────────────────────────────────────

def _metrics_fingerprint(root: Path) -> Tuple[int, float]:
    """(count, max_mtime) of every metrics.json under *root*. Cheap cache key:
    changes whenever any metrics file is added/removed/rewritten."""
    if not root.is_dir():
        return (0, 0.0)
    paths = list(root.rglob("metrics.json"))
    if not paths:
        return (0, 0.0)
    return (len(paths), max(p.stat().st_mtime for p in paths))


@st.cache_data(show_spinner=False)
def _load_index(root_str: str, fingerprint: Tuple[int, float]) -> pd.DataFrame:  # noqa: ARG001
    """Walk demo_sweep and return one DataFrame row per clustering.

    Joins manifest.yaml (params) with metrics.json (scores).
    """
    root = Path(root_str)
    if not root.is_dir():
        return pd.DataFrame()
    rows: List[Dict] = []
    for labels_path in sorted(root.rglob("cluster_labels.npy")):
        d = labels_path.parent
        try:
            with open(d / "manifest.yaml") as f:
                m = yaml.safe_load(f) or {}
        except Exception:
            continue
        try:
            with open(d / "metrics.json") as f:
                mx = json.load(f)
        except Exception:
            mx = {}
        # Walk path to find task: data/demo_sweep/<task>/run_clustering/...
        task = "?"
        for i, part in enumerate(d.parts):
            if part == "demo_sweep" and i + 1 < len(d.parts):
                task = d.parts[i + 1]
                break
        # Ordering: prefer manifest.pipeline_steps, fall back to parent dir name.
        steps = m.get("pipeline_steps") or []
        if "window" in steps and "umap" in steps:
            ordering = (
                "aggregate_first" if steps.index("window") < steps.index("umap")
                else "umap_first"
            )
        else:
            parent = d.parent.name
            ordering = {"agg_first": "aggregate_first",
                        "umap_first": "umap_first"}.get(parent, "umap_first")
        rk = m.get("rep_kwargs") or {}
        layer = rk.get("layer", "") if isinstance(rk, dict) else ""
        rep = m.get("influence_source") or m.get("slice_representation") or "?"
        rep_full = f"{rep}/{layer}" if layer else rep
        row = {
            "task":     task,
            "ordering": ordering,
            "rep":      rep_full,
            "k":        int(m.get("n_clusters", 0) or 0),
            "w":        int(m.get("window_width", 0) or 0),
            "s":        int(m.get("stride", 0) or 0),
            "agg":      str(m.get("aggregation", "?") or "?"),
            "path":     str(d),
        }
        # Pull every metric key on disk so the columns reflect the current
        # metrics.json schema, not whatever METRICS dict happened to be cached.
        for k, v in (mx or {}).items():
            row[k] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    # Ensure every metric the UI references is present (NaN if absent on disk).
    for k in METRICS:
        if k not in df.columns:
            df[k] = np.nan
    return df


df = _load_index(str(_DEMO_SWEEP), _metrics_fingerprint(_DEMO_SWEEP))
# Backfill any UI-referenced metric column missing on disk so dropna() / [] won't crash.
for _k in METRICS:
    if _k not in df.columns:
        df[_k] = np.nan

if df.empty:
    st.error(f"No clusterings found under `{_DEMO_SWEEP}`.")
    st.stop()


# ── Task selector ────────────────────────────────────────────────────────────

tasks = sorted(df["task"].unique().tolist())
task_sel = st.selectbox(
    "Task", tasks,
    index=0,
    help="Pick a task; all configs swept on that task are scored below.",
)
sub = df[df["task"] == task_sel].copy()
st.caption(
    f"**{task_sel}**: {len(sub)} clusterings  "
    f"({sub['ordering'].nunique()} ordering(s), {sub['rep'].nunique()} rep(s), "
    f"{sub['k'].nunique()} K value(s), {sub['w'].nunique()} W value(s), "
    f"{sub['s'].nunique()} S value(s))"
)


# ── 1. Optimal config per metric ─────────────────────────────────────────────

st.subheader("Optimal config per metric")

winners_rows: List[Dict] = []
for metric, info in METRICS.items():
    if metric not in sub.columns:
        continue
    if info.get("direction") is None:
        continue  # ambiguous metric — no "best" to report
    valid = sub.dropna(subset=[metric])
    if valid.empty:
        continue
    higher = info["direction"] == "higher"
    idx = valid[metric].idxmax() if higher else valid[metric].idxmin()
    best = valid.loc[idx]
    winners_rows.append({
        "metric": info["label"],
        "value":  f"{best[metric]:{info['fmt']}}",
        "rep":    best["rep"],
        "ordering": best["ordering"],
        "K":      int(best["k"]),
        "W":      int(best["w"]),
        "S":      int(best["s"]),
        "agg":    best["agg"],
    })

if winners_rows:
    winners_df = pd.DataFrame(winners_rows).set_index("metric")
    st.dataframe(winners_df, use_container_width=True)
    ambig = [info["label"] for m, info in METRICS.items() if info.get("direction") is None]
    if ambig:
        st.caption(
            "Excluded (no unambiguous direction — diagnostic only): "
            + ", ".join(f"**{x}**" for x in ambig)
            + "."
        )
else:
    st.info("No metric values available for this task.")


# ── 2. Pareto frontier per parameter ─────────────────────────────────────────

st.subheader("Pareto frontier per parameter")
st.caption(
    "For each (metric, parameter) pair, scatter shows every clustering's score. "
    "Metrics with a clear direction (↑/↓) get an overlay line showing the best "
    "achievable score at each parameter value; ambiguous/diagnostic metrics "
    "(distinct/ep, swap rate) get scatter only."
)

metric_pick = st.selectbox(
    "Metric",
    list(METRICS.keys()),
    format_func=lambda k: METRICS[k]["label"],
    index=0,
)


# ── Per-metric formula / definition ─────────────────────────────────────────
_METRIC_FORMULAS: Dict[str, str] = {
    "silhouette_mean": r"""
**Silhouette** — geometric cluster cohesion in the UMAP-reduced embedding space.

Per-sample score:
$$
a_i = \frac{1}{|C_{c_i}| - 1} \sum_{j \in C_{c_i},\, j \neq i} d(i,j),
\qquad
b_i = \min_{c \neq c_i} \frac{1}{|C_c|} \sum_{j \in C_c} d(i,j),
\qquad
s_i = \frac{b_i - a_i}{\max(a_i,\, b_i)} \in [-1, 1]
$$

where $d(i,j) = \|\mathbf{z}_i - \mathbf{z}_j\|_2$ is Euclidean distance in the embedding space.

Reported value: $\bar s = \frac{1}{N} \sum_i s_i$, computed via `sklearn.metrics.silhouette_score`.

**Implementation note** — to keep cost tractable ($O(N^2)$ pairwise distances), the score is
computed on a random subsample of ≤ 2000 points per clustering (`_SILHOUETTE_MAX_SAMPLES`
in `compute_clustering_metrics.py`).
""",
    "mi_success": r"""
**MI(label; success)** — mutual information between a window's cluster label and
its episode's success outcome, in nats.

Let $L_i \in \{0, \dots, K-1\}$ be the cluster of window $i$ (noise windows excluded),
and $S_i \in \{0, 1\}$ be the success outcome of the episode containing $i$. With
empirical counts $n_{c,o} = |\{i : L_i = c,\, S_i = o\}|$ and $N = \sum_{c,o} n_{c,o}$:

$$
\mathrm{MI}(L; S) = \sum_{c=0}^{K-1} \sum_{o \in \{0,1\}}
    \frac{n_{c,o}}{N}\, \log\frac{n_{c,o}\, N}{n_c\, n_o}
$$

(natural log; the result is in nats.) Skipped when either outcome is absent
(degenerate marginal $\Rightarrow$ MI is undefined).

**Interpretation** — average reduction in outcome uncertainty given the cluster label:
$\mathrm{MI}(L;S) = H(S) - H(S \mid L)$. A free-passenger cluster that appears equally
in successes and failures contributes 0; clusters enriched in one outcome contribute
positively.
""",
    "swap_rate_per_frame": r"""
**Swap rate per frame** — stride-fair temporal coherence.

For each episode $e$ with $N_e$ windows in temporal order, count label changes
between adjacent windows:
$$
\mathrm{swaps}(e) = \sum_{k=0}^{N_e - 2} \mathbf{1}\!\left[\hat\ell_{k+1} \neq \hat\ell_k\right]
$$

Reported value: total label changes over total frames across all episodes,
$$
\mathrm{swap}_f \;=\; \frac{\sum_e \mathrm{swaps}(e)}{\sum_e F_e}
$$
where $F_e$ is the highest `window_end` value seen in episode $e$ (≈ episode length
in frames). Dividing by total frames (not total windows) makes the metric independent
of the stride $S$ — at stride $S$ there are roughly $S\times$ fewer adjacent-window
pairs but each represents $S\times$ more frames.

**Diagnostic only** — zero swaps means degenerate single-cluster collapse; high swaps
mean flicker within a behavioral phase. There's no monotonic "best".
""",
    "distinct_per_episode": r"""
**Distinct clusters per episode** — mean number of distinct behavioral phases visited
per episode after run-length collapse.

For each episode $e$, run-length-encode its window-label sequence
$(\hat\ell_1, \dots, \hat\ell_{N_e})$ to keep only the label changes:
$\mathrm{RLE}(e) = (c_1, c_2, \dots, c_{T_e})$ with $c_k \neq c_{k+1}$. The episode's
distinct-cluster count is:
$$
D_e = \bigl|\{c : c \in \mathrm{RLE}(e)\}\bigr|
$$

Reported value: $\bar D = \frac{1}{|E|} \sum_{e \in E} D_e$.

**Diagnostic only** — for a task with $\sim 5$ true phases, $\bar D$ near 5 is healthy;
much lower means phase collapse, much higher means over-fragmentation. No single
direction is "better".
""",
}

with st.expander(f"How is **{METRICS[metric_pick]['label']}** computed?", expanded=False):
    st.markdown(_METRIC_FORMULAS.get(
        metric_pick,
        f"_(no formula registered for `{metric_pick}`)_",
    ))


info = METRICS[metric_pick]
direction = info.get("direction")          # "higher" | "lower" | None
higher_better = direction == "higher"
has_direction = direction is not None
valid = sub.dropna(subset=[metric_pick]).copy()

if valid.empty:
    st.warning(f"No `{metric_pick}` values for {task_sel}.")
    st.stop()


def _frontier_numeric(d: pd.DataFrame, axis: str, metric: str, higher: bool) -> Tuple[np.ndarray, np.ndarray]:
    grouped = d.groupby(axis)[metric].agg("max" if higher else "min").sort_index()
    return grouped.index.values, grouped.values


def _hover_cols(d: pd.DataFrame) -> np.ndarray:
    return np.stack([
        d["rep"].values, d["ordering"].values,
        d["k"].values, d["w"].values, d["s"].values, d["agg"].values,
    ], axis=-1)


_HOVER_TMPL = (
    "<b>%{y:.4f}</b><br>"
    "rep: %{customdata[0]}<br>"
    "ordering: %{customdata[1]}<br>"
    "K=%{customdata[2]}, W=%{customdata[3]}, S=%{customdata[4]}, agg=%{customdata[5]}"
    "<extra></extra>"
)


for axis, axis_label, kind in PARAM_AXES:
    if valid[axis].nunique() <= 1:
        continue
    fig = go.Figure()

    if kind == "numeric":
        # Jitter x for visibility on discrete axes
        rng = np.random.default_rng(0)
        x_jitter = valid[axis].astype(float).values + rng.uniform(-0.15, 0.15, len(valid))
        fig.add_trace(go.Scatter(
            x=x_jitter,
            y=valid[metric_pick].values,
            mode="markers",
            marker=dict(size=6, opacity=0.4, color="#888"),
            name="all configs",
            customdata=_hover_cols(valid),
            hovertemplate=_HOVER_TMPL,
        ))
        if has_direction:
            xs, ys = _frontier_numeric(valid, axis, metric_pick, higher_better)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                line=dict(color="crimson", width=2),
                marker=dict(size=9, color="crimson"),
                name=f"best {'↑' if higher_better else '↓'}",
            ))
        fig.update_xaxes(title=axis_label, tickmode="array",
                         tickvals=sorted(valid[axis].unique().tolist()))
    else:
        # Categorical: box per category + best marker on top
        cats = sorted(valid[axis].unique().tolist())
        for c in cats:
            sel = valid[valid[axis] == c]
            fig.add_trace(go.Box(
                y=sel[metric_pick].values,
                name=str(c),
                boxpoints="all",
                jitter=0.4,
                pointpos=0.0,
                marker=dict(size=4, color="#888", opacity=0.4),
                line=dict(color="#444"),
                fillcolor="rgba(150,150,150,0.15)",
                showlegend=False,
                customdata=_hover_cols(sel),
                hovertemplate=_HOVER_TMPL,
            ))
        if has_direction:
            bests = [
                valid[valid[axis] == c][metric_pick].agg("max" if higher_better else "min")
                for c in cats
            ]
            fig.add_trace(go.Scatter(
                x=cats, y=bests, mode="lines+markers",
                line=dict(color="crimson", width=2),
                marker=dict(size=11, color="crimson", symbol="diamond"),
                name=f"best {'↑' if higher_better else '↓'}",
            ))
        fig.update_xaxes(title=axis_label)

    yaxis_kwargs: Dict = {"title": info["label"]}
    if info["fmt"].endswith("%"):
        yaxis_kwargs["tickformat"] = ".0%"
    fig.update_yaxes(**yaxis_kwargs)
    fig.update_layout(
        height=320,
        margin=dict(l=40, r=20, t=30, b=40),
        title=f"{info['label']} vs {axis_label}",
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── 3. Joint Pareto frontier (configurable axes + highlight) ────────────────

st.subheader("Joint Pareto frontier")
st.caption(
    "Each point is one clustering configuration. The **Pareto frontier** is the "
    "set of configurations that are not dominated by any other — you can't "
    "improve one axis without giving up the other. Pick the two metrics to "
    "compare on, and optionally highlight a subset of points by parameter value."
)

# Axis pickers
_metric_keys = list(METRICS.keys())
_col_x, _col_y = st.columns(2)
with _col_x:
    x_metric = st.selectbox(
        "X axis",
        _metric_keys,
        index=_metric_keys.index("silhouette_mean") if "silhouette_mean" in _metric_keys else 0,
        format_func=lambda k: METRICS[k]["label"],
        key="joint_x_metric",
    )
with _col_y:
    _default_y = "mi_success" if "mi_success" in _metric_keys else _metric_keys[-1]
    y_metric = st.selectbox(
        "Y axis",
        _metric_keys,
        index=_metric_keys.index(_default_y),
        format_func=lambda k: METRICS[k]["label"],
        key="joint_y_metric",
    )

# Highlight picker — by which parameter (or none), and which values
_PARAMS = [("rep", "Representation"), ("ordering", "UMAP ordering"),
           ("k", "K"), ("w", "W"), ("s", "S"), ("agg", "Aggregation")]
_col_hp, _col_hv = st.columns([1, 2])
with _col_hp:
    hl_param_label = st.selectbox(
        "Highlight by",
        ["(none)"] + [p[1] for p in _PARAMS],
        key="joint_hl_param",
    )
hl_param = next((p[0] for p in _PARAMS if p[1] == hl_param_label), None)
with _col_hv:
    if hl_param is None:
        hl_values: List = []
        st.caption("Pick a parameter to enable value highlighting.")
    else:
        all_vals = sorted(sub[hl_param].dropna().unique().tolist(),
                          key=lambda v: (str(type(v)), v))
        hl_values = st.multiselect(
            f"Highlight values of `{hl_param}`",
            all_vals,
            default=[],
            key="joint_hl_values",
        )

joint = sub.dropna(subset=[x_metric, y_metric]).copy()
if joint.empty or x_metric == y_metric:
    st.info(
        "No configurations with both metrics available, or the two axes are "
        "the same metric."
    )
else:
    # Direction-aware Pareto. Convert each axis so "higher is better" for the
    # dominance check; metrics marked None are treated as "higher is better"
    # by convention (frontier is best-guess only — caveat in caption below).
    def _sign(direction):
        return 1.0 if direction != "lower" else -1.0
    x_sgn = _sign(METRICS[x_metric]["direction"])
    y_sgn = _sign(METRICS[y_metric]["direction"])
    xs_dom = joint[x_metric].values * x_sgn
    ys_dom = joint[y_metric].values * y_sgn
    n = len(joint)
    is_front = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_front[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if (xs_dom[j] >= xs_dom[i] and ys_dom[j] >= ys_dom[i]
                    and (xs_dom[j] > xs_dom[i] or ys_dom[j] > ys_dom[i])):
                is_front[i] = False
                break
    joint["on_front"] = is_front
    if hl_param is not None and hl_values:
        joint["highlighted"] = joint[hl_param].isin(hl_values)
    else:
        joint["highlighted"] = True  # no highlight = everyone "selected"

    # Color: if highlighting active, highlighted points keep an ordering-based
    # color; everyone else fades to gray. If no highlight, use ordering colors.
    _ORDERING_COLOR = {
        "umap_first":      "#1f77b4",
        "aggregate_first": "#ff7f0e",
    }

    def _point_color(o: str, highlighted: bool) -> str:
        if hl_values and not highlighted:
            return "#cccccc"
        return _ORDERING_COLOR.get(o, "#888")

    def _point_opacity(highlighted: bool, faint: bool) -> float:
        # faint = "dominated"; highlighted overrides faintness.
        if hl_values:
            if not highlighted:
                return 0.12
            return 1.0 if faint else 1.0  # always vivid when highlighted
        return 0.25 if faint else 1.0

    fig_joint = go.Figure()

    _x_label = METRICS[x_metric]["label"]
    _y_label = METRICS[y_metric]["label"]
    _hover = (
        f"{_x_label}: %{{x:{METRICS[x_metric]['fmt']}}}<br>"
        f"{_y_label}: %{{y:{METRICS[y_metric]['fmt']}}}<br>"
        "rep: %{customdata[0]}<br>"
        "ordering: %{customdata[1]}<br>"
        "K=%{customdata[2]}, W=%{customdata[3]}, S=%{customdata[4]}, agg=%{customdata[5]}"
        "<extra></extra>"
    )

    def _hover_cd(d: pd.DataFrame) -> np.ndarray:
        return np.stack([
            d["rep"].values, d["ordering"].values,
            d["k"].values, d["w"].values, d["s"].values, d["agg"].values,
        ], axis=-1)

    # Background: dominated points
    dom = joint[~joint["on_front"]]
    if not dom.empty:
        fig_joint.add_trace(go.Scatter(
            x=dom[x_metric], y=dom[y_metric],
            mode="markers",
            marker=dict(
                size=7,
                color=[_point_color(o, h) for o, h in zip(dom["ordering"], dom["highlighted"])],
                opacity=[_point_opacity(h, faint=True) for h in dom["highlighted"]],
                line=dict(width=0),
            ),
            name="dominated",
            customdata=_hover_cd(dom),
            hovertemplate=_hover,
        ))

    # Foreground: frontier staircase + points
    fr = joint[joint["on_front"]].sort_values([x_metric, y_metric])
    if not fr.empty:
        xs_fr = fr[x_metric].tolist()
        ys_fr = fr[y_metric].tolist()
        # Staircase shape depends on the directions of x and y. We sorted by
        # x ascending; along the frontier the y value moves in the dis-preferred
        # direction (per direction-aware ordering). We just connect with
        # right-angle steps: vertical then horizontal between adjacent points.
        step_x: List[float] = [xs_fr[0]]
        step_y: List[float] = [ys_fr[0]]
        for i in range(1, len(xs_fr)):
            step_x.extend([xs_fr[i], xs_fr[i]])
            step_y.extend([step_y[-1], ys_fr[i]])
        fig_joint.add_trace(go.Scatter(
            x=step_x, y=step_y,
            mode="lines",
            line=dict(color="crimson", width=3),
            name="Pareto frontier",
            hoverinfo="skip",
            showlegend=True,
        ))
        fig_joint.add_trace(go.Scatter(
            x=xs_fr, y=ys_fr,
            mode="markers",
            marker=dict(
                size=13,
                color=[_point_color(o, h) for o, h in zip(fr["ordering"], fr["highlighted"])],
                opacity=[_point_opacity(h, faint=False) for h in fr["highlighted"]],
                line=dict(width=2, color="crimson"),
                symbol="diamond",
            ),
            name="Pareto frontier (points)",
            customdata=_hover_cd(fr),
            hovertemplate="<b>FRONTIER</b><br>" + _hover,
        ))

    # Legend proxies for ordering colors (always shown)
    for ord_name, ord_col in _ORDERING_COLOR.items():
        if ord_name in joint["ordering"].unique():
            fig_joint.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=ord_col),
                name=ord_name, showlegend=True,
            ))

    # Title + axis direction arrows
    def _direction_arrow(direction, axis: str) -> str:
        if direction == "higher":
            return "higher better →" if axis == "x" else "higher better ↑"
        if direction == "lower":
            return "← lower better" if axis == "x" else "lower better ↓"
        return "diagnostic"

    x_title = f"{_x_label} ({_direction_arrow(METRICS[x_metric]['direction'], 'x')})"
    y_title = f"{_y_label} ({_direction_arrow(METRICS[y_metric]['direction'], 'y')})"

    title_suffix = ""
    if hl_param and hl_values:
        title_suffix = (
            f" — highlighting {hl_param}∈" + "{" + ", ".join(str(v) for v in hl_values) + "}"
        )

    fig_joint.update_layout(
        title=f"{_x_label} × {_y_label} — joint Pareto frontier" + title_suffix,
        xaxis_title=x_title,
        yaxis=dict(title=y_title),
        height=480,
        margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )

    if METRICS[x_metric]["direction"] is None or METRICS[y_metric]["direction"] is None:
        st.caption(
            "⚠️ At least one axis is a diagnostic-only metric (no clear best "
            "direction). The Pareto frontier defaults to *higher is better* for "
            "those axes — interpret the frontier accordingly."
        )

    st.plotly_chart(fig_joint, use_container_width=True)

    n_front = int(joint["on_front"].sum())
    n_hl = int(joint["highlighted"].sum()) if hl_values else len(joint)
    if hl_values:
        st.caption(
            f"**{n_front} of {len(joint)}** configurations are Pareto-optimal. "
            f"**{n_hl}** match the highlight filter."
        )
    else:
        st.caption(f"**{n_front} of {len(joint)}** configurations are Pareto-optimal.")

    with st.expander(f"Pareto-frontier configs ({n_front} rows)"):
        front_cols = ["rep", "ordering", "k", "w", "s", "agg", x_metric, y_metric]
        front_df = (
            joint[joint["on_front"]][front_cols]
            .sort_values(x_metric, ascending=(METRICS[x_metric]["direction"] == "lower"))
            .reset_index(drop=True)
        )
        st.dataframe(front_df, use_container_width=True)


# ── 4. Raw table (collapsed) ─────────────────────────────────────────────────

with st.expander(f"Raw scores ({len(valid)} rows)"):
    show_cols = ["rep", "ordering", "k", "w", "s", "agg"] + list(METRICS.keys())
    sort_ascending = direction == "lower"  # for "higher": False; for "lower": True; for None: False
    st.dataframe(
        valid[show_cols].sort_values(
            metric_pick, ascending=sort_ascending
        ).reset_index(drop=True),
        use_container_width=True,
        height=400,
    )


# ── 4. Markov detail (per-cluster drill-down) ────────────────────────────────

with st.expander("Markov detail — per-cluster diagnostics", expanded=False):
    st.caption(
        "Pick a clustering to see *which* cluster nodes carry history-dependence. "
        "Left: p-value per cluster from the chi² test of "
        "`P(next | curr) ≟ P(next | prev, curr)` — anything below the dashed α=0.05 "
        "line is non-Markovian. Right: chi² statistic vs cluster size — bubbles in "
        "the upper-right are large clusters with strong non-Markov behavior "
        "(worst offenders). Note: these per-clustering diagnostics are *not* a "
        "valid basis for ranking clusterings — the testable-node subset varies."
    )

    # Pick a clustering — default to the lowest Markov violation.
    markov_sorted = sub.dropna(subset=["markov_violation_mean"]).sort_values(
        "markov_violation_mean", ascending=True
    )
    if markov_sorted.empty:
        st.info("No Markov-violation values for this task.")
        st.stop()


    def _label_row(r) -> str:
        return (
            f"{r['rep']} · {r['ordering']} · K={r['k']} W={r['w']} S={r['s']} "
            f"(violation = {r['markov_violation_mean']:.3f})"
        )


    options = markov_sorted.index.tolist()
    default_idx = 0
    pick_idx = st.selectbox(
        "Clustering",
        options,
        index=default_idx,
        format_func=lambda i: _label_row(sub.loc[i]),
    )
    chosen = sub.loc[pick_idx]


    # ── 4·0a. Trajectory tree for the selected clustering ───────────────────────

    with st.spinner("Rendering trajectory tree…"):
        _, _g_labels, _g_meta, _g_level = _graph_for_clustering(chosen["path"])
        _mp4_dir, _mp4_index = _resolve_mp4_for_task(task_sel)

        from policy_doctor.streamlit_app.components.trajectory_tree_view import (
            render_trajectory_tree,
        )

        render_trajectory_tree(
            labels=_g_labels,
            metadata=_g_meta,
            view_mode="native_svg",
            level=_g_level,
            mp4_dir=_mp4_dir,
            mp4_index=_mp4_index,
            key_prefix=f"sa_tree_{chosen['task']}_{Path(chosen['path']).name}",
            min_branch=2,
            height=600,
        )


    @st.cache_data(show_spinner="Running Markov test…")
    def _per_state_markov(path_str: str) -> Dict:
        """Re-run the chi² test and return per-state results as plain dicts."""
        p = Path(path_str)
        labels = np.load(p / "cluster_labels.npy").astype(np.int64)
        with open(p / "metadata.json") as f:
            meta = json.load(f)
        with open(p / "manifest.yaml") as f:
            manifest_local = yaml.safe_load(f) or {}
        level = manifest_local.get("level", "rollout")
        from policy_doctor.behaviors.behavior_graph import test_markov_property
        result = test_markov_property(
            labels, meta, level=level,
            significance_level=0.05, method="chi2",
        )
        # Convert MarkovTestResult dataclasses to dicts (cacheable + small).
        out_per_state = {}
        for sid, r in (result.get("per_state") or {}).items():
            if not r.testable:
                out_per_state[int(sid)] = {
                    "testable": False,
                    "reason": r.reason or "untestable",
                }
                continue
            ct = r.contingency_table
            out_per_state[int(sid)] = {
                "testable": True,
                "chi2": float(r.chi2),
                "p_value": float(r.p_value),
                "dof": int(r.dof) if r.dof is not None else None,
                "markov_holds": bool(r.markov_holds),
                "contingency_table": ct.tolist() if ct is not None else None,
                "previous_states": list(r.previous_states or []),
                "next_states": list(r.next_states or []),
                "n_transitions": int(ct.sum()) if ct is not None else 0,
            }
        # Per-cluster total count (incl. non-testable)
        cluster_sizes = {int(c): int((labels == c).sum())
                         for c in np.unique(labels) if c >= 0}
        return {
            "per_state": out_per_state,
            "cluster_sizes": cluster_sizes,
            "level": level,
        }


    markov = _per_state_markov(chosen["path"])
    per_state = markov["per_state"]
    cluster_sizes = markov["cluster_sizes"]

    if not per_state:
        st.warning("No per-state results (cluster too small or no transitions).")
        st.stop()


    # Human-readable explanations for `MarkovTestResult.reason` codes.
    _REASON_EXPLANATIONS = {
        "no_transitions":          "Cluster never appears as a current state with both a predecessor and a successor.",
        "no_interior_transitions": "Cluster never appears as an interior state (only at episode boundaries).",
        "only_one_predecessor":    "Only one upstream cluster — independence from history is trivially undefined.",
        "only_one_successor":      "Only one downstream cluster — successor distribution is deterministic.",
        "insufficient_data":       "Fewer than 5 transitions through this cluster (chi² needs ≥ 5 observations).",
        "test_failed":             "chi²_contingency raised on the contingency table (e.g. degenerate table).",
        "untestable":              "Unspecified reason.",
    }


    # Tidy per-state DataFrame for plotting
    ps_rows: List[Dict] = []
    for sid, r in per_state.items():
        if r.get("testable"):
            ps_rows.append({
                "cluster": sid,
                "size": cluster_sizes.get(sid, 0),
                "n_transitions": r["n_transitions"],
                "chi2": r["chi2"],
                "p_value": r["p_value"],
                "dof": r["dof"],
                "markov_holds": r["markov_holds"],
                "status": "Markovian" if r["markov_holds"] else "Non-Markovian",
                "reason": "",
            })
        else:
            reason = r.get("reason") or "untestable"
            ps_rows.append({
                "cluster": sid,
                "size": cluster_sizes.get(sid, 0),
                "n_transitions": 0,
                "chi2": None,
                "p_value": None,
                "dof": None,
                "markov_holds": None,
                "status": "Not testable",
                "reason": reason,
            })
    ps_df = pd.DataFrame(ps_rows).sort_values("cluster").reset_index(drop=True)

    _STATUS_COLORS = {
        "Markovian":     "#2ca02c",
        "Non-Markovian": "#d62728",
        "Not testable":  "#bbbbbb",
    }


    # ── 4·0. Testability summary ─────────────────────────────────────────────────

    n_total = len(ps_df)
    n_test = int((ps_df["status"] != "Not testable").sum())
    n_mark = int((ps_df["status"] == "Markovian").sum())
    n_nonmark = int((ps_df["status"] == "Non-Markovian").sum())
    n_untest = n_total - n_test

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clusters", n_total)
    c2.metric("Testable", f"{n_test} / {n_total}",
              delta=f"{n_test / max(n_total, 1):.0%}", delta_color="off")
    c3.metric("Markovian", n_mark, delta=f"of {n_test} testable", delta_color="off")
    c4.metric("Non-Markovian", n_nonmark, delta=f"of {n_test} testable", delta_color="off")

    if n_untest:
        untest = ps_df[ps_df["status"] == "Not testable"].copy()
        untest["explanation"] = untest["reason"].map(
            lambda r: _REASON_EXPLANATIONS.get(r, r)
        )
        reason_counts = (
            untest.groupby("reason")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        reason_counts["explanation"] = reason_counts["reason"].map(
            lambda r: _REASON_EXPLANATIONS.get(r, r)
        )
        summary_bits = [
            f"{int(row['count'])} × `{row['reason']}`"
            for _, row in reason_counts.iterrows()
        ]
        st.caption(
            f"**{n_untest} clusters could not be tested** — "
            + ", ".join(summary_bits)
            + f". The Markov-violation score is computed only over the "
              f"{n_test} testable clusters."
        )
        with st.expander(f"Untestable clusters — breakdown ({n_untest})"):
            st.markdown("**Reason summary:**")
            st.dataframe(
                reason_counts[["reason", "explanation", "count"]],
                use_container_width=True, hide_index=True,
            )
            st.markdown("**Per-cluster:**")
            st.dataframe(
                untest[["cluster", "size", "reason", "explanation"]]
                .reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )

    col_bar, col_bubble = st.columns(2)

    # ── 4a. p-value bar chart ────────────────────────────────────────────────────
    with col_bar:
        # Use 1.0 for non-testable so they're visually grouped at the top in gray;
        # mark them clearly via color.
        y_vals = [r["p_value"] if r["p_value"] is not None else 1.0 for _, r in ps_df.iterrows()]
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=ps_df["cluster"].astype(str),
            y=y_vals,
            marker=dict(color=[_STATUS_COLORS[s] for s in ps_df["status"]]),
            customdata=np.stack([
                ps_df["status"].values,
                ps_df["size"].values,
                ps_df["n_transitions"].values,
                ps_df["chi2"].fillna(-1).values,
                ps_df["dof"].fillna(-1).values,
                ps_df["reason"].fillna("").values,
            ], axis=-1),
            hovertemplate=(
                "<b>Cluster %{x}</b><br>"
                "Status: %{customdata[0]}<br>"
                "Reason (if untestable): %{customdata[5]}<br>"
                "p-value: %{y:.4g}<br>"
                "Cluster size: %{customdata[1]}<br>"
                "Transitions through node: %{customdata[2]}<br>"
                "chi²: %{customdata[3]:.2f}<br>"
                "dof: %{customdata[4]}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))
        fig_bar.add_hline(
            y=0.05, line_dash="dash", line_color="black",
            annotation_text="α = 0.05", annotation_position="top right",
        )
        fig_bar.update_layout(
            title="p-value per cluster node",
            xaxis_title="Cluster ID",
            yaxis_title="p-value (chi²)",
            yaxis=dict(range=[0, 1.05]),
            height=400,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── 4b. chi² vs cluster size bubble plot ─────────────────────────────────────
    with col_bubble:
        testable = ps_df[ps_df["chi2"].notna()].copy()
        if testable.empty:
            st.info("No testable clusters for the bubble plot.")
        else:
            # Bubble area ∝ n_transitions
            max_n = max(testable["n_transitions"].max(), 1)
            sizeref = 2.0 * max_n / (40.0 ** 2)
            fig_bub = go.Figure()
            fig_bub.add_trace(go.Scatter(
                x=testable["size"],
                y=testable["chi2"],
                mode="markers+text",
                text=testable["cluster"].astype(str),
                textposition="middle center",
                textfont=dict(size=9, color="white"),
                marker=dict(
                    size=testable["n_transitions"],
                    sizemode="area",
                    sizeref=sizeref,
                    sizemin=8,
                    color=testable["p_value"],
                    colorscale="RdYlGn",
                    cmin=0.0, cmax=0.2,
                    colorbar=dict(title="p-value", tickformat=".2f"),
                    line=dict(width=1, color="#333"),
                ),
                customdata=np.stack([
                    testable["status"].values,
                    testable["p_value"].values,
                    testable["dof"].fillna(-1).values,
                    testable["n_transitions"].values,
                ], axis=-1),
                hovertemplate=(
                    "<b>Cluster %{text}</b><br>"
                    "Status: %{customdata[0]}<br>"
                    "Cluster size: %{x}<br>"
                    "chi²: %{y:.2f}<br>"
                    "p-value: %{customdata[1]:.4g}<br>"
                    "dof: %{customdata[2]}<br>"
                    "Transitions: %{customdata[3]}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))
            fig_bub.update_layout(
                title="chi² vs cluster size (bubble = #transitions, color = p)",
                xaxis_title="Cluster size (samples)",
                yaxis_title="chi² statistic",
                height=400,
                margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig_bub, use_container_width=True)


    # ── 4c. Per-predecessor next-state distribution ──────────────────────────────

    nonmark = ps_df[ps_df["status"] == "Non-Markovian"]["cluster"].tolist()
    if not nonmark:
        st.success("All testable clusters are Markovian.")
    else:
        st.markdown("**Where does history matter?**")
        detail_choice = st.selectbox(
            "Non-Markovian cluster",
            nonmark,
            index=0,
            format_func=lambda c: (
                f"Cluster {c}  "
                f"(chi²={ps_df.set_index('cluster').loc[c, 'chi2']:.1f}, "
                f"p={ps_df.set_index('cluster').loc[c, 'p_value']:.3g})"
            ),
        )
        r = per_state[int(detail_choice)]
        ct = np.array(r["contingency_table"], dtype=float)  # rows=prev, cols=next
        prev_states = r["previous_states"]
        next_states = r["next_states"]

        # Marginal P(next | curr): sum over prev rows
        col_sums = ct.sum(axis=0)
        total = col_sums.sum()
        marginal = col_sums / total if total > 0 else col_sums

        # Conditional P(next | prev, curr) per row
        row_sums = ct.sum(axis=1, keepdims=True)
        cond = np.divide(ct, row_sums, where=row_sums > 0)  # shape (n_prev, n_next)

        # Build x-axis: one tick per predecessor, plus a final "any prev" marginal.
        x_labels = [f"prev={p}\n(n={int(ct[i].sum())})" for i, p in enumerate(prev_states)]
        x_labels.append(f"any prev\n(n={int(total)})")

        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

        fig_detail = go.Figure()
        # One stacked-bar trace per next-state, so each column (predecessor)
        # adds up to 1.0 and the segments show where transitions go.
        for j, nxt in enumerate(next_states):
            ys = list(cond[:, j]) + [marginal[j]]
            fig_detail.add_trace(go.Bar(
                x=x_labels,
                y=ys,
                name=f"next={nxt}",
                marker=dict(color=palette[j % len(palette)],
                            line=dict(color="white", width=0.5)),
                hovertemplate=(
                    "%{x}<br>"
                    f"next={nxt}: "
                    "%{y:.1%}"
                    "<extra></extra>"
                ),
            ))
        fig_detail.update_layout(
            title=(
                f"Cluster {detail_choice}: next-state distribution per predecessor "
                "(stacked → each column sums to 1.0)"
            ),
            xaxis_title="Predecessor cluster",
            yaxis_title="P(next | prev, curr)",
            yaxis=dict(tickformat=".0%", range=[0, 1.0]),
            barmode="stack",
            height=420,
            margin=dict(l=40, r=20, t=50, b=60),
            legend=dict(title="Next cluster", orientation="v",
                        yanchor="top", y=1, xanchor="left", x=1.02),
        )
        # Visually separate the marginal column from the per-predecessor ones.
        fig_detail.add_vline(
            x=len(prev_states) - 0.5,
            line_dash="dot", line_color="#888",
        )
        st.plotly_chart(fig_detail, use_container_width=True)
        st.caption(
            f"Each column is one predecessor cluster (the rightmost is the "
            f"marginal, ignoring history). Colored segments show how cluster "
            f"{detail_choice} sends transitions to each next cluster, conditioned "
            f"on where they came from. The more the colored stacks differ across "
            "columns, the more history-dependence the chi² test detected."
        )
