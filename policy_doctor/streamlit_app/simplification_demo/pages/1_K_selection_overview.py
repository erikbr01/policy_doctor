"""K selection overview — all (task, rep, w, s) settings.

Companion page to simplification_demo/app.py.

Grid layout: one column per task, one row per (rep, w, s) setting.
Each cell shows:
  - MV₁ with 95% bootstrap CI band (blue, left y-axis)
  - Silhouette score (orange, right y-axis)
  - Coverage fraction (grey dashed, left y-axis)
  - γ-knee K* (green star)
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — same worktree root as app.py
# ---------------------------------------------------------------------------

_WORKTREE = Path(__file__).resolve().parents[4]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

_KSWEEP_FLAT_ROOT = Path("/Users/erik/stanford/asl_rotation/data/graph_simplification/clusterings")
_KSWEEP_RESULTS_DIR = Path("/Users/erik/stanford/asl_rotation/data/graph_simplification/results/k_sweep")

_SLUG_RE = re.compile(
    r"^(?P<task>[a-z][a-z0-9_]+?)__(?P<rep>[a-z][a-z0-9_]+?)__w(?P<w>\d+)_s(?P<s>\d+)__K(?P<K>\d+)$"
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _list_clusterings() -> List[Tuple[str, str, int, int, int]]:
    out: List[Tuple[str, str, int, int, int]] = []
    if not _KSWEEP_FLAT_ROOT.is_dir():
        return out
    for d in sorted(_KSWEEP_FLAT_ROOT.iterdir()):
        if not (d / "cluster_labels.npy").exists():
            continue
        m = _SLUG_RE.match(d.name)
        if not m:
            continue
        out.append((m["task"], m["rep"], int(m["w"]), int(m["s"]), int(m["K"])))
    return out


@st.cache_data(show_spinner=False)
def _load_mv_summary() -> List[Dict]:
    out: List[Dict] = []
    if not _KSWEEP_RESULTS_DIR.is_dir():
        return out
    for p in sorted(_KSWEEP_RESULTS_DIR.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            continue
    return out


@st.cache_data(show_spinner=False)
def _silhouette_family(task: str, rep: str, w: int, s: int) -> List[Dict]:
    from sklearn.metrics import silhouette_score

    results: List[Dict] = []
    for t, r, ww, ss, K in sorted(_list_clusterings(), key=lambda x: x[4]):
        if t != task or r != rep or ww != w or ss != s:
            continue
        cdir = _KSWEEP_FLAT_ROOT / f"{task}__{rep}__w{w}_s{s}__K{K}"
        emb_p = cdir / "embeddings_reduced.npy"
        lbl_p = cdir / "cluster_labels.npy"
        if not emb_p.exists() or not lbl_p.exists():
            continue
        try:
            emb = np.load(emb_p)
            lbl = np.load(lbl_p).astype(np.int64)
        except Exception:
            continue
        if len(emb) > 2000:
            rng = np.random.RandomState(0)
            idx = rng.choice(len(emb), size=2000, replace=False)
            emb, lbl = emb[idx], lbl[idx]
        mask = lbl >= 0
        if mask.sum() < 2 or len(np.unique(lbl[mask])) < 2:
            continue
        try:
            sil = float(silhouette_score(emb[mask], lbl[mask]))
        except Exception:
            continue
        results.append({"K": K, "silhouette": sil})
    return results


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="K selection overview — Policy Doctor",
    layout="wide",
)
st.title("K selection overview")
st.caption(
    "MV₁ (blue, left axis) · silhouette score (orange, right axis) · "
    "coverage fraction (grey dashed, left axis) across all K for every "
    "(task, rep, w, s) setting. Green star = γ-knee K* (γ=0.5, cov≥0.80, K≥5)."
)

all_clust = _list_clusterings()
if not all_clust:
    st.error(f"No clusterings found under {_KSWEEP_FLAT_ROOT}")
    st.stop()

all_tasks = sorted({t for t, *_ in all_clust})
all_reps  = sorted({r for _, r, *_ in all_clust})
all_ws    = sorted({w for _, _, w, *_ in all_clust})

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")
sel_tasks = st.sidebar.multiselect("Tasks", all_tasks, default=all_tasks)
sel_reps  = st.sidebar.multiselect("Representations", all_reps, default=all_reps)
sel_ws    = st.sidebar.multiselect("Window widths (w)", all_ws,
                                    default=[w for w in all_ws if w in (3, 5, 8)] or all_ws)
gamma     = st.sidebar.slider("γ (knee fraction of MV_asymp)", 0.05, 1.0, 0.5, 0.05)
cov_min   = st.sidebar.slider("Min coverage for knee / gate", 0.5, 1.0, 0.80, 0.05)
load_sil  = st.sidebar.toggle("Load silhouette scores", value=True,
                               help="Silhouette is computed from 50-D embeddings (2 000-sample subsample). "
                                    "First load per setting takes ~1 s; cached thereafter.")

if not sel_tasks or not sel_reps:
    st.info("Select at least one task and one representation in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Build grid: rows = (rep, w, s), cols = task
# ---------------------------------------------------------------------------
all_mv = _load_mv_summary()

# Collect all (rep, w, s) combos present in selected data
settings_set = {
    (r, w, s)
    for t, r, w, s, K in all_clust
    if t in sel_tasks and r in sel_reps and w in sel_ws
}
settings = sorted(settings_set)

if not settings:
    st.info("No data matches the current filters.")
    st.stop()

import plotly.graph_objects as go
from plotly.subplots import make_subplots

n_rows = len(settings)
n_cols = len(sel_tasks)

# Compute row heights dynamically
fig = make_subplots(
    rows=n_rows, cols=n_cols,
    subplot_titles=[
        f"{task.split('_')[0].capitalize()} — {rep}, w={w}, s={s}"
        for (rep, w, s) in settings
        for task in sel_tasks
    ],
    vertical_spacing=0.06 / max(n_rows, 1),
    horizontal_spacing=0.08 / max(n_cols, 1),
    specs=[[{"secondary_y": True}] * n_cols for _ in range(n_rows)],
)

# Index MV data: (task, rep, w, s) → sorted list of row dicts
mv_index: Dict[Tuple, List[Dict]] = defaultdict(list)
for r in all_mv:
    key = (r["task"], r["rep"], r["w"], r["s"])
    mv_index[key].append(r)
for v in mv_index.values():
    v.sort(key=lambda x: x["K"])

def _knee(family: List[Dict], gamma_v: float, cov_threshold: float) -> Optional[int]:
    gated = [r for r in family if r.get("mv1_coverage_fraction", 1.0) >= cov_threshold and r["K"] >= 5]
    if not gated:
        return None
    asymp = max(r["mv1_point"] for r in gated)
    target = gamma_v * asymp
    for r in gated:
        if r["mv1_point"] >= target:
            return r["K"]
    return gated[-1]["K"]


# Palette
_MV_COLOR   = "#60a5fa"   # blue
_SIL_COLOR  = "#fb923c"   # orange
_COV_COLOR  = "#94a3b8"   # slate grey
_KNEE_COLOR = "#22c55e"   # green

show_legend_mv  = True
show_legend_sil = True
show_legend_cov = True

for row_i, (rep, w, s) in enumerate(settings, start=1):
    for col_j, task in enumerate(sel_tasks, start=1):
        key = (task, rep, w, s)
        mv_family = mv_index.get(key, [])
        Ks_mv  = [r["K"] for r in mv_family]
        mv1    = [r["mv1_point"] for r in mv_family]
        ci_lo  = [r.get("mv1_ci_lo", r["mv1_point"]) for r in mv_family]
        ci_hi  = [r.get("mv1_ci_hi", r["mv1_point"]) for r in mv_family]
        cov    = [r.get("mv1_coverage_fraction", 1.0) for r in mv_family]

        # --- CI band ---
        if Ks_mv:
            fig.add_trace(go.Scatter(
                x=Ks_mv + Ks_mv[::-1],
                y=ci_lo + ci_hi[::-1],
                fill="toself", fillcolor=_MV_COLOR, opacity=0.13,
                line=dict(width=0), hoverinfo="skip", showlegend=False,
            ), row=row_i, col=col_j, secondary_y=False)

        # --- MV₁ line ---
        fig.add_trace(go.Scatter(
            x=Ks_mv, y=mv1, mode="lines+markers", name="MV₁",
            line=dict(color=_MV_COLOR, width=1.8),
            marker=dict(size=5, color=_MV_COLOR),
            showlegend=show_legend_mv,
            legendgroup="mv1",
            hovertemplate="K=%{x}<br>MV₁=%{y:.3f}<extra></extra>",
        ), row=row_i, col=col_j, secondary_y=False)
        show_legend_mv = False

        # --- Coverage fraction ---
        if Ks_mv:
            fig.add_trace(go.Scatter(
                x=Ks_mv, y=cov, mode="lines", name="Coverage",
                line=dict(color=_COV_COLOR, width=1.2, dash="dot"),
                showlegend=show_legend_cov,
                legendgroup="cov",
                hovertemplate="K=%{x}<br>cov₁=%{y:.2f}<extra></extra>",
            ), row=row_i, col=col_j, secondary_y=False)
            show_legend_cov = False

        # --- Silhouette ---
        if load_sil:
            sil_data = _silhouette_family(task, rep, w, s)
            Ks_sil  = [r["K"] for r in sil_data]
            sil_vals = [r["silhouette"] for r in sil_data]
            if Ks_sil:
                fig.add_trace(go.Scatter(
                    x=Ks_sil, y=sil_vals, mode="lines+markers", name="Silhouette",
                    line=dict(color=_SIL_COLOR, width=1.8),
                    marker=dict(size=5, color=_SIL_COLOR, symbol="diamond"),
                    showlegend=show_legend_sil,
                    legendgroup="sil",
                    hovertemplate="K=%{x}<br>sil=%{y:.3f}<extra></extra>",
                ), row=row_i, col=col_j, secondary_y=True)
                show_legend_sil = False

        # --- γ-knee star ---
        K_star = _knee(mv_family, gamma, cov_min)
        if K_star is not None and Ks_mv:
            mv_at_knee = next((r["mv1_point"] for r in mv_family if r["K"] == K_star), None)
            if mv_at_knee is not None:
                fig.add_trace(go.Scatter(
                    x=[K_star], y=[mv_at_knee], mode="markers",
                    marker=dict(size=12, color=_KNEE_COLOR, symbol="star",
                                line=dict(color="#052e16", width=1)),
                    name=f"γ-knee K*",
                    showlegend=(row_i == 1 and col_j == 1),
                    legendgroup="knee",
                    hovertemplate=f"K*={K_star}<br>MV₁={mv_at_knee:.3f}<extra></extra>",
                ), row=row_i, col=col_j, secondary_y=False)

        # --- Coverage gate line ---
        fig.add_hline(
            y=cov_min, line=dict(color=_COV_COLOR, width=0.8, dash="dash"),
            opacity=0.4, row=row_i, col=col_j,
        )

# Axis labels on left/bottom edges only to reduce clutter
for row_i in range(1, n_rows + 1):
    for col_j in range(1, n_cols + 1):
        axis_idx = (row_i - 1) * n_cols + col_j
        yaxis_key = f"yaxis{axis_idx if axis_idx > 1 else ''}"
        if col_j == 1:
            fig.update_layout(**{yaxis_key: dict(title="MV₁ / cov₁")})

fig.update_layout(
    height=max(300, 260 * n_rows),
    template="plotly_dark",
    margin=dict(l=50, r=50, t=60, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.01,
                xanchor="left", x=0, font=dict(size=11)),
    font=dict(size=10),
)
# Right y-axis label for silhouette (first subplot)
if load_sil:
    fig.update_layout(yaxis2=dict(title="Silhouette", showgrid=False))

st.plotly_chart(fig, use_container_width=True, key="overview_grid")

st.caption(
    f"**Blue** = MV₁ with 95% CI · **Orange** ◆ = silhouette (right axis, 2 000-sample subsample) · "
    f"**Grey ···** = coverage fraction · **Green ★** = γ-knee K* (γ={gamma:.2f}, cov≥{cov_min:.2f}). "
    "The horizontal grey dashed line marks the coverage gate. "
    "Cells with no MV data have no k-sweep JSON results yet."
)
