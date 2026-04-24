"""Runtime Monitor tab: visualize per-timestep behavior graph assignments and interventions.

Workflow:
  1. Load monitor assignments CSV (from monitor_online.py / monitor_offline.py)
  2. Configure intervention threshold (uses behavior graph V(s) from session state)
  3. Browse episodes with color-coded assignment timeline + intervention markers
  4. Play through rollout frames with node overlay (if image data is available)
  5. Explore top-K training demos linked to each intervention via influence scores
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from policy_doctor.config import VisualizerConfig


# ──────────────────────────────────────────────────────────
# Helpers: CSV loading and validation
# ──────────────────────────────────────────────────────────

_REQUIRED_COLS = {"timestep", "node_name"}
_OPTIONAL_COLS = {
    "episode", "cluster_id", "node_id", "distance", "total_ms",
    "intervention_triggered", "intervention_value", "intervention_reason",
    "env_idx",
}


def _parse_monitor_csv(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    if "episode" not in df.columns:
        df["episode"] = 0
    if "env_idx" not in df.columns:
        df["env_idx"] = 0
    df["episode"] = df["episode"].fillna(0).astype(int)
    df["env_idx"] = df["env_idx"].fillna(0).astype(int)
    df["timestep"] = df["timestep"].astype(int)
    df["node_name"] = df["node_name"].fillna("N/A").astype(str)
    if "node_id" in df.columns:
        df["node_id"] = pd.to_numeric(df["node_id"], errors="coerce").fillna(-1).astype(int)
    else:
        df["node_id"] = -1
    if "distance" in df.columns:
        df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    if "cluster_id" in df.columns:
        df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").fillna(-1).astype(int)
    return df


def _compute_interventions(
    df_ep: pd.DataFrame,
    node_values: Dict[int, float],
    threshold: float,
) -> np.ndarray:
    """Return boolean mask — True where V(node_id) < threshold."""
    mask = np.zeros(len(df_ep), dtype=bool)
    if "node_id" not in df_ep.columns:
        return mask
    for i, row in enumerate(df_ep.itertuples(index=False)):
        nid = getattr(row, "node_id", -1)
        v = node_values.get(int(nid))
        if v is not None and v < threshold:
            mask[i] = True
    return mask


def _ep_abs_sample_idx(rollout_episodes, ep_idx: int, local_t: int) -> Optional[int]:
    """Convert (episode_idx, within-episode timestep) → absolute sample index."""
    if rollout_episodes is None or ep_idx >= len(rollout_episodes):
        return None
    ep = rollout_episodes[ep_idx]
    abs_idx = ep.sample_start_idx + local_t
    if abs_idx >= ep.sample_end_idx:
        return None
    return int(abs_idx)


def _demo_abs_sample_idx(demo_episodes, ep_idx: int, local_t: int) -> Optional[int]:
    if demo_episodes is None or ep_idx >= len(demo_episodes):
        return None
    ep = demo_episodes[ep_idx]
    abs_idx = ep.sample_start_idx + local_t
    if abs_idx >= ep.sample_end_idx:
        return None
    return int(abs_idx)


# ──────────────────────────────────────────────────────────
# Node color map — persisted in session state so colors
# are stable across reruns for the same set of node names
# ──────────────────────────────────────────────────────────

def _get_node_color_map(df: pd.DataFrame) -> Dict[str, str]:
    key = "rm_node_color_map"
    if key not in st.session_state:
        st.session_state[key] = {}
    cm = st.session_state[key]
    from policy_doctor.plotting.common import get_label_color
    for name in df["node_name"].unique():
        get_label_color(str(name), cm)
    return cm


# ──────────────────────────────────────────────────────────
# Sub-renderers
# ──────────────────────────────────────────────────────────

def _suggested_csv_path(config: VisualizerConfig) -> Optional[Path]:
    """Return the most likely monitor_assignments.csv path from config.eval_dir."""
    if not config.eval_dir:
        return None
    from policy_doctor.paths import REPO_ROOT
    p = Path(config.eval_dir)
    if not p.is_absolute():
        p = REPO_ROOT / p
    candidate = p / "monitor_assignments.csv"
    return candidate


def _render_load_section(config: VisualizerConfig) -> Optional[pd.DataFrame]:
    """Section 1 — CSV upload or path input. Returns parsed DataFrame or None."""
    st.subheader("1. Load Monitor Data")

    suggested = _suggested_csv_path(config)
    default_path = st.session_state.get("rm_path", "")
    if not default_path and suggested is not None:
        default_path = str(suggested)

    col_upload, col_path = st.columns([1, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "Upload monitor CSV",
            type=["csv"],
            key="rm_upload",
            help="Output of monitor_online.py or monitor_offline.py. "
                 "Must have `timestep` and `node_name` columns. "
                 "An `episode` column is expected; if absent, all rows are treated as episode 0.",
        )
    with col_path:
        path_str = st.text_input(
            "…or path",
            value=default_path,
            key="rm_path_input",
            placeholder="/path/to/monitor_assignments.csv",
            help="Defaults to <eval_dir>/monitor_assignments.csv from the sidebar task config.",
        )
        if suggested is not None:
            if suggested.exists():
                st.caption(f"✓ found at `{suggested}`")
            else:
                st.caption(f"Not yet at `{suggested}`")

    df: Optional[pd.DataFrame] = st.session_state.get("rm_df")

    load_clicked = st.button("Load", key="rm_load_btn")

    if load_clicked:
        try:
            if uploaded is not None:
                raw = uploaded.read()
                df = _parse_monitor_csv(raw)
            elif path_str.strip():
                p = Path(path_str.strip())
                if not p.exists():
                    st.error(f"File not found: {p}")
                    return st.session_state.get("rm_df")
                df = _parse_monitor_csv(p.read_bytes())
            else:
                st.warning("Provide an uploaded file or a path.")
                return st.session_state.get("rm_df")
            st.session_state["rm_df"] = df
            st.session_state["rm_path"] = path_str.strip()
            st.session_state.pop("rm_episode", None)
            st.session_state.pop("rm_timestep", None)
            st.success(
                f"Loaded {len(df)} rows — "
                f"{df['episode'].nunique()} episode(s), "
                f"{df['node_name'].nunique()} unique nodes."
            )
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")
            return st.session_state.get("rm_df")

    if df is not None:
        n_ep = df["episode"].nunique()
        n_ts = len(df)
        n_nodes = df["node_name"].nunique()
        has_intv = "intervention_triggered" in df.columns

        cols = st.columns(4)
        cols[0].metric("Episodes", n_ep)
        cols[1].metric("Timesteps", n_ts)
        cols[2].metric("Nodes", n_nodes)
        cols[3].metric("Intervention col", "✓" if has_intv else "—")

        with st.expander("Preview (first 20 rows)", expanded=False):
            st.dataframe(df.head(20), use_container_width=True)

    return df


def _render_intervention_section(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, Optional[float]]:
    """Section 2 — Intervention settings.

    Returns:
        (global_intervention_mask, threshold_or_None)
    """
    st.subheader("2. Interventions")

    node_values: Optional[Dict[int, float]] = st.session_state.get("bg_values")
    has_csv_intv = "intervention_triggered" in df.columns

    source_options = []
    if node_values is not None:
        source_options.append("Behavior graph V(s) + threshold")
    if has_csv_intv:
        source_options.append("Column from CSV (intervention_triggered)")
    if not source_options:
        st.info(
            "No intervention data available. Build a behavior graph with computed values "
            "in the **Behavior Graph** tab (section 3, 'Compute values…'), or load a CSV "
            "that includes an `intervention_triggered` column."
        )
        return np.zeros(len(df), dtype=bool), None

    source = st.radio(
        "Intervention source",
        source_options,
        key="rm_intv_source",
        horizontal=True,
    )

    global_mask = np.zeros(len(df), dtype=bool)
    threshold: Optional[float] = None

    if "V(s)" in source and node_values is not None:
        v_list = [v for v in node_values.values() if v is not None]
        v_min = float(min(v_list)) if v_list else -1.0
        v_max = float(max(v_list)) if v_list else 1.0
        threshold = st.slider(
            "Threshold — intervene when V(node) <",
            min_value=round(v_min, 3),
            max_value=round(v_max, 3),
            value=st.session_state.get("rm_threshold", 0.0),
            step=0.01,
            key="rm_threshold",
            help="Any timestep whose assigned node has V(s) below this value is flagged.",
        )
        # Compute per-row
        nids = df["node_id"].values if "node_id" in df.columns else np.full(len(df), -1)
        for i, nid in enumerate(nids):
            v = node_values.get(int(nid))
            if v is not None and v < threshold:
                global_mask[i] = True
    else:
        # Use CSV column
        col = df["intervention_triggered"]
        global_mask = col.fillna(False).astype(bool).values

    n_intv = int(global_mask.sum())
    n_ep_with = int(df[global_mask]["episode"].nunique()) if n_intv > 0 else 0
    col1, col2 = st.columns(2)
    col1.metric("Intervention timesteps", n_intv)
    col2.metric("Episodes with interventions", n_ep_with)

    return global_mask, threshold


def _render_timeline_section(
    df: pd.DataFrame,
    global_mask: np.ndarray,
    data: Any,
) -> Tuple[int, int]:
    """Section 3 — Episode selector + timeline.

    Returns:
        (selected_episode_idx, selected_timestep)
    """
    st.subheader("3. Episode Timeline")

    mode = st.radio(
        "Data mode",
        ["Rollout", "Demo"],
        horizontal=True,
        key="rm_mode",
        help=(
            "Rollout: assignments from env rollouts (monitor_online.py / rollout PKLs). "
            "Demo: assignments from training demonstrations (monitor_offline.py --hdf5)."
        ),
    )

    episode_ids = sorted(df["episode"].unique().tolist())

    # Episode filter
    ep_filter = st.selectbox(
        "Show episodes",
        ["All", "With interventions", "Without interventions"],
        key="rm_ep_filter",
    )
    mask_df = pd.Series(global_mask, index=df.index)
    if ep_filter == "With interventions":
        ep_with_intv = set(df[mask_df]["episode"].tolist())
        episode_ids = [e for e in episode_ids if e in ep_with_intv]
    elif ep_filter == "Without interventions":
        ep_with_intv = set(df[mask_df]["episode"].tolist())
        episode_ids = [e for e in episode_ids if e not in ep_with_intv]

    if not episode_ids:
        st.warning("No episodes match the current filter.")
        return 0, 0

    n_ep = len(episode_ids)
    ep_idx_in_list = st.session_state.get("rm_episode_list_idx", 0)
    ep_idx_in_list = min(ep_idx_in_list, n_ep - 1)

    col_nav1, col_nav2, col_nav3 = st.columns([1, 3, 1])
    with col_nav1:
        if st.button("◀ Prev", key="rm_ep_prev") and ep_idx_in_list > 0:
            ep_idx_in_list -= 1
    with col_nav2:
        ep_idx_in_list = st.number_input(
            f"Episode (1–{n_ep})",
            min_value=1, max_value=n_ep,
            value=ep_idx_in_list + 1,
            step=1,
            key="rm_episode",
            label_visibility="collapsed",
        ) - 1
    with col_nav3:
        if st.button("Next ▶", key="rm_ep_next") and ep_idx_in_list < n_ep - 1:
            ep_idx_in_list += 1

    st.session_state["rm_episode_list_idx"] = int(ep_idx_in_list)
    selected_ep = episode_ids[int(ep_idx_in_list)]

    # Filter to env_idx=0 for clarity
    df_ep = df[(df["episode"] == selected_ep) & (df["env_idx"] == 0)].copy().sort_values("timestep")
    mask_ep = global_mask[df_ep.index]

    st.caption(
        f"Episode **{selected_ep}** ({len(df_ep)} timesteps, "
        f"env_idx=0{', ⚡ ' + str(int(mask_ep.sum())) + ' interventions' if mask_ep.any() else ''})"
    )

    # Check for success info
    if data is not None and mode == "Rollout":
        rollout_eps = getattr(data, "rollout_episodes", None)
        if rollout_eps is not None and selected_ep < len(rollout_eps):
            ep_info = rollout_eps[selected_ep]
            success_str = "✓ success" if getattr(ep_info, "success", None) else "✗ failure"
            st.caption(f"Outcome: {success_str}")

    # Build timeline
    node_color_map = _get_node_color_map(df)
    ts_arr = df_ep["timestep"].values
    names_arr = df_ep["node_name"].values
    dists_arr = df_ep["distance"].values if "distance" in df_ep.columns else None

    from policy_doctor.plotting.plotly.monitoring import create_monitoring_timeline

    fig_timeline = create_monitoring_timeline(
        timesteps=ts_arr,
        node_names=names_arr,
        intervention_mask=mask_ep,
        current_t=st.session_state.get("rm_timestep"),
        distances=dists_arr,
        node_color_map=node_color_map,
        height=130,
        title="",
    )
    st.plotly_chart(fig_timeline, use_container_width=True, key=f"rm_timeline_{selected_ep}")

    # Node value scatter (if values available)
    bg_values: Optional[Dict[int, float]] = st.session_state.get("bg_values")
    if bg_values is not None and "node_id" in df_ep.columns:
        vals = np.array(
            [bg_values.get(int(nid), np.nan) for nid in df_ep["node_id"].values],
            dtype=float,
        )
        from policy_doctor.plotting.plotly.monitoring import create_intervention_scatter

        fig_scatter = create_intervention_scatter(
            timesteps=ts_arr,
            node_names=names_arr,
            values=vals,
            distances=dists_arr,
            intervention_mask=mask_ep,
            node_color_map=node_color_map,
            height=180,
            title="V(s) over episode",
        )
        st.plotly_chart(fig_scatter, use_container_width=True, key=f"rm_scatter_{selected_ep}")

    # Timestep scrubber
    t_min = int(ts_arr.min()) if len(ts_arr) else 0
    t_max = int(ts_arr.max()) if len(ts_arr) else 0
    default_t = int(st.session_state.get("rm_timestep", t_min))
    default_t = max(t_min, min(t_max, default_t))

    selected_t = st.slider(
        "Timestep",
        min_value=t_min,
        max_value=max(t_min, t_max),
        value=default_t,
        step=1,
        key="rm_timestep",
    )

    return selected_ep, selected_t


def _render_frame_player(
    df: pd.DataFrame,
    selected_ep: int,
    selected_t: int,
    global_mask: np.ndarray,
    data: Any,
    mode: str,
) -> None:
    """Section 4 — Rollout or demo frame player at the selected timestep."""
    st.subheader("4. Frame Player")

    df_ep = df[(df["episode"] == selected_ep) & (df["env_idx"] == 0)].sort_values("timestep")
    row = df_ep[df_ep["timestep"] == selected_t]

    # Info overlay text
    if not row.empty:
        r = row.iloc[0]
        node_name = str(r["node_name"])
        node_id = int(r.get("node_id", -1)) if "node_id" in row.columns else -1
        dist_str = f"  dist={r['distance']:.3f}" if "distance" in row.columns else ""

        bg_values: Optional[Dict[int, float]] = st.session_state.get("bg_values")
        v_str = ""
        if bg_values is not None and node_id != -1:
            v = bg_values.get(node_id)
            if v is not None:
                v_str = f"  V={v:.3f}"

        mask_ep = global_mask[df_ep.index]
        is_intv = bool(mask_ep[df_ep["timestep"] == selected_t].values[0]) if len(mask_ep) > 0 else False
        intv_str = "  **⚡ INTERVENTION**" if is_intv else ""

        st.caption(
            f"t={selected_t}  Node: **{node_name}**{dist_str}{v_str}{intv_str}"
        )
    else:
        st.caption(f"t={selected_t}  (no assignment row found)")

    # Frame display
    frame = None
    if data is not None:
        try:
            if mode == "Rollout":
                rollout_eps = getattr(data, "rollout_episodes", None)
                abs_idx = _ep_abs_sample_idx(rollout_eps, selected_ep, selected_t)
                if abs_idx is not None:
                    frame = data.get_rollout_frame(abs_idx)
            else:
                demo_eps = getattr(data, "demo_episodes", None)
                abs_idx = _demo_abs_sample_idx(demo_eps, selected_ep, selected_t)
                if abs_idx is not None:
                    frame = data.get_demo_frame(abs_idx)
        except Exception as e:
            st.caption(f"Frame load error: {e}")

    if frame is not None:
        from policy_doctor.plotting.plotly.frames import create_annotated_frame

        label = f"t={selected_t}  {node_name if not row.empty else ''}"
        img = create_annotated_frame(np.asarray(frame), label=label, font_size=14)
        st.image(img, use_container_width=False, width=320)
    else:
        if data is None:
            st.info(
                "No frame data — load influence data via the sidebar (set eval_dir / train_dir "
                "in the task config and reload the page)."
            )
        else:
            st.info("Frame not available for this timestep (index out of range or no image dataset).")


def _render_linked_demos(
    df: pd.DataFrame,
    selected_ep: int,
    selected_t: int,
    global_mask: np.ndarray,
    data: Any,
) -> None:
    """Section 5 — Top-K influential training demos linked to the selected timestep."""
    st.subheader("5. Linked Training Demos")

    col_k, col_w, col_agg = st.columns(3)
    with col_k:
        top_k = st.number_input(
            "Top-K demos",
            min_value=1, max_value=20, value=st.session_state.get("rm_top_k", 5),
            step=1, key="rm_top_k",
        )
    with col_w:
        window = st.number_input(
            "Influence window (±t)",
            min_value=0, max_value=30, value=st.session_state.get("rm_influence_window", 3),
            step=1, key="rm_influence_window",
        )
    with col_agg:
        agg = st.selectbox(
            "Aggregation",
            ["sum", "mean", "max"],
            key="rm_agg",
        )

    if data is None:
        st.info(
            "No influence data loaded. Set eval_dir / train_dir in the sidebar task config "
            "to enable linked demo exploration."
        )
        return

    influence_matrix = getattr(data, "influence_matrix", None)
    rollout_eps = getattr(data, "rollout_episodes", None)
    demo_eps = getattr(data, "demo_episodes", None)
    demo_sample_infos = getattr(data, "demo_sample_infos", None)

    if influence_matrix is None:
        st.info("Influence matrix not available in the loaded data object.")
        return

    # Build the row window [t-window, t+window] in absolute indices
    df_ep = df[(df["episode"] == selected_ep) & (df["env_idx"] == 0)].sort_values("timestep")
    t_min_ep = int(df_ep["timestep"].min())
    t_max_ep = int(df_ep["timestep"].max())
    t_lo = max(t_min_ep, selected_t - int(window))
    t_hi = min(t_max_ep, selected_t + int(window))

    abs_indices = []
    if rollout_eps is not None and selected_ep < len(rollout_eps):
        ep_info = rollout_eps[selected_ep]
        for local_t in range(t_lo, t_hi + 1):
            abs_idx = ep_info.sample_start_idx + local_t
            if ep_info.sample_start_idx <= abs_idx < ep_info.sample_end_idx:
                abs_indices.append(abs_idx)

    if not abs_indices:
        st.warning(
            "Could not map episode timesteps to influence matrix rows. "
            "Check that rollout_episodes are correctly loaded and the episode index matches."
        )
        return

    # Slice influence matrix rows
    try:
        block = np.array(
            [influence_matrix[idx] for idx in abs_indices], dtype=np.float32
        )  # (window_size, N_demo)
    except Exception as e:
        st.warning(f"Could not read influence matrix rows: {e}")
        return

    # Aggregate
    if agg == "sum":
        scores = block.sum(axis=0)
    elif agg == "mean":
        scores = block.mean(axis=0)
    else:
        scores = block.max(axis=0)

    sorted_indices = np.argsort(scores)[::-1]
    top_indices = sorted_indices[:int(top_k)]
    top_scores = scores[top_indices]

    # Build demo labels
    demo_labels = []
    for demo_sample_idx in top_indices:
        if demo_sample_infos is not None and demo_sample_idx < len(demo_sample_infos):
            si = demo_sample_infos[demo_sample_idx]
            ep_i = getattr(si, "episode_idx", "?")
            ts_i = getattr(si, "timestep", "?")
            succ = ""
            if demo_eps is not None:
                try:
                    ep_obj = demo_eps[int(ep_i)]
                    succ = " ✓" if getattr(ep_obj, "success", False) else " ✗"
                except Exception:
                    pass
            demo_labels.append(f"ep{ep_i} t={ts_i}{succ}")
        else:
            demo_labels.append(f"sample {demo_sample_idx}")

    # Influence bar chart
    from policy_doctor.plotting.plotly.monitoring import create_demo_influence_bar

    fig_bar = create_demo_influence_bar(
        demo_indices=top_indices,
        scores=top_scores,
        demo_labels=demo_labels,
        top_k=int(top_k),
        height=max(180, 30 * int(top_k)),
        title=f"Top-{int(top_k)} demos influential at t={selected_t} ±{window}",
    )
    st.plotly_chart(fig_bar, use_container_width=True, key=f"rm_demo_bar_{selected_ep}_{selected_t}")

    # Per-demo frame players
    if data.get_demo_frame(0) is None:
        st.info("Demo frames not available (no image dataset loaded).")
        return

    st.markdown("---")
    for rank, (demo_sample_idx, score, label) in enumerate(
        zip(top_indices, top_scores, demo_labels)
    ):
        si = demo_sample_infos[demo_sample_idx] if (
            demo_sample_infos is not None and demo_sample_idx < len(demo_sample_infos)
        ) else None

        ep_i = int(getattr(si, "episode_idx", 0)) if si is not None else 0
        ts_i = int(getattr(si, "timestep", 0)) if si is not None else 0

        # Compute valid timestep range for this demo episode
        if demo_eps is not None and ep_i < len(demo_eps):
            demo_ep = demo_eps[ep_i]
            n_demo_ts = demo_ep.sample_end_idx - demo_ep.sample_start_idx
        else:
            n_demo_ts = 1

        with st.expander(
            f"#{rank + 1}  {label}  (score={score:.4f})",
            expanded=(rank < 3),
        ):
            scrubber_key = f"rm_demo_scrub_{rank}_{selected_ep}_{selected_t}"
            demo_t = st.slider(
                "Demo timestep",
                min_value=0,
                max_value=max(0, n_demo_ts - 1),
                value=min(ts_i, max(0, n_demo_ts - 1)),
                step=1,
                key=scrubber_key,
            )
            abs_demo_idx = _demo_abs_sample_idx(demo_eps, ep_i, demo_t)
            demo_frame = None
            if abs_demo_idx is not None:
                try:
                    demo_frame = data.get_demo_frame(abs_demo_idx)
                except Exception:
                    pass

            col_img, col_meta = st.columns([2, 1])
            with col_img:
                if demo_frame is not None:
                    from policy_doctor.plotting.plotly.frames import create_annotated_frame

                    ann = create_annotated_frame(
                        np.asarray(demo_frame),
                        label=f"ep{ep_i} t={demo_t}",
                        font_size=12,
                    )
                    st.image(ann, use_container_width=False, width=280)
                else:
                    st.caption("Frame unavailable")
            with col_meta:
                st.caption(f"**Demo ep {ep_i}**, t={demo_t}")
                st.caption(f"Influence score: `{score:.4f}`")
                st.caption(f"Sample idx: `{demo_sample_idx}`")
                if demo_eps is not None and ep_i < len(demo_eps):
                    ep_obj = demo_eps[ep_i]
                    succ = getattr(ep_obj, "success", None)
                    if succ is not None:
                        st.caption("Outcome: " + ("✓ success" if succ else "✗ failure"))


# ──────────────────────────────────────────────────────────
# Intervention event log
# ──────────────────────────────────────────────────────────

def _render_intervention_log(
    df: pd.DataFrame,
    global_mask: np.ndarray,
) -> None:
    """Collapsible table of all intervention events with jump links."""
    n_intv = int(global_mask.sum())
    if n_intv == 0:
        return

    with st.expander(f"Intervention event log ({n_intv} events)", expanded=False):
        intv_df = df[global_mask][["episode", "timestep", "node_name"]].copy()
        if "node_id" in df.columns:
            intv_df["node_id"] = df[global_mask]["node_id"].values
        bg_values: Optional[Dict[int, float]] = st.session_state.get("bg_values")
        if bg_values is not None and "node_id" in intv_df.columns:
            intv_df["V(s)"] = intv_df["node_id"].apply(
                lambda nid: round(bg_values.get(int(nid), float("nan")), 4)
            )
        if "distance" in df.columns:
            intv_df["distance"] = df[global_mask]["distance"].round(4).values
        st.dataframe(intv_df.reset_index(drop=True), use_container_width=True)


# ──────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────

def render_tab(
    config: VisualizerConfig,
    data: Any,
    task_config_stem: Optional[str] = None,
) -> None:
    """Render the Runtime Monitor tab."""
    st.header("Runtime Monitor")
    st.caption(
        "Visualize per-timestep behavior graph assignments and intervention triggers "
        "from saved monitor CSV files. Frame players and demo linking require image "
        "data to be loaded via the sidebar."
    )

    # ── 1. Load data ─────────────────────────────────────
    df = _render_load_section(config)
    if df is None:
        return

    st.divider()

    # ── 2. Interventions ─────────────────────────────────
    global_mask, threshold = _render_intervention_section(df)

    st.divider()

    # ── 3. Timeline ───────────────────────────────────────
    selected_ep, selected_t = _render_timeline_section(df, global_mask, data)

    # Intervention log (collapsible, between timeline and frame player)
    _render_intervention_log(df, global_mask)

    st.divider()

    # ── 4. Frame player ───────────────────────────────────
    mode = st.session_state.get("rm_mode", "Rollout")
    _render_frame_player(df, selected_ep, selected_t, global_mask, data, mode)

    st.divider()

    # ── 5. Linked demos ───────────────────────────────────
    _render_linked_demos(df, selected_ep, selected_t, global_mask, data)
