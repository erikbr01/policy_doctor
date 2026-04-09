"""Learning tab for managing data curation configurations.

This module provides:
1. Curation config selector (create/load configs)
2. Config viewer showing curated slices with video player and slice highlighting
3. Behavior slice search (reuses logic from render_behaviors/render_local_behaviors)
4. "Add to curation config" button to save search results
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import pandas as pd
import numpy as np
import streamlit as st
from scipy import stats as scipy_stats

import plotly.graph_objects as go

from influence_visualizer import plotting
from influence_visualizer.curation_config import (
    CurationConfig,
    CurationSlice,
    SelectionConfig,
    compute_dataset_fingerprint,
    create_curation_config,
    list_curation_configs,
    load_curation_config,
    load_selection_config,
    remove_selection_from_config,
    add_selection_to_config,
    save_curation_config,
    selection_source_distribution,
    selection_target_distribution,
    selection_attribution_breakdown,
)
from influence_visualizer.data_loader import InfluenceData
from influence_visualizer.render_annotation import (
    get_episode_annotations,
    get_label_for_frame,
    load_annotations,
)
from influence_visualizer.render_frames import (
    frame_player,
    render_annotated_frame,
    render_label_timeline,
)
from influence_visualizer.profiling import get_profiler, profile
from influence_visualizer.render_heatmaps import SplitType, get_split_data
from influence_visualizer.render_influences import (
    AGGREGATION_METHODS,
    _render_influence_detail,
    rank_demos_by_slice_influence,
)

# ---------------------------------------------------------------------------
# Section 1: Curation Config Selector
# ---------------------------------------------------------------------------


def _render_config_selector_fragment(
    task_config: str,
    data: InfluenceData,
    demo_split: SplitType,
    curation_mode: str = "filter",
) -> Optional[str]:
    """Select or create a curation config. Not a fragment so dropdown change triggers full rerun."""
    st.subheader("Curation Config")

    configs = list_curation_configs(task_config)

    col_select, col_create = st.columns([2, 1])

    with col_select:
        if configs:
            selected = st.selectbox(
                "Select curation config",
                options=configs,
                key=f"learning_config_select_{task_config}",
            )
            st.session_state[f"learning_active_config_{task_config}"] = selected
            # Show badge for selected config's mode
            try:
                cfg = load_curation_config(task_config, selected)
                cfg_mode = (cfg.metadata or {}).get("curation_mode", "filter")
                cfg_split = (cfg.metadata or {}).get("split", "train")
                if cfg_mode == "selection":
                    st.caption("Mode: **Curation-selection** (holdout)")
                else:
                    st.caption(f"Mode: **Curation-filtering** ({cfg_split})")
            except Exception:
                pass
        else:
            st.info("No curation configs found. Create one below.")
            st.session_state[f"learning_active_config_{task_config}"] = None

    with col_create:
        default_new_name = f"{task_config}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        new_name = st.text_input(
            "New config name",
            value=default_new_name,
            key=f"learning_new_config_name_{task_config}",
            placeholder=f"e.g. {default_new_name}",
        )
        if st.button(
            "Create Config",
            key=f"learning_create_config_{task_config}",
            disabled=not new_name,
        ):
            if new_name in configs:
                st.error(f"Config '{new_name}' already exists.")
            else:
                create_curation_config(
                    task_config, new_name, split=demo_split, mode=curation_mode
                )
                st.session_state[f"learning_active_config_{task_config}"] = new_name
                st.rerun()

    active_config = st.session_state.get(f"learning_active_config_{task_config}")

    return active_config


# ---------------------------------------------------------------------------
# Section 2: Selection distribution charts (source + target)
# ---------------------------------------------------------------------------


def _render_selection_distributions(
    task_config: str,
    selection_analyze_name: Optional[str],
    title_prefix: str = "",
    data: Optional[InfluenceData] = None,
    demo_split: SplitType = "train",
):
    """Render source (rollout) and target (demo) distribution charts for a selection config."""
    if not selection_analyze_name:
        return
    try:
        sel_config = load_selection_config(task_config, selection_analyze_name)
    except Exception:
        return
    if not sel_config.get_all_slices():
        st.caption(f"No slices in selection config '{selection_analyze_name}'.")
        return
    # Use the config's own split as authoritative source so holdout selection
    # configs are always analysed against the holdout split, regardless of the
    # UI-level curation-mode selectbox.
    effective_split: SplitType = demo_split
    config_split = (sel_config.metadata or {}).get("split")
    if config_split in ("train", "holdout", "both"):
        effective_split = config_split

    per_slice, per_rollout_ep = selection_source_distribution(sel_config)
    per_demo_ep = selection_target_distribution(sel_config)
    prefix = f"{title_prefix} " if title_prefix else ""

    key_slug = "existing" if title_prefix else "sel"
    col_src, col_tgt = st.columns(2)
    with col_src:
        if per_rollout_ep:
            rollout_eps = sorted(per_rollout_ep.keys())
            labels = [f"Rollout ep{k}" for k in rollout_eps]
            counts = [per_rollout_ep[k] for k in rollout_eps]
            fig_src = plotting.create_selection_source_distribution_plot(
                rollout_labels=labels,
                counts=counts,
                title=f"{prefix}Source: linked demo slices per rollout episode",
            )
            st.plotly_chart(
                fig_src,
                use_container_width=True,
                key=f"learning_plot_src_{task_config}_{selection_analyze_name}_{key_slug}",
            )
        else:
            st.caption("No rollout link data for source distribution.")
    with col_tgt:
        if per_demo_ep:
            demo_eps = sorted(per_demo_ep.keys(), key=lambda k: per_demo_ep[k], reverse=True)
            labels = [f"Demo ep{k}" for k in demo_eps]
            counts = [per_demo_ep[k] for k in demo_eps]
            fig_tgt = plotting.create_selection_target_distribution_plot(
                episode_labels=labels,
                counts=counts,
                title=f"{prefix}Target: selected samples per demo episode",
            )
            st.plotly_chart(
                fig_tgt,
                use_container_width=True,
                key=f"learning_plot_tgt_{task_config}_{selection_analyze_name}_{key_slug}",
            )
        else:
            st.caption("No target distribution.")
    if per_rollout_ep:
        _render_attribution_breakdown(
            sel_config,
            key_slug=f"{task_config}_{selection_analyze_name}_{key_slug}",
            data=data,
            demo_split=effective_split,
        )


def _format_selection_params(meta: Optional[Dict[str, Any]]) -> str:
    """Format saved selection_mode / n_sigma / percentile / global_top_k for display."""
    if not meta:
        return "—"
    mode = meta.get("selection_mode") or "global_top_k"
    if mode == "per_slice_n_sigma":
        n = meta.get("n_sigma")
        return f"per_slice_n_sigma σ={n}" if n is not None else "per_slice_n_sigma"
    if mode == "per_slice_percentile":
        p = meta.get("percentile")
        return f"per_slice_percentile p={p}" if p is not None else "per_slice_percentile"
    k = meta.get("global_top_k")
    return f"global_top_k k={k}" if k is not None else "global_top_k"


def _get_metadata_for_segment(
    sel_config: SelectionConfig,
    rollout_ep: int,
    ro_start: int,
    ro_end: int,
) -> Optional[Dict[str, Any]]:
    """Return selection_method_metadata from the selection that contributed the most slices to this segment."""
    best_meta: Optional[Dict[str, Any]] = None
    best_count = 0
    for sel in sel_config.selections or []:
        count = 0
        for s in sel.slices:
            if (
                s.rollout_episode_idx == rollout_ep
                and (s.rollout_start or 0) == ro_start
                and (s.rollout_end or 0) == ro_end
            ):
                count += 1
        if count > best_count and getattr(sel, "selection_method_metadata", None):
            best_count = count
            best_meta = sel.selection_method_metadata
    return best_meta


@dataclass
class SegmentScoreResult:
    """Full score data for a rollout segment: all demo samples + which are linked."""
    all_scores: np.ndarray        # scores for ALL demo samples (same as rank_demos_by_slice_influence raw_scores)
    linked_mask: np.ndarray       # boolean mask: True for samples that were linked/selected
    linked_scores: np.ndarray     # scores for only the linked samples
    seg_label: str
    meta: Optional[Dict[str, Any]]
    n_total: int                  # total demo samples
    n_linked: int                 # number linked


def _get_scores_for_segment(
    sel_config: SelectionConfig,
    rollout_ep: int,
    ro_start: int,
    ro_end: int,
    data: InfluenceData,
    demo_split: SplitType,
    aggregation_method: str = "mean",
) -> Optional[SegmentScoreResult]:
    """For one rollout segment, compute influence scores of ALL demo samples and mark which were linked.

    Uses selection_method_metadata from the selection that contributed the most to this segment
    (window_width, aggregation_method) when available; otherwise defaults.

    The effective split is determined from the selection config metadata when available
    (``sel_config.metadata["split"]``), falling back to *demo_split*.  This prevents a
    mismatch when the UI-level curation mode selectbox differs from the config's actual
    split (e.g. a holdout selection config viewed while the selectbox is on "filtering").
    """
    # Authoritative split: prefer the config's own metadata over the caller's value.
    effective_split: SplitType = demo_split
    config_split = (sel_config.metadata or {}).get("split")
    if config_split in ("train", "holdout", "both"):
        effective_split = config_split

    linked_sample_idxs: List[Optional[int]] = []
    for sel in sel_config.selections or []:
        for s in sel.slices:
            if (
                s.rollout_episode_idx == rollout_ep
                and (s.rollout_start or 0) == ro_start
                and (s.rollout_end or 0) == ro_end
            ):
                linked_sample_idxs.append(s.local_sample_idx)
    if not linked_sample_idxs:
        return None
    if rollout_ep >= len(data.rollout_episodes):
        return None
    meta = _get_metadata_for_segment(sel_config, rollout_ep, ro_start, ro_end)
    window_width = (
        int(meta["window_width"]) if (meta and meta.get("window_width") is not None) else max(1, ro_end - ro_start + 1)
    )
    agg_method = (
        str(meta["aggregation_method"]) if (meta and meta.get("aggregation_method") is not None) else aggregation_method
    )
    rollout_ep_info = data.rollout_episodes[rollout_ep]
    rollout_start_idx = rollout_ep_info.sample_start_idx + ro_start
    # ro_end is INCLUSIVE; convert to exclusive to match _run_slice_search_one
    rollout_end_idx = rollout_ep_info.sample_start_idx + ro_end + 1
    _, _, all_scores = rank_demos_by_slice_influence(
        data=data,
        rollout_start_idx=rollout_start_idx,
        rollout_end_idx=rollout_end_idx,
        window_width=window_width,
        aggregation_method=agg_method,
        split=effective_split,
        ascending=False,
    )
    # Build linked mask using stored local_sample_idx (direct index into all_scores).
    linked_mask = np.zeros(len(all_scores), dtype=bool)
    n_scores = len(all_scores)
    for idx in linked_sample_idxs:
        if idx is not None and 0 <= idx < n_scores:
            linked_mask[idx] = True
    label = f"ep{rollout_ep} [{ro_start}:{ro_end}]"
    return SegmentScoreResult(
        all_scores=all_scores,
        linked_mask=linked_mask,
        linked_scores=all_scores[linked_mask],
        seg_label=label,
        meta=meta,
        n_total=len(all_scores),
        n_linked=int(linked_mask.sum()),
    )


def _render_attribution_breakdown(
    sel_config: SelectionConfig,
    key_slug: str = "",
    data: Optional[InfluenceData] = None,
    demo_split: SplitType = "train",
) -> None:
    """Render expander with attribution breakdown: per-rollout-episode and per-selection."""
    breakdown = selection_attribution_breakdown(sel_config)
    total_slices = breakdown["total_demo_slices"]
    total_linked = breakdown["total_with_rollout_link"]
    per_rollout = breakdown["per_rollout_ep"]
    per_sel = breakdown["per_selection_rollout"]
    per_segment = breakdown.get("per_rollout_segment") or []
    per_sel_primary = breakdown.get("per_selection_primary") or []
    if not per_rollout:
        return
    with st.expander("Attribution breakdown (linked demo slices per rollout episode)", expanded=False):
        pct_linked = 100.0 * total_linked / total_slices if total_slices else 0
        st.caption(
            f"Total selected demo slices: **{total_slices}**. "
            f"With rollout link: **{total_linked}** ({pct_linked:.1f}%)."
        )
        # Bar chart: linked demo slices per rollout episode (always show a plot)
        rollout_eps = sorted({row["rollout_ep"] for row in per_rollout})
        labels = [f"Rollout ep{k}" for k in rollout_eps]
        counts = [next(row["count"] for row in per_rollout if row["rollout_ep"] == k) for k in rollout_eps]
        fig_src = plotting.create_selection_source_distribution_plot(
            rollout_labels=labels,
            counts=counts,
            title="Linked demo slices per rollout episode",
            height=320,
        )
        st.plotly_chart(fig_src, use_container_width=True, key=f"learning_attribution_bars_{key_slug}")
        # Table: rollout_ep | count | pct | distinct_rollout_segments
        df = pd.DataFrame(per_rollout)
        df = df.rename(columns={
            "rollout_ep": "Rollout ep",
            "count": "Linked demo slices",
            "pct": "% of linked",
            "distinct_rollout_segments": "Distinct rollout segments",
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
        # Per-selection: which selection contributed how many from which rollout ep
        st.markdown("**Per selection (slices with rollout link)**")
        rollout_eps = sorted({ep for row in per_rollout for ep in [row["rollout_ep"]]})
        sel_rows = []
        for s in per_sel:
            row = {"Selection": f"#{s['selection_id']}", "Label": s["label"] or ""}
            for ep in rollout_eps:
                row[f"Rollout ep{ep}"] = s["rollout_ep_counts"].get(ep, 0)
            row["Total"] = s["total_slices_with_link"]
            sel_rows.append(row)
        if sel_rows:
            st.dataframe(pd.DataFrame(sel_rows), use_container_width=True, hide_index=True)
        # Stacked bar: contributions by selection per rollout ep
        if len(per_sel) > 1 and rollout_eps:
            fig_stack = plotting.create_selection_contribution_stacked_plot(
                per_selection_rollout=per_sel,
                rollout_eps=rollout_eps,
                title="Contributions by selection per rollout episode",
            )
            if fig_stack is not None:
                st.plotly_chart(
                    fig_stack,
                    use_container_width=True,
                    key=f"learning_attribution_stacked_{key_slug}",
                )
        # Dive deeper: at which points in the selection process is one episode different?
        st.markdown("---")
        st.markdown("**Where does the bias come from?**")
        st.caption(
            "Each \"Add to curation\" is one selection. Below: which rollout slice you were viewing when you added (primary source), and which specific rollout time windows produced the most linked demo slices."
        )
        if per_sel_primary:
            st.markdown("**Primary source per selection**")
            st.caption("When you clicked Add to curation, which rollout episode and time window were you viewing? **Selection params** shows the saved mode and threshold (e.g. global_top_k k=1000 vs per_slice_n_sigma σ=3.5) that produced this selection.")
            sel_meta_by_id = {sel.id: getattr(sel, "selection_method_metadata", None) for sel in (sel_config.selections or [])}
            primary_rows = []
            for s in per_sel_primary:
                if s["primary_rollout_ep"] is None:
                    prim_src = "—"
                    prim_count = 0
                else:
                    prim_src = f"Rollout ep {s['primary_rollout_ep']} [{s['primary_start']}:{s['primary_end']}]"
                    prim_count = s["primary_count"]
                primary_rows.append({
                    "Selection": f"#{s['selection_id']}",
                    "Label": (s["label"] or "")[:24],
                    "Selection params": _format_selection_params(sel_meta_by_id.get(s["selection_id"])),
                    "Primary source": prim_src,
                    "Slices from this source": prim_count,
                    "Total linked": s["total_slices_with_link"],
                })
            st.dataframe(pd.DataFrame(primary_rows), use_container_width=True, hide_index=True)
        if per_segment:
            st.markdown("**Top rollout segments by linked demo slices**")
            st.caption("Each row is one rollout time window (episode + start:end). These are the specific points in the selection process that produced the most linked demo slices.")
            top_n = 25
            seg_df = pd.DataFrame(per_segment[:top_n])
            seg_df["Segment"] = seg_df.apply(
                lambda r: f"ep{r['rollout_ep']} [{r['rollout_start']}:{r['rollout_end']}]", axis=1
            )
            seg_display = seg_df[["Segment", "rollout_ep", "count"]].rename(
                columns={"rollout_ep": "Rollout ep", "count": "Linked demo slices"}
            )
            st.dataframe(seg_display, use_container_width=True, hide_index=True)
            # Bar chart of top segments
            seg_labels = seg_df["Segment"].tolist()
            seg_counts = seg_df["count"].tolist()
            fig_seg = plotting.create_selection_source_distribution_plot(
                rollout_labels=seg_labels,
                counts=seg_counts,
                title="Top rollout segments by linked demo slices",
                height=max(320, min(500, 28 * len(seg_labels))),
            )
            fig_seg.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_seg, use_container_width=True, key=f"learning_attribution_segments_{key_slug}")
        # Segment selector: full score distribution for ALL demo samples + linked overlay
        if data is not None and per_segment:
            st.markdown("**Score distribution for a rollout segment**")
            st.caption(
                "Select a rollout segment and click 'Compute' to see the influence-score distribution "
                "of **all** demo samples for that rollout slice. Linked (selected) demo slices are highlighted."
            )
            seg_options = [
                f"ep{s['rollout_ep']} [{s['rollout_start']}:{s['rollout_end']}] ({s['count']} slices)"
                for s in per_segment
            ]
            selected_seg_idx = st.selectbox(
                "Rollout segment",
                options=list(range(len(seg_options))),
                format_func=lambda i: seg_options[i],
                key=f"learning_attribution_segment_sel_{key_slug}",
            )
            _show_hist_key = f"learning_attribution_show_hist_{key_slug}"
            if st.button("Compute score distribution", key=f"learning_attribution_compute_hist_{key_slug}"):
                st.session_state[_show_hist_key] = True
            if selected_seg_idx is not None and 0 <= selected_seg_idx < len(per_segment) and st.session_state.get(_show_hist_key, False):
                seg = per_segment[selected_seg_idx]
                with st.spinner("Computing scores for all demo samples..."):
                    result = _get_scores_for_segment(
                        sel_config,
                        rollout_ep=seg["rollout_ep"],
                        ro_start=seg["rollout_start"],
                        ro_end=seg["rollout_end"],
                        data=data,
                        demo_split=demo_split,
                    )
                if result is not None:
                    meta_used = result.meta
                    all_scores = result.all_scores
                    linked_scores = result.linked_scores
                    n_total = result.n_total
                    n_linked = result.n_linked
                    seg_label = result.seg_label
                    # Summary
                    cap = (
                        f"**{seg_label}**: {n_linked} linked / {n_total} total demo samples. "
                        f"All scores: mean={all_scores.mean():.6f}, std={all_scores.std():.6f}. "
                        f"Rollout range (exclusive): [{seg['rollout_start']}, {seg['rollout_end'] + 1})"
                    )
                    if meta_used and (meta_used.get("window_width") is not None or meta_used.get("aggregation_method")):
                        cap += f". Saved params: window_width={meta_used.get('window_width', '?')}, aggregation={meta_used.get('aggregation_method', '?')}"
                    if meta_used:
                        params_str = _format_selection_params(meta_used)
                        if params_str != "—":
                            cap += f". **Selection params:** {params_str}"
                    st.caption(cap)
                    all_mean = float(np.mean(all_scores))
                    all_std = float(np.std(all_scores))
                    if all_std == 0:
                        all_std = 1.0
                    all_z = (all_scores - all_mean) / all_std
                    linked_z = all_z[result.linked_mask]
                    # Determine cutoff (sigma or percentile mode)
                    n_sigma_saved = None
                    percentile_saved = None
                    cutoff_raw: Optional[float] = None
                    saved_mode = meta_used.get("selection_mode") if meta_used else None
                    if saved_mode == "per_slice_n_sigma" and meta_used and meta_used.get("n_sigma") is not None:
                        n_sigma_saved = float(meta_used["n_sigma"])
                    elif saved_mode == "per_slice_percentile" and meta_used and meta_used.get("percentile") is not None:
                        percentile_saved = float(meta_used["percentile"])
                    if n_sigma_saved is not None:
                        cutoff_raw = all_mean + n_sigma_saved * all_std
                        n_above_cutoff = int((all_z >= n_sigma_saved).sum())
                        st.caption(
                            f"σ cutoff at {n_sigma_saved}: raw score ≥ {cutoff_raw:.6f}. "
                            f"Samples above cutoff: **{n_above_cutoff}** / {n_total}. "
                            f"Linked samples above cutoff: **{int((linked_z >= n_sigma_saved).sum())}** / {n_linked}."
                        )
                    elif percentile_saved is not None:
                        cutoff_raw = float(np.percentile(all_scores, percentile_saved))
                        n_above_cutoff = int((all_scores >= cutoff_raw).sum())
                        st.caption(
                            f"Empirical {percentile_saved}th percentile cutoff: raw score ≥ {cutoff_raw:.6f}. "
                            f"Samples above cutoff: **{n_above_cutoff}** / {n_total} "
                            f"(top {100 * n_above_cutoff / n_total:.2f}%). "
                            f"Linked samples above cutoff: **{int((linked_scores >= cutoff_raw).sum())}** / {n_linked}."
                        )
                    # --- Raw score histogram: all samples + linked overlay ---
                    nbins = min(80, max(20, n_total // 500))
                    not_linked_scores = all_scores[~result.linked_mask]
                    fig_raw = go.Figure()
                    fig_raw.add_trace(go.Histogram(
                        x=not_linked_scores, nbinsx=nbins, name="Not linked",
                        marker=dict(color="gray", opacity=0.5),
                    ))
                    fig_raw.add_trace(go.Histogram(
                        x=linked_scores, nbinsx=nbins, name=f"Linked ({n_linked})",
                        marker=dict(color="blue", opacity=0.8),
                    ))
                    if cutoff_raw is not None:
                        cutoff_label = (
                            f"σ={n_sigma_saved} cutoff" if n_sigma_saved is not None
                            else f"p{percentile_saved} cutoff"
                        )
                        fig_raw.add_vline(
                            x=cutoff_raw, line_dash="dash", line_color="red",
                            annotation_text=cutoff_label,
                            annotation_position="top right",
                        )
                    fig_raw.update_layout(
                        barmode="overlay",
                        title=f"Full score distribution (raw) — {seg_label}",
                        xaxis_title="Aggregated influence score",
                        yaxis_title="Number of demo samples",
                        height=400,
                    )
                    st.plotly_chart(fig_raw, use_container_width=True, key=f"learning_attribution_scores_hist_{key_slug}_{selected_seg_idx}")
                    # --- Z-score histogram: all samples + linked overlay ---
                    fig_z = go.Figure()
                    not_linked_z = all_z[~result.linked_mask]
                    fig_z.add_trace(go.Histogram(
                        x=not_linked_z, nbinsx=nbins, name="Not linked",
                        marker=dict(color="gray", opacity=0.5),
                    ))
                    fig_z.add_trace(go.Histogram(
                        x=linked_z, nbinsx=nbins, name=f"Linked ({n_linked})",
                        marker=dict(color="steelblue", opacity=0.8),
                    ))
                    if n_sigma_saved is not None:
                        fig_z.add_vline(
                            x=n_sigma_saved, line_dash="dash", line_color="red",
                            annotation_text=f"σ={n_sigma_saved}",
                            annotation_position="top right",
                        )
                    elif percentile_saved is not None:
                        z_cutoff = (cutoff_raw - all_mean) / all_std if cutoff_raw is not None else None
                        if z_cutoff is not None:
                            fig_z.add_vline(
                                x=z_cutoff, line_dash="dash", line_color="red",
                                annotation_text=f"p{percentile_saved} (z={z_cutoff:.2f})",
                                annotation_position="top right",
                            )
                    fig_z.update_layout(
                        barmode="overlay",
                        title=f"Z-score distribution (over all {n_total} samples) — {seg_label}",
                        xaxis_title="Z-score (computed over all demo samples)",
                        yaxis_title="Number of demo samples",
                        height=400,
                    )
                    st.plotly_chart(fig_z, use_container_width=True, key=f"learning_attribution_scores_z_hist_{key_slug}_{selected_seg_idx}")
                else:
                    st.caption("Could not compute scores for this segment (e.g. demo split mismatch).")


def _render_current_search_distributions(
    resolved_results: List[Dict],
    key_suffix: str = "current",
):
    """Render source and target distribution charts for current search results (would-be selection)."""
    if not resolved_results:
        return
    per_rollout_ep: Dict[int, int] = {}
    per_demo_ep: Dict[int, int] = {}
    for r in resolved_results:
        ep = r["episode"]
        demo_start = r["demo_start"]
        demo_end = r["demo_end"]
        cand = r["candidate"]
        ro_ep = cand.get("source_episode_idx", 0)
        per_rollout_ep[ro_ep] = per_rollout_ep.get(ro_ep, 0) + 1
        n = demo_end - demo_start + 1
        per_demo_ep[ep.index] = per_demo_ep.get(ep.index, 0) + n
    col_src, col_tgt = st.columns(2)
    with col_src:
        if per_rollout_ep:
            rollout_eps = sorted(per_rollout_ep.keys())
            labels = [f"Rollout ep{k}" for k in rollout_eps]
            counts = [per_rollout_ep[k] for k in rollout_eps]
            fig_src = plotting.create_selection_source_distribution_plot(
                rollout_labels=labels,
                counts=counts,
                title="Current search: linked demo slices per rollout episode",
            )
            st.plotly_chart(
                fig_src,
                use_container_width=True,
                key=f"learning_plot_src_current_{key_suffix}",
            )
    with col_tgt:
        if per_demo_ep:
            demo_eps = sorted(per_demo_ep.keys(), key=lambda k: per_demo_ep[k], reverse=True)
            labels = [f"Demo ep{k}" for k in demo_eps]
            counts = [per_demo_ep[k] for k in demo_eps]
            fig_tgt = plotting.create_selection_target_distribution_plot(
                episode_labels=labels,
                counts=counts,
                title="Current search: selected samples per demo episode",
            )
            st.plotly_chart(
                fig_tgt,
                use_container_width=True,
                key=f"learning_plot_tgt_current_{key_suffix}",
            )


# ---------------------------------------------------------------------------
# Section 3: Config Viewer with Episode Video Player
# ---------------------------------------------------------------------------


def _build_curation_timeline(
    curation_slices: List[CurationSlice],
    num_frames: int,
) -> List[Dict]:
    """Convert curation slices into annotation-style dicts for render_label_timeline.

    Args:
        curation_slices: List of CurationSlice for this episode
        num_frames: Number of frames in the episode

    Returns:
        List of dicts with 'start', 'end', 'label' keys
    """
    annotations = []
    for s in curation_slices:
        annotations.append(
            {
                "start": s.start,
                "end": s.end,
                "label": f"curated: {s.label}" if s.label else "curated",
            }
        )
    return annotations


def _compute_config_stats(
    config: CurationConfig,
    demo_split: str,
    data: InfluenceData,
) -> Tuple[int, int, Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Compute stats for a curation config: dataset share, quality label dist, slice label dist.

    Returns:
        (total_in_config, total_in_split, quality_label_counts, slice_label_counts,
         total_in_split_by_quality)
        - quality_label_counts: demo quality labels (from data.demo_quality_labels) in config
        - slice_label_counts: config slice labels (s.label) by sample count
        - total_in_split_by_quality: total samples in the split per quality label (for % curated)
    """
    total_in_config = sum(s.end - s.start + 1 for s in config.slices)
    demo_episodes = (
        data.demo_episodes if demo_split == "train" else data.holdout_episodes
    )
    # Use raw sample count (same as training) so "share" is comparable to training log
    total_in_split = sum(
        getattr(ep, "raw_length", None) or ep.num_samples for ep in demo_episodes
    )

    quality_map = data.demo_quality_labels or {}
    total_in_split_by_quality: Dict[str, int] = {}
    for ep in demo_episodes:
        label = quality_map.get(ep.index)
        if label is None or (isinstance(label, str) and not label.strip()):
            label = "(no quality label)"
        else:
            label = str(label).strip()
        ep_samples = getattr(ep, "raw_length", None) or ep.num_samples
        total_in_split_by_quality[label] = (
            total_in_split_by_quality.get(label, 0) + ep_samples
        )

    quality_label_counts: Dict[str, int] = {}
    for s in config.slices:
        label = quality_map.get(s.episode_idx)
        if label is None or (isinstance(label, str) and not label.strip()):
            label = "(no quality label)"
        else:
            label = str(label).strip()
        n = s.end - s.start + 1
        quality_label_counts[label] = quality_label_counts.get(label, 0) + n

    slice_label_counts: Dict[str, int] = {}
    for s in config.slices:
        label = s.label.strip() if s.label else "(no label)"
        n = s.end - s.start + 1
        slice_label_counts[label] = slice_label_counts.get(label, 0) + n

    return (
        total_in_config,
        total_in_split,
        quality_label_counts,
        slice_label_counts,
        total_in_split_by_quality,
    )


def _selection_overlap_matrix(
    sel_config: SelectionConfig,
) -> Tuple[np.ndarray, List[str]]:
    """Compute sample-overlap matrix between each pair of selections.

    Returns:
        overlap_matrix: (n, n) array; entry (i, j) = number of (episode_idx, timestep)
            samples in both selection i and selection j.
        labels: List of selection labels (each selection's label, or "Sel #id" if empty).
    """
    selections = sel_config.selections or []
    if not selections:
        return np.zeros((0, 0), dtype=np.int64), []
    sample_sets: List[Set[Tuple[int, int]]] = []
    for sel in selections:
        s: Set[Tuple[int, int]] = set()
        for sl in sel.slices:
            for t in range(sl.start, sl.end + 1):
                s.add((sl.episode_idx, t))
        sample_sets.append(s)
    n = len(selections)
    matrix = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            matrix[i, j] = len(sample_sets[i] & sample_sets[j])
    labels = [sel.label or f"Sel #{sel.id}" for sel in selections]
    return matrix, labels


def _selection_quality_distributions(
    sel_config: SelectionConfig,
    quality_map: Dict[int, str],
) -> List[Tuple[str, Dict[str, int]]]:
    """Compute demo quality label distribution per selection (deduplicated samples).

    Each (episode_idx, timestep) is counted at most once per selection, so overlapping
    slices do not double-count.

    Returns:
        List of (selection_display_label, quality_label -> unique_sample_count).
    """
    selections = sel_config.selections or []
    out: List[Tuple[str, Dict[str, int]]] = []
    for sel in selections:
        # Unique (episode_idx, timestep) in this selection
        samples: Set[Tuple[int, int]] = set()
        for sl in sel.slices:
            for t in range(sl.start, sl.end + 1):
                samples.add((sl.episode_idx, t))
        counts: Dict[str, int] = {}
        for ep_idx, _ in samples:
            label = quality_map.get(ep_idx)
            if label is None or (isinstance(label, str) and not label.strip()):
                label = "(no quality label)"
            else:
                label = str(label).strip()
            counts[label] = counts.get(label, 0) + 1
        sel_label = sel.label or f"Sel #{sel.id}"
        out.append((sel_label, counts))
    return out


def _selection_episode_distributions(
    sel_config: SelectionConfig,
    quality_map: Dict[int, str],
    demo_episodes: List[Any],
) -> List[Tuple[str, List[str], List[int], List[int], List[str]]]:
    """Compute deduplicated sample count per episode, per selection.

    Returns:
        List of (selection_display_label, episode_labels, counts, episode_totals,
                 episode_quality_labels) for use with create_curation_episode_histogram.
    """
    ep_len_map = {
        ep.index: getattr(ep, "raw_length", None) or ep.num_samples
        for ep in demo_episodes
    }
    out: List[Tuple[str, List[str], List[int], List[int], List[str]]] = []
    for sel in sel_config.selections or []:
        samples: Set[Tuple[int, int]] = set()
        for sl in sel.slices:
            for t in range(sl.start, sl.end + 1):
                samples.add((sl.episode_idx, t))
        ep_counts: Dict[int, int] = {}
        for ep_idx, _ in samples:
            ep_counts[ep_idx] = ep_counts.get(ep_idx, 0) + 1
        sorted_eps = sorted(ep_counts.keys(), key=lambda e: ep_counts[e], reverse=True)
        episode_labels = [f"Episode {e}" for e in sorted_eps]
        counts = [ep_counts[e] for e in sorted_eps]
        episode_totals = [ep_len_map.get(e, 0) for e in sorted_eps]
        ep_qualities = []
        for e in sorted_eps:
            q = quality_map.get(e)
            if q is None or (isinstance(q, str) and not q.strip()):
                ep_qualities.append("(no quality label)")
            else:
                ep_qualities.append(str(q).strip())
        sel_label = sel.label or f"Sel #{sel.id}"
        out.append((sel_label, episode_labels, counts, episode_totals, ep_qualities))
    return out


@st.fragment
def _render_config_viewer_fragment(
    data: InfluenceData,
    task_config: str,
    config_name: str,
    obs_key: str,
    demo_split: SplitType,
    annotation_file: str,
):
    """Fragment showing the contents of a curation config with episode video players."""
    # Determine the correct split from the config's own metadata so that
    # selection configs always compute scores/linked masks over holdout, even if
    # the UI-level selectbox happens to be set to "curation-filtering" (train).
    try:
        _cfg_meta = load_curation_config(task_config, config_name).metadata or {}
    except FileNotFoundError:
        _cfg_meta = {}
    config_demo_split: SplitType = _cfg_meta.get("split", demo_split)

    # Selection distributions for this config (selections live in same file)
    st.markdown("**Selection distributions**")
    _render_selection_distributions(
        task_config, config_name, title_prefix="", data=data, demo_split=config_demo_split
    )
    st.divider()

    # View / Edit Selections (list and remove)
    with st.expander("View Selections", expanded=True):
        try:
            sel_config = load_selection_config(task_config, config_name)
        except Exception:
            sel_config = SelectionConfig()
        if not sel_config.selections:
            st.caption(
                f"No selections yet for '{config_name}'. "
                "Add results from Behavior Slice Search below."
            )
        else:
            for sel in sel_config.selections:
                n_slices = len(sel.slices)
                n_samples = sum(s.end - s.start + 1 for s in sel.slices)
                created = sel.created or ""
                col_info, col_btn = st.columns([4, 1])
                with col_info:
                    st.caption(
                        f"Selection #{sel.id}: label '{sel.label}', "
                        f"{n_slices} slices, {n_samples} samples — {created}"
                    )
                with col_btn:
                    if st.button(
                        "Remove",
                        key=f"learning_remove_selection_{task_config}_{config_name}_{sel.id}",
                    ):
                        episode_ends = None
                        if hasattr(data, "demo_dataset") and hasattr(data.demo_dataset, "replay_buffer"):
                            episode_ends = data.demo_dataset.replay_buffer.episode_ends[:]
                        remove_selection_from_config(
                            task_config, config_name, sel.id,
                            episode_ends=episode_ends,
                        )
                        st.rerun()

            st.markdown("**Selection overlap (IoU)**")
            st.caption(
                "Cell (i, j) = intersection over union of selection i and j (symmetric). "
                "Hover shows IoU % and intersection in samples."
            )
            overlap_matrix, overlap_labels = _selection_overlap_matrix(sel_config)
            fig_overlap = plotting.create_selection_overlap_heatmap(
                overlap_matrix, overlap_labels, show_iou=True
            )
            st.plotly_chart(
                fig_overlap,
                use_container_width=True,
                key=f"learning_plot_overlap_{task_config}_{config_name}",
            )

            # Per-selection demo quality distribution
            st.markdown("**Demo quality distribution per selection**")
            quality_map = data.demo_quality_labels or {}
            per_sel_quality = _selection_quality_distributions(sel_config, quality_map)
            if per_sel_quality:
                sel_labels = [x[0] for x in per_sel_quality]
                quality_counts = [x[1] for x in per_sel_quality]
                # Total training samples per quality (for hover "% of all X samples in this quality")
                demo_episodes = (
                    data.demo_episodes
                    if demo_split == "train"
                    else data.holdout_episodes
                )
                total_samples_by_quality: Dict[str, int] = {}
                for ep in demo_episodes:
                    label = quality_map.get(ep.index)
                    if label is None or (
                        isinstance(label, str) and not label.strip()
                    ):
                        label = "(no quality label)"
                    else:
                        label = str(label).strip()
                    ep_samples = getattr(ep, "raw_length", None) or ep.num_samples
                    total_samples_by_quality[label] = (
                        total_samples_by_quality.get(label, 0) + ep_samples
                    )
                fig_qual = plotting.create_selection_quality_distributions_plot(
                    selection_labels=sel_labels,
                    per_selection_quality_counts=quality_counts,
                    title="Demo quality distribution per selection",
                    total_samples_by_quality=total_samples_by_quality,
                )
                st.plotly_chart(
                    fig_qual,
                    use_container_width=True,
                    key=f"learning_plot_qual_{task_config}_{config_name}",
                )
            else:
                st.caption("No selections to show quality distribution for.")

            # Per-selection samples by episode (stacked bar by quality, like main curation chart)
            st.markdown("**Samples by episode (per selection)**")
            demo_episodes_list = (
                data.demo_episodes if demo_split == "train" else data.holdout_episodes
            )
            per_sel_ep = _selection_episode_distributions(
                sel_config, quality_map, demo_episodes_list
            )
            for idx, (sel_label, ep_labels, counts, ep_totals, ep_qualities) in enumerate(per_sel_ep):
                fig_sel = plotting.create_curation_episode_histogram(
                    episode_labels=ep_labels,
                    counts=counts,
                    episode_totals=ep_totals,
                    title=f"Samples by episode — {sel_label}",
                    episode_quality_labels=ep_qualities,
                )
                st.plotly_chart(
                    fig_sel,
                    use_container_width=True,
                    key=f"learning_plot_sel_ep_{task_config}_{config_name}_{idx}",
                )

    with st.expander("View Curation Config", expanded=True):
        try:
            config = load_curation_config(task_config, config_name)
        except FileNotFoundError:
            st.error(f"Config '{config_name}' not found.")
            return

        if not config.slices:
            st.info(
                "This curation config is empty. "
                "Add selections via Behavior Slice Search below; slices are updated automatically when you add or remove selections."
            )
            return

        # Use config's split from metadata so selection configs show holdout, filter configs show train
        display_split: str = (config.metadata or {}).get("split", "train")
        is_selection_config = (config.metadata or {}).get("curation_mode") == "selection"
        if is_selection_config:
            st.caption("**Curation-selection**: these slices from holdout are added to training.")
        else:
            st.caption("**Curation-filtering**: these slices are excluded from training.")

        # Validate against data
        demo_episodes = (
            data.demo_episodes if display_split == "train" else data.holdout_episodes
        )
        errors = config.validate_against_data(demo_episodes)
        if errors:
            st.error("Validation errors found:")
            for err in errors:
                st.markdown(f"- {err}")

        # Summary
        num_slices = len(config.slices)
        num_episodes = len(config.get_unique_episode_indices())
        total_samples = sum(s.end - s.start + 1 for s in config.slices)
        action_word = "selected" if is_selection_config else "excluded"
        st.markdown(
            f"**{num_slices}** slices across **{num_episodes}** episodes, "
            f"**{total_samples}** total samples {action_word}"
        )

        # Load annotations for episode viewer (used below)
        annotations = load_annotations(annotation_file, task_config=task_config)

        # Config stats: % of dataset, quality label dist, slice (behavioral) label dist
        (
            total_in_config,
            total_in_split,
            quality_label_counts,
            slice_label_counts,
            total_in_split_by_quality,
        ) = _compute_config_stats(config, display_split, data)

        if total_in_split > 0:
            pct = 100.0 * total_in_config / total_in_split
            st.metric(
                f"Share of {display_split} split (samples)",
                f"{pct:.1f}%",
                help=(
                    f"{total_in_config} of {total_in_split} samples in the {display_split} split. "
                    + (
                        "At training, these selected holdout samples are added to the training set."
                        if is_selection_config
                        else "At training, the mask is applied to the full buffer (train+val+holdout), "
                        "so the training log may show a lower % (excluded / full buffer)."
                    )
                ),
            )

        # Percentage of each quality label (in the split) that is curated
        if total_in_split_by_quality:
            st.markdown(
                "**% of each quality label "
                + ("selected" if is_selection_config else "curated out")
                + "**"
            )
            for q in sorted(total_in_split_by_quality.keys()):
                total_q = total_in_split_by_quality[q]
                in_config_q = quality_label_counts.get(q, 0)
                if total_q > 0:
                    pct_q = 100.0 * in_config_q / total_q
                    st.caption(
                        f"**{q}:** {pct_q:.1f}% ({in_config_q} of {total_q} samples in this quality)"
                    )

        # Histogram: samples marked for curation per episode (sorted by count, biggest left)
        ep_indices = config.get_unique_episode_indices()
        if ep_indices:
            ep_data = []
            for ep_idx in ep_indices:
                slices_in_ep = config.get_slices_for_episode(ep_idx)
                num_curated = sum(s.end - s.start + 1 for s in slices_in_ep)
                ep_total = config.episode_lengths.get(ep_idx, 0)
                ep_data.append((ep_idx, num_curated, ep_total))
            ep_data.sort(key=lambda x: x[1], reverse=True)
            ep_labels = [f"Episode {x[0]}" for x in ep_data]
            ep_counts = [x[1] for x in ep_data]
            ep_totals = [x[2] for x in ep_data]
            quality_map = data.demo_quality_labels or {}
            ep_qualities = []
            ep_slice_breakdowns = []
            for x in ep_data:
                q = quality_map.get(x[0])
                if q is None or (isinstance(q, str) and not q.strip()):
                    ep_qualities.append("(no quality label)")
                else:
                    ep_qualities.append(str(q).strip())
                # Per-episode slice label -> sample count for hover
                breakdown: Dict[str, int] = {}
                for s in config.get_slices_for_episode(x[0]):
                    lbl = s.label.strip() if s.label else "(no label)"
                    n = s.end - s.start + 1
                    breakdown[lbl] = breakdown.get(lbl, 0) + n
                ep_slice_breakdowns.append(breakdown)
            fig_hist = plotting.create_curation_episode_histogram(
                episode_labels=ep_labels,
                counts=ep_counts,
                episode_totals=ep_totals,
                title=(
                    "Samples selected by episode"
                    if is_selection_config
                    else "Samples marked for curation by episode"
                ),
                episode_quality_labels=ep_qualities,
                episode_slice_label_breakdown=ep_slice_breakdowns,
            )
            st.plotly_chart(
                fig_hist,
                use_container_width=True,
                key=f"learning_plot_hist_{task_config}_{config_name}",
            )

        # Side-by-side: quality labels of demos (slice label dist) | current slice labels (behavioral)
        col_quality, col_behavior = st.columns(2)
        with col_quality:
            if quality_label_counts:
                labels_sorted = sorted(quality_label_counts.keys())
                fig_quality = plotting.create_behavior_pie_chart(
                    labels=labels_sorted,
                    values=[quality_label_counts[k] for k in labels_sorted],
                    title="Slice label distribution (demo quality labels)",
                )
                st.plotly_chart(
                    fig_quality,
                    use_container_width=True,
                    key=f"learning_plot_quality_{task_config}_{config_name}",
                )
            else:
                st.caption("No quality label data for curated samples.")
        with col_behavior:
            if slice_label_counts:
                labels_sorted = sorted(slice_label_counts.keys())
                total_n = sum(slice_label_counts.values())
                fig_slice = plotting.create_behavior_pie_chart(
                    labels=labels_sorted,
                    values=[slice_label_counts[k] for k in labels_sorted],
                    title=f"Behavioral labels (slice labels, {total_n} samples)",
                )
                st.plotly_chart(
                    fig_slice,
                    use_container_width=True,
                    key=f"learning_plot_behavior_{task_config}_{config_name}",
                )
            else:
                st.caption("No slice labels.")

        # Group slices by episode for display
        unique_eps = config.get_unique_episode_indices()

        # Episode selector
        if not unique_eps:
            return

        selected_ep_idx = st.selectbox(
            "Select episode to view",
            options=unique_eps,
            format_func=lambda x: f"Episode {x} ({len(config.get_slices_for_episode(x))} slices)",
            key=f"learning_view_ep_{task_config}_{config_name}",
        )

        if selected_ep_idx is not None:
            _render_episode_with_curation_highlight(
                data=data,
                episode_idx=selected_ep_idx,
                curation_config=config,
                obs_key=obs_key,
                demo_split=cast(SplitType, display_split),
                annotations=annotations,
                task_config=task_config,
                config_name=config_name,
            )


def _render_episode_with_curation_highlight(
    data: InfluenceData,
    episode_idx: int,
    curation_config: CurationConfig,
    obs_key: str,
    demo_split: SplitType,
    annotations: Dict,
    task_config: str,
    config_name: str,
):
    """Render an episode with video player, highlighting curated slices."""
    # Find the episode
    demo_episodes = (
        data.demo_episodes if demo_split == "train" else data.holdout_episodes
    )
    episode = None
    for ep in demo_episodes:
        if ep.index == episode_idx:
            episode = ep
            break

    if episode is None:
        st.warning(f"Episode {episode_idx} not found in {demo_split} split.")
        return

    # Get curation slices for this episode
    curation_slices = curation_config.get_slices_for_episode(episode_idx)
    curation_timeline = _build_curation_timeline(curation_slices, episode.num_samples)

    # Get behavior annotations for this episode
    ep_annotations = get_episode_annotations(
        annotations,
        str(episode_idx),
        split=demo_split,
    )

    # Show slice details
    st.markdown(f"**Episode {episode_idx}** — {episode.num_samples} samples")
    for i, s in enumerate(curation_slices):
        label_str = f" ({s.label})" if s.label else ""
        st.markdown(
            f"  - Slice {i + 1}: samples {s.start}–{s.end}{label_str} "
            f"({s.end - s.start + 1} samples), source: {s.source}"
        )

    # Check if frame is within a curated slice
    def _is_in_curation_slice(frame_idx: int) -> bool:
        for s in curation_slices:
            if s.start <= frame_idx <= s.end:
                return True
        return False

    # Render function for frame player
    # episode.sample_start_idx is always the global index in all_demo_sample_infos (train and holdout)
    def _render_frame(sample_offset):
        abs_idx = episode.sample_start_idx + sample_offset
        frame = data.get_demo_frame(abs_idx, obs_key=obs_key)

        # Build caption
        in_curation = _is_in_curation_slice(sample_offset)
        caption = f"sample {sample_offset}"
        if in_curation:
            caption += " [IN CURATION CONFIG]"

        # Get behavior label if available
        behavior_label = get_label_for_frame(sample_offset, ep_annotations)

        label_text = f"sample {sample_offset}"
        if behavior_label:
            label_text += f" | {behavior_label}"
        if in_curation:
            label_text += " | CURATED"

        render_annotated_frame(frame, label_text, caption=caption)

        # Show curation timeline
        render_label_timeline(
            curation_timeline,
            num_frames=episode.num_samples,
            current_frame=sample_offset,
            unique_key=f"curation_timeline_{task_config}_{config_name}_{episode_idx}",
        )

        # Show behavior annotation timeline if available
        if ep_annotations:
            render_label_timeline(
                ep_annotations,
                num_frames=episode.num_samples,
                current_frame=sample_offset,
                unique_key=f"behavior_timeline_{task_config}_{config_name}_{episode_idx}",
            )

    frame_player(
        label="Sample (index in episode):",
        min_value=0,
        max_value=episode.num_samples - 1,
        key=f"learning_ep_player_{task_config}_{config_name}_{episode_idx}",
        default_value=0,
        default_fps=3.0,
        render_fn=_render_frame,
        fragment_scope=True,
    )


# ---------------------------------------------------------------------------
# Section 3: Behavior Slice Search (reuses render_local_behaviors logic)
# ---------------------------------------------------------------------------


def _render_advantage_based_curation(
    data: InfluenceData,
    demo_split: SplitType,
    task_config: str,
    obs_key: str,
    active_config_name: Optional[str],
    annotation_file: str,
    is_selection_mode: bool = False,
) -> None:
    """Render advantage-based curation: filter = low-advantage edges (exclude); selection = high-advantage edges (include)."""
    from influence_visualizer.behavior_value_loader import (
        get_behavior_graph_and_slice_values,
        slice_indices_to_rollout_slices,
    )
    from influence_visualizer.clustering_results import (
        list_clustering_results,
        load_clustering_result,
    )

    if not task_config:
        st.info("Select a task config to use advantage-based curation.")
        return

    saved_names = list_clustering_results(task_config)
    if not saved_names:
        st.info(
            "No clustering results found. Run clustering in the **Cluster Generation** tab "
            "(Clustering Algorithms), save a result, then return here."
        )
        return

    selected_name = st.selectbox(
        "Clustering result",
        options=saved_names,
        key=f"learning_adv_clustering_{demo_split}",
        help="Saved clustering result to use for value/advantage computation.",
    )
    try:
        cluster_labels, metadata, _ = load_clustering_result(task_config, selected_name)
    except Exception as e:
        st.error(f"Failed to load clustering result: {e}")
        return

    st.subheader("Value parameters")
    col_gamma, col_rs, col_rf, col_re = st.columns(4)
    with col_gamma:
        gamma = st.number_input(
            "Gamma (γ)",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.05,
            key=f"learning_adv_gamma_{demo_split}",
        )
    with col_rs:
        reward_success = st.number_input(
            "R(SUCCESS)",
            value=1.0,
            key=f"learning_adv_rs_{demo_split}",
        )
    with col_rf:
        reward_failure = st.number_input(
            "R(FAILURE)",
            value=-1.0,
            key=f"learning_adv_rf_{demo_split}",
        )
    with col_re:
        reward_end = st.number_input(
            "R(END)",
            value=0.0,
            key=f"learning_adv_re_{demo_split}",
        )

    graph, values, q_values, advantages = get_behavior_graph_and_slice_values(
        cluster_labels,
        metadata,
        gamma=gamma,
        reward_success=reward_success,
        reward_failure=reward_failure,
        reward_end=reward_end,
    )

    # Value V(s) per slice (state value of the cluster for each transition)
    value_per_slice = np.full(len(cluster_labels), np.nan, dtype=np.float64)
    for node_id, v in values.items():
        value_per_slice[cluster_labels == node_id] = v

    st.subheader("Behavior graph (value-colored)")
    criterion = st.radio(
        "Criterion",
        options=["advantage", "value"],
        format_func=lambda x: "Advantage A(s,a) (edge)" if x == "advantage" else "Value V(s) (state)",
        key=f"learning_adv_criterion_{demo_split}",
        horizontal=True,
        help="Select rollout slices by advantage (transition value) or by behavior state value.",
    )
    use_advantage = criterion == "advantage"

    if use_advantage:
        if is_selection_mode:
            highlight_label = "Highlight edges to select"
            threshold_label = "Advantage threshold (select slices with A above)"
            threshold_help = "Rollout slices on edges with advantage above this value are selected to add to training."
            threshold_default = 0.3
        else:
            highlight_label = "Highlight edges to curate out"
            threshold_label = "Advantage threshold (curate out slices with A below)"
            threshold_help = "Rollout slices on edges with advantage below this value are selected for curation (excluded from training)."
            threshold_default = -0.3
    else:
        if is_selection_mode:
            highlight_label = "Highlight nodes to select"
            threshold_label = "Value threshold (select slices in states with V above)"
            threshold_help = "Rollout slices in behavior states with value above this are selected to add to training."
            threshold_default = 0.0
        else:
            highlight_label = "Highlight nodes to curate out"
            threshold_label = "Value threshold (curate out slices in states with V below)"
            threshold_help = "Rollout slices in behavior states with value below this are selected for curation (excluded from training)."
            threshold_default = -0.2

    highlight_curate = st.checkbox(
        highlight_label,
        value=True,
        key=f"learning_highlight_curate_edges_{demo_split}",
    )
    threshold = st.number_input(
        threshold_label,
        value=threshold_default,
        step=0.1,
        key=f"learning_advantage_threshold_{demo_split}",
        help=threshold_help,
    )

    if use_advantage:
        highlight_edges_above = threshold if highlight_curate and is_selection_mode else None
        highlight_edges_below = threshold if highlight_curate and not is_selection_mode else None
        highlight_nodes_above = None
        highlight_nodes_below = None
    else:
        highlight_edges_above = None
        highlight_edges_below = None
        highlight_nodes_above = threshold if highlight_curate and is_selection_mode else None
        highlight_nodes_below = threshold if highlight_curate and not is_selection_mode else None

    html = plotting.create_value_colored_interactive_graph(
        graph,
        values,
        gamma=gamma,
        highlight_edges_below_advantage=highlight_edges_below,
        highlight_edges_above_advantage=highlight_edges_above,
        highlight_nodes_below_value=highlight_nodes_below,
        highlight_nodes_above_value=highlight_nodes_above,
    )
    st.components.v1.html(html, height=650, scrolling=False)

    # Distribution and mask by criterion
    valid = cluster_labels != -1
    if use_advantage:
        dist_values = advantages[valid]
        dist_title = "Advantage distribution (all rollout transitions)"
        dist_x = "Advantage"
        vline_label = "Threshold (select above)" if is_selection_mode else "Threshold (curate out below)"
    else:
        dist_values = value_per_slice[valid]
        dist_values = dist_values[~np.isnan(dist_values)]
        dist_title = "Value V(s) distribution (all rollout transitions)"
        dist_x = "Value V(s)"
        vline_label = "Threshold (select above)" if is_selection_mode else "Threshold (curate out below)"

    if len(dist_values) > 0:
        fig_dist = plotting.create_histogram(
            dist_values,
            title=dist_title,
            xaxis_title=dist_x,
            yaxis_title="Count",
            nbins=40,
            vline_at=threshold,
            vline_label=vline_label,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    if use_advantage:
        if is_selection_mode:
            mask = valid & (advantages >= threshold)
            metric_title = "Rollout slices selected (A above threshold)"
            no_slices_msg = "No rollout slices with advantage above threshold. Adjust the threshold."
        else:
            mask = valid & (advantages < threshold)
            metric_title = "Rollout slices selected (A below threshold)"
            no_slices_msg = "No rollout slices with advantage below threshold. Adjust the threshold."
    else:
        if is_selection_mode:
            mask = valid & (~np.isnan(value_per_slice)) & (value_per_slice >= threshold)
            metric_title = "Rollout slices selected (V above threshold)"
            no_slices_msg = "No rollout slices in states with value above threshold. Adjust the threshold."
        else:
            mask = valid & (~np.isnan(value_per_slice)) & (value_per_slice < threshold)
            metric_title = "Rollout slices selected (V below threshold)"
            no_slices_msg = "No rollout slices in states with value below threshold. Adjust the threshold."
    indices = np.where(mask)[0]
    rollout_slices = slice_indices_to_rollout_slices(
        metadata, data, cluster_labels, indices
    )
    total_rollout_slices = (cluster_labels != -1).sum()
    pct = (
        100.0 * len(rollout_slices) / total_rollout_slices
        if total_rollout_slices else 0.0
    )
    st.metric(
        metric_title,
        f"{len(rollout_slices):,} ({pct:.1f}%)",
    )
    if total_rollout_slices:
        st.caption(f"Of {total_rollout_slices:,} rollout slices (assigned to clusters).")
    if not rollout_slices:
        st.warning(no_slices_msg)
        return

    st.divider()
    st.subheader(f"Demo selection from **{demo_split}** (top-k or percentile)")
    influence_matrix, demo_episodes, ep_idxs, _ = get_split_data(data, demo_split)
    window_width = st.number_input(
        "Window width",
        min_value=1,
        max_value=50,
        value=5,
        key=f"learning_adv_window_{demo_split}",
    )
    agg_options = list(AGGREGATION_METHODS.keys())
    agg_default_idx = agg_options.index("mean") if "mean" in agg_options else 0
    aggregation_method = st.selectbox(
        "Aggregation method",
        options=agg_options,
        index=agg_default_idx,
        key=f"learning_adv_agg_{demo_split}",
    )
    selection_mode = st.selectbox(
        "Selection mode",
        options=["global_top_k", "per_slice_n_sigma"],
        format_func=lambda x: {
            "global_top_k": "Global top-k",
            "per_slice_n_sigma": "Per-slice threshold",
        }[x],
        key=f"learning_adv_selection_mode_{demo_split}",
    )
    per_slice_top_k = 20
    global_top_k_input = 10
    ascending = False
    n_sigma_input = None
    percentile_input: Optional[float] = None
    threshold_type = "sigma"
    if selection_mode == "global_top_k":
        col_per_k, col_global_k, col_asc = st.columns(3)
        with col_per_k:
            per_slice_top_k = st.number_input(
                "Per-Slice Top K",
                min_value=1,
                value=20,
                key=f"learning_adv_per_slice_topk_{demo_split}",
            )
        with col_global_k:
            global_top_k_input = st.number_input(
                "Global Top K",
                min_value=1,
                value=10,
                key=f"learning_adv_global_topk_{demo_split}",
            )
        with col_asc:
            if is_selection_mode:
                ascending_help = (
                    "Leave unchecked (default) to select demos with the HIGHEST positive "
                    "influence on the selected (good) rollout transitions — these are the demos "
                    "most valuable to add. Checking this selects demos with the LOWEST influence."
                )
            else:
                ascending_help = (
                    "Leave unchecked (default) to select demos with the HIGHEST positive "
                    "influence on bad rollout transitions — these are the demos most responsible "
                    "for the bad behavior and should be excluded. "
                    "Checking this selects demos with the LOWEST influence, which is the "
                    "opposite of the intended exclusion logic."
                )
            ascending = st.checkbox(
                "Sort ascending (lowest influence)",
                value=False,
                key=f"learning_adv_ascending_{demo_split}",
                help=ascending_help,
            )
    else:
        threshold_type = st.radio(
            "Threshold by",
            options=["sigma", "percentile"],
            format_func=lambda x: "Sigma (σ)" if x == "sigma" else "Percentile (empirical)",
            key=f"learning_adv_threshold_type_{demo_split}",
            horizontal=True,
        )
        if threshold_type == "sigma":
            n_sigma_input = st.number_input(
                "n (sigma)",
                min_value=0.0,
                value=2.0,
                step=0.5,
                key=f"learning_adv_n_sigma_{demo_split}",
            )
            pct = _sigma_to_percentile(n_sigma_input)
            st.caption(f"≈ {pct:.1f}th percentile (top {100 - pct:.1f}%) assuming normal")
        else:
            percentile_input = st.number_input(
                "Percentile",
                min_value=50.0,
                max_value=99.99,
                value=97.7,
                step=0.5,
                key=f"learning_adv_percentile_{demo_split}",
                help="Keep demo slices at or above this empirical percentile within each slice. Robust to heavy-tailed distributions.",
            )
            st.caption(f"Top {100 - percentile_input:.2f}% of scores per slice (empirical)")
    selection_normalization = st.selectbox(
        "Score normalization for selection",
        options=["none", "scale-std", "scale-range", "z-score", "min-max"],
        format_func=lambda x: {
            "none": "None (raw scores)",
            "scale-std": "Scale by std (per slice)",
            "scale-range": "Divide by range (per slice)",
            "z-score": "Z-score (per slice)",
            "min-max": "Min-max → [0,1] (per slice)",
        }[x],
        key=f"learning_adv_norm_{demo_split}",
    )
    remove_negative_influence = st.checkbox(
        "Remove pairs with negative (unnormalized) influence",
        value=False,
        key=f"learning_adv_remove_neg_{demo_split}",
    )

    run_key = f"learning_adv_run_search_{demo_split}"
    show_search_key = f"learning_adv_show_search_{demo_split}"
    # Include all search-defining parameters in the cache key so cached results are
    # automatically invalidated when the threshold, clustering result, or selection mode changes.
    # selection_mode must be in the key because use_all_demos_per_slice depends on it.
    _adv_params = f"{selected_name}_{criterion}_{round(threshold, 8)}_{demo_split}_{selection_mode}"
    candidates_key = f"learning_adv_candidates_{_adv_params}"
    per_slice_key = f"learning_adv_per_slice_{_adv_params}"
    if st.button("Link rollout slices to demos", key=run_key, type="primary"):
        st.session_state[show_search_key] = True
        st.session_state.pop(candidates_key, None)
        st.session_state.pop(per_slice_key, None)

    if not st.session_state.get(show_search_key, False):
        return

    if candidates_key not in st.session_state:
        with st.spinner("Computing influence scores..."):
            all_candidates, per_slice_candidates = _run_slice_search(
                data=data,
                rollout_slices=rollout_slices,
                demo_split=demo_split,
                window_width=window_width,
                aggregation_method=aggregation_method,
                per_slice_top_k=per_slice_top_k,
                ascending=ascending,
                use_all_demos_per_slice=(selection_mode != "global_top_k"),
            )
            st.session_state[candidates_key] = all_candidates
            st.session_state[per_slice_key] = per_slice_candidates

    all_candidates = st.session_state[candidates_key]
    per_slice_candidates = st.session_state[per_slice_key]
    _apply_slice_normalization_for_selection(
        per_slice_candidates, selection_normalization
    )
    sort_key = lambda x: x.get("score_for_selection", x["score"])
    all_candidates_sorted = sorted(
        all_candidates, key=sort_key, reverse=not ascending
    )
    if selection_mode == "global_top_k":
        raw_selection = all_candidates_sorted[:global_top_k_input]
    elif threshold_type == "percentile" and percentile_input is not None:
        raw_selection = _per_slice_percentile_selection(
            per_slice_candidates, percentile_input
        )
    else:
        raw_selection = _per_slice_n_sigma_selection(
            per_slice_candidates, n_sigma_input or 2.0
        )
    if not raw_selection:
        st.warning("No results found.")
        return
    if remove_negative_influence:
        final_selection = [c for c in raw_selection if c["score"] >= 0]
    else:
        final_selection = raw_selection
    if not final_selection:
        st.warning("No results after filters.")
        return

    effective_sel_mode = (
        "per_slice_percentile" if threshold_type == "percentile" and percentile_input is not None
        else selection_mode
    )
    selected_label = f"{criterion}_based_{'A' if use_advantage else 'V'}{threshold}"
    st.divider()
    st.subheader("Results")
    _render_search_results_with_add_button(
        data=data,
        global_top_k=final_selection,
        demo_split=demo_split,
        demo_episodes=demo_episodes,
        ep_idxs=ep_idxs,
        obs_key=obs_key,
        annotation_file=annotation_file,
        task_config=task_config,
        active_config_name=active_config_name,
        selected_label=selected_label,
        window_width=window_width,
        aggregation_method=aggregation_method,
        selection_method="advantage_based",
        selection_normalization=selection_normalization,
        selection_mode=effective_sel_mode,
        n_sigma=n_sigma_input,
        global_top_k_input=global_top_k_input,
        percentile=percentile_input,
    )


def _render_path_based_curation(
    data: InfluenceData,
    demo_split: SplitType,
    task_config: str,
    obs_key: str,
    active_config_name: Optional[str],
    annotation_file: str,
    is_selection_mode: bool = False,
) -> None:
    """Render path-based curation: paths to FAILURE (filter) or SUCCESS (selection), then attribute rollout slices to demos."""
    from influence_visualizer.behavior_graph import FAILURE_NODE_ID, SUCCESS_NODE_ID
    from influence_visualizer.behavior_value_loader import (
        get_behavior_graph_and_slice_values,
        get_path_based_rollout_slices,
    )
    from influence_visualizer.clustering_results import (
        list_clustering_results,
        load_clustering_result,
    )

    if not task_config:
        st.info("Select a task config to use path-based curation.")
        return

    saved_names = list_clustering_results(task_config)
    if not saved_names:
        st.info(
            "No clustering results found. Run clustering in the **Cluster Generation** tab "
            "(Clustering Algorithms), save a result, then return here."
        )
        return

    selected_name = st.selectbox(
        "Clustering result",
        options=saved_names,
        key=f"learning_path_clustering_{demo_split}",
        help="Saved clustering result (rollout-level) for behavior graph paths.",
    )
    try:
        cluster_labels, metadata, _ = load_clustering_result(task_config, selected_name)
    except Exception as e:
        st.error(f"Failed to load clustering result: {e}")
        return

    if metadata and "rollout_idx" not in metadata[0]:
        st.warning(
            "Path-based curation requires rollout-level clustering. "
            "Use a clustering result built from rollout episodes."
        )
        return

    graph, _values, _q_values, _advantages = get_behavior_graph_and_slice_values(
        cluster_labels, metadata, gamma=0.95
    )
    if not graph.has_outcome_split:
        st.warning(
            "Clustering result has no success/failure outcome split. "
            "Path-based curation needs episode outcomes to distinguish paths to SUCCESS vs FAILURE."
        )
        return

    st.subheader("Path target")
    if is_selection_mode:
        terminal_label = "Paths to SUCCESS (select demos attributed to these paths)"
        terminal_id = SUCCESS_NODE_ID
    else:
        terminal_label = "Paths to FAILURE (filter: exclude demos attributed to these paths)"
        terminal_id = FAILURE_NODE_ID
    st.caption(terminal_label)

    st.subheader("Path parameters")
    col_k, col_min_prob, col_min_edge = st.columns(3)
    with col_k:
        top_k_paths = st.number_input(
            "Number of top paths",
            min_value=1,
            max_value=100,
            value=10,
            key=f"learning_path_top_k_{demo_split}",
            help="Use the top-K highest-probability paths to the target terminal.",
        )
    with col_min_prob:
        min_path_probability = st.number_input(
            "Min path probability",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            format="%.3f",
            key=f"learning_path_min_prob_{demo_split}",
            help="Ignore paths with total probability below this.",
        )
    with col_min_edge:
        min_edge_probability = st.number_input(
            "Min edge probability",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            format="%.3f",
            key=f"learning_path_min_edge_{demo_split}",
            help="Ignore transitions below this during path enumeration.",
        )

    path_compute_key = f"learning_path_compute_{demo_split}"
    if st.button("Compute path-based rollout slices", key=path_compute_key, type="primary"):
        with st.spinner("Enumerating paths and collecting rollout slices (one per clustering sample on path)..."):
            rollout_slices = get_path_based_rollout_slices(
                graph=graph,
                cluster_labels=cluster_labels,
                metadata=metadata,
                data=data,
                terminal_id=terminal_id,
                top_k_paths=top_k_paths,
                min_path_probability=min_path_probability,
                min_edge_probability=min_edge_probability,
            )
        st.session_state[f"learning_path_rollout_slices_{demo_split}"] = rollout_slices
        st.session_state[f"learning_path_computed_{demo_split}"] = True
        st.rerun()

    if not st.session_state.get(f"learning_path_computed_{demo_split}", False):
        st.info("Click **Compute path-based rollout slices** to find rollout segments on the selected paths.")
        return

    rollout_slices = st.session_state.get(f"learning_path_rollout_slices_{demo_split}", [])
    if not rollout_slices:
        st.warning(
            "No rollout slices found on the selected paths. Try lowering min path probability "
            "or min edge probability, or increase the number of top paths."
        )
        return

    st.success(f"Found **{len(rollout_slices)}** rollout slices on the top-{top_k_paths} paths (slice-level, not segment-level).")

    st.subheader("Attribution to demo slices")
    st.caption(
        "Use the influence matrix to attribute each path rollout slice to demo slices; "
        "then choose how many demo slices to add to the curation config."
    )
    _im, demo_episodes, ep_idxs, _ep_lens = get_split_data(data, demo_split)
    window_width = st.number_input(
        "Window width (demo aggregation)",
        min_value=1,
        max_value=64,
        value=1,
        key=f"learning_path_window_{demo_split}",
        help="Sliding window width when aggregating influence from rollout slice to demo samples.",
    )
    aggregation_method = st.selectbox(
        "Aggregation",
        options=list(AGGREGATION_METHODS.keys()),
        key=f"learning_path_agg_{demo_split}",
    )
    per_slice_top_k = st.number_input(
        "Top demo slices per rollout slice",
        min_value=1,
        max_value=500,
        value=50,
        key=f"learning_path_per_slice_k_{demo_split}",
        help="For each path rollout slice, keep this many top-attributed demo slices.",
    )
    global_top_k_input = st.number_input(
        "Global top-K demo slices to add",
        min_value=1,
        max_value=5000,
        value=min(500, 5000),
        key=f"learning_path_global_k_{demo_split}",
        help="After attributing all segments to demos, take the global top-K by influence to add.",
    )

    _path_params = f"{selected_name}_{terminal_id}_{top_k_paths}_{demo_split}"
    candidates_key = f"learning_path_candidates_{_path_params}"
    per_slice_key = f"learning_path_per_slice_{_path_params}"
    link_btn_key = f"learning_path_run_attribution_{demo_split}"

    if st.button("Link path rollout slices to demos", key=link_btn_key, type="primary"):
        with st.spinner("Computing influence scores for path rollout slices..."):
            all_candidates, per_slice_candidates = _run_slice_search(
                data=data,
                rollout_slices=rollout_slices,
                demo_split=demo_split,
                window_width=window_width,
                aggregation_method=aggregation_method,
                per_slice_top_k=per_slice_top_k,
                ascending=False,
                use_all_demos_per_slice=False,
            )
            for slice_cands in per_slice_candidates:
                for c in slice_cands:
                    c["score_for_selection"] = c["score"]
        st.session_state[candidates_key] = all_candidates
        st.session_state[per_slice_key] = per_slice_candidates
        st.rerun()

    if candidates_key not in st.session_state:
        st.info("Click **Link path rollout slices to demos** to compute influence-based attribution.")
        return

    all_candidates = st.session_state[candidates_key]
    per_slice_candidates = st.session_state[per_slice_key]
    sort_key_fn = lambda x: x.get("score_for_selection", x["score"])
    all_candidates_sorted = sorted(all_candidates, key=sort_key_fn, reverse=True)
    final_selection = all_candidates_sorted[:global_top_k_input]
    if not final_selection:
        st.warning("No demo slices attributed. Try increasing per-slice top-K or window width.")
        return

    selected_label = f"path_based_{'success' if is_selection_mode else 'failure'}_k{top_k_paths}"
    st.divider()
    st.subheader("Results")
    _render_search_results_with_add_button(
        data=data,
        global_top_k=final_selection,
        demo_split=demo_split,
        demo_episodes=demo_episodes,
        ep_idxs=ep_idxs,
        obs_key=obs_key,
        annotation_file=annotation_file,
        task_config=task_config,
        active_config_name=active_config_name,
        selected_label=selected_label,
        window_width=window_width,
        aggregation_method=aggregation_method,
        selection_method="path_based",
        selection_normalization="none",
        selection_mode="global_top_k",
        n_sigma=None,
        global_top_k_input=global_top_k_input,
        percentile=None,
    )


@st.fragment
def _render_slice_search_fragment(
    data: InfluenceData,
    demo_split: SplitType,
    annotation_file: str,
    task_config: str,
    obs_key: str,
    active_config_name: Optional[str],
    is_selection_mode: bool = False,
):
    """Fragment for searching demo slices by behavior and adding them to a curation config.

    This is closely modeled on render_behavior_slice_search from render_local_behaviors.py
    but adds the "Add to curation config" functionality.
    """
    with st.expander("Behavior Slice Search", expanded=True):
        # Split is fixed: holdout for selection, train for filtering
        st.markdown(f"**Dataset split:** **{demo_split}** (fixed for this curation mode)")
        if demo_split == "holdout":
            st.caption("Selecting from holdout. These slices will be added to training (curation-selection).")
        else:
            st.caption("Selecting from train. These slices will be excluded from training (curation-filtering).")
        st.divider()

        selection_method = st.selectbox(
            "Selection method",
            options=["ranking_based", "advantage_based", "path_based"],
            format_func=lambda x: {
                "ranking_based": "Ranking-based curation",
                "advantage_based": "Advantage-based curation",
                "path_based": "Path-based curation (behavior graph paths)",
            }[x],
            key=f"learning_selection_method_{demo_split}",
        )

        if selection_method == "path_based":
            _render_path_based_curation(
                data=data,
                demo_split=demo_split,
                task_config=task_config,
                obs_key=obs_key,
                active_config_name=active_config_name,
                annotation_file=annotation_file,
                is_selection_mode=is_selection_mode,
            )
            return

        if selection_method == "advantage_based":
            _render_advantage_based_curation(
                data=data,
                demo_split=demo_split,
                task_config=task_config,
                obs_key=obs_key,
                active_config_name=active_config_name,
                annotation_file=annotation_file,
                is_selection_mode=is_selection_mode,
            )
            return

        # --- Ranking-based: annotation-driven flow ---
        # Load annotations
        with profile("learning_load_annotations"):
            annotations = load_annotations(annotation_file, task_config=task_config)

        if not annotations:
            st.warning(
                "No annotations found. Please annotate some demonstrations first "
                "using the Annotation tab."
            )
            return

        # Get influence data
        with profile("learning_get_split_data"):
            influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(
                data, demo_split
            )

        # Collect all unique labels from ROLLOUT annotations
        all_labels = set()
        for rollout_ep in data.rollout_episodes:
            episode_id = str(rollout_ep.index)
            ep_annotations = get_episode_annotations(annotations, episode_id, "rollout")
            for slice_info in ep_annotations:
                all_labels.add(slice_info["label"])

        if not all_labels:
            st.warning(
                "No labeled rollout slices found. "
                "Please annotate some rollout episodes first in the Annotation tab."
            )
            return

        # 1. Label selection
        st.subheader("1. Select Behavior Label")

        col_label, col_count = st.columns([2, 1])

        with col_label:
            selected_label = st.selectbox(
                "Rollout behavior to search for",
                options=sorted(all_labels),
                key=f"learning_search_label_{demo_split}",
            )

        # Collect rollout slices with this label
        rollout_slices_with_label = []
        for rollout_ep in data.rollout_episodes:
            episode_id = str(rollout_ep.index)
            ep_annotations = get_episode_annotations(annotations, episode_id, "rollout")
            for slice_info in ep_annotations:
                if slice_info["label"] == selected_label:
                    rollout_slices_with_label.append(
                        {
                            "rollout_idx": rollout_ep.index,
                            "rollout_ep": rollout_ep,
                            "start": slice_info["start"],
                            "end": slice_info["end"],
                        }
                    )

        with col_count:
            st.metric("Rollout slices with this label", len(rollout_slices_with_label))

        if not rollout_slices_with_label:
            st.info(f"No rollout slices found with label '{selected_label}'")
            return

        st.divider()

        # 2. Search parameters
        st.subheader("2. Configure Search Parameters")

        col_agg, col_window = st.columns(2)

        with col_agg:
            agg_options = list(AGGREGATION_METHODS.keys())
            agg_default_idx = agg_options.index("mean") if "mean" in agg_options else 0
            aggregation_method = st.selectbox(
                "Aggregation method",
                options=agg_options,
                index=agg_default_idx,
                key=f"learning_search_agg_{demo_split}",
            )

        with col_window:
            window_width = st.number_input(
                "Window width",
                min_value=1,
                max_value=50,
                value=5,
                key=f"learning_search_window_{demo_split}",
                help="Width of sliding window over demo timesteps",
            )

        selection_mode = st.selectbox(
            "Selection mode",
            options=["global_top_k", "per_slice_n_sigma"],
            format_func=lambda x: {
                "global_top_k": "Global top-k",
                "per_slice_n_sigma": "Per-slice threshold",
            }[x],
            key=f"learning_selection_mode_{demo_split}",
        )

        n_sigma_input = None
        percentile_input: Optional[float] = None
        threshold_type = "sigma"
        per_slice_top_k = 20
        global_top_k_input = 10
        ascending = False
        if selection_mode == "global_top_k":
            col_per_k, col_global_k, col_ascending = st.columns(3)
            with col_per_k:
                per_slice_top_k = st.number_input(
                    "Per-Slice Top K",
                    min_value=1,
                    value=20,
                    key=f"learning_search_per_slice_topk_{demo_split}",
                    help="Number of top demos to collect from each behavior slice",
                )
            with col_global_k:
                global_top_k_input = st.number_input(
                    "Global Top K",
                    min_value=1,
                    value=10,
                    key=f"learning_search_global_topk_{demo_split}",
                    help="Number of top demos to display after global re-ranking",
                )
            with col_ascending:
                ascending = st.checkbox(
                    "Sort ascending (lowest influence)",
                    value=False,
                    key=f"learning_search_ascending_{demo_split}",
                    help="If checked, find slices with lowest influence instead of highest",
                )
        else:
            threshold_type = st.radio(
                "Threshold by",
                options=["sigma", "percentile"],
                format_func=lambda x: "Sigma (σ)" if x == "sigma" else "Percentile (empirical)",
                key=f"learning_threshold_type_{demo_split}",
                horizontal=True,
            )
            if threshold_type == "sigma":
                n_sigma_input = st.number_input(
                    "n (sigma)",
                    min_value=0.0,
                    value=2.0,
                    step=0.5,
                    key=f"learning_n_sigma_{demo_split}",
                    help="Keep demo slices with z-score ≥ n within each rollout slice.",
                )
                pct = _sigma_to_percentile(n_sigma_input)
                st.caption(f"≈ {pct:.1f}th percentile (top {100 - pct:.1f}%) assuming normal")
            else:
                percentile_input = st.number_input(
                    "Percentile",
                    min_value=50.0,
                    max_value=99.99,
                    value=97.7,
                    step=0.5,
                    key=f"learning_percentile_{demo_split}",
                    help="Keep demo slices at or above this empirical percentile within each slice. Robust to heavy-tailed distributions.",
                )
                st.caption(f"Top {100 - percentile_input:.2f}% of scores per slice (empirical)")

        st.caption("Selection: how to compare scores across rollout slices (Global top-k only)")
        _norm_format = lambda x: {
            "none": "None (raw scores)",
            "scale-std": "Scale by std (per slice)",
            "scale-range": "Divide by range (per slice)",
            "z-score": "Z-score (per slice)",
            "min-max": "Min-max → [0,1] (per slice)",
        }[x]
        if selection_mode == "per_slice_n_sigma":
            st.selectbox(
                "Score normalization for selection",
                options=["z-score"],
                index=0,
                format_func=_norm_format,
                key=f"learning_selection_normalization_{demo_split}",
                disabled=True,
                help=(
                    "Per-slice threshold selection computes z-scores (sigma mode) or "
                    "empirical percentiles (percentile mode) internally from raw scores "
                    "within each slice. Normalization is fixed to z-score for chart display."
                ),
            )
            selection_normalization = "z-score"
        else:
            selection_normalization = st.selectbox(
                "Score normalization for selection",
                options=["none", "scale-std", "scale-range", "z-score", "min-max"],
                format_func=_norm_format,
                key=f"learning_selection_normalization_{demo_split}",
                help="Scale-by-std/range: rescale only (no centering). Min-max: map each slice to [0,1]. Z-score: center at 0, scale by std.",
            )

        remove_negative_influence = st.checkbox(
            "Remove pairs with negative (unnormalized) influence",
            value=False,
            key=f"learning_remove_negative_{demo_split}_{selected_label}",
            help="Drop demo slices whose raw influence for that rollout–demo pair is < 0.",
        )

        st.divider()

        # 3. Run search
        st.subheader("3. Run Search")

        search_key = f"learning_search_results_{demo_split}_{selected_label}_{selection_mode}"
        candidates_key = f"learning_search_candidates_{demo_split}_{selected_label}_{selection_mode}"

        per_slice_key = f"learning_search_per_slice_{demo_split}_{selected_label}_{selection_mode}"
        slices_info_key = f"learning_search_slices_info_{demo_split}_{selected_label}_{selection_mode}"

        if st.button(
            f"Search for '{selected_label}' slices",
            key=f"learning_btn_search_{demo_split}",
            type="primary",
        ):
            st.session_state[search_key] = True
            # Clear previous candidates (for this selection mode)
            st.session_state.pop(candidates_key, None)
            st.session_state.pop(per_slice_key, None)
            st.session_state.pop(slices_info_key, None)

        if not st.session_state.get(search_key, False):
            return

        use_all_demos_per_slice = selection_mode != "global_top_k"

        # Compute if not cached
        if candidates_key not in st.session_state:
            with st.spinner("Computing influence scores..."):
                with profile("run_slice_search"):
                    all_candidates, per_slice_candidates = _run_slice_search(
                        data=data,
                        rollout_slices=rollout_slices_with_label,
                        demo_split=demo_split,
                        window_width=window_width,
                        aggregation_method=aggregation_method,
                        per_slice_top_k=per_slice_top_k,
                        ascending=ascending,
                        use_all_demos_per_slice=use_all_demos_per_slice,
                    )
                st.session_state[candidates_key] = all_candidates
                st.session_state[per_slice_key] = per_slice_candidates
                st.session_state[slices_info_key] = rollout_slices_with_label

        all_candidates = st.session_state[candidates_key]
        per_slice_candidates = st.session_state.get(per_slice_key, [])
        rollout_slices_info = st.session_state.get(
            slices_info_key, rollout_slices_with_label
        )

        # Apply per-slice normalization for selection and for chart display
        _apply_slice_normalization_for_selection(
            per_slice_candidates, selection_normalization
        )

        # Build raw selection: global top-k, per-slice n-sigma, or per-slice percentile
        sort_key = lambda x: x.get("score_for_selection", x["score"])
        all_candidates_sorted = sorted(
            all_candidates, key=sort_key, reverse=not ascending
        )
        if selection_mode == "global_top_k":
            raw_selection = all_candidates_sorted[:global_top_k_input]
        elif threshold_type == "percentile" and percentile_input is not None:
            raw_selection = _per_slice_percentile_selection(
                per_slice_candidates, percentile_input
            )
        else:
            raw_selection = _per_slice_n_sigma_selection(
                per_slice_candidates, n_sigma_input or 2.0
            )

        if not raw_selection:
            st.warning("No results found.")
            return

        if selected_label is None:
            return

        # Apply negative-influence filter (option set in section 2)
        if remove_negative_influence:
            final_selection = [c for c in raw_selection if c["score"] >= 0]
            removed_negative_count = len(raw_selection) - len(final_selection)
        else:
            final_selection = raw_selection
            removed_negative_count = 0

        if remove_negative_influence:
            st.info(
                f"Removed **{removed_negative_count}** demo slice(s) with negative influence (in total across all rollout slices)."
            )

        if not final_selection:
            st.warning("No results after filters.")
            return

        st.divider()

        # 4. Ranking Score Visualizations (lazy: only build/plot when user clicks)
        st.subheader("4. Ranking Score Charts")
        st.markdown(
            "These charts help you choose appropriate top-k cutoff values by "
            "visualizing the distribution of influence scores."
        )
        show_ranking_charts_key = f"learning_show_ranking_charts_{demo_split}_{selected_label}"
        with st.expander("Load ranking charts", expanded=False):
            st.caption(
                "Build global and per-slice ranking charts (can take a minute with many slices)."
            )
            if st.button(
                "Show ranking charts",
                key=f"learning_btn_ranking_charts_{demo_split}_{selected_label}",
            ):
                st.session_state[show_ranking_charts_key] = True

        if st.session_state.get(show_ranking_charts_key, False):
            subsample_plots = st.checkbox(
                "Subsample points in plots",
                value=True,
                key=f"learning_subsample_plots_{demo_split}_{selected_label}",
                help="Limit plotted points for performance; when enabled, show a random sample of up to the max points below.",
            )
            col_global_pts, col_slice_pts = st.columns(2)
            with col_global_pts:
                global_max_plot_points = st.number_input(
                    "Max points (global plot)",
                    min_value=1,
                    value=5000,
                    key=f"learning_global_max_plot_points_{demo_split}_{selected_label}",
                    help="Max number of points to show in the global ranking chart (random sample).",
                    disabled=not subsample_plots,
                )
            with col_slice_pts:
                per_slice_max_plot_points = st.number_input(
                    "Max points (per-slice plot)",
                    min_value=1,
                    value=500,
                    key=f"learning_per_slice_max_plot_points_{demo_split}_{selected_label}",
                    help="Max number of points to show in each per-slice ranking chart (random sample).",
                    disabled=not subsample_plots,
                )

            with profile("learning_render_ranking_charts"):
                _render_ranking_charts(
                    data=data,
                    all_candidates=all_candidates_sorted,
                    per_slice_candidates=per_slice_candidates,
                    rollout_slices_info=rollout_slices_info,
                    demo_split=demo_split,
                    ep_idxs=ep_idxs,
                    demo_episodes=demo_episodes,
                    global_top_k_input=global_top_k_input,
                    per_slice_top_k=per_slice_top_k,
                    selected_label=selected_label,
                    current_selection_candidates=final_selection,
                    window_width=window_width,
                    selection_normalization=selection_normalization,
                    selection_mode=selection_mode,
                    subsample_plots=subsample_plots,
                    global_max_plot_points=global_max_plot_points,
                    per_slice_max_plot_points=per_slice_max_plot_points,
                    n_sigma_input=n_sigma_input,
                )

        st.divider()

        # 5. Display results with "Add to curation config" button and selection distributions
        st.subheader("5. Results")

        effective_sel_mode = (
            "per_slice_percentile" if threshold_type == "percentile" and percentile_input is not None
            else selection_mode
        )
        _render_search_results_with_add_button(
            data=data,
            global_top_k=final_selection,
            demo_split=demo_split,
            demo_episodes=demo_episodes,
            ep_idxs=ep_idxs,
            obs_key=obs_key,
            annotation_file=annotation_file,
            task_config=task_config,
            active_config_name=active_config_name,
            selected_label=selected_label,
            window_width=window_width,
            aggregation_method=aggregation_method,
            selection_method="ranking_based",
            selection_normalization=selection_normalization,
            selection_mode=effective_sel_mode,
            n_sigma=n_sigma_input,
            global_top_k_input=global_top_k_input,
            percentile=percentile_input,
        )

    # When this fragment runs (e.g. after "Show ranking charts"), main() did not run,
    # so the sidebar metrics block at the end of main() is stale. Update sidebar here.
    get_profiler().print_to_streamlit(title_suffix=" (latest)")


def _build_local_sample_lookup(
    data: InfluenceData,
    demo_split: SplitType,
    ep_idxs: list,
    demo_episodes: list,
    window_width: int,
) -> Dict[int, Tuple[int, int, int, int]]:
    """Build local_sample_idx -> (ep_idx, episode_idx, start, end) using vectorized numpy."""
    num_train_samples = len(data.demo_sample_infos)
    n_total = sum(len(s) for s in ep_idxs)

    # Precompute global arrays once (avoids 125k+ Python object lookups)
    timesteps = np.array(
        [s.timestep for s in data.all_demo_sample_infos], dtype=np.int64
    )
    episode_ids_global = np.array(
        [s.episode_idx for s in data.all_demo_sample_infos], dtype=np.int64
    )
    ep_lens = np.array(
        [
            getattr(ep, "raw_length", None) or ep.num_samples
            for ep in demo_episodes
        ],
        dtype=np.int64,
    )

    # Which episode each local sample belongs to (one short loop over episodes)
    ep_idx_for_sample = np.empty(n_total, dtype=np.int64)
    for ep_idx, sample_idxs in enumerate(ep_idxs):
        ep_idx_for_sample[sample_idxs] = ep_idx

    # Vectorized: all local indices and their global position
    local_sample_idx_arr = np.arange(n_total, dtype=np.int64)
    global_sample_idx = local_sample_idx_arr + (
        num_train_samples if demo_split == "holdout" else 0
    )
    valid = global_sample_idx < len(data.all_demo_sample_infos)

    demo_start = np.zeros(n_total, dtype=np.int64)
    demo_start[valid] = timesteps[global_sample_idx[valid]]
    episode_idx = np.zeros(n_total, dtype=np.int64)
    episode_idx[valid] = episode_ids_global[global_sample_idx[valid]]

    ep_end = np.maximum(0, ep_lens[ep_idx_for_sample] - 1)
    demo_end = np.minimum(demo_start + window_width - 1, ep_end)

    # Build dict (loop is over n_total but only dict assigns, no Python object access)
    lookup: Dict[int, Tuple[int, int, int, int]] = {}
    for i in range(n_total):
        if not valid[i]:
            continue
        lookup[int(i)] = (
            int(ep_idx_for_sample[i]),
            int(episode_idx[i]),
            int(demo_start[i]),
            int(demo_end[i]),
        )
    return lookup


def _build_local_sample_lookup_for_indices(
    data: InfluenceData,
    demo_split: SplitType,
    ep_idxs: list,
    demo_episodes: list,
    window_width: int,
    needed_indices: set,
) -> Dict[int, Tuple[int, int, int, int]]:
    """Build lookup only for local_sample_idx in needed_indices. O(len(needed) * num_episodes) instead of O(total_samples)."""
    num_train_samples = len(data.demo_sample_infos)
    lookup: Dict[int, Tuple[int, int, int, int]] = {}
    for local_sample_idx in needed_indices:
        for ep_idx, sample_idxs in enumerate(ep_idxs):
            if local_sample_idx not in sample_idxs:
                continue
            episode = demo_episodes[ep_idx]
            ep_len = getattr(episode, "raw_length", None) or episode.num_samples
            if demo_split == "train":
                global_sample_idx = local_sample_idx
            elif demo_split == "holdout":
                global_sample_idx = local_sample_idx + num_train_samples
            else:
                global_sample_idx = local_sample_idx
            if global_sample_idx >= len(data.all_demo_sample_infos):
                break
            sample_info = data.all_demo_sample_infos[global_sample_idx]
            demo_start = int(sample_info.timestep)
            demo_end = min(demo_start + window_width - 1, ep_len - 1)
            lookup[int(local_sample_idx)] = (
                ep_idx,
                int(sample_info.episode_idx),
                demo_start,
                demo_end,
            )
            break
    return lookup


def _resolve_candidate_demo_slice(
    candidate: Dict,
    data: InfluenceData,
    demo_split: SplitType,
    ep_idxs: list,
    demo_episodes: list,
    window_width: int,
    lookup: Optional[Dict[int, Tuple[int, int, int, int]]] = None,
) -> Optional[Tuple[int, int, int]]:
    """Resolve candidate to (demo_episode_idx, demo_start, demo_end) in raw timestep space.

    Uses the same episode index and timestep as stored when adding to the curation config
    (sample_info.episode_idx and sample_info.timestep). If lookup is provided, uses O(1) lookup.
    """
    local_sample_idx = candidate["local_sample_idx"]
    if lookup is not None:
        t = lookup.get(local_sample_idx)
        return (t[1], t[2], t[3]) if t is not None else None
    num_train_samples = len(data.demo_sample_infos)

    for ep_idx, sample_idxs in enumerate(ep_idxs):
        if local_sample_idx in sample_idxs:
            episode = demo_episodes[ep_idx]
            if demo_split == "train":
                global_sample_idx = local_sample_idx
            elif demo_split == "holdout":
                global_sample_idx = local_sample_idx + num_train_samples
            else:
                global_sample_idx = local_sample_idx

            if global_sample_idx >= len(data.all_demo_sample_infos):
                return None
            sample_info = data.all_demo_sample_infos[global_sample_idx]
            demo_start = int(sample_info.timestep)
            ep_len = getattr(episode, "raw_length", None) or episode.num_samples
            demo_end = min(demo_start + window_width - 1, ep_len - 1)
            # Use sample_info.episode_idx so we match exactly what the config stores
            return (int(sample_info.episode_idx), demo_start, demo_end)
    return None


def _candidate_in_curation(
    candidate: Dict,
    config: CurationConfig,
    data: InfluenceData,
    demo_split: SplitType,
    ep_idxs: list,
    demo_episodes: list,
    window_width: int,
) -> bool:
    """True if the candidate's demo slice overlaps any slice in the curation config."""
    resolved = _resolve_candidate_demo_slice(
        candidate, data, demo_split, ep_idxs, demo_episodes, window_width
    )
    if resolved is None:
        return False
    ep_idx, start, end = resolved
    start, end = int(start), int(end)
    for s in config.slices:
        # Normalize to int (YAML may give int or str)
        s_ep = int(s.episode_idx)
        s_start = int(s.start)
        s_end = int(s.end)
        if s_ep != ep_idx:
            continue
        # Overlap: [start, end] and [s_start, s_end] intersect (inclusive)
        if s_start <= end and s_end >= start:
            return True
    return False


def _build_candidate_label(
    candidate: Dict,
    data: InfluenceData,
    demo_split: SplitType,
    ep_idxs: list,
    demo_episodes: list,
    lookup: Optional[Dict[int, Tuple[int, int, int, int]]] = None,
) -> str:
    """Build a human-readable label for a ranking chart entry. Use lookup for O(1) when provided."""
    local_sample_idx = candidate["local_sample_idx"]
    if lookup is not None:
        t = lookup.get(local_sample_idx)
        if t is not None:
            ep_idx, _episode_idx, start, _end = t
            demo_ep = demo_episodes[ep_idx]
            return (
                f"Rollout ep{candidate['source_episode_idx']}"
                f"[{candidate['source_start']}:{candidate['source_end']}] "
                f"→ Demo ep{demo_ep.index} t={start}"
            )
        return (
            f"Rollout ep{candidate['source_episode_idx']}"
            f"[{candidate['source_start']}:{candidate['source_end']}] "
            f"→ Demo sample {local_sample_idx}"
        )
    num_train_samples = len(data.demo_sample_infos)
    demo_ep = None
    demo_timestep = None
    for ep_idx, sample_idxs in enumerate(ep_idxs):
        if local_sample_idx in sample_idxs:
            demo_ep = demo_episodes[ep_idx]
            if demo_split == "train":
                global_sample_idx = local_sample_idx
            elif demo_split == "holdout":
                global_sample_idx = local_sample_idx + num_train_samples
            else:
                global_sample_idx = local_sample_idx

            if global_sample_idx < len(data.all_demo_sample_infos):
                sample_info = data.all_demo_sample_infos[global_sample_idx]
                demo_timestep = sample_info.timestep
            break

    if demo_ep is not None and demo_timestep is not None:
        return (
            f"Rollout ep{candidate['source_episode_idx']}"
            f"[{candidate['source_start']}:{candidate['source_end']}] "
            f"→ Demo ep{demo_ep.index} t={demo_timestep}"
        )
    return (
        f"Rollout ep{candidate['source_episode_idx']}"
        f"[{candidate['source_start']}:{candidate['source_end']}] "
        f"→ Demo sample {local_sample_idx}"
    )


def _render_ranking_charts(
    data: InfluenceData,
    all_candidates: List[Dict],
    per_slice_candidates: List[List[Dict]],
    rollout_slices_info: List[Dict],
    demo_split: SplitType,
    ep_idxs: list,
    demo_episodes: list,
    global_top_k_input: int,
    per_slice_top_k: int,
    selected_label: str,
    current_selection_candidates: Optional[List[Dict]] = None,
    window_width: int = 5,
    selection_normalization: str = "none",
    selection_mode: str = "global_top_k",
    subsample_plots: bool = True,
    global_max_plot_points: int = 5000,
    per_slice_max_plot_points: int = 500,
    n_sigma_input: Optional[float] = None,
):
    """Render ranking score charts for both per-slice and global rankings.

    current_selection_candidates: The current selection (results that would be added
        if user clicks "Add to curation config"). Used to color per-slice bars by
        "in current selection" vs "not in current selection".
    selection_normalization: "none" | "scale-std" | "scale-range" | "z-score" | "min-max".
        When not "none", charts show normalized/scaled scores and y-axis label is "Normalized score".
    selection_mode: "global_top_k" | "per_slice_n_sigma" | "per_slice_percentile".
        When per-slice, top-k cutoff line is not shown.
    subsample_plots: If True, limit plotted points to global_max_plot_points / per_slice_max_plot_points.
    """
    use_normalized = selection_normalization != "none"
    yaxis_title = "Normalized score" if use_normalized else "Score"
    score_key = "score_for_selection" if use_normalized else "score"

    # One-time O(samples) lookup so resolve/label are O(1) per candidate
    with profile("learning_ranking_build_lookup"):
        lookup = _build_local_sample_lookup(
            data, demo_split, ep_idxs, demo_episodes, window_width
        )

    # Build set of (episode_idx, start, end) for the current selection (used for global and per-slice in-selection coloring)
    with profile("learning_ranking_build_selection_slices"):
        current_selection_slices: set = set()
        if current_selection_candidates:
            for c in current_selection_candidates:
                resolved = _resolve_candidate_demo_slice(
                    c, data, demo_split, ep_idxs, demo_episodes, window_width, lookup=lookup
                )
                if resolved is not None:
                    ep_idx, start, end = resolved
                    current_selection_slices.add((int(ep_idx), int(start), int(end)))

    # --- Global ranking chart ---
    with st.expander("Global Ranking", expanded=True):
        if all_candidates:
            if subsample_plots and len(all_candidates) > global_max_plot_points:
                rng = np.random.default_rng(seed=hash((selected_label, demo_split)) % (2**32))
                idx = rng.choice(len(all_candidates), size=global_max_plot_points, replace=False)
                plot_candidates = [all_candidates[i] for i in np.sort(idx)]
            else:
                plot_candidates = all_candidates
            all_scores = np.array([c.get(score_key, c["score"]) for c in plot_candidates])
            with profile("learning_ranking_global_labels"):
                all_labels = [
                    _build_candidate_label(c, data, demo_split, ep_idxs, demo_episodes, lookup=lookup)
                    for c in plot_candidates
                ]
            # μ, σ only meaningful per-slice when normalized; at global level don't show them
            raw_all = np.array([c["score"] for c in plot_candidates])
            raw_mean_global = float(np.mean(raw_all))
            raw_std_global = float(np.std(raw_all)) if len(raw_all) > 1 else 0.0
            show_global_raw_stats = not use_normalized
            color_by_rollout = st.checkbox(
                "Color bars by source rollout",
                value=False,
                key=f"learning_global_ranking_color_by_rollout_{demo_split}_{selected_label}",
                help="Assign a different color to each source rollout episode.",
            )
            color_global_by_selection = st.checkbox(
                "Color bars by in-selection (current search results)",
                value=(selection_mode != "global_top_k"),
                key=f"learning_global_ranking_color_by_selection_{demo_split}_{selected_label}",
                help="Green = in current selection; gray = not in selection.",
            )
            source_ids = (
                [c["source_episode_idx"] for c in plot_candidates]
                if color_by_rollout and not color_global_by_selection
                else None
            )
            with profile("learning_ranking_global_in_selection"):
                in_selection_global = None
                if color_global_by_selection and current_selection_slices:
                    in_selection_global = []
                    for c in plot_candidates:
                        resolved = _resolve_candidate_demo_slice(
                            c, data, demo_split, ep_idxs, demo_episodes, window_width, lookup=lookup
                        )
                        in_selection_global.append(
                            resolved is not None
                            and (int(resolved[0]), int(resolved[1]), int(resolved[2]))
                            in current_selection_slices
                        )
            with profile("learning_ranking_global_plot"):
                fig_global = plotting.create_ranking_scores_plot(
                    scores=all_scores,
                    labels=all_labels,
                    title=f"Global Demo Influence on Rollout Behavior '{selected_label}'",
                    highlight_top_k=global_top_k_input if selection_mode == "global_top_k" else None,
                    show_cumulative=False,
                    color_by_source=source_ids,
                    color_by_in_selection=in_selection_global,
                    yaxis_title=yaxis_title,
                    raw_mean=raw_mean_global if show_global_raw_stats else None,
                    raw_std=raw_std_global if show_global_raw_stats else None,
                )
                st.plotly_chart(
                    fig_global,
                    width='stretch',
                    key=f"learning_plot_global_rank_{demo_split}_{selected_label}",
                )
        else:
            st.info("No candidates to display.")

    # --- Per-slice ranking charts (with pagination) ---
    if len(per_slice_candidates) > 0:
        num_slices = len(per_slice_candidates)
        charts_per_page = 5
        num_pages = max(1, (num_slices + charts_per_page - 1) // charts_per_page)

        with st.expander(
            f"Per-Slice Rankings ({num_slices} slices)", expanded=True
        ):
            color_per_slice_by_selection = st.checkbox(
                "Color bars by in-selection (current search results)",
                value=(selection_mode != "global_top_k"),
                key=f"learning_per_slice_color_by_selection_{demo_split}_{selected_label}",
                help="Green = in global selection, blue = in this slice's local selection only, gray = not in selection.",
            )
            show_global_in_per_slice = True
            show_local_in_per_slice = True
            if color_per_slice_by_selection:
                show_global_in_per_slice = st.checkbox(
                    "Show global selection (green) in per-slice rankings",
                    value=True,
                    key=f"learning_per_slice_show_global_{demo_split}_{selected_label}",
                    help="Highlight bars that are in the current search result (would be added to curation).",
                )
                show_local_in_per_slice = st.checkbox(
                    "Show local selection (blue) in per-slice rankings",
                    value=True,
                    key=f"learning_per_slice_show_local_{demo_split}_{selected_label}",
                    help="Highlight bars that are in this slice's local selection (e.g. top-k or n-sigma for this slice only).",
                )
                st.caption(
                    "Local selection (blue) = this slice's rule-based selection only. You may see no blue bars if the "
                    "random sample doesn't include any demos that are in this slice's selection but not in the global search."
                )

            page = st.number_input(
                "Page",
                min_value=1,
                max_value=num_pages,
                value=1,
                key=f"learning_ranking_charts_page_{demo_split}_{selected_label}",
                help=f"Show {charts_per_page} charts per page ({num_pages} total pages).",
            )
            page = max(1, min(page, num_pages))
            start_idx = (page - 1) * charts_per_page
            end_idx = min(start_idx + charts_per_page, num_slices)

            st.caption(
                f"Showing slices {start_idx + 1}–{end_idx} of {num_slices}."
            )

            with profile("learning_ranking_per_slice_loop"):
                for slice_idx in range(start_idx, end_idx):
                    slice_candidates = per_slice_candidates[slice_idx]
                    if not slice_candidates:
                        continue

                    if subsample_plots and len(slice_candidates) > per_slice_max_plot_points:
                        rng_slice = np.random.default_rng(
                            seed=(hash((selected_label, demo_split)) + slice_idx) % (2**32)
                        )
                        idx_slice = rng_slice.choice(
                            len(slice_candidates), size=per_slice_max_plot_points, replace=False
                        )
                        plot_slice_candidates = [slice_candidates[i] for i in np.sort(idx_slice)]
                    else:
                        plot_slice_candidates = slice_candidates

                    # Get slice info for the title
                    if slice_idx < len(rollout_slices_info):
                        s_info = rollout_slices_info[slice_idx]
                        slice_title = (
                            f"Rollout ep{s_info['rollout_idx']} "
                            f"t[{s_info['start']}:{s_info['end']}]"
                        )
                    else:
                        slice_title = f"Slice {slice_idx + 1}"

                    # Candidates are already sorted from rank_demos_by_slice_influence
                    slice_scores = np.array(
                        [c.get(score_key, c["score"]) for c in plot_slice_candidates]
                    )
                    slice_raw = np.array([c["score"] for c in plot_slice_candidates])
                    slice_raw_mean = float(np.mean(slice_raw))
                    slice_raw_std = float(np.std(slice_raw)) if len(slice_raw) > 1 else 0.0
                    slice_labels = [
                        _build_candidate_label(c, data, demo_split, ep_idxs, demo_episodes, lookup=lookup)
                        for c in plot_slice_candidates
                    ]
                    color_by_category = None
                    in_global_mask = None
                    in_local_mask = None
                    in_selection = None
                    if color_per_slice_by_selection:
                        local_resolved: set[tuple[int, int, int]] = set()
                        full_mean, full_std, n_sigma = 0.0, 1.0, 1.0
                        if selection_mode == "global_top_k":
                            for c in slice_candidates[: min(per_slice_top_k, len(slice_candidates))]:
                                r = _resolve_candidate_demo_slice(
                                    c, data, demo_split, ep_idxs, demo_episodes, window_width, lookup=lookup
                                )
                                if r is not None:
                                    local_resolved.add((int(r[0]), int(r[1]), int(r[2])))
                        else:
                            full_mean = float(np.mean([c["score"] for c in slice_candidates]))
                            full_std = float(np.std([c["score"] for c in slice_candidates])) or 1.0
                            n_sigma = n_sigma_input if n_sigma_input is not None else 2.0

                        in_global_list: list[bool] = []
                        in_local_list: list[bool] = []
                        for c in plot_slice_candidates:
                            resolved = _resolve_candidate_demo_slice(
                                c, data, demo_split, ep_idxs, demo_episodes, window_width, lookup=lookup
                            )
                            if resolved is None:
                                in_global_list.append(False)
                                in_local_list.append(False)
                                continue
                            key = (int(resolved[0]), int(resolved[1]), int(resolved[2]))
                            in_global = key in current_selection_slices
                            if selection_mode == "global_top_k":
                                in_local = key in local_resolved
                            else:
                                z = (c["score"] - full_mean) / full_std
                                in_local = z >= n_sigma
                            in_global_list.append(in_global)
                            in_local_list.append(in_local)
                        in_global_mask = in_global_list
                        in_local_mask = in_local_list

                    slice_title_suffix = (
                        f"{len(slice_candidates)} demos"
                        if selection_mode != "global_top_k"
                        else f"Top {len(slice_candidates)} demos"
                    )
                    if subsample_plots and len(slice_candidates) > per_slice_max_plot_points:
                        slice_title_suffix += f" (showing {len(plot_slice_candidates)} random)"
                    fig_slice = plotting.create_ranking_scores_plot(
                        scores=slice_scores,
                        labels=slice_labels,
                        title=f"{slice_title} — {slice_title_suffix}",
                        highlight_top_k=min(per_slice_top_k, len(slice_candidates)) if selection_mode == "global_top_k" else None,
                        show_cumulative=False,
                        in_global_mask=in_global_mask,
                        in_local_mask=in_local_mask,
                        color_by_in_selection=in_selection,
                        show_global_selection=show_global_in_per_slice,
                        show_local_selection=show_local_in_per_slice,
                        yaxis_title=yaxis_title,
                        raw_mean=slice_raw_mean,
                        raw_std=slice_raw_std,
                        n_sigma_reference=(n_sigma_input if n_sigma_input is not None else 2.0) if selection_mode == "per_slice_n_sigma" else None,  # No sigma line for percentile mode
                        sigma_lines_in_z_space=use_normalized,
                        show_histogram=True,
                    )
                    st.plotly_chart(
                        fig_slice,
                        width='stretch',
                        key=f"learning_plot_slice_rank_{demo_split}_{selected_label}_{slice_idx}",
                    )


def _sigma_to_percentile(sigma: float) -> float:
    """Convert z-score (sigma) to approximate percentile (assuming normal). E.g. 2 -> ~97.7."""
    return 100.0 * float(scipy_stats.norm.cdf(sigma))


def _percentile_to_sigma(percentile: float) -> float:
    """Convert percentile to z-score (assuming normal). E.g. 95 -> ~1.645."""
    p = max(1e-6, min(1 - 1e-6, percentile / 100.0))
    return float(scipy_stats.norm.ppf(p))


def _per_slice_n_sigma_selection(
    per_slice_candidates: List[List[Dict]],
    n_sigma: float,
) -> List[Dict]:
    """Select candidates with z-score ≥ n_sigma within each rollout slice (union).

    Uses raw ``c["score"]`` per slice to compute z-scores; the "score_for_selection"
    field (set by _apply_slice_normalization_for_selection) is intentionally NOT used
    here because that normalization is already z-score by definition for this mode.
    Using score_for_selection would double-normalize (z-score of z-scores).
    The normalization dropdown is therefore disabled/fixed to "z-score" in the UI.

    Keeps candidates where z >= n_sigma and returns the union across all slices.
    """
    selected = []
    for slice_candidates in per_slice_candidates:
        if not slice_candidates:
            continue
        scores = np.array([c["score"] for c in slice_candidates], dtype=np.float64)
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        if std == 0:
            std = 1.0
        for c in slice_candidates:
            z = (c["score"] - mean) / std
            if z >= n_sigma:
                selected.append(c)
    return selected


def _per_slice_percentile_selection(
    per_slice_candidates: List[List[Dict]],
    percentile: float,
) -> List[Dict]:
    """Select candidates at or above the given empirical percentile within each slice.

    Unlike ``_per_slice_n_sigma_selection`` which assumes normality when interpreting
    the sigma threshold, this function uses ``np.percentile`` directly on raw scores.
    This is robust to heavy-tailed distributions where z-score thresholds select far
    more samples than the normal approximation would suggest.
    """
    selected = []
    for slice_candidates in per_slice_candidates:
        if not slice_candidates:
            continue
        scores = np.array([c["score"] for c in slice_candidates], dtype=np.float64)
        cutoff = float(np.percentile(scores, percentile))
        for c in slice_candidates:
            if c["score"] >= cutoff:
                selected.append(c)
    return selected


def _apply_slice_normalization_for_selection(
    per_slice_candidates: List[List[Dict]],
    normalization: str,
) -> None:
    """Set score_for_selection on each candidate (in-place) for global ranking.

    normalization: "none" | "scale-std" | "scale-range" | "z-score" | "min-max"
    - none: score_for_selection = score
    - scale-std: per slice, score / std (std=1 if std==0). Rescale only; no centering.
    - scale-range: per slice, score / (max - min) (span=1 if constant). Rescale only.
    - z-score: per slice, (score - mean) / std (std=1 if std==0)
    - min-max: per slice, (score - min) / (max - min) (0 if max==min)
    """
    for slice_candidates in per_slice_candidates:
        if not slice_candidates:
            continue
        scores = np.array([c["score"] for c in slice_candidates], dtype=np.float64)
        if normalization == "none":
            for c in slice_candidates:
                c["score_for_selection"] = c["score"]
        elif normalization == "scale-std":
            std = float(np.std(scores))
            if std == 0:
                std = 1.0
            for c in slice_candidates:
                c["score_for_selection"] = c["score"] / std
        elif normalization == "scale-range":
            lo, hi = float(np.min(scores)), float(np.max(scores))
            span = (hi - lo) if hi > lo else 1.0
            for c in slice_candidates:
                c["score_for_selection"] = c["score"] / span
        elif normalization == "z-score":
            mean = float(np.mean(scores))
            std = float(np.std(scores))
            if std == 0:
                std = 1.0
            for c in slice_candidates:
                c["score_for_selection"] = (c["score"] - mean) / std
        elif normalization == "min-max":
            lo, hi = float(np.min(scores)), float(np.max(scores))
            span = (hi - lo) if hi > lo else 1.0
            for c in slice_candidates:
                c["score_for_selection"] = (c["score"] - lo) / span
        else:
            for c in slice_candidates:
                c["score_for_selection"] = c["score"]


def _run_slice_search_one(
    data: InfluenceData,
    rollout_slice: Dict,
    demo_split: SplitType,
    window_width: int,
    aggregation_method: str,
    per_slice_top_k: int,
    ascending: bool,
    use_all_demos_per_slice: bool,
) -> Tuple[int, int, int, List[Dict]]:
    """Process one rollout slice; used for parallel slice search.

    Returns:
        (rollout_idx, start, end, slice_candidates)
    """
    rollout_ep = rollout_slice["rollout_ep"]
    rollout_idx = rollout_slice["rollout_idx"]
    start = rollout_slice["start"]
    end = rollout_slice["end"]
    rollout_start_idx = rollout_ep.sample_start_idx + start
    rollout_end_idx = rollout_ep.sample_start_idx + end + 1

    sorted_indices, sorted_scores, _ = rank_demos_by_slice_influence(
        data=data,
        rollout_start_idx=rollout_start_idx,
        rollout_end_idx=rollout_end_idx,
        window_width=window_width,
        aggregation_method=aggregation_method,
        split=demo_split,
        ascending=ascending,
    )

    n_take = len(sorted_indices) if use_all_demos_per_slice else min(per_slice_top_k, len(sorted_indices))
    slice_candidates = []
    for i in range(n_take):
        local_sample_idx = sorted_indices[i]
        score = float(sorted_scores[i])
        slice_candidates.append({
            "local_sample_idx": local_sample_idx,
            "score": score,
            "source_episode_idx": rollout_idx,
            "source_start": start,
            "source_end": end,
        })
    return (rollout_idx, start, end, slice_candidates)


def _run_slice_search(
    data: InfluenceData,
    rollout_slices: List[Dict],
    demo_split: SplitType,
    window_width: int,
    aggregation_method: str,
    per_slice_top_k: int,
    ascending: bool,
    use_all_demos_per_slice: bool = False,
) -> Tuple[List[Dict], List[List[Dict]]]:
    """Run the actual slice search computation (parallel over rollout slices).

    use_all_demos_per_slice: If True (e.g. for per-slice n-sigma selection), include
        all demo slices per rollout slice so z-score stats are computed over the full
        set, not just top-k.

    Returns:
        Tuple of (all_candidates, per_slice_candidates) where per_slice_candidates
        is a list of lists, one per rollout slice, each containing the candidates
        from that slice (for per-slice ranking charts).
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    max_workers = min(len(rollout_slices), (os.cpu_count() or 4))

    def task(slice_idx: int):
        return _run_slice_search_one(
            data=data,
            rollout_slice=rollout_slices[slice_idx],
            demo_split=demo_split,
            window_width=window_width,
            aggregation_method=aggregation_method,
            per_slice_top_k=per_slice_top_k,
            ascending=ascending,
            use_all_demos_per_slice=use_all_demos_per_slice,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        ordered_results = list(
            executor.map(task, range(len(rollout_slices)))
        )

    per_slice_candidates = [r[3] for r in ordered_results]
    all_candidates = [c for slice_cands in per_slice_candidates for c in slice_cands]
    return all_candidates, per_slice_candidates


def _resolve_candidates_to_unique_slices(
    candidates: List[Dict],
    data: InfluenceData,
    demo_split: SplitType,
    demo_episodes: list,
    ep_idxs: list,
    window_width: int,
) -> List[Dict]:
    """Resolve candidate dicts to demo (episode, start, end) and de-duplicate by slice.

    Uses the vectorized O(total_samples) lookup rather than an O(C×E) nested loop.

    Returns:
        List of resolved result dicts (each has episode, demo_start, demo_end, candidate, etc.).
    """
    num_train_samples = len(data.demo_sample_infos)
    lookup = _build_local_sample_lookup(data, demo_split, ep_idxs, demo_episodes, window_width)

    resolved_results = []
    for candidate in candidates:
        local_sample_idx = candidate["local_sample_idx"]
        t = lookup.get(local_sample_idx)
        if t is None:
            continue
        ep_idx, _episode_idx, demo_start, demo_end = t
        episode = demo_episodes[ep_idx]
        if demo_split == "holdout":
            global_sample_idx = local_sample_idx + num_train_samples
        else:
            global_sample_idx = local_sample_idx
        if global_sample_idx >= len(data.all_demo_sample_infos):
            continue
        sample_info = data.all_demo_sample_infos[global_sample_idx]
        resolved_results.append(
            {
                "candidate": candidate,
                "episode": episode,
                "sample_info": sample_info,
                "global_sample_idx": global_sample_idx,
                "local_sample_idx": local_sample_idx,
                "demo_start": demo_start,
                "demo_end": demo_end,
            }
        )

    seen_slices: set = set()
    resolved_unique = []
    for result in resolved_results:
        key = (result["episode"].index, result["demo_start"], result["demo_end"])
        if key not in seen_slices:
            seen_slices.add(key)
            resolved_unique.append(result)
    return resolved_unique


def _render_search_results_with_add_button(
    data: InfluenceData,
    global_top_k: List[Dict],
    demo_split: SplitType,
    demo_episodes: list,
    ep_idxs: list,
    obs_key: str,
    annotation_file: str,
    task_config: str,
    active_config_name: Optional[str],
    selected_label: str,
    window_width: int,
    aggregation_method: str = "mean",
    selection_method: str = "ranking_based",
    selection_normalization: str = "z-score",
    selection_mode: str = "global_top_k",
    n_sigma: Optional[float] = None,
    global_top_k_input: int = 10,
    percentile: Optional[float] = None,
):
    """Render search results and provide an 'Add to Curation Config' button."""
    with st.spinner(f"Resolving {len(global_top_k)} candidates to demo slices..."):
        resolved_results_unique = _resolve_candidates_to_unique_slices(
            global_top_k, data, demo_split, demo_episodes, ep_idxs, window_width
        )
    if not resolved_results_unique:
        st.warning("Could not resolve any results to demo episodes.")
        return

    pct_already_in: Optional[float] = None
    if active_config_name and resolved_results_unique:
        try:
            curation_config = load_curation_config(task_config, active_config_name)
            config_slices = curation_config.slices or []
            config_by_ep: Dict[int, List[Tuple[int, int]]] = {}
            for cs in config_slices:
                config_by_ep.setdefault(cs.episode_idx, []).append((cs.start, cs.end))
            already_in = 0
            for result in resolved_results_unique:
                ep_idx = result["episode"].index
                start, end = result["demo_start"], result["demo_end"]
                for cs_start, cs_end in config_by_ep.get(ep_idx, []):
                    if start <= cs_end and end >= cs_start:
                        already_in += 1
                        break
            pct_already_in = 100.0 * already_in / len(resolved_results_unique)
        except Exception:
            pct_already_in = None

    # Add to curation config button (uses unique demo slices)
    if active_config_name:
        st.markdown(
            f"**Active curation config:** `{active_config_name}` "
            f"— Add selected results to this config."
        )

        button_label = f"Add {len(resolved_results_unique)} unique demo slice(s) to '{active_config_name}'"
        if pct_already_in is not None:
            button_label += f" ({pct_already_in:.0f}% already in config)"
        if st.button(
            button_label,
            key=f"learning_add_to_config_{demo_split}_{selected_label}",
            type="primary",
        ):
            selection_method_metadata: Dict[str, Any] = {
                "selection_method": selection_method,
                "window_width": window_width,
                "aggregation_method": aggregation_method,
                "selection_normalization": selection_normalization,
                "selection_mode": selection_mode,
                "n_sigma": n_sigma,
                "global_top_k": global_top_k_input,
            }
            if percentile is not None:
                selection_method_metadata["percentile"] = percentile
            with st.spinner("Adding results to config..."):
                _add_results_to_config(
                    task_config=task_config,
                    config_name=active_config_name,
                    resolved_results=resolved_results_unique,
                    selected_label=selected_label,
                    demo_split=demo_split,
                    data=data,
                    selection_method_metadata=selection_method_metadata,
                )
    else:
        st.info(
            "No curation config selected. Create or select one above to add results."
        )

    # Selection distributions — gated behind button to avoid expensive eager computation
    _dist_key = f"learning_show_sel_distributions_{demo_split}_{selected_label}"
    if st.button(
        f"Show selection distributions ({len(resolved_results_unique)} unique slices)",
        key=f"learning_btn_sel_distributions_{demo_split}_{selected_label}",
    ):
        st.session_state[_dist_key] = not st.session_state.get(_dist_key, False)
    if st.session_state.get(_dist_key, False):
        if active_config_name:
            st.caption(f"Existing: curation config '{active_config_name}'")
            _render_selection_distributions(
                task_config,
                active_config_name,
                title_prefix="Existing:",
                data=data,
                demo_split=demo_split,
            )
        st.caption("Current search (results above, de-duplicated by demo slice)")
        _render_current_search_distributions(
            resolved_results_unique,
            key_suffix=f"{demo_split}_{selected_label}",
        )

    st.divider()

    # Render each result (unique demo slices only) — gated behind button
    direction = (
        "Most"
        if not st.session_state.get(f"learning_search_ascending_{demo_split}", False)
        else "Least"
    )
    n_unique = len(resolved_results_unique)
    n_total = len(global_top_k)
    subheader_suffix = f" ({n_unique} unique)" if n_unique != n_total else ""

    _results_key = f"learning_show_results_list_{demo_split}_{selected_label}"
    if st.button(
        f"Show top {n_unique} {direction.lower()} influential demo slices{subheader_suffix}",
        key=f"learning_btn_results_list_{demo_split}_{selected_label}",
    ):
        st.session_state[_results_key] = not st.session_state.get(_results_key, False)
    if st.session_state.get(_results_key, False):
        results_per_page = 10
        num_results = n_unique
        num_pages = max(1, (num_results + results_per_page - 1) // results_per_page)
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=num_pages,
            value=1,
            key=f"learning_topk_results_page_{demo_split}_{selected_label}",
            help=f"Show {results_per_page} results per page ({num_pages} total pages).",
        )
        page = max(1, min(page, num_pages))
        start_idx = (page - 1) * results_per_page
        end_idx = min(start_idx + results_per_page, num_results)
        st.caption(f"Showing results {start_idx + 1}–{end_idx} of {num_results}.")

        for i, result in enumerate(resolved_results_unique[start_idx:end_idx]):
            rank = start_idx + i + 1
            candidate = result["candidate"]
            episode = result["episode"]
            sample_info = result["sample_info"]

            influence = {
                "influence_score": candidate["score"],
                "demo_episode_idx": episode.index,
                "demo_timestep": sample_info.timestep,
                "local_demo_sample_idx": result["local_sample_idx"],
                "global_demo_sample_idx": result["global_sample_idx"],
                "sample_info": sample_info,
                "episode": episode,
            }

            st.markdown(
                f"**Source:** Rollout ep{candidate['source_episode_idx']} "
                f"t[{candidate['source_start']}:{candidate['source_end']}] | "
                f"**Demo slice:** ep{episode.index} "
                f"t[{result['demo_start']}:{result['demo_end']}]"
            )

            _render_influence_detail(
                data=data,
                influence=influence,
                rank=rank,
                rollout_episode_idx=0,
                rollout_sample_idx=0,
                obs_key=obs_key,
                split=demo_split,
                demo_episodes=demo_episodes,
                key_prefix=f"learning_result_{selected_label}_",
                annotation_file=annotation_file,
                task_config=task_config,
                show_trajectory_heatmap=False,
            )


def _add_results_to_config(
    task_config: str,
    config_name: str,
    resolved_results: List[Dict],
    selected_label: str,
    demo_split: SplitType,
    data: InfluenceData,
    selection_method_metadata: Optional[Dict[str, Any]] = None,
):
    """Add search results as one selection to the selection config (same name as curation config)."""
    slices_with_rollout: List[Tuple] = []
    for result in resolved_results:
        episode = result["episode"]
        demo_start = result["demo_start"]
        demo_end = result["demo_end"]
        candidate = result["candidate"]
        cs = CurationSlice(
            episode_idx=episode.index,
            start=demo_start,
            end=demo_end,
            label=selected_label,
            source=f"behavior_search_{demo_split}",
        )
        ro_ep = candidate.get("source_episode_idx", 0)
        ro_start = candidate.get("source_start", 0)
        ro_end = candidate.get("source_end", 0)
        local_idx = candidate.get("local_sample_idx")
        slices_with_rollout.append((cs, ro_ep, ro_start, ro_end, local_idx))
    # Build episode_lengths from resolved results so config stays valid
    episode_lengths = {}
    for result in resolved_results:
        ep = result["episode"]
        ep_len = getattr(ep, "raw_length", None) or ep.num_samples
        episode_lengths[ep.index] = ep_len
    episode_ends = None
    if hasattr(data, "demo_dataset") and hasattr(data.demo_dataset, "replay_buffer"):
        episode_ends = data.demo_dataset.replay_buffer.episode_ends[:]
    if episode_ends is None:
        st.warning(
            "Dataset fingerprint unavailable (replay buffer not accessible). "
            "The curation config will be saved **without** dataset integrity protection — "
            "training will not verify the config matches the dataset before applying the mask. "
            "Ensure the same dataset is used in the visualizer and during training."
        )
    try:
        add_selection_to_config(
            task_config_name=task_config,
            config_name=config_name,
            slices_with_rollout=slices_with_rollout,
            label=selected_label,
            demo_split=demo_split,
            source=f"behavior_search_{demo_split}",
            episode_lengths=episode_lengths,
            episode_ends=episode_ends,
            selection_method_metadata=selection_method_metadata,
        )
        st.success(
            f"Added 1 selection ({len(slices_with_rollout)} slices) to '{config_name}'."
        )
    except Exception as e:
        st.error(str(e))


# ---------------------------------------------------------------------------
# Section 2b: Add behaviors by sigma (batch)
# ---------------------------------------------------------------------------


@st.fragment
def _render_batch_add_behaviors_fragment(
    data: InfluenceData,
    demo_split: SplitType,
    annotation_file: str,
    task_config: str,
    active_config_name: Optional[str],
):
    """GUI: list all behaviors with checkboxes, one sigma for all, add selected to config."""
    annotations = load_annotations(annotation_file, task_config=task_config)
    if not annotations:
        return

    all_labels = set()
    for rollout_ep in data.rollout_episodes:
        episode_id = str(rollout_ep.index)
        ep_annotations = get_episode_annotations(annotations, episode_id, "rollout")
        for slice_info in ep_annotations:
            all_labels.add(slice_info["label"])

    if not all_labels:
        return

    with st.expander("Add behaviors by sigma (batch)", expanded=False):
        st.caption(
            "Select one or more behaviors, set a threshold by sigma or percentile (applied to all), "
            "then add each selected behavior's selection to the current curation config."
        )

        sorted_labels = sorted(all_labels)
        selected_labels: List[str] = []
        cols = 4
        for i in range(0, len(sorted_labels), cols):
            row_labels = sorted_labels[i : i + cols]
            row_cols = st.columns(len(row_labels))
            for j, label in enumerate(row_labels):
                with row_cols[j]:
                    if st.checkbox(
                        label,
                        key=f"learning_batch_label_{demo_split}_{label}",
                        help=f"Add behavior '{label}' to config at chosen threshold",
                    ):
                        selected_labels.append(label)

        batch_threshold_type = st.radio(
            "Threshold by",
            options=["sigma", "percentile"],
            format_func=lambda x: "Sigma (σ)" if x == "sigma" else "Percentile (empirical)",
            key=f"learning_batch_threshold_type_{demo_split}",
            horizontal=True,
        )
        sigma_batch: Optional[float] = None
        percentile_batch: Optional[float] = None
        if batch_threshold_type == "sigma":
            sigma_batch = st.number_input(
                "Sigma (applied to all selected behaviors)",
                min_value=0.1,
                max_value=10.0,
                value=2.0,
                step=0.5,
                key=f"learning_batch_sigma_{demo_split}",
                help="Per-slice z-score threshold: keep demo slices with z ≥ this value",
            )
            st.caption(f"≈ {_sigma_to_percentile(sigma_batch):.1f}th percentile assuming normal")
        else:
            percentile_batch = st.number_input(
                "Percentile (applied to all selected behaviors)",
                min_value=50.0,
                max_value=99.99,
                value=97.7,
                step=0.5,
                key=f"learning_batch_percentile_{demo_split}",
                help="Keep demo slices at or above this empirical percentile within each slice. Robust to heavy-tailed distributions.",
            )
            st.caption(f"Top {100 - percentile_batch:.2f}% of scores per slice (empirical)")

        if not active_config_name:
            st.info("Select a curation config above to add behaviors.")
            return

        if st.button(
            f"Add {len(selected_labels)} selected behavior(s) to '{active_config_name}'",
            key=f"learning_batch_add_btn_{demo_split}",
            type="primary",
            disabled=len(selected_labels) == 0,
        ):
            _, demo_episodes, ep_idxs, _ = get_split_data(data, demo_split)
            window_width = 5
            aggregation_method = "mean"
            per_slice_top_k = 20
            ascending = False

            added = 0
            total_slices = 0
            errors: List[str] = []
            with st.spinner("Computing and adding selections..."):
                for label in selected_labels:
                    rollout_slices_with_label = []
                    for rollout_ep in data.rollout_episodes:
                        episode_id = str(rollout_ep.index)
                        ep_annotations = get_episode_annotations(
                            annotations, episode_id, "rollout"
                        )
                        for slice_info in ep_annotations:
                            if slice_info["label"] == label:
                                rollout_slices_with_label.append(
                                    {
                                        "rollout_idx": rollout_ep.index,
                                        "rollout_ep": rollout_ep,
                                        "start": slice_info["start"],
                                        "end": slice_info["end"],
                                    }
                                )
                    if not rollout_slices_with_label:
                        continue
                    all_candidates, per_slice_candidates = _run_slice_search(
                        data=data,
                        rollout_slices=rollout_slices_with_label,
                        demo_split=demo_split,
                        window_width=window_width,
                        aggregation_method=aggregation_method,
                        per_slice_top_k=per_slice_top_k,
                        ascending=ascending,
                        use_all_demos_per_slice=True,
                    )
                    if batch_threshold_type == "percentile" and percentile_batch is not None:
                        raw_selection = _per_slice_percentile_selection(
                            per_slice_candidates, percentile_batch
                        )
                    else:
                        raw_selection = _per_slice_n_sigma_selection(
                            per_slice_candidates, sigma_batch or 2.0
                        )
                    if not raw_selection:
                        continue
                    resolved_unique = _resolve_candidates_to_unique_slices(
                        raw_selection,
                        data,
                        demo_split,
                        demo_episodes,
                        ep_idxs,
                        window_width,
                    )
                    if not resolved_unique:
                        continue
                    batch_sel_mode = (
                        "per_slice_percentile" if batch_threshold_type == "percentile"
                        else "per_slice_n_sigma"
                    )
                    batch_meta: Dict[str, Any] = {
                        "selection_method": "ranking_based",
                        "window_width": window_width,
                        "aggregation_method": aggregation_method,
                        "selection_normalization": "z-score",
                        "selection_mode": batch_sel_mode,
                        "n_sigma": sigma_batch,
                    }
                    if percentile_batch is not None:
                        batch_meta["percentile"] = percentile_batch
                    try:
                        _add_results_to_config(
                            task_config=task_config,
                            config_name=active_config_name,
                            resolved_results=resolved_unique,
                            selected_label=label,
                            demo_split=demo_split,
                            data=data,
                            selection_method_metadata=batch_meta,
                        )
                        added += 1
                        total_slices += len(resolved_unique)
                    except Exception as e:
                        errors.append(f"{label}: {e}")
            if added > 0:
                st.success(
                    f"Added {added} selection(s) ({total_slices} slices total) to '{active_config_name}'."
                )
            if errors:
                for err in errors:
                    st.error(err)
            if added > 0 or errors:
                st.rerun()


# ---------------------------------------------------------------------------
# Main tab render function
# ---------------------------------------------------------------------------


def render_learning_tab(
    data: InfluenceData,
    demo_split: SplitType,
    top_k: int,
    obs_key: str,
    annotation_file: str,
    task_config: str = "",
):
    """Render the Learning tab for data curation management."""
    st.header("Data Curation")
    st.markdown("""
    Manage sample-level data curation configurations. Use this tab to:
    1. **Create/select** a curation config for this task
    2. **View** the current curation config with episode video players
    3. **Search** for influential behavior slices and add them to the config
    """)

    if not task_config:
        st.warning(
            "No task config selected. Please select a task config in the sidebar."
        )
        return

    # Curation mode: filtering (exclude slices from split) vs selection (include slices from holdout)
    curation_mode_choice = st.selectbox(
        "Curation mode",
        options=["curation-filtering", "curation-selection"],
        format_func=lambda x: (
            "Curation-filtering (exclude selected slices from training split)"
            if x == "curation-filtering"
            else "Curation-selection (select slices from holdout to add to training)"
        ),
        key=f"learning_curation_mode_{task_config}",
    )
    is_selection_mode = curation_mode_choice == "curation-selection"
    # Split is fixed: holdout for selection (add slices to training), train for filtering (exclude slices)
    effective_demo_split: SplitType = "holdout" if is_selection_mode else "train"
    curation_mode_value = "selection" if is_selection_mode else "filter"
    if is_selection_mode:
        st.caption(
            "Selection mode: you are selecting valuable slices from the **holdout** split to add to training. "
            "All configs and search below use holdout demonstrations."
        )

    # Section 1: Config selector
    active_config = _render_config_selector_fragment(
        data=data,
        task_config=task_config,
        demo_split=effective_demo_split,
        curation_mode=curation_mode_value,
    )

    if not active_config:
        st.info("Select or create a curation config to get started.")
        return

    st.divider()

    # Section 2: Config viewer (with View Selections and distribution charts)
    _render_config_viewer_fragment(
        data=data,
        task_config=task_config,
        config_name=active_config,
        obs_key=obs_key,
        demo_split=effective_demo_split,
        annotation_file=annotation_file,
    )

    st.divider()

    # Section 2b: Add behaviors by sigma (batch)
    _render_batch_add_behaviors_fragment(
        data=data,
        demo_split=effective_demo_split,
        annotation_file=annotation_file,
        task_config=task_config,
        active_config_name=active_config,
    )

    st.divider()

    # Section 3: Behavior slice search with add-to-config and distributions under top-k
    _render_slice_search_fragment(
        data=data,
        demo_split=effective_demo_split,
        annotation_file=annotation_file,
        task_config=task_config,
        obs_key=obs_key,
        active_config_name=active_config,
        is_selection_mode=is_selection_mode,
    )
