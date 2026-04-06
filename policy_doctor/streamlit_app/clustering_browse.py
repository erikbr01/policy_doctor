"""Streamlit: episode frame player with cluster timelines (aligned with influence_visualizer)."""

from __future__ import annotations

import base64
from html import escape
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st

from policy_doctor import plotting
from policy_doctor.behaviors.clustering_temporal import (
    build_cluster_timeline,
    build_episode_cluster_map,
    resolve_rollout_episode,
)
from policy_doctor.data.influence_loader import InfluenceDataContainer
from policy_doctor.data.structures import EpisodeInfo


def _resolve_demo_episode(
    data: InfluenceDataContainer,
    demo_split: str,
    ep_id: int,
) -> Optional[EpisodeInfo]:
    if demo_split == "holdout":
        episodes = data.holdout_episodes
    elif demo_split == "both":
        episodes = data.all_demo_episodes
    else:
        episodes = data.demo_episodes
    for ep in episodes:
        if ep.index == ep_id:
            return ep
    return None


def _render_annotated_frame_height_capped(
    frame: Any,
    label: str,
    caption: Optional[str],
    *,
    max_height_px: int = 600,
    max_width_px: int = 800,
    font_size: int = 12,
    max_upscale: float = 12.0,
) -> None:
    """Annotated frame scaled to fit a box (downscale *or upscale*), centered in HTML.

    Small dataset frames are enlarged up to ``max_width_px`` × ``max_height_px`` so
    previews are readable. Large frames are shrunk to fit that box. ``max_upscale``
    limits pathological tiny inputs.
    """
    from influence_visualizer.plotting import create_annotated_frame
    from PIL import Image

    if frame is None:
        st.warning("Frame not available")
        return
    pil_img = create_annotated_frame(frame, label, font_size=font_size)
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS

    w, h = pil_img.size
    scale = min(max_width_px / float(w), max_height_px / float(h))
    scale = min(scale, max_upscale)
    if abs(scale - 1.0) > 1e-6:
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        pil_img = pil_img.resize((nw, nh), resample)

    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    w = int(pil_img.width)
    cap_html = (
        f'<p style="text-align:center;color:#666;font-size:0.9rem;margin:0.5rem 0 0 0">{escape(caption)}</p>'
        if caption
        else ""
    )
    st.markdown(
        f'<div style="display:flex;flex-direction:column;align-items:center;width:100%">'
        f'<img src="data:image/png;base64,{b64}" width="{w}" alt="frame" '
        f'style="width:{w}px;max-width:100%;height:auto;" />'
        f"{cap_html}</div>",
        unsafe_allow_html=True,
    )


@st.fragment
def render_cluster_episode_browser(
    data: InfluenceDataContainer,
    cluster_labels: np.ndarray,
    metadata: List[Dict[str, Any]],
    *,
    representation: str = "sliding_window",
    level: str = "rollout",
    demo_split: str = "train",
    key_prefix: str = "pd_cl_browse",
    annotations: Optional[Dict[str, Any]] = None,
) -> None:
    """Paginated expanders per episode: cluster timeline + optional human labels + frame player."""
    # Accept either a fully backed InfluenceDataContainer (preferred) or any compatible object
    # that provides frame/action accessors. Also support older cached containers that keep
    # the underlying IV object in ``_iv_source`` but lack proxy methods.
    source = data
    if not (hasattr(source, "get_rollout_frame") and hasattr(source, "get_demo_frame")):
        maybe_iv = getattr(data, "_iv_source", None)
        if maybe_iv is not None and hasattr(maybe_iv, "get_rollout_frame") and hasattr(maybe_iv, "get_demo_frame"):
            source = maybe_iv
        else:
            st.warning(
                "Video frames require influence data with frame accessors. "
                "Ensure eval_dir/train_dir/image_dataset_path are set in the task YAML and reload the app."
            )
            st.caption(f"Debug: data type = `{type(data).__name__}`")
            return

    from influence_visualizer.render_frames import frame_player

    episode_cluster_map = build_episode_cluster_map(
        cluster_labels, metadata, representation, level
    )
    if not episode_cluster_map:
        st.warning("No episodes found in clustering metadata.")
        return

    all_cluster_ids = sorted(set(int(x) for x in cluster_labels))
    has_true_noise = bool(np.any(cluster_labels == -1))

    st.subheader("Browse episodes (video + cluster timeline)")
    st.caption(
        f"{'Rollouts' if level == 'rollout' else 'Demonstrations'} — cluster assignments over time "
        "(same idea as influence_visualizer temporal coherence browsing)."
    )

    sorted_episodes = sorted(episode_cluster_map.keys())

    episodes_per_page = st.slider(
        "Episodes per page",
        min_value=1,
        max_value=20,
        value=5,
        key=f"{key_prefix}_episodes_per_page",
    )
    total_pages = max(1, (len(sorted_episodes) + episodes_per_page - 1) // episodes_per_page)
    col_page1, col_page2, col_page3 = st.columns([1, 2, 1])
    with col_page2:
        current_page = st.number_input(
            f"Page (1–{total_pages})",
            min_value=1,
            max_value=total_pages,
            value=1,
            key=f"{key_prefix}_page_num",
        )

    start_idx = (int(current_page) - 1) * episodes_per_page
    end_idx = min(start_idx + episodes_per_page, len(sorted_episodes))
    page_episodes = sorted_episodes[start_idx:end_idx]

    st.markdown(f"**Episodes {start_idx + 1}–{end_idx} of {len(sorted_episodes)}**")

    for ep_id in page_episodes:
        if level == "rollout":
            ep = resolve_rollout_episode(data.rollout_episodes, int(ep_id))
            ep_label = f"Rollout {ep_id}"
            if ep is None:
                with st.expander(f"{ep_label} (missing episode)", expanded=False):
                    st.error("Episode not found in loaded rollout list.")
                continue
            status = "Success" if ep.success else "Failure"
        else:
            ep = _resolve_demo_episode(data, demo_split, int(ep_id))
            ep_label = f"Demo {ep_id}"
            if ep is None:
                with st.expander(f"{ep_label} (missing episode)", expanded=False):
                    st.error("Episode not found for this demo split.")
                continue
            ql = data.demo_quality_labels
            status = ql.get(ep.index, "unknown") if isinstance(ql, dict) else "unknown"

        num_frames = ep.num_samples
        cluster_timeline = build_cluster_timeline(
            num_frames, int(ep_id), episode_cluster_map, representation
        )
        num_unassigned = sum(1 for c in cluster_timeline if c == -1)

        episode_annotations: List[Dict[str, Any]] = []
        if annotations:
            from influence_visualizer.render_annotation import get_episode_annotations

            ann_split = "rollout" if level == "rollout" else demo_split
            try:
                episode_annotations = get_episode_annotations(
                    annotations, str(ep_id), split=ann_split
                )
            except ValueError:
                episode_annotations = []

        with st.expander(
            f"{ep_label} | {status} | {num_frames} frames",
            expanded=False,
        ):
            if num_unassigned > 0 and not has_true_noise:
                st.caption(
                    f"{num_unassigned}/{num_frames} frames are unassigned (not covered by any window). "
                    "Normal when stride > 1 or edges have no full window."
                )

            def _make_render_fn(
                _level: str,
                _episode: EpisodeInfo,
                _timeline: List[int],
                _ep_id: int,
                _has_noise: bool,
                _nframes: int,
                _all_ids: List[int],
                _ann: List[Dict[str, Any]],
            ):
                def _render_frame(t: int) -> None:
                    st.caption("Automatic cluster assignments")
                    fig_t = plotting.create_cluster_timeline(
                        cluster_assignments=_timeline,
                        num_frames=_nframes,
                        current_frame=t,
                        has_true_noise=_has_noise,
                        all_cluster_ids=_all_ids,
                    )
                    st.plotly_chart(fig_t, use_container_width=True, key=f"{key_prefix}_tl_{_ep_id}_{t}")

                    st.caption("Human annotations")
                    fig_a = plotting.create_label_timeline(
                        _ann,
                        _nframes,
                        current_frame=t,
                        unlabeled_name="Unassigned",
                    )
                    st.plotly_chart(fig_a, use_container_width=True, key=f"{key_prefix}_ann_{_ep_id}_{t}")

                    abs_idx = _episode.sample_start_idx + t
                    if _level == "rollout":
                        frame = source.get_rollout_frame(abs_idx)
                    else:
                        frame = source.get_demo_frame(abs_idx)

                    cid = _timeline[t]
                    if cid == -1:
                        if _has_noise:
                            st.info(f"Frame t={t} | Cluster: noise (label −1)")
                        else:
                            st.warning(f"Frame t={t} | Unassigned (no window)")
                    else:
                        st.success(f"Frame t={t} | Cluster: {cid}")

                    _render_annotated_frame_height_capped(
                        frame,
                        f"t={t}",
                        f"{'Rollout' if _level == 'rollout' else 'Demo'} {_ep_id}",
                        max_height_px=600,
                        max_width_px=800,
                    )

                return _render_frame

            render_fn = _make_render_fn(
                level,
                ep,
                cluster_timeline,
                int(ep_id),
                has_true_noise,
                num_frames,
                all_cluster_ids,
                episode_annotations,
            )

            frame_player(
                label="Timestep",
                min_value=0,
                max_value=max(0, num_frames - 1),
                key=f"{key_prefix}_player_{ep_id}",
                default_value=0,
                default_fps=3.0,
                render_fn=render_fn,
                fragment_scope=True,
            )
