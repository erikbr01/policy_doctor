from __future__ import annotations

from pathlib import Path

import streamlit as st

from policy_doctor.streamlit_app.components.mp4_player import mp4_player


def render_video_browser(
    mp4_dir: Path,
    index: dict,
    page_size: int = 5,
    key_prefix: str = "vbrow",
) -> None:
    episodes = index["episodes"]
    fps = index.get("fps", 10)

    filter_key = f"{key_prefix}_filter"
    page_key = f"{key_prefix}_page"

    filter_choice = st.radio(
        "Filter episodes",
        options=["All", "Success only", "Failure only"],
        horizontal=True,
        key=filter_key,
    )

    if filter_choice == "Success only":
        filtered = [ep for ep in episodes if ep.get("success") is True]
    elif filter_choice == "Failure only":
        filtered = [ep for ep in episodes if ep.get("success") is False]
    else:
        filtered = episodes

    if st.session_state.get(f"{key_prefix}_last_filter") != filter_choice:
        st.session_state[page_key] = 0
        st.session_state[f"{key_prefix}_last_filter"] = filter_choice

    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    current_page = int(st.session_state[page_key])
    current_page = max(0, min(current_page, total_pages - 1))
    st.session_state[page_key] = current_page

    col_prev, col_label, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("← Prev", key=f"{key_prefix}_prev", disabled=current_page == 0):
            st.session_state[page_key] = current_page - 1
            st.rerun()
    with col_label:
        st.markdown(
            f"<div style='text-align:center;padding-top:6px;'>Page {current_page + 1} / {total_pages}</div>",
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button("Next →", key=f"{key_prefix}_next", disabled=current_page >= total_pages - 1):
            st.session_state[page_key] = current_page + 1
            st.rerun()

    start = current_page * page_size
    page_episodes = filtered[start : start + page_size]

    for ep in page_episodes:
        success = ep.get("success")
        if success is True:
            status = "✓ Success"
        elif success is False:
            status = "✗ Failure"
        else:
            status = "Unknown"

        with st.expander(f"Episode {ep['index']} — {status}"):
            video_path = mp4_dir / ep["path"]
            mp4_player(
                video_path,
                label="",
                key=f"{key_prefix}_ep{ep['index']}",
            )
            st.caption(f"{ep['frame_count']} frames at {fps} fps")
