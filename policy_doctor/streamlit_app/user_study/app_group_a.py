from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st
import yaml

from policy_doctor.streamlit_app.user_study.intro import gate_or_render
from policy_doctor.streamlit_app.user_study.likert_survey import (
    render_block2_strategy,
    render_block3_final,
)
from policy_doctor.streamlit_app.user_study.nasa_tlx import render_nasa_tlx
from policy_doctor.streamlit_app.user_study.strategies import (
    load_study_config,
    render_strategy_allocator,
    render_strategy_summary,
)
from policy_doctor.streamlit_app.user_study.video_browser import render_video_browser

st.set_page_config(page_title="User Study — Group A", layout="wide")

gate_or_render()

st.title("User Study: Teaching Robots from Examples")
st.markdown(
    "**Welcome!** In this study, you will help improve a robot arm that learns by watching "
    "human demonstrations — much like how a person learns a new skill by watching someone else do it. "
    "The more demonstrations the robot sees, and the more carefully chosen they are, "
    "the better it gets at the task."
)

with st.expander("📖 Background: How does the robot learn?", expanded=False):
    st.markdown("""
**Learning from Demonstrations (LfD)**

Instead of programming the robot with explicit rules, we show it many examples of the task being done correctly.
The robot learns a *policy* — a mapping from what it sees (camera images, joint positions) to what action to take next.

**The task:** The robot must pick up an object from a table and transport it to a goal location.
- ✓ **Success** — the object reaches the goal
- ✗ **Failure** — the robot drops it, misses, or runs out of time

**Your role:** You'll watch videos of the robot's current behavior, then decide how to allocate a *data collection budget* — choosing how many new demonstrations to collect, and of what kind. Think of it like a coach deciding which drills to run before the next game.
""")

st.sidebar.header("Configuration")

_REPO_ROOT = Path(__file__).parents[3]
_SESSIONS_DIR = _REPO_ROOT / "policy_doctor" / "configs" / "user_study" / "sessions"
_session_files = sorted(_SESSIONS_DIR.glob("*.yaml")) if _SESSIONS_DIR.is_dir() else []
_session_labels = {f.stem: yaml.safe_load(f.read_text()).get("label", f.stem) for f in _session_files}

participant_id = st.sidebar.text_input("Participant ID", value="anonymous")

if not _session_files:
    st.sidebar.warning("No session configs found in configs/user_study/sessions/")
    st.stop()

session_choice = st.sidebar.selectbox(
    "Session",
    options=[f.stem for f in _session_files],
    format_func=lambda k: _session_labels.get(k, k),
)


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else _REPO_ROOT / path


# Auto-load on session change (no Load button click required).
if st.session_state.get("ga_loaded_session") != session_choice:
    sess_path = _SESSIONS_DIR / f"{session_choice}.yaml"
    sess = yaml.safe_load(sess_path.read_text())
    mp4_dir = _resolve(sess["mp4_dir"])
    config_path = _resolve(sess["study_config"])

    errors = []
    index_path = mp4_dir / "index.json"
    if not mp4_dir.is_dir():
        errors.append(f"MP4 directory not found: {mp4_dir}")
    elif not index_path.exists():
        errors.append("index.json not found in MP4 directory")
    if not config_path.exists():
        errors.append(f"Study config not found: {config_path}")

    if errors:
        for e in errors:
            st.sidebar.error(e)
    else:
        with open(index_path) as f:
            st.session_state["ga_index"] = json.load(f)
        cfg = load_study_config(config_path)
        st.session_state["ga_strategies"] = cfg["strategies"]
        st.session_state["ga_budget"] = cfg.get("budget", {}).get("total_demos", 500)
        st.session_state["ga_alloc_step"] = cfg.get("budget", {}).get("allocation_step", 25)
        st.session_state["ga_mp4_dir"] = str(mp4_dir)
        st.session_state["ga_loaded_session"] = session_choice

if "ga_index" in st.session_state:
    st.sidebar.caption(
        f"{len(st.session_state['ga_index']['episodes'])} episodes loaded"
    )

index = st.session_state.get("ga_index")
strategies = st.session_state.get("ga_strategies")
mp4_dir_str = st.session_state.get("ga_mp4_dir")

if index is None or strategies is None:
    st.error("Failed to load session — check the sidebar for errors.")
    st.stop()

mp4_dir = Path(mp4_dir_str)
total_budget = st.session_state.get("ga_budget", 500)
alloc_step = st.session_state.get("ga_alloc_step", 25)

st.header("1. Watch the Robot in Action")
st.markdown(
    "These videos show **rollouts** — the robot attempting the task from scratch. "
    "Watch several to get a feel for what it does well and where it struggles. "
    "Pay attention to how it handles the object and whether it reaches the goal."
)

render_video_browser(mp4_dir, index, page_size=9, key_prefix="vbrow")

st.divider()

st.header("2. Design Your Data Collection Strategy")
st.markdown(
    """
Allocate your data collection budget across the strategies below.
Each strategy corresponds to a specific data collection protocol.
The total budget is shown in the progress bar — stay within it.
"""
)

allocations = render_strategy_allocator(
    strategies,
    total_budget=total_budget,
    allocation_step=alloc_step,
    key_prefix="ga_alloc",
)
render_strategy_summary(allocations, strategies, total_budget)

st.divider()

st.header("3. Strategy-Selection Survey")
likert_strategy = render_block2_strategy(key_prefix="ga_likert")

st.divider()

st.header("4. NASA Task Load Index")
tlx_responses = render_nasa_tlx(key_prefix="ga_tlx")

st.divider()

st.header("5. Final Assessment")
likert_final = render_block3_final(
    key_prefix="ga_likert", include_graph_questions=False,
)

st.divider()

st.header("6. Submit")

notes = st.text_area("Any additional notes or reasoning", value="", height=120)

if st.button("Submit", type="primary"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = mp4_dir / "study_responses"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"group_a_{timestamp}.json"

    result = {
        "participant_id": participant_id,
        "group": "A",
        "allocations": allocations,
        "nasa_tlx": tlx_responses,
        "likert_strategy": likert_strategy,
        "likert_final": likert_final,
        "notes": notes,
        "timestamp": timestamp,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    st.success(f"Response saved to {out_path}")
    st.download_button(
        label="Download your response",
        data=json.dumps(result, indent=2),
        file_name=f"group_a_{timestamp}.json",
        mime="application/json",
    )
