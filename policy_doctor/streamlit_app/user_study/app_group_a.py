from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st
import yaml

from policy_doctor.streamlit_app.user_study.likert_survey import (
    render_block2_strategy,
    render_block3_final,
)
from policy_doctor.streamlit_app.user_study.nasa_tlx import render_nasa_tlx
from policy_doctor.streamlit_app.user_study.response_store import get_store
from policy_doctor.streamlit_app.user_study.strategies import (
    load_study_config,
    render_strategy_allocator,
    render_strategy_summary,
)
from policy_doctor.streamlit_app.user_study.survey_steps import (
    N_STEPS,
    STEP_LABELS,
    advance_step,
    get_step_durations,
    record_step_entry,
    render_progress_bar,
    render_rollout_timer,
)
from policy_doctor.streamlit_app.user_study.video_browser import render_video_browser

st.set_page_config(page_title="User Study — Group A", layout="wide")

# ── Sidebar: config + session loading ────────────────────────────────────────

st.sidebar.header("Study configuration")

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

PFX = "ga"
STEP_KEY = f"{PFX}_step"


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else _REPO_ROOT / path


if st.session_state.get(f"{PFX}_loaded_session") != session_choice:
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
            st.session_state[f"{PFX}_index"] = json.load(f)
        cfg = load_study_config(config_path)
        st.session_state[f"{PFX}_strategies"] = cfg["strategies"]
        st.session_state[f"{PFX}_budget"] = cfg.get("budget", {}).get("total_demos", 500)
        st.session_state[f"{PFX}_alloc_step"] = cfg.get("budget", {}).get("allocation_step", 25)
        st.session_state[f"{PFX}_mp4_dir"] = str(mp4_dir)
        st.session_state[f"{PFX}_rollout_limit"] = sess.get("rollout_time_limit_seconds", 600)
        st.session_state[f"{PFX}_loaded_session"] = session_choice

if f"{PFX}_index" in st.session_state:
    st.sidebar.caption(f"{len(st.session_state[f'{PFX}_index']['episodes'])} episodes loaded")

index = st.session_state.get(f"{PFX}_index")
strategies = st.session_state.get(f"{PFX}_strategies")
mp4_dir_str = st.session_state.get(f"{PFX}_mp4_dir")

if index is None or strategies is None:
    st.error("Failed to load session — check the sidebar for errors.")
    st.stop()

mp4_dir = Path(mp4_dir_str)
total_budget = st.session_state.get(f"{PFX}_budget", 500)
alloc_step = st.session_state.get(f"{PFX}_alloc_step", 25)
rollout_limit = st.session_state.get(f"{PFX}_rollout_limit", 600)

# ── Step routing ──────────────────────────────────────────────────────────────

step = st.session_state.get(STEP_KEY, 0)

render_progress_bar(step, STEP_LABELS)
record_step_entry(step, STEP_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Consent
# ─────────────────────────────────────────────────────────────────────────────
if step == 0:
    st.title("Invitation to Participate in a Research Study")
    st.markdown("""
We would like to invite you to participate in a research study on **robot learning and data collection**.

The research investigates whether a visual representation of a robot policy's behavioral modes helps
human planners make more targeted and effective data-collection decisions.

**What you will do:**
- Watch videos of a robot attempting a manipulation task
- Decide how to allocate a data collection budget across different strategies
- Complete a brief survey about your experience

**Time commitment:** approximately **25–35 minutes** in total.

**Participation is voluntary.** You may withdraw at any time without any disadvantage.
You will not receive compensation for participation.

**Group assignment.** Participants are randomly assigned to one of two conditions.
Regardless of your condition, please proceed as instructed.

**Data privacy.** Your responses are recorded anonymously with your chosen participant ID.
""")

    agreed = st.checkbox(
        "I have read the above information and voluntarily agree to participate.",
        key=f"{PFX}_consent_agreed",
    )
    if st.button("Continue →", type="primary", disabled=not agreed):
        advance_step(0, STEP_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Introduction
# ─────────────────────────────────────────────────────────────────────────────
elif step == 1:
    st.title("Study Introduction")

    st.markdown("""
### Background: How does the robot learn?

Instead of programming the robot with explicit rules, we show it many examples of the task being done correctly.
The robot learns a *policy* — a mapping from what it sees (camera images, joint positions) to what action to take next.

This approach is called **Learning from Demonstrations (LfD)**.

### The task

The robot must pick up an object from a table and transport it to a goal location.
- ✓ **Success** — the object reaches the goal
- ✗ **Failure** — the robot drops it, misses, or runs out of time

### Your role

You will watch videos of the robot's current behavior, then decide how to allocate a
**demo collection budget** across different data collection strategies.
Think of it like a coach deciding which drills to run before the next game.
""")

    limit_min = rollout_limit // 60
    limit_sec = rollout_limit % 60
    limit_str = f"{limit_min} minute{'s' if limit_min != 1 else ''}" + (
        f" {limit_sec}s" if limit_sec else ""
    )

    st.warning(
        f"**About the next section — Rollout Info:**  \n"
        f"You will have **{limit_str}** to watch the robot videos. "
        f"A countdown timer will be shown at the top of the page. "
        f"Once the timer expires (or you click Proceed), you will move on automatically "
        f"and **cannot return** to the video page. "
        f"Use your time wisely — watch a variety of episodes, including successes and failures."
    )

    if st.button("Continue →", type="primary"):
        advance_step(1, STEP_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Rollout Info  (timer-gated)
# ─────────────────────────────────────────────────────────────────────────────
elif step == 2:
    st.title("Watch the Robot in Action")

    start_key = f"{PFX}_rollout_start"
    if start_key not in st.session_state:
        st.session_state[start_key] = st.session_state.get(
            f"{PFX}_times", {}).get(2, {}).get("start")
        if st.session_state[start_key] is None:
            import time as _time
            st.session_state[start_key] = _time.time()

    remaining, expired = render_rollout_timer(
        st.session_state[start_key],
        rollout_limit,
        key=f"{PFX}_rtimer",
    )

    st.markdown(
        "These videos show **rollouts** — the robot attempting the task from scratch. "
        "Watch several to get a feel for what it does well and where it struggles. "
        "Pay attention to how it handles the object and whether it reaches the goal."
    )

    n_ep = len(index["episodes"])
    n_succ = sum(1 for ep in index["episodes"] if ep.get("success") is True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total rollouts", n_ep)
    c2.metric("Successes", n_succ)
    c3.metric("Success rate", f"{n_succ / n_ep:.0%}" if n_ep else "—")

    render_video_browser(mp4_dir, index, page_size=9, key_prefix=f"{PFX}_vbrow")

    st.divider()
    if expired or st.button("Proceed to Data Collection →", type="primary"):
        advance_step(2, STEP_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Data Collection
# ─────────────────────────────────────────────────────────────────────────────
elif step == 3:
    st.title("Design Your Data Collection Strategy")
    st.markdown(
        "Allocate your demo budget across the strategies below. "
        "Each strategy corresponds to a specific data collection protocol. "
        f"Total budget: **{total_budget} demos**."
    )

    allocations = render_strategy_allocator(
        strategies,
        total_budget=total_budget,
        allocation_step=alloc_step,
        key_prefix=f"{PFX}_alloc",
    )
    render_strategy_summary(allocations, strategies, total_budget)

    st.divider()
    if st.button("Continue to Survey →", type="primary"):
        st.session_state[f"{PFX}_allocations"] = allocations
        advance_step(3, STEP_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Survey + Submit
# ─────────────────────────────────────────────────────────────────────────────
elif step == 4:
    st.title("Survey")

    st.header("1 — Strategy-Selection Survey")
    likert_strategy = render_block2_strategy(key_prefix=f"{PFX}_likert")

    st.divider()
    st.header("2 — NASA Task Load Index")
    tlx_responses = render_nasa_tlx(key_prefix=f"{PFX}_tlx")

    st.divider()
    st.header("3 — Final Assessment")
    likert_final = render_block3_final(
        key_prefix=f"{PFX}_likert", include_graph_questions=False,
    )

    st.divider()
    notes = st.text_area(
        "Any additional notes or reasoning",
        value="",
        height=120,
        placeholder="e.g. I focused on failures because ...",
    )

    if st.button("Submit", type="primary"):
        record_step_exit(4, STEP_KEY)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = {
            "participant_id": participant_id,
            "group": "A",
            "session": session_choice,
            "allocations": st.session_state.get(f"{PFX}_allocations", allocations),
            "nasa_tlx": tlx_responses,
            "likert_strategy": likert_strategy,
            "likert_final": likert_final,
            "notes": notes,
            "timestamp": timestamp,
            "step_durations_seconds": get_step_durations(STEP_KEY),
        }

        store = get_store(mp4_dir / "study_responses")
        response_id = store.save(result)
        st.success(f"Response saved (ID: {response_id})")
        st.download_button(
            label="Download your response",
            data=json.dumps(result, indent=2),
            file_name=f"group_a_{timestamp}.json",
            mime="application/json",
        )
        st.balloons()
