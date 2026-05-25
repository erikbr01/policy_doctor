from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from policy_doctor.streamlit_app.appearance import get_theme, render_appearance_sidebar
from policy_doctor.streamlit_app.user_study.likert_survey import (
    render_block2_strategy,
    render_block3_final,
)
from policy_doctor.streamlit_app.user_study.nasa_tlx import render_nasa_tlx
from policy_doctor.streamlit_app.user_study.response_store import get_store
from policy_doctor.streamlit_app.user_study.task_setup import load_task
from policy_doctor.streamlit_app.user_study.strategies import (
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
    rollout_time_remaining,
    watch_rollout_expiry,
)
from policy_doctor.streamlit_app.user_study.video_browser import render_video_browser

st.set_page_config(page_title="User Study — Group A", layout="wide")

# ── Sidebar: appearance + task loading ──────────────────────────────────────

render_appearance_sidebar(show_colorblind=True)

PFX = "ga"
STEP_KEY = f"{PFX}_step"

participant_id, task_name, _load_errors = load_task(PFX, needs_graph=False)
if _load_errors:
    for _err in _load_errors:
        st.error(_err)
    st.stop()

index = st.session_state[f"{PFX}_index"]
strategies = st.session_state[f"{PFX}_strategies"]
mp4_dir = Path(st.session_state[f"{PFX}_mp4_dir"])
total_budget = st.session_state.get(f"{PFX}_budget", 500)
alloc_step = st.session_state.get(f"{PFX}_alloc_step", 25)
rollout_limit = st.session_state.get(f"{PFX}_rollout_limit", 600)
_dvd_str = st.session_state.get(f"{PFX}_demo_videos_dir")
demo_videos_dir = Path(_dvd_str) if _dvd_str else None

# ── Step routing ──────────────────────────────────────────────────────────────

step = st.session_state.get(STEP_KEY, 0)
_theme = get_theme()

render_progress_bar(step, STEP_LABELS, theme=_theme)
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

**Data privacy.** Your responses are recorded anonymously.
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

    _, expired = rollout_time_remaining(st.session_state[start_key], rollout_limit)
    if expired:
        advance_step(2, STEP_KEY)

    watch_rollout_expiry(start_key, rollout_limit, STEP_KEY, 2)

    render_rollout_timer(
        st.session_state[start_key],
        rollout_limit,
        key=f"{PFX}_rtimer",
        theme=_theme,
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
    if st.button("Proceed to Data Collection →", type="primary"):
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
        demo_videos_dir=demo_videos_dir,
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
            "task": task_name,
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
