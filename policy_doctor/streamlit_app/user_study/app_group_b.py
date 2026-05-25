from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from policy_doctor.streamlit_app.appearance import get_theme, render_appearance_sidebar
from policy_doctor.behaviors.behavior_graph import BehaviorGraph
from policy_doctor.streamlit_app.user_study.likert_survey import (
    render_block1_graph_interaction,
    render_block2_strategy,
    render_block3_final,
)
from policy_doctor.streamlit_app.user_study.nasa_tlx import render_nasa_tlx
from policy_doctor.streamlit_app.user_study.response_store import get_store
from policy_doctor.streamlit_app.user_study.clustering_loader import apply_clustering_for_k
from policy_doctor.streamlit_app.user_study.task_setup import load_task, repo_root
from policy_doctor.streamlit_app.user_study.strategies import (
    render_strategy_allocator,
    render_strategy_summary,
)
from policy_doctor.streamlit_app.user_study.survey_steps import (
    STEP_LABELS,
    advance_step,
    get_step_durations,
    record_step_entry,
    record_step_exit,
    render_progress_bar,
    render_rollout_timer,
    rollout_time_remaining,
    watch_rollout_expiry,
)
from policy_doctor.streamlit_app.user_study.video_browser import render_video_browser

st.set_page_config(page_title="User Study — Group B", layout="wide")

# ── Sidebar: appearance + task loading ──────────────────────────────────────

_, colorblind_mode = render_appearance_sidebar(show_colorblind=True)

PFX = "gb"
STEP_KEY = f"{PFX}_step"

participant_id, task_name, _load_errors = load_task(PFX, needs_graph=True)
if _load_errors:
    for _err in _load_errors:
        st.error(_err)
    st.stop()

index = st.session_state[f"{PFX}_index"]
strategies = st.session_state[f"{PFX}_strategies"]
mp4_dir = Path(st.session_state[f"{PFX}_mp4_dir"])
labels = st.session_state[f"{PFX}_labels"]
metadata = st.session_state[f"{PFX}_metadata"]
graph: BehaviorGraph = st.session_state[f"{PFX}_graph"]
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
- Explore an interactive **behavior graph** showing the robot's movement patterns
- Decide how to allocate a data collection budget across different strategies
- Complete a brief survey about your experience

**Time commitment:** approximately **25–35 minutes** in total.

**Participation is voluntary.** You may withdraw at any time without any disadvantage.
You will not receive compensation for participation.

**Group assignment.** Participants are randomly assigned to one of two conditions.
You have been assigned to the condition with the behavior graph visualization.

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
    st.title("Your Mission")

    st.markdown("""
### Your role

You are a **data designer** at a robotics company.
Your robots are controlled by machine learning models trained on *expert demonstrations*:
human data collectors teleoperate the robots to record example task executions.

### The task

Your latest project is a partnership with a **kendama toy manufacturer** who wants to automate
the packing of their kendama toys. You need to train a robot policy to pick up and pack
kendama toys reliably.

To collect training data, you hired **two data-collection companies**:

- **Company A** recorded demonstrations that pick up the kendama **in one continuous motion**.
- **Company B** recorded demonstrations that pick up the kendama **in two steps** (a brief
  intermediate re-grip before completing the lift).

You trained a first policy on this combined dataset and ran it on the robot.
Now you want to answer two questions:

1. **Which collection strategy is more robust?**
2. **Where do you need more data** — which scenarios are causing the most failures?

### How you'll explore the data

You will watch videos of the robot and explore a **behavior graph** — an automatic grouping of
the robot's movement patterns into labeled behavioral modes.
Arrows show how often the robot transitions between behaviors and which transitions lead to
success or failure. **You can click any node or edge** to see video clips for that behavior.
""")

    limit_min = rollout_limit // 60
    limit_sec = rollout_limit % 60
    limit_str = f"{limit_min} minute{'s' if limit_min != 1 else ''}" + (
        f" {limit_sec}s" if limit_sec else ""
    )

    st.warning(
        f"**About the next section — Rollout Info:**  \n"
        f"You will have **{limit_str}** to explore the robot videos and behavior graph. "
        f"A countdown timer will be shown at the top of the page. "
        f"Once the timer expires (or you click Proceed), you will move on automatically "
        f"and **cannot return** to this page. "
        f"Use your time wisely — browse videos, explore the graph, and note patterns "
        f"that will inform your data collection decisions."
    )

    if st.button("Continue →", type="primary"):
        advance_step(1, STEP_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Rollout Info  (timer-gated)
# ─────────────────────────────────────────────────────────────────────────────
elif step == 2:
    st.title("Explore the Robot's Behavior")

    # Timer setup
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

    # ── Section A: Videos ────────────────────────────────────────────────────
    n_total = len(index["episodes"])
    n_success = sum(1 for ep in index["episodes"] if ep.get("success") is True)
    ov_c1, ov_c2, ov_c3 = st.columns(3)
    ov_c1.metric("Total rollouts", n_total)
    ov_c2.metric("Successes", n_success)
    ov_c3.metric("Success rate", f"{n_success / n_total:.0%}" if n_total else "—")

    st.subheader("Watch the Robot in Action")
    st.markdown(
        "These videos show **rollouts** — the robot attempting the task from scratch. "
        "Watch several to understand what it does well and where it struggles."
    )
    render_video_browser(mp4_dir, index, page_size=4, key_prefix=f"{PFX}_vbrow")

    # ── Section B: Behavior graph ─────────────────────────────────────────────
    st.divider()
    st.subheader("Explore the Behavior Graph")
    st.markdown(
        "The robot's rollouts have been automatically grouped into **behavioral modes** — "
        "recurring movement patterns. "
        "The graph shows how often the robot transitions between modes and which lead to "
        "**success ✓** or **failure ✗**."
    )
    with st.expander("❓ How to read this graph", expanded=False):
        st.markdown("""
- **Each circle** is a behavioral mode
- **Arrows** show transitions (thickness = probability)
- **Larger circles** = more episodes passed through this mode
- **Click any node or edge** to see example clips
- **Click the background** to deselect
""")

    _study_graph = st.session_state.get(f"{PFX}_study_graph", {})
    _viz_type = _study_graph.get("visualization", "tree_native_svg")
    _min_branch = int(_study_graph.get("min_transition_count", 2))
    _clust_cfg = _study_graph.get("clustering") or {}
    _k_options = _study_graph.get("k_options") or [st.session_state.get(f"{PFX}_loaded_k", 15)]

    _color_opts = ["outcome", "id"]
    _color_labels = {
        "outcome": "Outcome (success rate)",
        "id": "Cluster ID",
    }
    _default_color = _study_graph.get("color_by", "outcome")
    if _default_color not in _color_opts:
        _default_color = "outcome"
    if f"{PFX}_color_by" not in st.session_state:
        st.session_state[f"{PFX}_color_by"] = _default_color

    if _clust_cfg and len(_k_options) > 1:
        if f"{PFX}_graph_k_idx" not in st.session_state:
            _loaded_k = st.session_state.get(
                f"{PFX}_loaded_k", _study_graph.get("default_k", _k_options[0]),
            )
            try:
                st.session_state[f"{PFX}_graph_k_idx"] = _k_options.index(int(_loaded_k))
            except ValueError:
                st.session_state[f"{PFX}_graph_k_idx"] = 0
        _c_k, _c_color = st.columns([2, 1])
        with _c_k:
            st.markdown("**Graph complexity**")
            _k_l, _k_s, _k_r = st.columns([1, 8, 1])
            with _k_l:
                st.markdown(
                    "<div style='padding-top:0.5rem;text-align:right;color:inherit'>Less</div>",
                    unsafe_allow_html=True,
                )
            with _k_s:
                _k_idx = st.slider(
                    "Graph complexity",
                    min_value=0,
                    max_value=len(_k_options) - 1,
                    key=f"{PFX}_graph_k_idx",
                    label_visibility="collapsed",
                )
            with _k_r:
                st.markdown(
                    "<div style='padding-top:0.5rem;color:inherit'>More</div>",
                    unsafe_allow_html=True,
                )
            _selected_k = _k_options[_k_idx]
        with _c_color:
            _color_by = st.selectbox(
                "Color nodes by",
                options=_color_opts,
                format_func=lambda v: _color_labels[v],
                key=f"{PFX}_color_by",
            )
        if st.session_state.get(f"{PFX}_loaded_k") != _selected_k:
            _k_errs = apply_clustering_for_k(PFX, int(_selected_k), _clust_cfg, repo_root())
            if _k_errs:
                for _e in _k_errs:
                    st.error(_e)
            else:
                graph = st.session_state[f"{PFX}_graph"]
                labels = st.session_state[f"{PFX}_labels"]
                metadata = st.session_state[f"{PFX}_metadata"]
                for _sk in (
                    f"{PFX}_tree_graph_selected",
                    f"{PFX}_tree_graph_selected_edge",
                    f"{PFX}_tree_graph_last_seq",
                ):
                    st.session_state.pop(_sk, None)
                st.rerun()
    else:
        _color_by = st.selectbox(
            "Color nodes by",
            options=_color_opts,
            format_func=lambda v: _color_labels[v],
            key=f"{PFX}_color_by",
        )

    _active_graph, _active_labels = graph, labels

    from policy_doctor.streamlit_app.components.trajectory_tree_view import (
        render_trajectory_tree,
    )
    render_trajectory_tree(
        labels=_active_labels,
        metadata=metadata,
        view_mode=_viz_type.replace("tree_", ""),
        min_branch=_min_branch,
        max_depth_cap=500,
        color_mode=_color_by,
        node_values={},
        cluster_names=None,
        mp4_dir=mp4_dir,
        mp4_index=index,
        height=600,
        level=getattr(_active_graph, "level", "rollout"),
        key_prefix=f"{PFX}_tree",
        edge_style="lines",
        edge_width_slope=5.0,
        node_size_slope=24.0,
        show_stats=False,
        layout_token=st.session_state.get(f"{PFX}_loaded_k"),
        theme=_theme,
    )

    st.divider()
    if st.button("Proceed to Data Collection →", type="primary"):
        advance_step(2, STEP_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Data Collection
# ─────────────────────────────────────────────────────────────────────────────
elif step == 3:
    st.title("Design Your Data Collection Strategy")
    st.markdown(
        "Based on what you observed, allocate your **{} demo budget** across the strategies below. "
        "Each strategy targets a specific data collection protocol.".format(total_budget)
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

    st.header("1 — Behavior-Graph Survey")
    likert_graph = render_block1_graph_interaction(key_prefix=f"{PFX}_likert")

    st.divider()
    st.header("2 — Strategy-Selection Survey")
    likert_strategy = render_block2_strategy(key_prefix=f"{PFX}_likert")

    st.divider()
    st.header("3 — NASA Task Load Index")
    tlx_responses = render_nasa_tlx(key_prefix=f"{PFX}_tlx")

    st.divider()
    st.header("4 — Final Assessment")
    likert_final = render_block3_final(
        key_prefix=f"{PFX}_likert", include_graph_questions=True,
    )

    st.divider()
    notes = st.text_area(
        "Any additional notes or reasoning about your choices",
        value="",
        height=120,
        placeholder="e.g. I noticed failures often occur when the arm approaches from the left, so I focused on...",
    )

    if st.button("Submit", type="primary"):
        record_step_exit(4, STEP_KEY)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = {
            "participant_id": participant_id,
            "group": "B",
            "task": task_name,
            "allocations": st.session_state.get(f"{PFX}_allocations", {}),
            "nasa_tlx": tlx_responses,
            "likert_graph": likert_graph,
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
            file_name=f"group_b_{timestamp}.json",
            mime="application/json",
        )
        st.balloons()
