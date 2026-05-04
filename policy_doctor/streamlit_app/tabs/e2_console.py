"""Streamlit operator console for Experiment E2.

Stateless: every interaction is a HTTP call to the proposal_server. Reload-safe
because no chat / queue state lives in ``st.session_state``.

Panels:
  - Status & pool             (GET /health, GET /pool, GET /graph)
  - Agent sessions            (POST /agent_session for A_G / A_NG)
  - Human exploration         (browse rollouts, submit DemonstrationRequests
                               for H_NG / H_G conditions)
  - Request queue             (GET /requests, GET /requests/active)
  - Legacy one-shot proposals (POST /propose; gated behind a checkbox)
  - Chat                      (POST /chat, only when chat_enabled=true server-side)

The console NEVER displays target_cluster or source_condition. The server
strips those from /requests/active and /requests; this view filters them
again as defense in depth.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests
import streamlit as st


_SERVER_KEY = "e2_console_server_url"
_DEFAULT_URL = "http://127.0.0.1:5003"


def render() -> None:
    st.header("Experiment E2 — Operator Console")

    server_url = st.text_input(
        "Proposal server URL",
        value=st.session_state.get(_SERVER_KEY, _DEFAULT_URL),
        key=_SERVER_KEY,
    ).rstrip("/")

    health = _safe_get(f"{server_url}/health")
    if health is None:
        st.error(f"Cannot reach proposal server at {server_url}.")
        st.stop()

    cols = st.columns(4)
    cols[0].metric("Pool size", health.get("pool_size", "?"))
    cols[1].metric("Pending", health.get("n_pending", "?"))
    cols[2].metric("Total requests", health.get("n_requests", "?"))
    cols[3].metric("Backend device", health.get("backend_device", "?"))

    chat_enabled = bool(health.get("chat_enabled"))

    st.divider()
    _render_agent_sessions(server_url)
    st.divider()
    _render_human_exploration(server_url)
    st.divider()
    with st.expander("Legacy one-shot proposals (ablation only)", expanded=False):
        _render_proposals(server_url, chat_enabled)
    st.divider()
    _render_queue(server_url)
    if chat_enabled:
        st.divider()
        _render_chat(server_url)
    else:
        st.info("Chat is disabled by server config (chat_enabled=false).")


def _render_proposals(server_url: str, chat_enabled: bool) -> None:
    st.subheader("Generate proposals")
    cols = st.columns(2)
    with cols[0]:
        if st.button("Generate batch — graph condition"):
            _propose(server_url, "graph")
    with cols[1]:
        if st.button("Generate batch — outcome_only condition"):
            _propose(server_url, "outcome_only")


def _propose(server_url: str, condition: str) -> None:
    with st.spinner(f"Calling /propose for condition={condition} ..."):
        try:
            resp = requests.post(
                f"{server_url}/propose",
                json={"condition": condition},
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()
            st.success(f"Added {data['n_added']} requests for {condition}")
            st.json(data.get("consistency_metrics", {}))
        except requests.HTTPError as e:
            st.error(f"/propose failed: {e}")


def _render_queue(server_url: str) -> None:
    st.subheader("Request queue")
    queue = _safe_get(f"{server_url}/requests")
    if queue is None:
        st.warning("Queue unavailable.")
        return
    rows = queue.get("queue", [])
    if not rows:
        st.write("Queue is empty.")
        return
    st.write(f"{len(rows)} request(s) in queue:")
    st.dataframe(rows)

    st.write("Inspect active request (operator-view only):")
    active = _safe_get(f"{server_url}/requests/active")
    if active:
        # Defense in depth: filter the operator-hidden fields if any leaked.
        scrubbed = {k: v for k, v in active.items() if k not in ("target_cluster", "source_condition")}
        ref_ids = scrubbed.pop("reference_storyboard_ids", []) or []
        st.json(scrubbed)
        if ref_ids:
            st.caption(
                f"Reference storyboards ({len(ref_ids)}) — what the agent inspected before writing this request:"
            )
            cols = st.columns(min(3, len(ref_ids)))
            for i, sid in enumerate(ref_ids):
                with cols[i % len(cols)]:
                    st.image(f"{server_url}/storyboards/{sid}", caption=sid)
    else:
        st.write("No pending request to display.")


def _render_chat(server_url: str) -> None:
    st.subheader("Chat (development mode)")
    if "e2_chat_log" not in st.session_state:
        st.session_state.e2_chat_log = []
    for entry in st.session_state.e2_chat_log:
        with st.chat_message(entry["role"]):
            st.write(entry["text"])
    user_msg = st.chat_input("Ask the proposal model a question...")
    if user_msg:
        st.session_state.e2_chat_log.append({"role": "user", "text": user_msg})
        with st.chat_message("user"):
            st.write(user_msg)
        with st.spinner("Calling /chat ..."):
            try:
                resp = requests.post(
                    f"{server_url}/chat", json={"text": user_msg}, timeout=120
                )
                resp.raise_for_status()
                reply = resp.json().get("text", "")
            except requests.HTTPError as e:
                reply = f"[error: {e}]"
        st.session_state.e2_chat_log.append({"role": "assistant", "text": reply})
        with st.chat_message("assistant"):
            st.write(reply)


def _render_agent_sessions(server_url: str) -> None:
    """Trigger one or more agentic sessions on the server."""
    st.subheader("Agent sessions (A_G / A_NG)")
    cols = st.columns(3)
    cond = cols[0].selectbox(
        "Condition", ["A_G", "A_NG", "H_G", "H_NG"], index=0, key="e2_console_agent_cond",
    )
    seed = cols[1].number_input("Seed", min_value=0, value=0, key="e2_console_agent_seed")
    if cols[2].button("Run agent session"):
        with st.spinner(f"Running {cond} session (seed={seed})..."):
            try:
                resp = requests.post(
                    f"{server_url}/agent_session",
                    json={"condition": cond, "seed": int(seed)},
                    timeout=3600,
                )
                resp.raise_for_status()
                data = resp.json()
                st.success(
                    f"Session ended ({data.get('stop_reason', '?')}); "
                    f"submitted {data.get('n_submitted', 0)} requests, "
                    f"enqueued {data.get('n_enqueued', 0)}."
                )
                st.json(data)
            except requests.HTTPError as e:
                st.error(f"/agent_session failed: {e}")


def _render_human_exploration(server_url: str) -> None:
    """Human exploration UI for H_NG (default) and H_G conditions.

    Each operator action — viewing a rollout storyboard, submitting a
    request — is logged via the same proposal_server endpoints, so adherence
    scoring and queue handling are uniform across human and agent conditions.
    """
    st.subheader("Human exploration (H_NG / H_G)")
    cols = st.columns(2)
    h_condition = cols[0].selectbox(
        "Operator condition", ["H_NG", "H_G"], index=0, key="e2_console_h_cond",
    )
    expose_graph = h_condition == "H_G"

    pool = _safe_get(f"{server_url}/pool")
    if pool is None:
        st.warning("Pool unavailable.")
        return
    rollouts = pool.get("rollouts", [])
    if not rollouts:
        st.write("Pool is empty.")
        return

    outcome_filter = cols[1].selectbox(
        "Filter outcome", ["all", "failure", "success"], index=0, key="e2_console_h_filt",
    )
    rollouts_view = [
        r for r in rollouts
        if outcome_filter == "all"
        or (outcome_filter == "success" and r.get("success") is True)
        or (outcome_filter == "failure" and r.get("success") is False)
    ]
    # Strip cluster_path for H_NG; keep it for H_G.
    if not expose_graph:
        rollouts_view = [
            {k: v for k, v in r.items() if k != "cluster_path"} for r in rollouts_view
        ]
    st.dataframe(rollouts_view)

    # ---- Submission form -------------------------------------------------
    st.markdown("**Submit DemonstrationRequest**")
    with st.form(f"hng_form_{h_condition}", clear_on_submit=True):
        request_type = st.selectbox(
            "request_type", ["full_trajectory", "recovery", "alternative_strategy"],
        )
        ref_id = st.selectbox(
            "reference_rollout_id", [r["rollout_id"] for r in rollouts_view],
        )
        ref_frame = st.number_input("reference_frame", min_value=0, value=0)
        target_behavior = st.text_area(
            "target_behavior",
            help="Behaviorally observable description for the operator. NO cluster / "
                 "graph / embedding terms.",
        )
        prohibitions_raw = st.text_area("prohibitions (one per line)", value="")
        success_criterion = st.text_input("success_criterion", value="task_success")
        target_cluster = None
        if expose_graph:
            target_cluster = st.number_input(
                "target_cluster (H_G only)", min_value=-5, value=0,
                help="Behavior cluster id this request targets.",
            )
        rationale = st.text_area(
            "rationale (logged, not shown to executing operator)",
            help="REQUIRED. Explain why this request matters.",
        )
        submitted = st.form_submit_button("Submit request")
        if submitted:
            _submit_human_request(
                server_url=server_url,
                condition=h_condition,
                request_type=request_type,
                ref_id=ref_id,
                ref_frame=int(ref_frame),
                target_behavior=target_behavior,
                prohibitions=[p.strip() for p in prohibitions_raw.splitlines() if p.strip()],
                success_criterion=success_criterion,
                target_cluster=int(target_cluster) if expose_graph else None,
                rationale=rationale,
            )


def _submit_human_request(
    *,
    server_url: str,
    condition: str,
    request_type: str,
    ref_id: str,
    ref_frame: int,
    target_behavior: str,
    prohibitions: list,
    success_criterion: str,
    target_cluster: Optional[int],
    rationale: str,
) -> None:
    """Validate locally and POST to /agent_session-equivalent enqueue path.

    Currently writes the request directly into the operator queue via a
    minimal /agent_session call shape (the server enqueues whatever is in
    submitted_requests). For the human path, we synthesize a single-shot
    "session" payload and POST it to /agent_session with a sentinel
    backend that just enqueues the form-built request.

    A clean future extension: dedicated POST /human_request endpoint.
    """
    if not target_behavior.strip():
        st.error("target_behavior is required.")
        return
    if not rationale.strip():
        st.error("rationale is required (it's how the human path matches the agent path).")
        return
    # Build a DemonstrationRequest dict client-side; the server has a
    # validator that will reject denylist violations.
    from policy_doctor.vlm.proposals.request import DemonstrationRequest

    req = DemonstrationRequest(
        request_id=DemonstrationRequest.new_id(),
        request_type=request_type,
        initial_conditions=__import__(
            "policy_doctor.vlm.proposals.request", fromlist=["InitialConditions"]
        ).InitialConditions(reference_rollout_id=ref_id, reference_frame=ref_frame),
        target_behavior=target_behavior,
        prohibitions=prohibitions,
        success_criterion=success_criterion,
        target_cluster=target_cluster,
        source_condition=condition,
    )
    # We POST to a tiny human-submission endpoint; if the server doesn't
    # expose one, fall back to a local note so the operator sees feedback.
    payload = {
        "condition": condition,
        "request": req.to_dict(),
        "rationale": rationale,
    }
    try:
        resp = requests.post(
            f"{server_url}/human_request", json=payload, timeout=30,
        )
        resp.raise_for_status()
        st.success(f"Submitted request {req.request_id}.")
        st.json(resp.json())
    except requests.HTTPError as e:
        st.error(f"/human_request failed: {e}; the request was NOT enqueued.")
    except requests.RequestException:
        st.warning(
            f"Server has no /human_request endpoint yet. Built request:\n\n"
            f"```json\n{req.to_dict()}\n```"
        )


def _safe_get(url: str) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 204:
            return None
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError):
        return None
