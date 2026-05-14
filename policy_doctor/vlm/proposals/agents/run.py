"""High-level orchestration: run N seeds × M conditions of agent sessions.

Used by both the standalone ``scripts/run_e2_agent.py`` entry point and the
server's ``POST /agent_session`` endpoint. Encapsulates the dance of:

1. Build SessionContext (graph, pool, classifier).
2. Build per-condition tool registry.
3. Build the user message from the rollout pool index.
4. Run :class:`AgentSession`.
5. Persist trace + submitted_requests + rationale + budget summary.
6. Optionally enqueue submitted requests on the server.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from policy_doctor.vlm.backends.base import VLMBackend
from policy_doctor.vlm.proposals.agents.budget import BudgetConfig
from policy_doctor.vlm.proposals.agents.conditions import (
    Condition,
    parse_condition,
)
from policy_doctor.vlm.proposals.agents.context import SessionContext
from policy_doctor.vlm.proposals.agents.session import AgentSession, SessionResult
from policy_doctor.vlm.proposals.agents.system_prompts import prompt_text
from policy_doctor.vlm.proposals.agents.tools.registry import (
    build_description_tool_registry,
    build_exploration_tool_registry,
    build_tool_registry,
)
from policy_doctor.vlm.proposals.agents.trace import SessionTrace


@dataclass
class AgentRunConfig:
    """Per-condition agent-run knobs. Frozen for pre-registration."""

    backend: VLMBackend
    budget_config: BudgetConfig
    max_turns: int = 100
    temperature: float = 0.3
    max_tokens: int = 4096
    n_sessions: int = 3
    base_seed: int = 0
    kinematic_summary_strategy: str = "raw_states"
    cache_enabled: bool = True
    # Storyboard rendering knobs forwarded to ctx.config['storyboard'] and
    # consumed by Layer 2 visual tools (get_slice_video, get_rollout_video).
    # Keys are documented at policy_doctor/vlm/proposals/agents/tools/access.py
    # in `_render_slice_storyboard`. None = use the function defaults.
    storyboard: Optional[Dict[str, Any]] = None


@dataclass
class ConditionRunOutputs:
    """Trace + submitted artefacts for one condition (across N sessions)."""

    condition: str
    seeds: List[int] = field(default_factory=list)
    session_results: List[SessionResult] = field(default_factory=list)
    out_dirs: List[Path] = field(default_factory=list)


def run_one_session(
    *,
    condition: Condition | str,
    seed: int,
    backend: VLMBackend,
    graph,
    pool,
    out_dir: Path,
    budget_config: Optional[BudgetConfig] = None,
    max_turns: int = 100,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    cluster_labels=None,
    cluster_metadata=None,
    cluster_centroids=None,
    classifier=None,
    raw_states_dir: Optional[Path] = None,
    storyboards_dir: Optional[Path] = None,
    videos_dir: Optional[Path] = None,
    task_hint: str = "",
    kinematic_summary_strategy: str = "raw_states",
    cache_enabled: bool = True,
    user_message: Optional[str] = None,
    storyboard: Optional[Dict[str, Any]] = None,
    exploration_taxonomy: Optional[Dict[str, Any]] = None,
    visual_descriptions: Optional[Dict[str, Any]] = None,
    pre_inspected_slices: Optional[set] = None,
    pre_inspected_nodes: Optional[set] = None,
) -> SessionResult:
    """Run a single session and persist its artefacts."""
    cond = parse_condition(condition)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session_config: Dict[str, Any] = {
        "kinematic_summary_strategy": kinematic_summary_strategy,
    }
    if storyboard:
        session_config["storyboard"] = dict(storyboard)

    ctx = SessionContext.build(
        condition=cond.value,
        graph=graph,
        pool=pool,
        cluster_labels=cluster_labels,
        cluster_metadata=cluster_metadata,
        cluster_centroids=cluster_centroids,
        classifier=classifier,
        raw_states_dir=raw_states_dir,
        storyboards_dir=storyboards_dir,
        videos_dir=videos_dir,
        budget_config=budget_config,
        cache_enabled=cache_enabled,
        task_hint=task_hint,
        config=session_config,
        backend=backend,
    )

    # Pre-populate inspection bookkeeping from a prior description session so
    # the evidence gate accepts slices the Stage 1 agent already watched.
    if pre_inspected_slices:
        ctx.inspected_slices.update(pre_inspected_slices)
    if pre_inspected_nodes:
        ctx.inspected_nodes.update(pre_inspected_nodes)

    tools = build_tool_registry(cond, ctx, backend=backend)
    system_prompt = prompt_text(cond)
    user_msg = user_message or _default_user_message(
        pool, task_hint,
        exploration_taxonomy=exploration_taxonomy,
        visual_descriptions=visual_descriptions,
    )

    trace_path = out_dir / "trace.jsonl"
    with SessionTrace(out_path=trace_path) as trace:
        session = AgentSession(
            backend=backend,
            ctx=ctx,
            tools=tools,
            system_prompt=system_prompt,
            user_message=user_msg,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            max_turns=max_turns,
            trace=trace,
            out_dir=out_dir,
        )
        result = session.run()

    return result


def run_condition(
    *,
    condition: Condition | str,
    cfg: AgentRunConfig,
    graph,
    pool,
    base_out_dir: Path,
    **session_kwargs: Any,
) -> ConditionRunOutputs:
    """Run all ``cfg.n_sessions`` for one condition."""
    cond = parse_condition(condition)
    base_out_dir = Path(base_out_dir)
    out = ConditionRunOutputs(condition=cond.value)

    for i in range(cfg.n_sessions):
        seed = cfg.base_seed + i
        out_dir = base_out_dir / cond.value / f"seed_{seed}"
        result = run_one_session(
            condition=cond,
            seed=seed,
            backend=cfg.backend,
            graph=graph,
            pool=pool,
            out_dir=out_dir,
            budget_config=cfg.budget_config,
            max_turns=cfg.max_turns,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            kinematic_summary_strategy=cfg.kinematic_summary_strategy,
            cache_enabled=cfg.cache_enabled,
            storyboard=cfg.storyboard,
            **session_kwargs,
        )
        out.seeds.append(seed)
        out.session_results.append(result)
        out.out_dirs.append(out_dir)

    return out


def run_exploration_session(
    *,
    backend: VLMBackend,
    graph,
    pool,
    out_dir: Path,
    budget_config: Optional[BudgetConfig] = None,
    max_turns: int = 60,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    cluster_labels=None,
    cluster_metadata=None,
    cluster_centroids=None,
    classifier=None,
    raw_states_dir: Optional[Path] = None,
    storyboards_dir: Optional[Path] = None,
    videos_dir: Optional[Path] = None,
    task_hint: str = "",
    kinematic_summary_strategy: str = "raw_states",
    storyboard: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the pre-stage exploration session and return the cluster taxonomy dict.

    Returns {} if the session ended without calling finalize_exploration.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    effective_budget = budget_config or BudgetConfig(
        max_tool_calls=80,
        max_visual_calls=10,
        max_video_calls=0,
    )

    session_config: Dict[str, Any] = {
        "kinematic_summary_strategy": kinematic_summary_strategy,
    }
    if storyboard:
        session_config["storyboard"] = dict(storyboard)

    ctx = SessionContext.build(
        condition="exploration",
        graph=graph,
        pool=pool,
        cluster_labels=cluster_labels,
        cluster_metadata=cluster_metadata,
        cluster_centroids=cluster_centroids,
        classifier=classifier,
        raw_states_dir=raw_states_dir,
        storyboards_dir=storyboards_dir,
        videos_dir=videos_dir,
        budget_config=effective_budget,
        task_hint=task_hint,
        config=session_config,
        backend=backend,
    )

    tools = build_exploration_tool_registry(ctx, out_dir=out_dir)

    from importlib.resources import files as _pkg_files
    import policy_doctor.vlm.proposals.agents.system_prompts as _sp_pkg
    system_prompt = (_pkg_files(_sp_pkg) / "exploration.md").read_text(encoding="utf-8")

    sample_ids = [e.rollout_id for e in pool.entries[:20]]
    user_msg = (
        f"Task: {task_hint or '(unspecified)'}\n\n"
        f"Pool contains {len(pool)} rollouts; sample ids: {' '.join(sample_ids)}.\n\n"
        "Survey all clusters and call finalize_exploration when done. "
        "Do not submit demonstration requests."
    )

    trace_path = out_dir / "trace.jsonl"
    with SessionTrace(out_path=trace_path) as trace:
        session = AgentSession(
            backend=backend,
            ctx=ctx,
            tools=tools,
            system_prompt=system_prompt,
            user_message=user_msg,
            seed=0,
            temperature=temperature,
            max_tokens=max_tokens,
            max_turns=max_turns,
            trace=trace,
            out_dir=out_dir,
            target_n_submissions=0,
        )
        session.run()

    taxonomy_path = out_dir / "cluster_taxonomy.json"
    if taxonomy_path.exists():
        return json.loads(taxonomy_path.read_text(encoding="utf-8"))
    return {}


def run_description_session(
    *,
    backend: VLMBackend,
    graph,
    pool,
    out_dir: Path,
    budget_config: Optional[BudgetConfig] = None,
    max_turns: int = 40,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    cluster_labels=None,
    cluster_metadata=None,
    cluster_centroids=None,
    raw_states_dir=None,
    storyboards_dir=None,
    videos_dir=None,
    task_hint: str = "",
    storyboard: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Stage 1 of the two-stage pipeline: visual description session.

    Runs an agent whose only job is to watch video/storyboard clips and write
    literal descriptions of what it sees — no failure-mode labels, no proposals.
    Returns the loaded ``visual_descriptions.json`` dict (or ``{}`` on failure).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    effective_budget = budget_config or BudgetConfig(
        max_tool_calls=80,
        max_visual_calls=12,
        max_video_calls=0,
        max_session_duration_s=1800,
    )

    # Stage 1 always uses storyboard frames — much lighter on context tokens
    # than inline MP4s. The model can still make literal descriptions
    # (gripper open/closed, contact/no-contact) from per-frame images.
    desc_storyboard = {
        "n_frames": 5,
        "pad_before": 12,
        "pad_after": 12,
        "target_size": (1024, 1024),
        "cameras": None,
        "mode": "frames",
        "include_state_text": True,
    }
    if storyboard:
        # Allow callers to override specific keys but keep mode=frames.
        desc_storyboard.update(storyboard)
        desc_storyboard["mode"] = "frames"
        desc_storyboard["include_state_text"] = True

    session_config: Dict[str, Any] = {"storyboard": desc_storyboard}

    ctx = SessionContext.build(
        condition="description",
        graph=graph,
        pool=pool,
        cluster_labels=cluster_labels,
        cluster_metadata=cluster_metadata,
        cluster_centroids=cluster_centroids,
        raw_states_dir=raw_states_dir,
        storyboards_dir=storyboards_dir,
        videos_dir=videos_dir,
        budget_config=effective_budget,
        task_hint=task_hint,
        config=session_config,
        backend=backend,
    )

    tools = build_description_tool_registry(ctx, out_dir=out_dir)

    from importlib.resources import files as _pkg_files
    import policy_doctor.vlm.proposals.agents.system_prompts as _sp_pkg
    system_prompt = (_pkg_files(_sp_pkg) / "description.md").read_text(encoding="utf-8")

    sample_ids = [e.rollout_id for e in pool.entries[:20]]
    user_msg = (
        f"Task: {task_hint or '(unspecified)'}\n\n"
        f"Pool contains {len(pool)} rollouts; sample ids: {' '.join(sample_ids)}.\n\n"
        "Watch video clips from the high-failure clusters and describe exactly what "
        "you see. Call finalize_descriptions when done."
    )

    trace_path = out_dir / "trace.jsonl"
    with SessionTrace(out_path=trace_path) as trace:
        session = AgentSession(
            backend=backend,
            ctx=ctx,
            tools=tools,
            system_prompt=system_prompt,
            user_message=user_msg,
            seed=0,
            temperature=temperature,
            max_tokens=max_tokens,
            max_turns=max_turns,
            trace=trace,
            out_dir=out_dir,
            target_n_submissions=0,
        )
        session.run()

    desc_path = out_dir / "visual_descriptions.json"
    if desc_path.exists():
        return json.loads(desc_path.read_text(encoding="utf-8"))
    return {}


def run_two_stage_session(
    *,
    condition: Condition | str,
    seed: int,
    backend: VLMBackend,
    graph,
    pool,
    out_dir: Path,
    budget_config: Optional[BudgetConfig] = None,
    max_turns: int = 60,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    cluster_labels=None,
    cluster_metadata=None,
    cluster_centroids=None,
    classifier=None,
    raw_states_dir=None,
    storyboards_dir=None,
    videos_dir=None,
    task_hint: str = "",
    storyboard: Optional[Dict[str, Any]] = None,
    description_budget: Optional[BudgetConfig] = None,
) -> "SessionResult":
    """Run a two-stage session: description first, then proposals from descriptions.

    Stage 1 (description): the agent watches video clips and writes literal
    observations — arm positions, gripper states, contact or lack thereof.
    No failure-mode labels. Outputs ``visual_descriptions.json``.

    Stage 2 (proposals): a fresh agent session with standard A_G/A_NG tools
    but NO visual budget. It receives the Stage 1 descriptions as context in
    its user message and must ground its submissions in those observations
    rather than in task priors. The inspected_slices / inspected_nodes from
    Stage 1 are pre-populated so the evidence gate passes.
    """
    out_dir = Path(out_dir)
    desc_dir = out_dir / "stage1_description"
    proposal_dir = out_dir / "stage2_proposals"

    # --- Stage 1: description ---
    desc_result = run_description_session(
        backend=backend,
        graph=graph,
        pool=pool,
        out_dir=desc_dir,
        budget_config=description_budget,
        max_turns=max_turns // 2,
        temperature=temperature,
        max_tokens=max_tokens,
        cluster_labels=cluster_labels,
        cluster_metadata=cluster_metadata,
        cluster_centroids=cluster_centroids,
        raw_states_dir=raw_states_dir,
        storyboards_dir=storyboards_dir,
        videos_dir=videos_dir,
        task_hint=task_hint,
        storyboard=storyboard,
    )

    # --- Stage 2: proposals from descriptions ---
    # Visual/video budget is always 0 — the agent must work from Stage 1 text.
    # We preserve max_tool_calls and session duration from the caller's config
    # but hard-zero any visual budget regardless of what was passed.
    _base = budget_config or BudgetConfig()
    s2_budget = BudgetConfig(
        max_tool_calls=_base.max_tool_calls,
        max_visual_calls=0,
        max_video_calls=0,
        max_session_duration_s=_base.max_session_duration_s,
    )

    result = run_one_session(
        condition=condition,
        seed=seed,
        backend=backend,
        graph=graph,
        pool=pool,
        out_dir=proposal_dir,
        budget_config=s2_budget,
        max_turns=max_turns,
        temperature=temperature,
        max_tokens=max_tokens,
        cluster_labels=cluster_labels,
        cluster_metadata=cluster_metadata,
        cluster_centroids=cluster_centroids,
        classifier=classifier,
        raw_states_dir=raw_states_dir,
        storyboards_dir=storyboards_dir,
        videos_dir=videos_dir,
        task_hint=task_hint,
        storyboard=storyboard,
        visual_descriptions=desc_result,
        # Pre-populate inspection bookkeeping from Stage 1 so the evidence
        # gate accepts the slices the description agent already watched.
        pre_inspected_slices=set(desc_result.get("inspected_slices") or []),
        pre_inspected_nodes=set(int(n) for n in (desc_result.get("inspected_nodes") or [])),
    )

    return result


def _default_user_message(
    pool,
    task_hint: str,
    exploration_taxonomy: Optional[Dict[str, Any]] = None,
    visual_descriptions: Optional[Dict[str, Any]] = None,
) -> str:
    """Initial user turn shown to the agent."""
    sample_ids = [e.rollout_id for e in pool.entries[:20]]
    sample_str = " ".join(sample_ids)
    msg = (
        f"Task: {task_hint or '(unspecified)'}\n\n"
        "You have a tool surface for inspecting the rollout pool and (when this "
        "condition includes the graph) its behavior structure. The pool contains "
        f"{len(pool)} rollouts; ids include: {sample_str}.\n\n"
        "Begin by orienting yourself with the cheapest tools, then submit a small "
        "set of demonstration requests via propose_collection_request, and end "
        "with finalize_strategy."
    )
    if exploration_taxonomy:
        msg += (
            "\n\n## Pre-exploration cluster taxonomy\n\n"
            "The following was produced by a prior text-only survey of all clusters.\n"
            "Use it to inform which clusters to target for visual evidence collection.\n"
            "Do not blindly trust it — verify with get_slice_video — but treat it as\n"
            "your starting prior.\n\n"
            f"{json.dumps(exploration_taxonomy, indent=2)}"
        )
    if visual_descriptions:
        descs = visual_descriptions.get("cluster_descriptions") or []
        msg += (
            "\n\n## Stage 1 visual observations\n\n"
            "The following are LITERAL descriptions of video clips from the high-failure "
            "clusters, written by an observer who watched the clips frame by frame. "
            "These are the only visual evidence available to you — you have NO visual "
            "budget to watch additional clips. Your demonstration requests must be "
            "grounded in these specific observations. Do not introduce failure modes "
            "or robot behaviors not described here.\n\n"
        )
        for d in descs:
            cid = d.get("cluster_id", "?")
            slices = d.get("slices_observed", [])
            informative = d.get("informative", True)
            contact = d.get("robot_object_contact", False)
            msg += f"### c{cid} (slices: {', '.join(slices)})\n"
            if not informative:
                msg += (
                    f"**UNINFORMATIVE** — the observer reported no robot-object contact "
                    f"in any clip. Do not submit requests citing c{cid} as evidence of "
                    f"a specific failure mode.\n\n"
                )
            else:
                msg += f"Robot-object contact: {'yes' if contact else 'no'}\n"
                msg += f"{d.get('literal_description', '')}\n"
                if d.get("gripper_states"):
                    msg += f"Gripper states: {d['gripper_states']}\n"
                if d.get("object_location"):
                    msg += f"Object location: {d['object_location']}\n"
                if d.get("sequence_of_events"):
                    msg += f"Sequence: {d['sequence_of_events']}\n"
                msg += "\n"
    return msg


# ---------------------------------------------------------------------------
# Pre-registration helpers
# ---------------------------------------------------------------------------


def write_pre_registration(
    out_path: Path,
    *,
    cfg: Dict[str, Any],
    conditions: List[str],
) -> None:
    """Write a frozen ``pre_registration.yaml`` describing this run.

    Captures: tool schema hash, system prompt hashes, budget config, and the
    full agentic config block so the experiment is reproducible from disk.
    """
    import yaml

    from policy_doctor.vlm.proposals.agents.system_prompts import all_prompt_hashes
    from policy_doctor.vlm.proposals.agents.tools.schema import schema_hash

    payload = {
        "schema_hash": schema_hash(),
        "prompt_hashes": all_prompt_hashes(),
        "conditions": list(conditions),
        "agentic": cfg.get("agentic"),
        "adherence": cfg.get("adherence"),
        "n_requests_per_type": cfg.get("n_requests_per_type"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=True)
