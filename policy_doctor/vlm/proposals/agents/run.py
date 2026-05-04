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

    tools = build_tool_registry(cond, ctx, backend=backend)
    system_prompt = prompt_text(cond)
    user_msg = user_message or _default_user_message(
        pool, task_hint, exploration_taxonomy=exploration_taxonomy
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


def _default_user_message(
    pool,
    task_hint: str,
    exploration_taxonomy: Optional[Dict[str, Any]] = None,
) -> str:
    """Initial user turn shown to the agent.

    Includes the task hint and a sampled list of rollout ids so the agent has
    concrete handles to query. The tool surface is the source of truth for
    everything else. If ``exploration_taxonomy`` is provided, it is appended as
    a structured prior from the pre-stage exploration session.
    """
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
