"""Assemble the per-condition tool registry.

The session loop never instantiates tools itself — it asks this module for the
right surface given (condition, context). That keeps condition-specific tool
inclusion / exclusion in one place and trivially testable.

Surface composition:

* **A_G** — Layer 1 (topology) + Layer 2 (graph-aware access) + Layer 3
  (analysis) + Layer 4 with target_cluster.
* **A_NG** — no_graph parallel surface + Layer 4 without target_cluster.
* **H_G / H_NG** — same surfaces as A_G / A_NG, but the "tool" handlers are
  instead exposed in the Streamlit UI (this module still produces the specs
  so the spec stays the source-of-truth).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

from policy_doctor.vlm.proposals.agents.conditions import (
    Condition,
    parse_condition,
)
from policy_doctor.vlm.proposals.agents.context import SessionContext
from policy_doctor.vlm.proposals.agents.tools.types import ToolSpec

if TYPE_CHECKING:
    from policy_doctor.vlm.backends.base import VLMBackend


def build_tool_registry(
    condition: Condition | str,
    ctx: SessionContext,
    backend: Optional["VLMBackend"] = None,
) -> Dict[str, ToolSpec]:
    """Build the ordered tool registry for ``condition``.

    Returns a dict (insertion-ordered) so the agent sees tools in a consistent
    order across runs — the spec-prescribed order is Layer 1 → 2 → 3 → 4.
    """
    cond = parse_condition(condition)
    specs: list[ToolSpec] = []

    if cond in (Condition.A_G, Condition.H_G):
        # Lazy imports keep the module import-light.
        from policy_doctor.vlm.proposals.agents.tools import access, submission, topology

        specs.extend(topology.build(ctx))
        specs.extend(access.build(ctx))
        try:
            from policy_doctor.vlm.proposals.agents.tools import analysis

            specs.extend(analysis.build(ctx))
        except ImportError:
            # Layer 3 lands later; A_G works without it during early development.
            pass
        specs.extend(submission.build(ctx, with_target_cluster=True))
        from policy_doctor.vlm.proposals.agents.tools import grounding
        specs.extend(grounding.build(ctx))

    elif cond in (Condition.A_NG, Condition.H_NG):
        from policy_doctor.vlm.proposals.agents.tools import no_graph, submission

        specs.extend(no_graph.build(ctx))
        specs.extend(submission.build(ctx, with_target_cluster=False))

    else:
        raise ValueError(f"unknown condition {condition!r}")

    by_name: Dict[str, ToolSpec] = {}
    for spec in specs:
        if spec.name in by_name:
            raise ValueError(f"duplicate tool name {spec.name!r} in registry for {cond}")
        by_name[spec.name] = spec
    return by_name


def build_description_tool_registry(
    ctx: SessionContext,
    out_dir: Optional[Path] = None,
) -> Dict[str, ToolSpec]:
    """Build the tool registry for the Stage 1 visual description session.

    Includes: all cheap topology tools, all access tools (including
    get_slice_video so the agent can watch clips), all analysis tools.
    Does NOT include submission or exploration tools.
    Terminal tool: finalize_descriptions.
    """
    from policy_doctor.vlm.proposals.agents.tools import (
        access,
        analysis,
        description,
        topology,
    )

    specs = []
    specs.extend(topology.build(ctx))
    specs.extend(access.build(ctx))
    specs.extend(analysis.build(ctx))
    specs.extend(description.build(ctx, out_dir=out_dir))

    by_name: Dict[str, ToolSpec] = {}
    for spec in specs:
        if spec.name in by_name:
            raise ValueError(f"duplicate tool name {spec.name!r} in description registry")
        by_name[spec.name] = spec
    return by_name


def build_exploration_tool_registry(
    ctx: SessionContext,
    out_dir: Optional[Path] = None,
) -> Dict[str, ToolSpec]:
    """Build the tool registry for the pre-stage exploration session.

    Includes: all cheap topology tools, all access tools (including
    get_slice_video for visual spot-checks), all analysis tools.
    Does NOT include submission tools.
    Terminal tool: finalize_exploration.
    """
    from policy_doctor.vlm.proposals.agents.tools import (
        access,
        analysis,
        exploration,
        topology,
    )

    specs = []
    specs.extend(topology.build(ctx))
    specs.extend(access.build(ctx))
    specs.extend(analysis.build(ctx))
    specs.extend(exploration.build(ctx, out_dir=out_dir))

    by_name: Dict[str, ToolSpec] = {}
    for spec in specs:
        if spec.name in by_name:
            raise ValueError(f"duplicate tool name {spec.name!r} in exploration registry")
        by_name[spec.name] = spec
    return by_name
