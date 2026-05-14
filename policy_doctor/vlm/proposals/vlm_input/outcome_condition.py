"""Outcome-only VLM input builder.

The VLM sees rollout storyboards labelled with outcome only (no graph image,
no graph text, no cluster paths). The schema does not require ``target_cluster``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from policy_doctor.vlm.proposals.registry import register_vlm_input_builder
from policy_doctor.vlm.proposals.vlm_input.base import Message, VLMInputBuilder

if TYPE_CHECKING:
    from policy_doctor.vlm.proposals.graph_representation.base import VLMArtefact
    from policy_doctor.vlm.proposals.pool import RolloutPool


_OPERATOR_FORBIDDEN_TERMS = (
    "cluster", "node", "graph", "umap", "kmeans", "centroid", "embedding",
)


def _outcome_str(success: Optional[bool]) -> str:
    if success is True:
        return "success"
    if success is False:
        return "failure"
    return "unknown"


def _system_prompt(
    *,
    task_hint: str,
    n_requests_per_type: Dict[str, int],
    json_schema: Dict[str, Any],
) -> str:
    total = sum(n_requests_per_type.values())
    lines = [
        "You are a robotics data-collection planner. You are given a pool of",
        "base-policy rollouts labelled with their success/failure outcomes only.",
        f"Your job is to propose {total} demonstration requests that, if",
        "executed by a trained operator, would teach the policy to recover from",
        "its observed failure modes and broaden its behavioral coverage.",
    ]
    if task_hint:
        lines += ["", f"Task: {task_hint}"]
    lines += [
        "",
        "Distribution constraint (must be exact):",
    ]
    for k, v in sorted(n_requests_per_type.items()):
        lines.append(f"  - {k}: {v}")
    lines += [
        "",
        "Operator-facing fields ('target_behavior', 'success_criterion',",
        "'prohibitions') must use only behaviorally-observable terms describing",
        "what the operator should do or avoid. Do NOT use any of these words in",
        "those fields: " + ", ".join(_OPERATOR_FORBIDDEN_TERMS) + ".",
        "",
        "Output: a single JSON object matching this schema (no surrounding text):",
        "```json",
        json.dumps(json_schema, indent=2),
        "```",
    ]
    return "\n".join(lines)


def _pool_table_block(pool: "RolloutPool") -> str:
    lines = ["## Rollout pool (full, outcomes only)"]
    lines.append("| rollout_id | length | outcome |")
    lines.append("|---|---|---|")
    for entry in pool.entries:
        lines.append(
            f"| {entry.rollout_id} | {entry.length} | {_outcome_str(entry.success)} |"
        )
    return "\n".join(lines)


def _sampled_rollouts_block(
    pool: "RolloutPool",
    *,
    sample_size: int,
    rng: random.Random,
) -> List[Path]:
    return []


class OutcomeConditionInputBuilder(VLMInputBuilder):
    def __init__(self, params: Dict[str, Any] | None = None):
        params = params or {}
        self.sample_size = int(params.get("sample_size", 30))
        self.seed = int(params.get("seed", 0))
        self.include_pool_table = bool(params.get("include_pool_table", True))

    def build_messages(
        self,
        *,
        graph_artefact: "VLMArtefact",
        pool: "RolloutPool",
        condition: str,
        n_requests_per_type: Dict[str, int],
        json_schema: Dict[str, Any],
        history: Optional[List[Message]] = None,
        task_hint: str = "",
    ) -> List[Message]:
        rng = random.Random(self.seed)

        system = Message(
            role="system",
            text_blocks=[
                _system_prompt(
                    task_hint=task_hint,
                    n_requests_per_type=n_requests_per_type,
                    json_schema=json_schema,
                )
            ],
        )

        user_blocks: List[str] = []
        user_images: List[Path] = []

        if pool.entries:
            n = min(self.sample_size, len(pool.entries))
            sample = rng.sample(pool.entries, n)
            user_blocks.append(
                f"## Sampled rollouts (n={n}, outcomes only)"
            )
            for e in sample:
                user_blocks.append(f"- {e.rollout_id} [{_outcome_str(e.success)}]")
                if e.storyboard_path is not None and Path(e.storyboard_path).exists():
                    user_images.append(Path(e.storyboard_path))

        if self.include_pool_table:
            user_blocks.append(_pool_table_block(pool))

        total = sum(n_requests_per_type.values())
        user_blocks.append(
            f"\nProduce a JSON object with a 'requests' array of {total} items "
            "matching the distribution above."
        )

        user = Message(role="user", text_blocks=user_blocks, images=user_images)

        msgs: List[Message] = [system]
        if history:
            msgs.extend(history)
        msgs.append(user)
        return msgs


def build_outcome_condition(params: Dict[str, Any]) -> OutcomeConditionInputBuilder:
    return OutcomeConditionInputBuilder(params)


register_vlm_input_builder("outcome_condition", build_outcome_condition)
