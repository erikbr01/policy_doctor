"""Graph-condition VLM input builder.

The VLM sees: the graph artefact (image + text), per-cluster example
storyboards, a sampled-rollouts table with cluster paths and outcomes, and the
full rollout pool. It is required to emit ``target_cluster`` for every request.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from policy_doctor.vlm.proposals.registry import register_vlm_input_builder
from policy_doctor.vlm.proposals.vlm_input.base import Message, VLMInputBuilder

if TYPE_CHECKING:
    from policy_doctor.vlm.proposals.graph_representation.base import VLMArtefact
    from policy_doctor.vlm.proposals.pool import RolloutEntry, RolloutPool


_OPERATOR_FORBIDDEN_TERMS = (
    "cluster", "node", "graph", "umap", "kmeans", "centroid", "embedding",
)


def _outcome_str(success: Optional[bool]) -> str:
    if success is True:
        return "success"
    if success is False:
        return "failure"
    return "unknown"


def _format_path(path: Optional[List[int]]) -> str:
    if not path:
        return "(none)"
    return " -> ".join(f"c{p}" if p >= 0 else str(p) for p in path)


def sample_per_cluster_examples(
    pool: "RolloutPool",
    *,
    n_per_cluster: int,
    rng: random.Random,
) -> Dict[int, List["RolloutEntry"]]:
    by_cluster: Dict[int, List["RolloutEntry"]] = defaultdict(list)
    for entry in pool.entries:
        if not entry.cluster_path:
            continue
        if entry.storyboard_path is None or not Path(entry.storyboard_path).exists():
            continue
        for cid in set(entry.cluster_path):
            if cid < 0:
                continue
            by_cluster[cid].append(entry)
    out: Dict[int, List["RolloutEntry"]] = {}
    for cid, entries in by_cluster.items():
        if len(entries) < n_per_cluster:
            continue
        out[cid] = rng.sample(entries, n_per_cluster)
    return out


def _system_prompt(
    *,
    task_hint: str,
    n_requests_per_type: Dict[str, int],
    json_schema: Dict[str, Any],
    require_target_cluster: bool,
) -> str:
    total = sum(n_requests_per_type.values())
    lines = [
        "You are a robotics data-collection planner. You are given a behavior",
        "graph and a pool of base-policy rollouts. Your job is to propose",
        f"{total} demonstration requests that, if executed by a trained operator,",
        "would teach the policy to recover from its observed failure modes and",
        "broaden its behavioral coverage.",
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
    ]
    if require_target_cluster:
        lines += [
            "",
            "Each request MUST include a 'target_cluster' integer (server-side",
            "metadata for adherence scoring; never shown to the operator). Pick",
            "the cluster id from the behavior graph that the request targets.",
        ]
    lines += [
        "",
        "Output: a single JSON object matching this schema (no surrounding text):",
        "```json",
        json.dumps(json_schema, indent=2),
        "```",
    ]
    return "\n".join(lines)


def _pool_table_block(pool: "RolloutPool", *, with_paths: bool) -> str:
    lines = ["## Rollout pool (full)"]
    header = "| rollout_id | length | outcome"
    if with_paths:
        header += " | cluster_path"
    header += " |"
    sep = "|---|---|---" + ("|---" if with_paths else "") + "|"
    lines.append(header)
    lines.append(sep)
    for entry in pool.entries:
        row = f"| {entry.rollout_id} | {entry.length} | {_outcome_str(entry.success)}"
        if with_paths:
            row += f" | {_format_path(entry.cluster_path)}"
        row += " |"
        lines.append(row)
    return "\n".join(lines)


def _sampled_rollouts_block(
    pool: "RolloutPool",
    *,
    sample_size: int,
    with_paths: bool,
    rng: random.Random,
) -> str:
    if not pool.entries:
        return "## Sampled rollouts\n(empty pool)"
    n = min(sample_size, len(pool.entries))
    sample = rng.sample(pool.entries, n)
    lines = [f"## Sampled rollouts (n={n})"]
    for e in sample:
        line = f"- {e.rollout_id} [{_outcome_str(e.success)}]"
        if with_paths:
            line += f": {_format_path(e.cluster_path)}"
        lines.append(line)
    return "\n".join(lines)


class GraphConditionInputBuilder(VLMInputBuilder):
    def __init__(self, params: Dict[str, Any] | None = None):
        params = params or {}
        self.sample_size = int(params.get("sample_size", 30))
        self.examples_per_cluster = int(params.get("examples_per_cluster", 3))
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
                    require_target_cluster=True,
                )
            ],
        )

        user_blocks: List[str] = []
        user_images: List[Path] = []

        user_blocks.append("## Behavior graph")
        user_blocks.extend(graph_artefact.text_blocks)
        user_images.extend(Path(p) for p in graph_artefact.images)

        per_cluster = sample_per_cluster_examples(
            pool,
            n_per_cluster=self.examples_per_cluster,
            rng=rng,
        )
        if per_cluster:
            user_blocks.append("")
            user_blocks.append(
                f"## Per-cluster example storyboards "
                f"(up to {self.examples_per_cluster} per cluster)"
            )
            for cid in sorted(per_cluster):
                user_blocks.append(f"Cluster c{cid}:")
                for e in per_cluster[cid]:
                    user_blocks.append(
                        f"  - {e.rollout_id} [{_outcome_str(e.success)}]: "
                        f"{_format_path(e.cluster_path)}"
                    )
                    if e.storyboard_path is not None and Path(e.storyboard_path).exists():
                        user_images.append(Path(e.storyboard_path))

        user_blocks.append(
            _sampled_rollouts_block(
                pool, sample_size=self.sample_size, with_paths=True, rng=rng,
            )
        )
        if self.include_pool_table:
            user_blocks.append(_pool_table_block(pool, with_paths=True))

        total = sum(n_requests_per_type.values())
        user_blocks.append(
            f"\nProduce a JSON object with a 'requests' array of {total} items "
            "matching the distribution above. Every request must include "
            "'target_cluster'."
        )

        user = Message(role="user", text_blocks=user_blocks, images=user_images)

        msgs: List[Message] = [system]
        if history:
            msgs.extend(history)
        msgs.append(user)
        return msgs


def build_graph_condition(params: Dict[str, Any]) -> GraphConditionInputBuilder:
    return GraphConditionInputBuilder(params)


register_vlm_input_builder("graph_condition", build_graph_condition)
