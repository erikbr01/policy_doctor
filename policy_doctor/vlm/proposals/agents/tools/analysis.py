"""Layer 3 — search and aggregation tools.

Convenience wrappers over Layer 1 + 2 that the agent could in principle
replicate from primitives, but that save tokens and make reasoning more
direct. Cheap by construction (no images, no disk I/O beyond what's already
in memory).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from policy_doctor.behaviors.behavior_graph import (
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
    TERMINAL_NODE_IDS,
)
from policy_doctor.vlm.proposals.agents.context import SessionContext
from policy_doctor.vlm.proposals.agents.tools import schema as S
from policy_doctor.vlm.proposals.agents.tools.topology import _node_name
from policy_doctor.vlm.proposals.agents.tools.types import ToolResult, ToolSpec


# ---------------------------------------------------------------------------
# find_failure_nodes
# ---------------------------------------------------------------------------


def _make_find_failure_nodes(ctx: SessionContext) -> ToolSpec:
    def _run(args: Dict[str, Any]) -> ToolResult:
        threshold = float(args.get("min_failure_prob", 0.3))
        failure_lik = ctx.failure_likelihoods()
        rows: List[Dict[str, Any]] = []
        for nid, node in ctx.graph.cluster_nodes.items():
            fl = failure_lik.get(nid, 0.0)
            if fl < threshold:
                continue
            # Best (highest-prob) path from this node to FAILURE.
            paths = ctx.graph.enumerate_paths_to_terminal(
                FAILURE_NODE_ID, max_paths=20
            )
            via = next(
                (
                    [_node_name(n) for n in p]
                    for p, _, _ in paths
                    if nid in p
                ),
                None,
            )
            rows.append({
                "node_id": nid,
                "name": _node_name(nid),
                "failure_likelihood": round(fl, 4),
                "n_episodes": node.num_episodes,
                "main_failure_path_via": via,
            })
        rows.sort(key=lambda r: -r["failure_likelihood"])
        return ToolResult.text(
            "find_failure_nodes",
            json.dumps({"n_nodes": len(rows), "nodes": rows}, default=str),
        )

    return ToolSpec(
        name="find_failure_nodes",
        description=(
            "List behavior cluster nodes with probability of reaching FAILURE at "
            "least min_failure_prob (default 0.3). Returns each with its "
            "best example path to FAILURE."
        ),
        input_schema=S.FIND_FAILURE_NODES,
        func=_run,
        cost="cheap",
    )


# ---------------------------------------------------------------------------
# find_recovery_paths
# ---------------------------------------------------------------------------


def _make_find_recovery_paths(ctx: SessionContext) -> ToolSpec:
    def _run(args: Dict[str, Any]) -> ToolResult:
        from_id = int(args.get("from_node"))
        top_k = int(args.get("top_k", 5))
        if from_id not in ctx.graph.nodes:
            return ToolResult.error(
                "find_recovery_paths", f"node {from_id} not in graph", code="not_found"
            )
        all_paths = ctx.graph.enumerate_paths_to_terminal(
            SUCCESS_NODE_ID, max_paths=200
        )
        # Filter to paths that pass through from_id.
        relevant = [(p, prob) for p, prob, _ in all_paths if from_id in p]
        relevant.sort(key=lambda x: -x[1])
        rows = [
            {
                "path": [_node_name(n) for n in p],
                "node_ids": list(p),
                "probability": round(prob, 4),
            }
            for p, prob in relevant[:top_k]
        ]
        return ToolResult.text(
            "find_recovery_paths",
            json.dumps({"from_node": from_id, "n_paths": len(rows), "paths": rows}, default=str),
        )

    return ToolSpec(
        name="find_recovery_paths",
        description=(
            "Top-k highest-probability paths from from_node that reach SUCCESS. "
            "Use after find_failure_nodes to learn what successful traversals "
            "through a high-failure node look like."
        ),
        input_schema=S.FIND_RECOVERY_PATHS,
        func=_run,
        cost="cheap",
        # Specific (from_node) lookup. Charges normally; remains callable after
        # exhaustion so the agent can verify a recovery target before submitting.
        bypass_when_exhausted=True,
    )


# ---------------------------------------------------------------------------
# find_underrepresented_modes
# ---------------------------------------------------------------------------


def _make_find_underrepresented_modes(ctx: SessionContext) -> ToolSpec:
    def _run(args: Dict[str, Any]) -> ToolResult:
        metric = str(args.get("metric", "rollout_count"))
        threshold = float(args.get("threshold", 5))
        values = ctx.node_values()
        rows: List[Dict[str, Any]] = []
        for nid, node in ctx.graph.cluster_nodes.items():
            if metric == "rollout_count":
                if node.num_episodes >= threshold:
                    continue
                reason = "low rollout count"
                value = node.num_episodes
            elif metric == "v":
                v = values.get(nid, 0.0)
                if v >= threshold:
                    continue
                reason = "low V-value"
                value = round(v, 4)
            else:
                return ToolResult.error(
                    "find_underrepresented_modes",
                    f"unknown metric {metric!r}",
                    code="bad_arg",
                )
            rows.append({
                "node_id": nid,
                "name": _node_name(nid),
                "value": value,
                "reason": reason,
            })
        return ToolResult.text(
            "find_underrepresented_modes",
            json.dumps({"metric": metric, "n_nodes": len(rows), "nodes": rows}, default=str),
        )

    return ToolSpec(
        name="find_underrepresented_modes",
        description=(
            "Find behavior nodes that are likely under-supported by demonstration data. "
            "metric='rollout_count' returns nodes touched by fewer than threshold rollouts; "
            "metric='v' returns nodes with low Bellman value."
        ),
        input_schema=S.FIND_UNDERREPRESENTED_MODES,
        func=_run,
        cost="cheap",
    )


# ---------------------------------------------------------------------------
# compare_paths
# ---------------------------------------------------------------------------


def _make_compare_paths(ctx: SessionContext) -> ToolSpec:
    def _run(args: Dict[str, Any]) -> ToolResult:
        a = list(args.get("path_a") or [])
        b = list(args.get("path_b") or [])
        if not a or not b:
            return ToolResult.error(
                "compare_paths", "path_a and path_b must be non-empty", code="bad_arg"
            )

        # Shared prefix.
        shared: List[int] = []
        for x, y in zip(a, b):
            if x == y:
                shared.append(x)
            else:
                break
        divergence_point = shared[-1] if shared else None

        # Per-rollout outcome distributions for rollouts following each path.
        # Pool cluster paths contain only real cluster ids — strip any START /
        # terminal sentinels the agent may have included.
        def _interior(path: List[int]) -> List[int]:
            return [n for n in path if n != START_NODE_ID and n not in TERMINAL_NODE_IDS]

        def _outcome_dist(path: List[int]) -> Dict[str, float]:
            interior = _interior(path)
            n_succ, n_fail, n_other = 0, 0, 0
            for entry in ctx.pool.entries:
                cp = entry.cluster_path or []
                if _is_subseq(interior, cp):
                    if entry.success is True:
                        n_succ += 1
                    elif entry.success is False:
                        n_fail += 1
                    else:
                        n_other += 1
            total = n_succ + n_fail + n_other
            if total == 0:
                return {"success": 0.0, "failure": 0.0, "other": 0.0, "n": 0}
            return {
                "success": round(n_succ / total, 4),
                "failure": round(n_fail / total, 4),
                "other": round(n_other / total, 4),
                "n": total,
            }

        payload = {
            "shared_prefix": [_node_name(n) for n in shared],
            "divergence_point": (_node_name(divergence_point) if divergence_point is not None else None),
            "path_a_outcome_distribution": _outcome_dist(a),
            "path_b_outcome_distribution": _outcome_dist(b),
        }
        return ToolResult.text("compare_paths", json.dumps(payload, default=str))

    return ToolSpec(
        name="compare_paths",
        description=(
            "Compare two graph paths: shared prefix, divergence point, and per-path "
            "outcome distributions over rollouts that traverse them. Use to verify "
            "that an intervention path actually diverges from the failure path."
        ),
        input_schema=S.COMPARE_PATHS,
        func=_run,
        cost="cheap",
        # Specific (path_a, path_b) verification. Charges normally; remains
        # callable after exhaustion to confirm path divergence pre-submit.
        bypass_when_exhausted=True,
    )


def _is_subseq(needle: List[int], haystack: List[int]) -> bool:
    it = iter(haystack)
    return all(any(x == n for x in it) for n in needle)


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------


def build(ctx: SessionContext) -> List[ToolSpec]:
    return [
        _make_find_failure_nodes(ctx),
        _make_find_recovery_paths(ctx),
        _make_find_underrepresented_modes(ctx),
        _make_compare_paths(ctx),
    ]
