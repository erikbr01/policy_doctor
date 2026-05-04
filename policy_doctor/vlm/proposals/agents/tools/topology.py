"""Layer 1 — graph topology tools (cheap, broad, no images).

These are the agent's first-line orientation tools. Almost every session
starts with ``get_graph_summary`` followed by ``list_nodes`` /
``list_paths``; the spec is explicit that this layer must be cheap so the
agent can form hypotheses *before* committing to visual budget.

All tools read the (frozen) :class:`policy_doctor.behaviors.behavior_graph.BehaviorGraph`
on the :class:`SessionContext`. None of them write state.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from policy_doctor.behaviors.behavior_graph import (
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
    BehaviorGraph,
)
from policy_doctor.vlm.proposals.agents.context import SessionContext
from policy_doctor.vlm.proposals.agents.tools import schema as S
from policy_doctor.vlm.proposals.agents.tools.kinematic_summary import (
    kinematic_summary_for_node,
)
from policy_doctor.vlm.proposals.agents.tools.types import ToolResult, ToolSpec


# Pretty names for the special node ids.
_SPECIAL_NAMES = {
    START_NODE_ID: "START",
    SUCCESS_NODE_ID: "SUCCESS",
    FAILURE_NODE_ID: "FAILURE",
    END_NODE_ID: "END",
}
_NAME_TO_ID = {v: k for k, v in _SPECIAL_NAMES.items()}


def _node_name(node_id: int) -> str:
    """Render a node id (special or cluster) as a human-readable string."""
    return _SPECIAL_NAMES.get(node_id, f"c{node_id}")


def _resolve_node_arg(value: Any) -> Optional[int]:
    """Accept either an int cluster id or a special string name."""
    if isinstance(value, bool):  # bool is int subclass
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip().upper()
        if s in _NAME_TO_ID:
            return _NAME_TO_ID[s]
        if s.startswith("C") and s[1:].isdigit():
            return int(s[1:])
        if s.lstrip("-").isdigit():
            return int(s)
    return None


# ---------------------------------------------------------------------------
# get_graph_summary
# ---------------------------------------------------------------------------


def _make_get_graph_summary(ctx: SessionContext) -> ToolSpec:
    def _run(_args: Dict[str, Any]) -> ToolResult:
        g = ctx.graph
        values = ctx.node_values()
        failure_lik = ctx.failure_likelihoods()

        cluster_node_ids = sorted(g.cluster_nodes.keys())
        v_values = [values[nid] for nid in cluster_node_ids] if cluster_node_ids else [0.0]

        # Paths to each terminal — count by terminal kind.
        paths = g.enumerate_paths(max_paths=200)
        n_to_success = sum(1 for p, _, _ in paths if p[-1] == SUCCESS_NODE_ID)
        n_to_failure = sum(1 for p, _, _ in paths if p[-1] == FAILURE_NODE_ID)
        n_to_end = sum(1 for p, _, _ in paths if p[-1] == END_NODE_ID)

        # Marginal terminal probability — sum of path probabilities by terminal.
        # (Approximation; bounded by enumerate_paths' max_paths cap.)
        terminal_prob: Dict[str, float] = {"success": 0.0, "failure": 0.0, "end": 0.0}
        for p, prob, _ in paths:
            if p[-1] == SUCCESS_NODE_ID:
                terminal_prob["success"] += prob
            elif p[-1] == FAILURE_NODE_ID:
                terminal_prob["failure"] += prob
            elif p[-1] == END_NODE_ID:
                terminal_prob["end"] += prob

        # Outcome counts from the pool (ground truth, exact).
        outcomes = {
            "success": sum(1 for e in ctx.pool.entries if e.success is True),
            "failure": sum(1 for e in ctx.pool.entries if e.success is False),
            "unknown": sum(1 for e in ctx.pool.entries if e.success is None),
        }

        summary: Dict[str, Any] = {
            "task_hint": ctx.task_hint,
            "n_cluster_nodes": len(cluster_node_ids),
            "n_total_nodes": len(g.nodes),
            "terminal_node_kinds": sorted(_node_name(t) for t in g.terminal_node_ids),
            "n_paths_enumerated": len(paths),
            "n_paths_to_success": n_to_success,
            "n_paths_to_failure": n_to_failure,
            "n_paths_to_end": n_to_end,
            "marginal_terminal_probability": {
                k: round(v, 4) for k, v in terminal_prob.items()
            },
            "v_value_range": [round(min(v_values), 4), round(max(v_values), 4)],
            "n_rollouts_in_pool": len(ctx.pool),
            "rollout_outcomes": outcomes,
            "max_failure_likelihood_among_clusters": round(
                max((failure_lik.get(nid, 0.0) for nid in cluster_node_ids), default=0.0), 4
            ),
        }
        return ToolResult(
            name="get_graph_summary",
            ok=True,
            content=[
                # JSON in a text block; the agent can parse, but it's also readable.
                _json_block(summary),
            ],
        )

    return ToolSpec(
        name="get_graph_summary",
        description=(
            "Return a high-level overview of the behavior graph and rollout pool: "
            "node counts, marginal terminal probabilities, V-value range, and pool "
            "outcome counts. Cheap. Almost always the first call."
        ),
        input_schema=S.GET_GRAPH_SUMMARY,
        func=_run,
        cost="cheap",
    )


# ---------------------------------------------------------------------------
# list_nodes
# ---------------------------------------------------------------------------


def _make_list_nodes(ctx: SessionContext) -> ToolSpec:
    def _run(args: Dict[str, Any]) -> ToolResult:
        min_fail = float(args.get("min_failure_likelihood", 0.0))
        min_v = float(args.get("min_v", -1.0))
        max_v = float(args.get("max_v", 1.0))
        g = ctx.graph
        values = ctx.node_values()
        failure_lik = ctx.failure_likelihoods()

        rows: List[Dict[str, Any]] = []
        for nid, node in sorted(g.cluster_nodes.items()):
            v = values.get(nid, 0.0)
            fl = failure_lik.get(nid, 0.0)
            if fl < min_fail or v < min_v or v > max_v:
                continue
            rows.append({
                "node_id": nid,
                "name": _node_name(nid),
                "v": round(v, 4),
                "failure_likelihood": round(fl, 4),
                "in_degree": len(g.get_incoming_transitions(nid)),
                "out_degree": len(g.get_outgoing_transitions(nid)),
                "n_episodes": node.num_episodes,
                "n_timesteps": node.num_timesteps,
            })
        return ToolResult(
            name="list_nodes",
            ok=True,
            content=[_json_block({"n_filtered": len(rows), "nodes": rows})],
        )

    return ToolSpec(
        name="list_nodes",
        description=(
            "List behavior-cluster nodes (excluding START and terminals) with summary "
            "stats (V-value, failure likelihood, in/out degrees, rollout count). "
            "Optional filters trim to high-failure or high/low-V regions. Cheap."
        ),
        input_schema=S.LIST_NODES,
        func=_run,
        cost="cheap",
    )


# ---------------------------------------------------------------------------
# list_paths
# ---------------------------------------------------------------------------


def _make_list_paths(ctx: SessionContext) -> ToolSpec:
    def _run(args: Dict[str, Any]) -> ToolResult:
        from_node_arg = args.get("from_node", "START")
        to_node_arg = args.get("to_node", "FAILURE")
        top_k = int(args.get("top_k", 10))

        from_id = _resolve_node_arg(from_node_arg)
        to_id = _resolve_node_arg(to_node_arg)
        if from_id is None or to_id is None:
            return ToolResult.error(
                "list_paths",
                f"unrecognized node names; got from_node={from_node_arg!r} to_node={to_node_arg!r}",
                code="bad_arg",
            )

        all_paths = ctx.graph.enumerate_paths(max_paths=max(top_k * 5, 50))
        # Filter by both endpoints.
        filtered = [
            (path, prob)
            for path, prob, _ in all_paths
            if path[0] == from_id and path[-1] == to_id
        ]
        # enumerate_paths returns sorted by descending probability.
        rows = []
        for path, prob in filtered[:top_k]:
            rows.append({
                "path": [_node_name(n) for n in path],
                "node_ids": list(path),
                "probability": round(prob, 4),
            })
        return ToolResult(
            name="list_paths",
            ok=True,
            content=[_json_block({"n_paths": len(rows), "paths": rows})],
        )

    return ToolSpec(
        name="list_paths",
        description=(
            "Enumerate the top-k highest-probability paths between two nodes "
            "(default START → FAILURE). Path probabilities are products of edge "
            "transition probabilities under the Markov chain. Cheap."
        ),
        input_schema=S.LIST_PATHS,
        func=_run,
        cost="cheap",
        # Specific (from, to) verification lookup. Charges normally during
        # exploration; remains available after budget exhaustion as a
        # recovery affordance so the agent can verify a path before submitting.
        bypass_when_exhausted=True,
    )


# ---------------------------------------------------------------------------
# get_node
# ---------------------------------------------------------------------------


def _make_get_node(ctx: SessionContext) -> ToolSpec:
    def _run(args: Dict[str, Any]) -> ToolResult:
        node_id = _resolve_node_arg(args.get("node_id"))
        if node_id is None or node_id not in ctx.graph.nodes:
            return ToolResult.error(
                "get_node",
                f"node_id={args.get('node_id')!r} not in graph",
                code="not_found",
            )
        # Track that the agent has inspected this cluster — submission
        # validator requires inspection before targeting.
        if node_id >= 0:
            ctx.inspected_nodes.add(int(node_id))

        g = ctx.graph
        node = g.nodes[node_id]
        v = ctx.node_values().get(node_id, 0.0)
        fl = ctx.failure_likelihoods().get(node_id, 0.0)

        in_edges = [
            {
                "from": _node_name(src),
                "from_id": src,
                "edge_prob": round(p, 4),
                "count": cnt,
            }
            for src, cnt, p in sorted(
                g.get_incoming_transitions(node_id), key=lambda x: -x[2]
            )
        ]
        out_edges = [
            {
                "to": _node_name(tgt),
                "to_id": tgt,
                "edge_prob": round(p, 4),
                "count": cnt,
            }
            for tgt, cnt, p in sorted(
                g.get_outgoing_transitions(node_id), key=lambda x: -x[2]
            )
        ]

        # Kinematic summary requires per-slice metadata + a raw-states dir; falls
        # back to a structural summary when either is missing.
        kin = kinematic_summary_for_node(ctx, node_id)

        payload = {
            "node_id": node_id,
            "name": _node_name(node_id),
            "v": round(v, 4),
            "failure_likelihood": round(fl, 4),
            "n_timesteps": node.num_timesteps,
            "n_episodes": node.num_episodes,
            "predecessors": in_edges,
            "successors": out_edges,
            "kinematic_summary": kin,
        }
        return ToolResult(
            name="get_node",
            ok=True,
            content=[_json_block(payload)],
        )

    return ToolSpec(
        name="get_node",
        description=(
            "Full information for one node: V-value, failure likelihood, in/out edges "
            "with counts and probabilities, and a textual kinematic_summary computed "
            "from raw state trajectories. Use this BEFORE spending visual budget on "
            "a node — the kinematic summary is often enough to form a hypothesis."
        ),
        input_schema=S.GET_NODE,
        func=_run,
        cost="cheap",
        # Specific node verification lookup. Charges normally during
        # exploration; remains available after budget exhaustion so the
        # agent can satisfy the cluster_not_inspected gate before submitting.
        bypass_when_exhausted=True,
    )


# ---------------------------------------------------------------------------
# get_edge
# ---------------------------------------------------------------------------


def _make_get_edge(ctx: SessionContext) -> ToolSpec:
    def _run(args: Dict[str, Any]) -> ToolResult:
        from_id = _resolve_node_arg(args.get("from_node"))
        to_id = _resolve_node_arg(args.get("to_node"))
        if from_id is None or to_id is None:
            return ToolResult.error(
                "get_edge",
                f"bad node ids: from={args.get('from_node')!r} to={args.get('to_node')!r}",
                code="bad_arg",
            )
        g = ctx.graph
        prob = g.transition_probs.get(from_id, {}).get(to_id)
        if prob is None:
            return ToolResult.error(
                "get_edge",
                f"no edge {_node_name(from_id)} → {_node_name(to_id)}",
                code="not_found",
            )
        count = g.transition_counts.get(from_id, {}).get(to_id, 0)

        # Example rollouts — find pool entries whose collapsed cluster path
        # contains this transition.
        examples: List[str] = []
        for entry in ctx.pool.entries:
            cp = entry.cluster_path
            if not cp:
                continue
            for i in range(len(cp) - 1):
                if cp[i] == from_id and cp[i + 1] == to_id:
                    examples.append(entry.rollout_id)
                    break
            if len(examples) >= 8:
                break

        # Advantage = V(to) - V(from) at this transition.
        v = ctx.node_values()
        advantage = v.get(to_id, 0.0) - v.get(from_id, 0.0)

        payload = {
            "from": _node_name(from_id),
            "to": _node_name(to_id),
            "from_id": from_id,
            "to_id": to_id,
            "probability": round(prob, 4),
            "count": count,
            "advantage": round(advantage, 4),
            "example_rollouts": examples,
        }
        return ToolResult(
            name="get_edge",
            ok=True,
            content=[_json_block(payload)],
        )

    return ToolSpec(
        name="get_edge",
        description=(
            "Information about one edge: probability, count, advantage = V(to) − V(from), "
            "and up to 8 example rollouts that traverse it. Cheap."
        ),
        input_schema=S.GET_EDGE,
        func=_run,
        cost="cheap",
        # Specific edge verification lookup. Charges normally; remains
        # callable after exhaustion as a recovery affordance.
        bypass_when_exhausted=True,
    )


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------


def build(ctx: SessionContext) -> List[ToolSpec]:
    """Build all Layer 1 tool specs for one session."""
    return [
        _make_get_graph_summary(ctx),
        _make_list_nodes(ctx),
        _make_list_paths(ctx),
        _make_get_node(ctx),
        _make_get_edge(ctx),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_block(payload: Dict[str, Any]):
    """Return a TextBlock with pretty-printed JSON content (stable key order)."""
    from policy_doctor.vlm.proposals.agents.tools.types import TextBlock

    return TextBlock(text=json.dumps(payload, indent=2, sort_keys=False, default=str))
