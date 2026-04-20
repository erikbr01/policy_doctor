"""Adapter: PMM → BehaviorGraph.

Converts the Probabilistic Mealy Machine produced by :class:`ExtendedLStar`
into the shared :class:`~policy_doctor.behaviors.behavior_graph.BehaviorGraph`
format consumed by all downstream pipeline steps.

Usage::

    pmm, node_assignments = ExtendedLStar(...).build_graph()
    graph = pmm_to_behavior_graph(pmm, node_assignments, actions, metadata, level="task")
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from policy_doctor.behaviors.behavior_graph import BehaviorGraph
from policy_doctor.enap.extended_l_star import PMM


def pmm_to_behavior_graph(
    pmm: PMM,
    node_assignments: np.ndarray,
    actions: np.ndarray,
    metadata: List[Dict],
    level: str,
    node_names: Optional[Dict[int, str]] = None,
) -> BehaviorGraph:
    """Convert a :class:`PMM` to a :class:`BehaviorGraph`.

    Builds ``pmm_edges`` (the ``Dict[int, Dict[int, Tuple]]`` expected by
    :meth:`~BehaviorGraph.from_enap_assignments`) from the PMM's edge
    attributes, then delegates all graph construction to
    :meth:`BehaviorGraph.from_enap_assignments`.

    Args:
        pmm: Probabilistic Mealy Machine returned by
            :meth:`~ExtendedLStar.build_graph`.
        node_assignments: Per-timestep node ID array of shape ``(N,)`` returned
            alongside the PMM.
        actions: ``(N, action_dim)`` continuous action array for all timesteps,
            in the same order as ``node_assignments``.
        metadata: Per-timestep metadata list of length ``N`` (dicts with at
            least ``"episode_id"``).
        level: Behaviour granularity label (e.g. ``"task"``).
        node_names: Optional ``{node_id: name_str}`` override map.  When
            provided, the names are injected into the node ``label`` field of
            the returned graph.

    Returns:
        A fully constructed :class:`BehaviorGraph` with ``builder="enap"``.
    """
    # Build pmm_edges dict: {src_id: {tgt_id: (input_symbol, next_input_set)}}
    pmm_edges: Dict[int, Dict[int, Tuple[Optional[int], Optional[List[int]]]]] = {}
    for node_id, node in pmm.nodes.items():
        if not node.outgoing:
            continue
        pmm_edges[node_id] = {}
        for sym, edge in node.outgoing.items():
            pmm_edges[node_id][edge.target_id] = (
                edge.input_symbol,
                edge.next_input_set,
            )

    graph = BehaviorGraph.from_enap_assignments(
        node_assignments=node_assignments,
        actions=actions,
        metadata=metadata,
        level=level,
        node_names=node_names,
        pmm_edges=pmm_edges,
    )

    return graph
