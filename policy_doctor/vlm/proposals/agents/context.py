"""Per-session shared state for the agent loop.

Tools never reach into the session loop and vice-versa: both sides communicate
through this dataclass. That makes tools straightforward to test in isolation
and keeps the loop free of tool-specific knowledge.

A :class:`SessionContext` is built once per agent session and threaded into
every tool registry call (see :mod:`.tools.registry`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

import numpy as np

from policy_doctor.vlm.proposals.agents.budget import (
    BudgetConfig,
    BudgetTracker,
    ResultCache,
)
from policy_doctor.vlm.proposals.request import DemonstrationRequest

if TYPE_CHECKING:
    from policy_doctor.behaviors.behavior_graph import BehaviorGraph
    from policy_doctor.vlm.proposals.pool import RolloutPool


@dataclass
class SubmittedRequest:
    """A request submitted via Layer 4 ``propose_collection_request``.

    Wraps the underlying :class:`DemonstrationRequest` with the agent's
    ``reasoning`` (server-side, never sent to operator) and a revision
    history that surfaces in the trace.
    """

    request: DemonstrationRequest
    reasoning: str
    revision_history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def request_id(self) -> str:
        return self.request.request_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "reasoning": self.reasoning,
            "revision_history": list(self.revision_history),
        }


@dataclass
class SessionContext:
    """Everything one tool needs to do its work.

    Attributes
    ----------
    condition:
        Pre-registered condition label. Drives the tool registry assembled by
        :func:`policy_doctor.vlm.proposals.agents.tools.registry.build_tool_registry`.
    graph:
        The behavior graph the agent reasons over. Always present, even for
        ``A_NG`` — the no-graph tools just refuse to expose it.
    pool:
        Rollout pool with storyboards / sim states / cluster paths.
    cluster_labels, cluster_metadata:
        Per-slice labels + metadata (window_start/end, rollout_idx, success).
        Needed by Layer 2 ``list_slices_in_node`` and the slice-id resolver.
    cluster_centroids:
        Optional ``(K, D)`` centroid matrix in the post-UMAP space. Required
        only when ``sort_by="centroid_distance"``; falls back to ``random``
        when absent.
    classifier:
        Optional :class:`policy_doctor.monitoring.TrajectoryClassifier`. The
        agent surface itself doesn't classify rollouts; classifier is plumbed
        through so submission / revision can validate references.
    raw_states_dir:
        Directory of ``{rollout_id}.npz`` raw state arrays (kinematic
        summaries pull from here when present).
    storyboards_dir, videos_dir:
        Sidecar directories produced by ``build_rollout_pool``. Layer 2 reads
        from them.
    budget, cache:
        See :mod:`.budget`.
    submitted:
        Live list of requests the agent has submitted.
    finalized:
        Set after :func:`finalize_strategy`. The session loop terminates.
    rationale:
        Set by :func:`finalize_strategy`.
    task_hint:
        Free-text task description, surfaced in the system prompt and the
        ``get_graph_summary`` tool result.
    config:
        Free-form pass-through for tool-specific settings (e.g.
        ``kinematic_summary_strategy``).
    """

    condition: str
    graph: "BehaviorGraph"
    pool: "RolloutPool"
    cluster_labels: Optional[np.ndarray] = None
    cluster_metadata: Optional[List[Dict[str, Any]]] = None
    cluster_centroids: Optional[np.ndarray] = None
    classifier: Any = None
    raw_states_dir: Optional[Path] = None
    storyboards_dir: Optional[Path] = None
    videos_dir: Optional[Path] = None
    slice_storyboards_dir: Optional[Path] = None

    budget: BudgetTracker = field(default_factory=BudgetTracker)
    cache: ResultCache = field(default_factory=ResultCache)

    submitted: List[SubmittedRequest] = field(default_factory=list)
    finalized: bool = False
    rationale: Optional[str] = None

    # Inspection bookkeeping. Used by the submission validator to require
    # the agent has actually *looked at* the cluster / rollout / slice it
    # cites before submitting:
    #   * ``inspected_nodes`` — clusters read via get_node / list_slices_in_node.
    #   * ``inspected_slices`` — slice_ids fetched via get_slice_video (visual).
    #   * ``inspected_rollouts`` — rollout_ids fetched via get_rollout_video (visual).
    # Each closes a different failure mode where the agent generates prose
    # from priors instead of from evidence.
    inspected_nodes: Set[int] = field(default_factory=set)
    inspected_slices: Set[str] = field(default_factory=set)
    inspected_rollouts: Set[str] = field(default_factory=set)

    task_hint: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    # ---- node value cache ----------------------------------------------------
    # Computed lazily on first access. Re-computed on demand if the graph is
    # rebuilt (rare; the spec requires the graph be frozen for the experiment).
    _node_values: Optional[Dict[int, float]] = None
    _failure_likelihoods: Optional[Dict[int, float]] = None

    def node_values(self) -> Dict[int, float]:
        """Cached Bellman V-values per node (gamma=0.99, +1 SUCCESS, -1 FAILURE)."""
        if self._node_values is None:
            self._node_values = self.graph.compute_values()
        return self._node_values

    def failure_likelihoods(self) -> Dict[int, float]:
        """Per-node probability of eventually reaching FAILURE.

        Computed by solving the linear system V_f = P V_f + p_terminal_F under
        the same transition matrix used by :meth:`BehaviorGraph.compute_values`,
        with reward 1 only at FAILURE.
        """
        if self._failure_likelihoods is None:
            self._failure_likelihoods = self.graph.compute_values(
                gamma=1.0,
                reward_success=0.0,
                reward_failure=1.0,
                reward_end=0.0,
            )
        return self._failure_likelihoods

    # ---- factory -------------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        condition: str,
        graph: "BehaviorGraph",
        pool: "RolloutPool",
        budget_config: Optional[BudgetConfig] = None,
        cache_enabled: bool = True,
        **kwargs: Any,
    ) -> "SessionContext":
        return cls(
            condition=condition,
            graph=graph,
            pool=pool,
            budget=BudgetTracker(config=budget_config or BudgetConfig()),
            cache=ResultCache(enabled=cache_enabled),
            **kwargs,
        )
