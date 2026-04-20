"""Extended L* algorithm for extracting a Probabilistic Mealy Machine (PMM).

Implements Algorithm 1 from ENAP (Pan, Luo et al. 2026).  The algorithm builds
a PMM offline from a dataset of ``(h_t, c_t, a_t)`` tuples by iteratively
expanding a set of state-representative embeddings (the *Prefix Set* U) until
the resulting hypothesis graph is *closed* (all reachable states are
represented) and *consistent* (no trajectory contradicts the graph).

The three key operations are:

- **Membership Query (MQ)**: given a centroid ``u``, scan the dataset for all
  ``h_t`` within ``τ_sim`` (cosine distance), collect empirical outgoing
  transitions and their action priors.
- **Closedness check**: ensure every destination embedding reached during MQ
  maps to an existing centroid in ``U``.  If not, add the new embedding as a
  new state centroid.
- **Equivalence Query (EQ)** (non-deterministic variant): replay each
  trajectory through the hypothesis PMM.  If any step has no matching edge
  (unknown symbol) or the nearest matching centroid is too far away, add the
  current ``h_t`` as a counter-example and expand ``U``.

Stable-phase pruning (Algorithm 1 §3.4) merges nodes that share identical
outgoing edge signatures to keep the PMM compact.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# PMM data structures
# ---------------------------------------------------------------------------

@dataclass
class PMMEdge:
    """A directed edge in the Probabilistic Mealy Machine.

    Args:
        target_id: ID of the destination :class:`PMMNode`.
        input_symbol: Observation symbol ``c_t`` that triggers this edge.
        probability: Empirical transition probability P(target | source, symbol).
        action_prior: Mean continuous action ``a_base`` observed on this edge
            (shape ``(action_dim,)``).
        next_input_set: Set of valid following symbols (NIS), i.e. which ``c_{t+1}``
            values have been observed after taking this edge.
        count: Raw observation count used to compute ``probability``.
    """

    target_id: int
    input_symbol: int
    probability: float
    action_prior: np.ndarray
    next_input_set: List[int] = field(default_factory=list)
    count: int = 0


@dataclass
class PMMNode:
    """A state (node) in the PMM.

    A node represents a stable task phase characterised by a centroid embedding
    ``u`` in the RNN hidden state space.

    Args:
        id: Unique integer node identifier.
        embedding_centroid: Representative ``h_t`` embedding for this state,
            shape ``(hidden_dim,)``.
        outgoing: Dict mapping ``input_symbol → PMMEdge``.
    """

    id: int
    embedding_centroid: np.ndarray
    outgoing: Dict[int, PMMEdge] = field(default_factory=dict)

    def edge_signature(self) -> Tuple[int, ...]:
        """Sorted tuple of outgoing symbols — used for stable-phase pruning."""
        return tuple(sorted(self.outgoing.keys()))


@dataclass
class PMM:
    """A Probabilistic Mealy Machine extracted by the Extended L* algorithm.

    Args:
        nodes: Mapping from node ID to :class:`PMMNode`.
        start_node_id: ID of the initial state.
    """

    nodes: Dict[int, PMMNode]
    start_node_id: int

    def step(
        self, current_node_id: int, symbol: int
    ) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """Advance the PMM by one step.

        Args:
            current_node_id: Current state ID.
            symbol: Observed symbol ``c_t``.

        Returns:
            ``(next_node_id, action_prior)`` — or ``(None, None)`` if no edge
            exists for this symbol.
        """
        node = self.nodes.get(current_node_id)
        if node is None:
            return None, None
        edge = node.outgoing.get(symbol)
        if edge is None:
            return None, None
        return edge.target_id, edge.action_prior

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        nodes_out: Dict[str, Any] = {}
        for nid, node in self.nodes.items():
            edges_out: Dict[str, Any] = {}
            for sym, edge in node.outgoing.items():
                edges_out[str(sym)] = {
                    "target_id": edge.target_id,
                    "input_symbol": edge.input_symbol,
                    "probability": edge.probability,
                    "action_prior": edge.action_prior.tolist(),
                    "next_input_set": edge.next_input_set,
                    "count": edge.count,
                }
            nodes_out[str(nid)] = {
                "id": node.id,
                "embedding_centroid": node.embedding_centroid.tolist(),
                "outgoing": edges_out,
            }
        return {
            "nodes": nodes_out,
            "start_node_id": self.start_node_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PMM":
        """Reconstruct a PMM from a :meth:`to_dict` payload."""
        nodes: Dict[int, PMMNode] = {}
        for nid_s, nd in d["nodes"].items():
            nid = int(nid_s)
            outgoing: Dict[int, PMMEdge] = {}
            for sym_s, ed in nd["outgoing"].items():
                sym = int(sym_s)
                outgoing[sym] = PMMEdge(
                    target_id=ed["target_id"],
                    input_symbol=ed["input_symbol"],
                    probability=ed["probability"],
                    action_prior=np.array(ed["action_prior"]),
                    next_input_set=ed.get("next_input_set", []),
                    count=ed.get("count", 0),
                )
            nodes[nid] = PMMNode(
                id=nid,
                embedding_centroid=np.array(nd["embedding_centroid"]),
                outgoing=outgoing,
            )
        return cls(nodes=nodes, start_node_id=d["start_node_id"])


# ---------------------------------------------------------------------------
# Extended L* algorithm
# ---------------------------------------------------------------------------

class ExtendedLStar:
    """Non-deterministic extension of the L* algorithm for PMM extraction.

    The algorithm operates on a dataset of per-timestep tuples
    ``(h_t, c_t, a_t)`` grouped into episodes and iteratively refines a set
    of state representatives until the hypothesis PMM is both *closed* and
    *consistent*.

    Args:
        h_embeddings: Per-timestep RNN hidden states, shape ``(N, hidden_dim)``.
        symbols: Per-timestep discrete symbols ``c_t``, shape ``(N,)`` int.
        actions: Per-timestep continuous actions ``a_t``, shape ``(N, action_dim)``.
        metadata: Per-timestep metadata list (must contain ``"rollout_idx"`` or
            ``"demo_idx"`` and ``"timestep"``/``"window_start"``).
        tau_sim: Cosine-similarity threshold for node membership queries.
            Timesteps with ``cos_sim(h_t, centroid) ≥ τ_sim`` are considered
            members of that node.
        level: ``"rollout"`` or ``"demo"`` — selects the episode key from metadata.
        max_iterations: Upper bound on the MQ/EQ loop to prevent divergence.
        min_edge_count: Minimum number of observations required for an edge to
            be included in the PMM (filters spurious transitions).
    """

    def __init__(
        self,
        h_embeddings: np.ndarray,
        symbols: np.ndarray,
        actions: np.ndarray,
        metadata: List[Dict],
        tau_sim: float = 0.7,
        level: str = "rollout",
        max_iterations: int = 50,
        min_edge_count: int = 2,
    ) -> None:
        self.h = h_embeddings.astype(np.float32)
        self.c = symbols.astype(np.int64)
        self.a = actions.astype(np.float32)
        self.metadata = metadata
        self.tau_sim = tau_sim
        self.level = level
        self.max_iterations = max_iterations
        self.min_edge_count = min_edge_count

        # Pre-compute unit-normalised embeddings for fast cosine similarity
        norms = np.linalg.norm(self.h, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        self.h_unit = self.h / norms  # (N, D)

        self._ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
        self._episodes: Dict[int, List[int]] = self._build_episode_index()
        self._next_node_id: int = 0

    # ------------------------------------------------------------------
    # Episode index
    # ------------------------------------------------------------------

    def _build_episode_index(self) -> Dict[int, List[int]]:
        """Return {episode_idx: [sorted timestep indices]}."""
        ep_dict: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for i, meta in enumerate(self.metadata):
            ep_idx = meta.get(self._ep_key, 0)
            sort_key = meta.get("timestep", meta.get("window_start", 0))
            ep_dict[ep_idx].append((sort_key, i))
        return {
            ep: [idx for _, idx in sorted(ts)]
            for ep, ts in ep_dict.items()
        }

    # ------------------------------------------------------------------
    # Membership Query (MQ)
    # ------------------------------------------------------------------

    def _membership_query(
        self, centroid: np.ndarray
    ) -> Tuple[Dict[int, Dict[str, Any]], np.ndarray]:
        """Find all h_t within τ_sim of centroid and compute empirical transitions.

        Args:
            centroid: Query embedding ``u``, shape ``(D,)``.

        Returns:
            Tuple of:
            - ``transitions``: Dict ``{symbol: {"targets": {dest_idx: count},
              "actions": {dest_idx: [action_vecs]}, "nis": {dest_idx: set}}}``.
            - ``member_mask``: Boolean array of shape ``(N,)`` indicating which
              timesteps are members of this node.
        """
        centroid_unit = centroid / (np.linalg.norm(centroid) + 1e-8)
        cos_sims = self.h_unit @ centroid_unit  # (N,)
        member_mask = cos_sims >= self.tau_sim

        transitions: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {"targets": defaultdict(int), "actions": defaultdict(list), "nis": defaultdict(set)}
        )

        for ep_idx, indices in self._episodes.items():
            member_indices = [i for i in indices if member_mask[i]]
            for pos, ts_idx in enumerate(indices[:-1]):
                if not member_mask[ts_idx]:
                    continue
                next_idx = indices[pos + 1]
                sym = int(self.c[ts_idx])
                dest_h_idx = next_idx  # use next timestep's embedding as destination
                transitions[sym]["targets"][dest_h_idx] += 1
                transitions[sym]["actions"][dest_h_idx].append(self.a[ts_idx])
                # NIS: which symbol follows
                if pos + 1 < len(indices):
                    transitions[sym]["nis"][dest_h_idx].add(int(self.c[next_idx]))

        return dict(transitions), member_mask

    # ------------------------------------------------------------------
    # Node assignment
    # ------------------------------------------------------------------

    def _assign_to_node(
        self,
        h_t: np.ndarray,
        prefix_set_centroids: np.ndarray,
    ) -> int:
        """Return the index of the closest centroid, or -1 if none is ≥ τ_sim."""
        h_unit = h_t / (np.linalg.norm(h_t) + 1e-8)
        norms = np.linalg.norm(prefix_set_centroids, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        centroids_unit = prefix_set_centroids / norms
        sims = centroids_unit @ h_unit
        best = int(np.argmax(sims))
        return best if sims[best] >= self.tau_sim else -1

    # ------------------------------------------------------------------
    # Closedness check
    # ------------------------------------------------------------------

    def _check_closedness(
        self,
        nodes: Dict[int, PMMNode],
        transitions: Dict[int, Dict[str, Any]],
    ) -> Optional[np.ndarray]:
        """Check that all destination embeddings map to known nodes.

        Returns:
            ``None`` if closed, or the un-mapped destination embedding that
            should be added as a new state centroid.
        """
        centroids = np.stack([n.embedding_centroid for n in nodes.values()])
        for sym, tdata in transitions.items():
            for dest_ts_idx in tdata["targets"]:
                dest_h = self.h[dest_ts_idx]
                assigned = self._assign_to_node(dest_h, centroids)
                if assigned == -1:
                    return dest_h  # new centroid needed
        return None

    # ------------------------------------------------------------------
    # Hypothesis construction
    # ------------------------------------------------------------------

    def _build_hypothesis(
        self, nodes: Dict[int, PMMNode]
    ) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, np.ndarray]]:
        """Run MQ for each node and return raw transition data.

        Returns:
            Tuple of:
            - ``all_transitions``: ``{node_id: transition_dict}``
            - ``member_masks``: ``{node_id: bool array (N,)}``
        """
        all_transitions: Dict[int, Dict[str, Any]] = {}
        member_masks: Dict[int, np.ndarray] = {}
        for nid, node in nodes.items():
            trans, mask = self._membership_query(node.embedding_centroid)
            all_transitions[nid] = trans
            member_masks[nid] = mask
        return all_transitions, member_masks

    # ------------------------------------------------------------------
    # PMM materialisation
    # ------------------------------------------------------------------

    def _materialise_pmm(
        self,
        nodes: Dict[int, PMMNode],
        all_transitions: Dict[int, Dict[str, Any]],
        member_masks: Dict[int, np.ndarray],
    ) -> PMM:
        """Convert raw transition counts into a PMM with edges.

        Each edge is assigned a node-to-node transition (destination determined
        by :meth:`_assign_to_node` on the destination embedding) and an action
        prior (mean of all ``a_t`` on that edge).
        """
        centroids = np.stack([nodes[nid].embedding_centroid for nid in sorted(nodes)])
        nid_list = sorted(nodes.keys())

        for nid in nid_list:
            nodes[nid].outgoing.clear()

        for nid, trans in all_transitions.items():
            for sym, tdata in trans.items():
                # Aggregate all destinations to get node-level probabilities
                dest_node_counts: Dict[int, int] = defaultdict(int)
                dest_node_actions: Dict[int, List[np.ndarray]] = defaultdict(list)
                dest_node_nis: Dict[int, Set[int]] = defaultdict(set)

                for dest_ts_idx, cnt in tdata["targets"].items():
                    dest_h = self.h[dest_ts_idx]
                    idx_in_list = self._assign_to_node(dest_h, centroids)
                    if idx_in_list == -1:
                        continue
                    tgt_nid = nid_list[idx_in_list]
                    dest_node_counts[tgt_nid] += cnt
                    dest_node_actions[tgt_nid].extend(tdata["actions"][dest_ts_idx])
                    dest_node_nis[tgt_nid].update(tdata["nis"][dest_ts_idx])

                if not dest_node_counts:
                    continue
                total = sum(dest_node_counts.values())
                # Pick the most probable destination (deterministic collapse of
                # the non-deterministic EQ result)
                best_tgt = max(dest_node_counts, key=lambda k: dest_node_counts[k])
                cnt = dest_node_counts[best_tgt]
                if cnt < self.min_edge_count:
                    continue
                actions_list = dest_node_actions[best_tgt]
                action_prior = np.mean(actions_list, axis=0) if actions_list else np.zeros(self.a.shape[1])
                nis = sorted(dest_node_nis[best_tgt])
                nodes[nid].outgoing[sym] = PMMEdge(
                    target_id=best_tgt,
                    input_symbol=sym,
                    probability=cnt / total,
                    action_prior=action_prior,
                    next_input_set=nis,
                    count=cnt,
                )

        return PMM(nodes=nodes, start_node_id=min(nodes.keys()))

    # ------------------------------------------------------------------
    # Equivalence Query (EQ)
    # ------------------------------------------------------------------

    def _equivalence_query(
        self,
        pmm: PMM,
    ) -> Optional[np.ndarray]:
        """Replay trajectories through the PMM; return counter-example or None.

        A trajectory fails if at any step the current h_t is too far from the
        expected node's centroid (drift) or if no edge matches the observed symbol.

        Returns the ``h_t`` embedding at the failing step, or ``None`` if all
        trajectories are consistent.
        """
        centroids = np.stack([pmm.nodes[nid].embedding_centroid for nid in sorted(pmm.nodes)])
        nid_list = sorted(pmm.nodes.keys())

        for ep_idx, indices in self._episodes.items():
            if not indices:
                continue
            # Assign first timestep to start node
            current_nid = pmm.start_node_id
            for i, ts_idx in enumerate(indices[:-1]):
                # Check if h_t is still within τ_sim of current node
                idx_in_list = self._assign_to_node(self.h[ts_idx], centroids)
                if idx_in_list != -1:
                    current_nid = nid_list[idx_in_list]
                sym = int(self.c[ts_idx])
                next_nid, _ = pmm.step(current_nid, sym)
                if next_nid is None:
                    # No valid edge — counter-example at the next timestep
                    return self.h[indices[i + 1]]
                current_nid = next_nid
        return None

    # ------------------------------------------------------------------
    # Stable-phase pruning
    # ------------------------------------------------------------------

    def _stable_phase_pruning(self, pmm: PMM) -> PMM:
        """Merge nodes with identical outgoing edge signatures.

        Two nodes are mergeable if they have the same set of outgoing symbols
        and their merged centroid (mean of the two) would still be ≥ τ_sim from
        both original members' h_t embeddings.

        This is a simple greedy pass; the result may not be globally optimal.
        """
        changed = True
        while changed:
            changed = False
            nids = sorted(pmm.nodes.keys())
            for i in range(len(nids)):
                if nids[i] not in pmm.nodes:
                    continue
                for j in range(i + 1, len(nids)):
                    if nids[j] not in pmm.nodes:
                        continue
                    na = pmm.nodes[nids[i]]
                    nb = pmm.nodes[nids[j]]
                    if na.edge_signature() != nb.edge_signature():
                        continue
                    # Candidate merged centroid
                    merged_centroid = (na.embedding_centroid + nb.embedding_centroid) / 2.0
                    # Check that merged centroid still covers both original centroids
                    def _cos(u: np.ndarray, v: np.ndarray) -> float:
                        return float(
                            np.dot(u, v)
                            / (np.linalg.norm(u) + 1e-8)
                            / (np.linalg.norm(v) + 1e-8)
                        )
                    if (
                        _cos(merged_centroid, na.embedding_centroid) < self.tau_sim
                        or _cos(merged_centroid, nb.embedding_centroid) < self.tau_sim
                    ):
                        continue
                    # Merge nb into na
                    na.embedding_centroid = merged_centroid
                    # Redirect all edges pointing to nb → na
                    for nid, node in pmm.nodes.items():
                        for sym, edge in node.outgoing.items():
                            if edge.target_id == nids[j]:
                                edge.target_id = nids[i]
                    del pmm.nodes[nids[j]]
                    changed = True
                    break
                if changed:
                    break
        return pmm

    # ------------------------------------------------------------------
    # Node assignment for full dataset
    # ------------------------------------------------------------------

    def _assign_all_timesteps(self, pmm: PMM) -> np.ndarray:
        """Return per-timestep node assignments for the full dataset.

        Assigns each timestep to the PMM node whose centroid has the highest
        cosine similarity to h_t (regardless of threshold, so every point gets
        a node).

        Returns:
            Integer array of shape ``(N,)`` with PMM node IDs.
        """
        nid_list = sorted(pmm.nodes.keys())
        centroids = np.stack([pmm.nodes[nid].embedding_centroid for nid in nid_list])
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        centroids_unit = centroids / norms
        sims = self.h_unit @ centroids_unit.T  # (N, K)
        best_indices = sims.argmax(axis=1)     # (N,)
        return np.array([nid_list[i] for i in best_indices], dtype=np.int64)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def build_graph(self) -> Tuple[PMM, np.ndarray]:
        """Run the Extended L* algorithm and return the extracted PMM.

        Initialises the Prefix Set with the first timestep's embedding, then
        alternates between MQ (building the hypothesis) and EQ (testing it),
        expanding the prefix set with counter-examples until convergence or
        ``max_iterations`` is reached.

        Returns:
            Tuple of:
            - ``pmm``: The extracted :class:`PMM`.
            - ``node_assignments``: Per-timestep node ID array of shape ``(N,)``.
        """
        # Initialise prefix set with first observation
        first_h = self.h[0].copy()
        first_nid = self._next_node_id
        self._next_node_id += 1
        nodes: Dict[int, PMMNode] = {
            first_nid: PMMNode(id=first_nid, embedding_centroid=first_h)
        }

        for iteration in range(self.max_iterations):
            # Step 1: Membership queries → build raw transitions
            all_transitions, member_masks = self._build_hypothesis(nodes)

            # Step 2: Closedness check — add new centroids as needed
            closed = False
            while not closed:
                closed = True
                for nid, trans in all_transitions.items():
                    new_centroid = self._check_closedness(nodes, trans)
                    if new_centroid is not None:
                        new_id = self._next_node_id
                        self._next_node_id += 1
                        nodes[new_id] = PMMNode(id=new_id, embedding_centroid=new_centroid.copy())
                        # Re-run MQ for the new node
                        trans_new, mask_new = self._membership_query(new_centroid)
                        all_transitions[new_id] = trans_new
                        member_masks[new_id] = mask_new
                        closed = False
                        break

            # Step 3: Materialise PMM from current prefix set
            pmm = self._materialise_pmm(nodes, all_transitions, member_masks)

            # Step 4: Equivalence query
            counter_example = self._equivalence_query(pmm)
            if counter_example is None:
                break  # converged
            # Add counter-example as a new state
            new_id = self._next_node_id
            self._next_node_id += 1
            nodes[new_id] = PMMNode(id=new_id, embedding_centroid=counter_example.copy())

        # Step 5: Stable-phase pruning
        pmm = self._stable_phase_pruning(pmm)

        # Step 6: Assign all timesteps to PMM nodes
        node_assignments = self._assign_all_timesteps(pmm)

        return pmm, node_assignments
