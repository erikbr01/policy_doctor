"""Behavior graph data structure for analyzing transitions between behavioral clusters.

Given cluster assignments for timesteps across episodes (rollouts or demos),
this module builds a directed graph where:
- Each cluster is a node representing a behavioral mode
- START and END sentinel nodes mark trajectory boundaries
- Edges represent transitions between behaviors with associated probabilities

Consecutive identical cluster assignments within an episode are collapsed,
so transitions only count actual behavior changes.
"""

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

START_NODE_ID = -2
END_NODE_ID = -3
SUCCESS_NODE_ID = -4
FAILURE_NODE_ID = -5

TERMINAL_NODE_IDS = frozenset({END_NODE_ID, SUCCESS_NODE_ID, FAILURE_NODE_ID})


@dataclass
class BehaviorNode:
    """A node in the behavior graph representing a behavioral cluster or sentinel."""

    cluster_id: int
    name: str
    num_timesteps: int
    num_episodes: int
    episode_indices: List[int] = field(default_factory=list)

    @property
    def is_start(self) -> bool:
        return self.cluster_id == START_NODE_ID

    @property
    def is_end(self) -> bool:
        return self.cluster_id in TERMINAL_NODE_IDS

    @property
    def is_success(self) -> bool:
        return self.cluster_id == SUCCESS_NODE_ID

    @property
    def is_failure(self) -> bool:
        return self.cluster_id == FAILURE_NODE_ID

    @property
    def is_special(self) -> bool:
        return self.is_start or self.is_end


@dataclass
class BehaviorGraph:
    """Directed graph of behavioral transitions between clusters.

    Attributes:
        nodes: Mapping from cluster_id to BehaviorNode.
        transition_counts: Raw counts: source_id -> {target_id -> count}.
        transition_probs: Probabilities: source_id -> {target_id -> P(target|source)}.
        num_episodes: Total episodes used to build the graph.
        level: Whether graph was built from "rollout" or "demo" data.
    """

    nodes: Dict[int, BehaviorNode]
    transition_counts: Dict[int, Dict[int, int]]
    transition_probs: Dict[int, Dict[int, float]]
    num_episodes: int
    level: str

    @classmethod
    def from_cluster_assignments(
        cls,
        cluster_labels: np.ndarray,
        metadata: List[Dict],
        level: str = "rollout",
        cluster_names: Optional[Dict[int, str]] = None,
    ) -> "BehaviorGraph":
        """Build a behavior graph from cluster assignments and episode metadata.

        For each episode, the cluster assignment sequence is collapsed (removing
        consecutive duplicates) and START/terminal markers are added. If the
        metadata contains ``"success"`` info the terminal is split into SUCCESS
        and FAILURE nodes; otherwise a single END node is used.

        Transition probabilities are computed from the aggregated counts across
        all episodes.  Noise points (cluster_id == -1) are excluded.

        Args:
            cluster_labels: Array of cluster assignments, shape (N,).
            metadata: Per-sample metadata dicts with episode index and timestep info.
            level: "rollout" or "demo" — determines which metadata key identifies episodes.
            cluster_names: Optional mapping from cluster_id to display name (e.g. human labels).
        """
        ep_key = "rollout_idx" if level == "rollout" else "demo_idx"

        # Group timesteps by episode and collect per-episode outcomes
        episodes: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        episode_outcomes: Dict[int, Optional[bool]] = {}
        for i, meta in enumerate(metadata):
            label = int(cluster_labels[i])
            if label == -1:
                continue
            ep_idx = meta[ep_key]
            sort_key = meta.get("timestep", meta.get("window_start", 0))
            episodes[ep_idx].append((sort_key, label))
            if "success" in meta and ep_idx not in episode_outcomes:
                episode_outcomes[ep_idx] = meta["success"]

        for ep_idx in episodes:
            episodes[ep_idx].sort(key=lambda x: x[0])

        # Decide whether to use SUCCESS/FAILURE split
        has_outcome_info = any(v is not None for v in episode_outcomes.values())

        # Collapse consecutive identical assignments within each episode
        collapsed: Dict[int, List[int]] = {}
        for ep_idx, seq in episodes.items():
            if not seq:
                continue
            result = [seq[0][1]]
            for _, label in seq[1:]:
                if label != result[-1]:
                    result.append(label)
            collapsed[ep_idx] = result

        # Count transitions (START -> first, internal, last -> terminal)
        trans_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for ep_idx, seq in collapsed.items():
            if not seq:
                continue
            trans_counts[START_NODE_ID][seq[0]] += 1
            # Route to SUCCESS / FAILURE / END based on outcome
            if has_outcome_info:
                outcome = episode_outcomes.get(ep_idx)
                if outcome is True:
                    trans_counts[seq[-1]][SUCCESS_NODE_ID] += 1
                elif outcome is False:
                    trans_counts[seq[-1]][FAILURE_NODE_ID] += 1
                else:
                    trans_counts[seq[-1]][END_NODE_ID] += 1
            else:
                trans_counts[seq[-1]][END_NODE_ID] += 1
            for j in range(len(seq) - 1):
                trans_counts[seq[j]][seq[j + 1]] += 1

        # Per-node statistics from original (non-collapsed) assignments
        node_timestep_counts: Dict[int, int] = defaultdict(int)
        node_episode_sets: Dict[int, Set[int]] = defaultdict(set)
        for ep_idx, seq in episodes.items():
            for _, label in seq:
                node_timestep_counts[label] += 1
                node_episode_sets[label].add(ep_idx)

        # Normalize to probabilities
        trans_probs: Dict[int, Dict[int, float]] = {}
        for src, targets in trans_counts.items():
            total = sum(targets.values())
            trans_probs[src] = {
                tgt: count / total for tgt, count in targets.items()
            }

        # Build nodes
        all_episode_ids = sorted(collapsed.keys())
        success_ids = [e for e in all_episode_ids if episode_outcomes.get(e) is True]
        failure_ids = [e for e in all_episode_ids if episode_outcomes.get(e) is False]
        unknown_ids = [
            e for e in all_episode_ids
            if episode_outcomes.get(e) is None
        ]

        nodes: Dict[int, BehaviorNode] = {
            START_NODE_ID: BehaviorNode(
                START_NODE_ID, "START", 0, len(all_episode_ids), all_episode_ids,
            ),
        }

        if has_outcome_info:
            if success_ids:
                nodes[SUCCESS_NODE_ID] = BehaviorNode(
                    SUCCESS_NODE_ID, "SUCCESS", 0, len(success_ids), success_ids,
                )
            if failure_ids:
                nodes[FAILURE_NODE_ID] = BehaviorNode(
                    FAILURE_NODE_ID, "FAILURE", 0, len(failure_ids), failure_ids,
                )
            if unknown_ids:
                nodes[END_NODE_ID] = BehaviorNode(
                    END_NODE_ID, "END", 0, len(unknown_ids), unknown_ids,
                )
        else:
            nodes[END_NODE_ID] = BehaviorNode(
                END_NODE_ID, "END", 0, len(all_episode_ids), all_episode_ids,
            )

        for c_id in sorted(set(int(c) for c in cluster_labels if c != -1)):
            name = (
                cluster_names.get(c_id, f"Behavior {c_id}")
                if cluster_names is not None
                else f"Behavior {c_id}"
            )
            nodes[c_id] = BehaviorNode(
                cluster_id=c_id,
                name=name,
                num_timesteps=node_timestep_counts[c_id],
                num_episodes=len(node_episode_sets[c_id]),
                episode_indices=sorted(node_episode_sets[c_id]),
            )

        return cls(
            nodes=nodes,
            transition_counts={k: dict(v) for k, v in trans_counts.items()},
            transition_probs=trans_probs,
            num_episodes=len(all_episode_ids),
            level=level,
        )

    def get_outgoing_transitions(
        self, node_id: int,
    ) -> List[Tuple[int, int, float]]:
        """Return (target_id, count, probability) sorted by descending probability."""
        if node_id not in self.transition_counts:
            return []
        counts = self.transition_counts[node_id]
        probs = self.transition_probs.get(node_id, {})
        return sorted(
            [(tgt, cnt, probs.get(tgt, 0.0)) for tgt, cnt in counts.items()],
            key=lambda x: -x[2],
        )

    def get_incoming_transitions(
        self, node_id: int,
    ) -> List[Tuple[int, int, float]]:
        """Return (source_id, count, P(node|source)) sorted by descending probability."""
        incoming = []
        for src, targets in self.transition_counts.items():
            if node_id in targets:
                prob = self.transition_probs.get(src, {}).get(node_id, 0.0)
                incoming.append((src, targets[node_id], prob))
        return sorted(incoming, key=lambda x: -x[2])

    @property
    def cluster_nodes(self) -> Dict[int, BehaviorNode]:
        """Cluster nodes only (excluding START and terminal nodes)."""
        return {k: v for k, v in self.nodes.items() if not v.is_special}

    @property
    def terminal_node_ids(self) -> Set[int]:
        """IDs of terminal nodes (END, SUCCESS, FAILURE) present in this graph."""
        return {
            nid for nid in self.nodes
            if nid in TERMINAL_NODE_IDS
        }

    @property
    def has_outcome_split(self) -> bool:
        """True if the graph has SUCCESS/FAILURE rather than a single END."""
        return SUCCESS_NODE_ID in self.nodes or FAILURE_NODE_ID in self.nodes

    def compute_values(
        self,
        gamma: float = 0.99,
        reward_success: float = 1.0,
        reward_failure: float = -1.0,
        reward_end: float = 0.0,
    ) -> Dict[int, float]:
        """Compute the Bellman value of every node in the graph.

        Terminal nodes have fixed values equal to their rewards.  Non-terminal
        node values satisfy V(s) = gamma * sum_{s'} P(s'|s) * V(s'), which is
        solved exactly via a linear system.

        Args:
            gamma: Discount factor in [0, 1].
            reward_success: Reward assigned to the SUCCESS terminal.
            reward_failure: Reward assigned to the FAILURE terminal.
            reward_end: Reward assigned to the generic END terminal.

        Returns:
            Dict mapping each node_id to its value V(s).
        """
        terminal_rewards = {
            SUCCESS_NODE_ID: reward_success,
            FAILURE_NODE_ID: reward_failure,
            END_NODE_ID: reward_end,
        }

        # Separate terminal and non-terminal node IDs
        terminal_ids = {
            nid for nid in self.nodes if nid in TERMINAL_NODE_IDS
        }
        nonterminal_ids = sorted(
            nid for nid in self.nodes if nid not in terminal_ids
        )
        nt_index = {nid: i for i, nid in enumerate(nonterminal_ids)}
        n = len(nonterminal_ids)

        if n == 0:
            return {nid: terminal_rewards.get(nid, 0.0) for nid in self.nodes}

        # P_nn: transition matrix among non-terminal nodes
        # b: gamma * P_nt @ V_t  (contributions from terminal transitions)
        P_nn = np.zeros((n, n))
        b = np.zeros(n)

        for nid in nonterminal_ids:
            i = nt_index[nid]
            probs = self.transition_probs.get(nid, {})
            for tgt, p in probs.items():
                if tgt in nt_index:
                    P_nn[i, nt_index[tgt]] = p
                elif tgt in terminal_ids:
                    b[i] += p * terminal_rewards.get(tgt, 0.0)

        # Solve (I - gamma * P_nn) V = gamma * b
        A = np.eye(n) - gamma * P_nn
        rhs = gamma * b
        V_nt = np.linalg.solve(A, rhs)

        values: Dict[int, float] = {}
        for nid in nonterminal_ids:
            values[nid] = float(V_nt[nt_index[nid]])
        for nid in terminal_ids:
            values[nid] = terminal_rewards.get(nid, 0.0)

        # Guarantee every possible terminal node ID is present in the returned dict,
        # even if it was never added to self.nodes (e.g. END_NODE_ID is absent when
        # every episode has a known success/failure outcome). compute_slice_values()
        # does node_values.get(terminal, 0.0) — without this the fallback is always
        # 0.0 regardless of reward_end, silently using the wrong value.
        for nid, reward in terminal_rewards.items():
            if nid not in values:
                values[nid] = reward

        return values

    def compute_slice_values(
        self,
        cluster_labels: np.ndarray,
        metadata: List[Dict],
        node_values: Dict[int, float],
        gamma: float = 0.99,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-slice Q-values and advantages.

        For each slice, the Q-value is based on its specific transition in the
        episode: Q(s, s') = gamma * V(s').  The advantage is A = Q - V(s).

        Slices labeled as noise (-1) get Q = 0, A = 0.

        Args:
            cluster_labels: Array of cluster assignments, shape (N,).
            metadata: Per-sample metadata dicts (same as used to build the graph).
            node_values: Dict from ``compute_values()``.
            gamma: Discount factor (should match the one used for ``compute_values``).

        Returns:
            Tuple of (q_values, advantages, next_cluster) arrays, each shape (N,).
            next_cluster[i] is the next distinct cluster for slice i in its
            episode (or the terminal node id).
        """
        ep_key = "rollout_idx" if self.level == "rollout" else "demo_idx"
        n = len(cluster_labels)

        # Group slice indices by episode, sorted by timestep
        episodes: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        episode_outcomes: Dict[int, Optional[bool]] = {}
        for i, meta in enumerate(metadata):
            label = int(cluster_labels[i])
            ep_idx = meta[ep_key]
            sort_key = meta.get("timestep", meta.get("window_start", 0))
            episodes[ep_idx].append((sort_key, i, label))
            # Only set outcome from non-noise timesteps, matching from_cluster_assignments()
            # which skips label==-1 entries entirely. Without this guard, a noise timestep
            # at the start of an episode could set the wrong outcome for that episode.
            if label != -1 and "success" in meta and ep_idx not in episode_outcomes:
                episode_outcomes[ep_idx] = meta["success"]

        has_outcome_info = any(v is not None for v in episode_outcomes.values())

        q_values = np.zeros(n)
        advantages = np.zeros(n)
        next_cluster = np.full(n, -1, dtype=int)

        for ep_idx, slices in episodes.items():
            slices.sort(key=lambda x: x[0])

            # Build the collapsed sequence with the slice indices that belong
            # to each segment
            segments: List[Tuple[int, List[int]]] = []
            for _, idx, label in slices:
                if label == -1:
                    continue
                if segments and segments[-1][0] == label:
                    segments[-1][1].append(idx)
                else:
                    segments.append((label, [idx]))

            if not segments:
                continue

            # Determine the terminal node for this episode
            if has_outcome_info:
                outcome = episode_outcomes.get(ep_idx)
                if outcome is True:
                    terminal = SUCCESS_NODE_ID
                elif outcome is False:
                    terminal = FAILURE_NODE_ID
                else:
                    terminal = END_NODE_ID
            else:
                terminal = END_NODE_ID

            # Assign Q-values: for each segment, next_state is the next
            # segment's cluster (or the terminal for the last segment)
            for seg_i, (seg_label, seg_indices) in enumerate(segments):
                if seg_i < len(segments) - 1:
                    next_state = segments[seg_i + 1][0]
                else:
                    next_state = terminal

                v_next = node_values.get(next_state, 0.0)
                v_current = node_values.get(seg_label, 0.0)
                q = gamma * v_next
                adv = q - v_current

                for idx in seg_indices:
                    q_values[idx] = q
                    advantages[idx] = adv
                    next_cluster[idx] = next_state

        return q_values, advantages, next_cluster

    def enumerate_paths(
        self,
        max_paths: int = 50,
        min_probability: float = 0.0,
        min_edge_probability: float = 0.0,
    ) -> List[Tuple[List[int], float, List[Tuple[int, int, float]]]]:
        """Enumerate simple paths from START to any terminal node.

        Uses best-first search (max-heap by path probability) so we expand
        highest-probability paths first and stop once we have max_paths.
        Avoids exhaustive DFS which can be exponential in graph size.

        For each path, computes:
        - path probability = product of transition probabilities along the chain
        - loop-back edges: transitions in the full graph from a node in the
          path back to an *earlier* node also in that path

        Args:
            max_paths: Maximum number of paths to return (search stops when reached).
            min_probability: Discard paths with total probability below this.
            min_edge_probability: Ignore transitions with probability below
                this threshold during search.

        Returns:
            List of (path, probability, loops) sorted by descending probability.
        """
        terminals = self.terminal_node_ids
        results: List[Tuple[List[int], float]] = []

        # Best-first: (-prob, path, visited) so highest prob is popped first
        # Tie-break by path so heap doesn't compare sets
        counter = 0
        heap: List[Tuple[float, int, List[int], Set[int]]] = [
            (-1.0, 0, [START_NODE_ID], {START_NODE_ID}),
        ]

        while heap and len(results) < max_paths:
            neg_prob, _, path, visited = heapq.heappop(heap)
            prob = -neg_prob
            current = path[-1]

            if current in terminals:
                if prob >= min_probability:
                    results.append((list(path), prob))
                continue

            targets = self.transition_probs.get(current, {})
            for tgt, edge_prob in targets.items():
                if edge_prob < min_edge_probability:
                    continue
                if tgt in visited and tgt not in terminals:
                    continue
                new_prob = prob * edge_prob
                counter += 1
                heapq.heappush(
                    heap,
                    (-new_prob, counter, path + [tgt], visited | {tgt}),
                )

        # Detect loop-back edges for each path
        output: List[Tuple[List[int], float, List[Tuple[int, int, float]]]] = []
        for path, prob in results:
            loops: List[Tuple[int, int, float]] = []
            path_set = set(path)
            for i, node in enumerate(path):
                for tgt, edge_prob in self.transition_probs.get(node, {}).items():
                    if tgt in path_set and tgt not in terminals:
                        tgt_idx = path.index(tgt)
                        if tgt_idx < i:
                            loops.append((node, tgt, edge_prob))
            output.append((path, prob, loops))

        return output

    def enumerate_paths_to_terminal(
        self,
        terminal_id: int,
        max_paths: int = 50,
        min_probability: float = 0.0,
        min_edge_probability: float = 0.0,
    ) -> List[Tuple[List[int], float, List[Tuple[int, int, float]]]]:
        """Enumerate simple paths from START to a specific terminal (SUCCESS or FAILURE).

        Same as enumerate_paths but restricted to paths ending at terminal_id.
        Used for path-based curation: paths to FAILURE for filtering, paths to SUCCESS for selection.

        Args:
            terminal_id: SUCCESS_NODE_ID or FAILURE_NODE_ID.
            max_paths: Maximum number of paths to return.
            min_probability: Discard paths with total probability below this.
            min_edge_probability: Ignore transitions below this during DFS.

        Returns:
            List of (path, probability, loops) sorted by descending probability.
        """
        if terminal_id not in self.nodes:
            return []
        all_paths = self.enumerate_paths(
            max_paths=max_paths,
            min_probability=min_probability,
            min_edge_probability=min_edge_probability,
        )
        return [(path, prob, loops) for path, prob, loops in all_paths if path[-1] == terminal_id]


def _slice_bounds(meta: Dict) -> Tuple[int, int]:
    """Inclusive start, inclusive end for one rollout slice (fixed-length window from clustering)."""
    start = meta.get("window_start", meta.get("timestep", 0))
    end = meta.get("window_end")
    if end is None:
        end = start + meta.get("window_width", 1) - 1
    else:
        end = end - 1  # exclusive -> inclusive
    return (start, end)


def get_rollout_slices_for_paths(
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    level: str,
    paths: List[List[int]],
) -> List[Tuple[int, int, int]]:
    """Get (rollout_idx, start, end) for every original rollout slice on the given paths.

    Hierarchy: full rollout → segments (contiguous same-label, arbitrary length) → slices
    (fixed length from clustering) → samples (raw timesteps). Segments are composed of
    the rollout slices we obtained during clustering; each slice has a fixed-length
    window (one row in the clustering result). We return those original slices—each
    slice's own (start, end) from metadata—so attribution is run per slice, not over
    whole segments.

    A path is [START, n1, n2, ..., nk, TERMINAL]. An episode follows a path if its
    collapsed label sequence equals [n1, n2, ..., nk] and outcome matches TERMINAL.
    For each such episode, we include every slice (every clustering sample index) that
    belongs to any segment on the path. Each segment is a contiguous run of same-label
    slices; we emit one (rollout_idx, start, end) per slice using that slice's metadata.

    Args:
        cluster_labels: Array of cluster assignments, shape (N,).
        metadata: Per-slice metadata (rollout_idx/demo_idx, window_start, window_end, etc.).
        level: "rollout" or "demo".
        paths: List of paths from enumerate_paths_to_terminal (each path ends at SUCCESS or FAILURE).

    Returns:
        Deduplicated list of (rollout_episode_idx, start_inclusive, end_inclusive), one
        per distinct (rollout_idx, start, end) among the slices on the path.
    """
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"

    # Per episode: list of (sort_key, slice_index, label) for each clustering slice
    episodes: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    episode_outcomes: Dict[int, Optional[bool]] = {}
    for i, meta in enumerate(metadata):
        label = int(cluster_labels[i])
        if label == -1:
            continue
        ep_idx = meta[ep_key]
        sort_key = meta.get("timestep", meta.get("window_start", 0))
        episodes[ep_idx].append((sort_key, i, label))
        if "success" in meta and ep_idx not in episode_outcomes:
            episode_outcomes[ep_idx] = meta["success"]

    for ep_idx in episodes:
        episodes[ep_idx].sort(key=lambda x: x[0])

    has_outcome_info = any(v is not None for v in episode_outcomes.values())

    # Segments = contiguous runs of same label. Each segment = (label, list of slice indices).
    # Those slice indices are the original clustering slices (fixed-length windows).
    collapsed_segments: Dict[int, List[Tuple[int, List[int]]]] = {}
    for ep_idx, seq in episodes.items():
        if not seq:
            continue
        segments: List[Tuple[int, List[int]]] = []
        for _, idx, label in seq:
            if segments and segments[-1][0] == label:
                segments[-1][1].append(idx)
            else:
                segments.append((label, [idx]))
        collapsed_segments[ep_idx] = segments

    # Path body = path without START and terminal (e.g. [n1, n2, n3])
    path_bodies = []
    for path in paths:
        if len(path) < 3:
            continue
        path_bodies.append((path[1:-1], path[-1]))

    # One (rollout_idx, start, end) per original slice on the path (each slice = one clustering row)
    result_set: Set[Tuple[int, int, int]] = set()
    for (body, terminal_id) in path_bodies:
        if has_outcome_info:
            if terminal_id == SUCCESS_NODE_ID:
                outcome_ok = True
            elif terminal_id == FAILURE_NODE_ID:
                outcome_ok = False
            else:
                outcome_ok = None
        else:
            outcome_ok = None

        for ep_idx, segments in collapsed_segments.items():
            if has_outcome_info and outcome_ok is not None:
                if episode_outcomes.get(ep_idx) != outcome_ok:
                    continue
            seg_labels = [s[0] for s in segments]
            if seg_labels != body:
                continue
            # Each segment is composed of slices (clustering samples). Emit each slice's bounds.
            for (_label, slice_indices) in segments:
                for slice_idx in slice_indices:
                    meta = metadata[slice_idx]
                    if meta.get(ep_key) is None:
                        continue
                    start, end = _slice_bounds(meta)
                    result_set.add((ep_idx, start, end))

    return sorted(result_set, key=lambda x: (x[0], x[1]))


def get_episodes_with_transition(
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    level: str,
    source_id: int,
    target_id: int,
) -> List[int]:
    """Return episode indices where the transition (source_id -> target_id) occurs.

    Uses the same grouping and collapse logic as BehaviorGraph.from_cluster_assignments:
    consecutive identical cluster assignments within an episode are collapsed, and
    START/terminal semantics apply (first behavior = transition from START, last
    behavior transitions to SUCCESS/FAILURE/END based on outcome).

    Args:
        cluster_labels: Array of cluster assignments, shape (N,).
        metadata: Per-sample metadata dicts with episode index and timestep info.
        level: "rollout" or "demo" — determines which metadata key identifies episodes.
        source_id: Source node ID (cluster or START_NODE_ID).
        target_id: Target node ID (cluster or END_NODE_ID, SUCCESS_NODE_ID, FAILURE_NODE_ID).

    Returns:
        Sorted list of episode indices (rollout_idx or demo_idx) where the transition appears.
    """
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"

    # Group timesteps by episode and collect per-episode outcomes
    episodes: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    episode_outcomes: Dict[int, Optional[bool]] = {}
    for i, meta in enumerate(metadata):
        label = int(cluster_labels[i])
        if label == -1:
            continue
        ep_idx = meta[ep_key]
        sort_key = meta.get("timestep", meta.get("window_start", 0))
        episodes[ep_idx].append((sort_key, label))
        if "success" in meta and ep_idx not in episode_outcomes:
            episode_outcomes[ep_idx] = meta["success"]

    for ep_idx in episodes:
        episodes[ep_idx].sort(key=lambda x: x[0])

    has_outcome_info = any(v is not None for v in episode_outcomes.values())

    # Collapse consecutive identical assignments within each episode
    collapsed: Dict[int, List[int]] = {}
    for ep_idx, seq in episodes.items():
        if not seq:
            continue
        result = [seq[0][1]]
        for _, label in seq[1:]:
            if label != result[-1]:
                result.append(label)
        collapsed[ep_idx] = result

    result_episodes: List[int] = []

    for ep_idx, seq in collapsed.items():
        if not seq:
            continue

        # Determine effective terminal for this episode
        if has_outcome_info:
            outcome = episode_outcomes.get(ep_idx)
            if outcome is True:
                terminal = SUCCESS_NODE_ID
            elif outcome is False:
                terminal = FAILURE_NODE_ID
            else:
                terminal = END_NODE_ID
        else:
            terminal = END_NODE_ID

        # START -> target: first behavior equals target_id
        if source_id == START_NODE_ID:
            if seq[0] == target_id:
                result_episodes.append(ep_idx)
            continue

        # source -> terminal: last behavior equals source_id and target is this episode's terminal
        if target_id in TERMINAL_NODE_IDS:
            if seq[-1] == source_id and target_id == terminal:
                result_episodes.append(ep_idx)
            continue

        # source -> target (both clusters): consecutive (source_id, target_id) in seq
        for j in range(len(seq) - 1):
            if seq[j] == source_id and seq[j + 1] == target_id:
                result_episodes.append(ep_idx)
                break

    return sorted(result_episodes)
