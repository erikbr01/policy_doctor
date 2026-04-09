"""Behavior graph: transitions between behavioral clusters. Ported from influence_visualizer."""

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.stats import chi2_contingency

START_NODE_ID = -2
END_NODE_ID = -3
SUCCESS_NODE_ID = -4
FAILURE_NODE_ID = -5

TERMINAL_NODE_IDS = frozenset({END_NODE_ID, SUCCESS_NODE_ID, FAILURE_NODE_ID})

# Graph nodes that must not appear in per-slice cluster label arrays.
_SPECIAL_GRAPH_NODE_IDS = frozenset({START_NODE_ID}) | TERMINAL_NODE_IDS


def _follow_merge_chain(cluster_id: int, merge_map: Dict[int, int]) -> int:
    """Resolve ``cluster_id`` through a merge map (path compression, cycle-safe)."""
    if cluster_id == -1:
        return -1
    seen: Set[int] = set()
    cur = cluster_id
    while cur in merge_map:
        if cur in seen:
            return cur
        seen.add(cur)
        cur = merge_map[cur]
    return cur


def _apply_merge_map_to_labels(
    labels: np.ndarray, merge_map: Dict[int, int]
) -> np.ndarray:
    """Rewrite ``labels`` so each id is replaced by its merge-chain root."""
    if not merge_map:
        return labels
    uniq = np.unique(labels)
    resolved = {int(u): _follow_merge_chain(int(u), merge_map) for u in uniq}
    return np.vectorize(lambda x: resolved[int(x)], otypes=[np.int64])(labels).astype(
        np.int64, copy=False
    )


@dataclass
class BehaviorNode:
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
        ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
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
        collapsed: Dict[int, List[int]] = {}
        for ep_idx, seq in episodes.items():
            if not seq:
                continue
            result = [seq[0][1]]
            for _, label in seq[1:]:
                if label != result[-1]:
                    result.append(label)
            collapsed[ep_idx] = result
        trans_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for ep_idx, seq in collapsed.items():
            if not seq:
                continue
            trans_counts[START_NODE_ID][seq[0]] += 1
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
        node_timestep_counts: Dict[int, int] = defaultdict(int)
        node_episode_sets: Dict[int, Set[int]] = defaultdict(set)
        for ep_idx, seq in episodes.items():
            for _, label in seq:
                node_timestep_counts[label] += 1
                node_episode_sets[label].add(ep_idx)
        trans_probs: Dict[int, Dict[int, float]] = {}
        for src, targets in trans_counts.items():
            total = sum(targets.values())
            trans_probs[src] = {tgt: count / total for tgt, count in targets.items()}
        all_episode_ids = sorted(collapsed.keys())
        success_ids = [e for e in all_episode_ids if episode_outcomes.get(e) is True]
        failure_ids = [e for e in all_episode_ids if episode_outcomes.get(e) is False]
        unknown_ids = [e for e in all_episode_ids if episode_outcomes.get(e) is None]
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

    @property
    def terminal_node_ids(self) -> Set[int]:
        return {nid for nid in self.nodes if nid in TERMINAL_NODE_IDS}

    @property
    def cluster_nodes(self) -> Dict[int, BehaviorNode]:
        """Cluster nodes only (excluding START and terminal nodes)."""
        return {k: v for k, v in self.nodes.items() if not v.is_special}

    def get_outgoing_transitions(self, node_id: int) -> List[Tuple[int, int, float]]:
        """Return list of (target_id, count, probability) for edges from node_id."""
        targets = self.transition_probs.get(node_id, {})
        counts = self.transition_counts.get(node_id, {})
        return [
            (tgt_id, counts.get(tgt_id, 0), prob)
            for tgt_id, prob in targets.items()
        ]

    def get_incoming_transitions(self, node_id: int) -> List[Tuple[int, int, float]]:
        """Return list of (source_id, count, probability) for edges into node_id."""
        result: List[Tuple[int, int, float]] = []
        for src_id, targets in self.transition_probs.items():
            if node_id in targets:
                prob = targets[node_id]
                cnt = self.transition_counts.get(src_id, {}).get(node_id, 0)
                result.append((src_id, cnt, prob))
        return result

    def compute_values(
        self,
        gamma: float = 0.99,
        reward_success: float = 1.0,
        reward_failure: float = -1.0,
        reward_end: float = 0.0,
    ) -> Dict[int, float]:
        terminal_rewards = {
            SUCCESS_NODE_ID: reward_success,
            FAILURE_NODE_ID: reward_failure,
            END_NODE_ID: reward_end,
        }
        terminal_ids = {nid for nid in self.nodes if nid in TERMINAL_NODE_IDS}
        nonterminal_ids = sorted(nid for nid in self.nodes if nid not in terminal_ids)
        nt_index = {nid: i for i, nid in enumerate(nonterminal_ids)}
        n = len(nonterminal_ids)
        if n == 0:
            return {nid: terminal_rewards.get(nid, 0.0) for nid in self.nodes}
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
        A = np.eye(n) - gamma * P_nn
        V_nt = np.linalg.solve(A, gamma * b)
        values: Dict[int, float] = {}
        for nid in nonterminal_ids:
            values[nid] = float(V_nt[nt_index[nid]])
        for nid in terminal_ids:
            values[nid] = terminal_rewards.get(nid, 0.0)
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
        ep_key = "rollout_idx" if self.level == "rollout" else "demo_idx"
        n = len(cluster_labels)
        episodes: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        episode_outcomes: Dict[int, Optional[bool]] = {}
        for i, meta in enumerate(metadata):
            label = int(cluster_labels[i])
            ep_idx = meta[ep_key]
            sort_key = meta.get("timestep", meta.get("window_start", 0))
            episodes[ep_idx].append((sort_key, i, label))
            if label != -1 and "success" in meta and ep_idx not in episode_outcomes:
                episode_outcomes[ep_idx] = meta["success"]
        has_outcome_info = any(v is not None for v in episode_outcomes.values())
        q_values = np.zeros(n)
        advantages = np.zeros(n)
        next_cluster = np.full(n, -1, dtype=int)
        for ep_idx, slices in episodes.items():
            slices.sort(key=lambda x: x[0])
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
            if has_outcome_info:
                outcome = episode_outcomes.get(ep_idx)
                terminal = (
                    SUCCESS_NODE_ID
                    if outcome is True
                    else (FAILURE_NODE_ID if outcome is False else END_NODE_ID)
                )
            else:
                terminal = END_NODE_ID
            for seg_i, (seg_label, seg_indices) in enumerate(segments):
                next_state = (
                    segments[seg_i + 1][0]
                    if seg_i < len(segments) - 1
                    else terminal
                )
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
        terminals = self.terminal_node_ids
        results: List[Tuple[List[int], float]] = []
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
            for tgt, edge_prob in self.transition_probs.get(current, {}).items():
                if edge_prob < min_edge_probability or (
                    tgt in visited and tgt not in terminals
                ):
                    continue
                counter += 1
                heapq.heappush(
                    heap,
                    (-prob * edge_prob, counter, path + [tgt], visited | {tgt}),
                )
        output: List[Tuple[List[int], float, List[Tuple[int, int, float]]]] = []
        for path, prob in results:
            path_set = set(path)
            loops: List[Tuple[int, int, float]] = []
            for i, node in enumerate(path):
                for tgt, edge_prob in self.transition_probs.get(node, {}).items():
                    if tgt in path_set and tgt not in TERMINAL_NODE_IDS:
                        if path.index(tgt) < i:
                            loops.append((node, tgt, edge_prob))
            output.append((path, prob, loops))
        return output

    def simplify_by_degree_one_pruning(
        self,
        cluster_labels: np.ndarray,
        metadata: List[Dict],
    ) -> Tuple["BehaviorGraph", np.ndarray, int, int]:
        """Prune degree-1 cluster nodes to fixed point; see :func:`degree_one_prune_to_fixed_point`."""
        return degree_one_prune_to_fixed_point(self, cluster_labels, metadata)

    def enumerate_paths_to_terminal(
        self,
        terminal_id: int,
        max_paths: int = 50,
        min_probability: float = 0.0,
        min_edge_probability: float = 0.0,
    ) -> List[Tuple[List[int], float, List[Tuple[int, int, float]]]]:
        if terminal_id not in self.nodes:
            return []
        all_paths = self.enumerate_paths(
            max_paths=max_paths,
            min_probability=min_probability,
            min_edge_probability=min_edge_probability,
        )
        return [
            (path, prob, loops)
            for path, prob, loops in all_paths
            if path[-1] == terminal_id
        ]


def degree_one_prune_to_fixed_point(
    graph: BehaviorGraph,
    cluster_labels: np.ndarray,
    metadata: List[Dict],
) -> Tuple[BehaviorGraph, np.ndarray, int, int]:
    """Prune non-special nodes with a single in- or out-neighbor until none remain.

    Same rules as :meth:`BehaviorGraph.simplify_by_degree_one_pruning`. Exposed
    as a module function so callers (e.g. Streamlit) always use the
    implementation from the loaded ``behavior_graph`` module, avoiding stale
    bound methods on ``BehaviorGraph`` from older installs.

    Returns:
        ``(simplified_graph, new_cluster_labels, num_rounds, num_nodes_merged)``.
    """
    labels = np.asarray(cluster_labels, dtype=np.int64).copy()
    level = graph.level
    num_rounds = 0
    num_nodes_merged = 0
    max_rounds = max(500, len(graph.nodes) * 20)
    while num_rounds < max_rounds:
        g = BehaviorGraph.from_cluster_assignments(
            labels, metadata, level=level,
        )
        merge_map: Dict[int, int] = {}
        for nid in sorted(g.nodes.keys()):
            if nid in _SPECIAL_GRAPH_NODE_IDS:
                continue
            inc = g.get_incoming_transitions(nid)
            out = g.get_outgoing_transitions(nid)
            inc_sources = [s for s, c, _ in inc if c > 0]
            out_tgts = [t for t, c, _ in out if c > 0]
            repl: Optional[int] = None
            if len(out_tgts) == 1:
                repl = out_tgts[0]
            elif len(inc_sources) == 1:
                repl = inc_sources[0]
            if repl is None or repl in _SPECIAL_GRAPH_NODE_IDS:
                continue
            merge_map[nid] = repl
        if not merge_map:
            return g, labels, num_rounds, num_nodes_merged
        num_rounds += 1
        num_nodes_merged += len(merge_map)
        new_labels = _apply_merge_map_to_labels(labels, merge_map)
        if np.array_equal(new_labels, labels):
            return g, labels, num_rounds, num_nodes_merged
        labels = new_labels
    return (
        BehaviorGraph.from_cluster_assignments(labels, metadata, level=level),
        labels,
        num_rounds,
        num_nodes_merged,
    )


def _slice_bounds(meta: Dict) -> Tuple[int, int]:
    start = meta.get("window_start", meta.get("timestep", 0))
    end = meta.get("window_end")
    if end is None:
        end = start + meta.get("window_width", 1) - 1
    else:
        end = end - 1
    return (start, end)


def get_rollout_slices_for_paths(
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    level: str,
    paths: List[List[int]],
) -> List[Tuple[int, int, int]]:
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
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
    collapsed_segments: Dict[int, List[Tuple[int, List[int]]]] = {}
    for ep_idx, seq in episodes.items():
        if not seq:
            continue
        segments_list: List[Tuple[int, List[int]]] = []
        for _, idx, label in seq:
            if segments_list and segments_list[-1][0] == label:
                segments_list[-1][1].append(idx)
            else:
                segments_list.append((label, [idx]))
        collapsed_segments[ep_idx] = segments_list
    path_bodies = []
    for path in paths:
        if len(path) < 3:
            continue
        path_bodies.append((path[1:-1], path[-1]))
    result_set: Set[Tuple[int, int, int]] = set()
    for (body, terminal_id) in path_bodies:
        outcome_ok = None
        if has_outcome_info:
            outcome_ok = (
                True
                if terminal_id == SUCCESS_NODE_ID
                else (False if terminal_id == FAILURE_NODE_ID else None)
            )
        for ep_idx, segments in collapsed_segments.items():
            if has_outcome_info and outcome_ok is not None:
                if episode_outcomes.get(ep_idx) != outcome_ok:
                    continue
            if [s[0] for s in segments] != body:
                continue
            for _label, slice_indices in segments:
                for slice_idx in slice_indices:
                    meta = metadata[slice_idx]
                    if meta.get(ep_key) is None:
                        continue
                    start, end = _slice_bounds(meta)
                    result_set.add((ep_idx, start, end))
    return sorted(result_set, key=lambda x: (x[0], x[1]))


@dataclass
class MarkovTestResult:
    """Result of a per-state Markov property test."""

    state: int
    testable: bool
    chi2: Optional[float] = None
    p_value: Optional[float] = None
    dof: Optional[int] = None
    markov_holds: Optional[bool] = None
    contingency_table: Optional[np.ndarray] = None
    previous_states: Optional[List[int]] = None
    next_states: Optional[List[int]] = None
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers: sequence extraction and second-order counts
# ---------------------------------------------------------------------------

def _extract_collapsed_sequences(
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    level: str,
) -> Tuple[Dict[int, List[int]], Dict[int, Optional[bool]]]:
    """Extract run-length-collapsed cluster sequences per episode.

    Returns (collapsed_sequences, episode_outcomes).
    """
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
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

    collapsed: Dict[int, List[int]] = {}
    for ep_idx, seq in episodes.items():
        if not seq:
            continue
        result = [seq[0][1]]
        for _, label in seq[1:]:
            if label != result[-1]:
                result.append(label)
        collapsed[ep_idx] = result
    return collapsed, episode_outcomes


SecondOrderCounts = Dict[int, Dict[int, Dict[int, int]]]


def _compute_second_order_counts(
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    level: str,
    exclude_terminals: bool,
) -> Tuple[SecondOrderCounts, Set[int]]:
    """Compute ``counts[current][prev][next]`` from collapsed episode sequences.

    Returns ``(second_order_counts, all_cluster_ids)``.
    """
    collapsed, episode_outcomes = _extract_collapsed_sequences(
        cluster_labels, metadata, level
    )
    has_outcome_info = any(v is not None for v in episode_outcomes.values())
    skip_nodes = (
        frozenset({START_NODE_ID}) | TERMINAL_NODE_IDS if exclude_terminals else frozenset()
    )

    second_order: SecondOrderCounts = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for ep_idx, seq in collapsed.items():
        if has_outcome_info:
            outcome = episode_outcomes.get(ep_idx)
            terminal = (
                SUCCESS_NODE_ID
                if outcome is True
                else (FAILURE_NODE_ID if outcome is False else END_NODE_ID)
            )
        else:
            terminal = END_NODE_ID

        full_seq = [START_NODE_ID] + seq + [terminal]
        for i in range(1, len(full_seq) - 1):
            prev_state = full_seq[i - 1]
            current_state = full_seq[i]
            next_state = full_seq[i + 1]
            if prev_state in skip_nodes or next_state in skip_nodes:
                continue
            second_order[current_state][prev_state][next_state] += 1

    all_cluster_ids = set(int(c) for c in cluster_labels if c != -1)
    return dict(second_order), all_cluster_ids


def _merge_second_order_counts(*count_dicts: SecondOrderCounts) -> SecondOrderCounts:
    """Element-wise sum of multiple second-order count dicts."""
    merged: SecondOrderCounts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for counts in count_dicts:
        for cur, prev_to_next in counts.items():
            for prev, next_counts in prev_to_next.items():
                for nxt, cnt in next_counts.items():
                    merged[cur][prev][nxt] += cnt
    return dict(merged)


# ---------------------------------------------------------------------------
# Statistical tests on contingency tables
# ---------------------------------------------------------------------------

def _chi2_stat(table: np.ndarray) -> float:
    """Chi-squared statistic without scipy overhead (for permutation loops)."""
    row_sums = table.sum(axis=1, keepdims=True).astype(float)
    col_sums = table.sum(axis=0, keepdims=True).astype(float)
    total = float(table.sum())
    if total == 0:
        return 0.0
    expected = row_sums * col_sums / total
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0, (table - expected) ** 2 / expected, 0.0)
    return float(terms.sum())


def _expand_table(table: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Expand a contingency table into observation-level (row, col) arrays."""
    rows_list: List[int] = []
    cols_list: List[int] = []
    n_rows, n_cols = table.shape
    for i in range(n_rows):
        for j in range(n_cols):
            c = int(table[i, j])
            rows_list.extend([i] * c)
            cols_list.extend([j] * c)
    return np.array(rows_list, dtype=int), np.array(cols_list, dtype=int)


def _permutation_independence_test(
    table: np.ndarray,
    n_permutations: int = 10000,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[float, float]:
    """Monte Carlo permutation test of independence for an r x c table.

    Permutes row labels to break any row-column association while preserving
    marginals, then compares the observed chi-squared statistic against the
    permutation distribution.

    Returns ``(observed_chi2, p_value)``.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    observed = _chi2_stat(table)
    rows, cols = _expand_table(table)
    n_rows, n_cols = table.shape
    count_ge = 0
    for _ in range(n_permutations):
        rng.shuffle(rows)
        perm_table = np.zeros((n_rows, n_cols))
        np.add.at(perm_table, (rows, cols), 1.0)
        if _chi2_stat(perm_table) >= observed - 1e-12:
            count_ge += 1
    p_value = (count_ge + 1) / (n_permutations + 1)
    return observed, p_value


def _permutation_modal_test(
    table: np.ndarray,
    n_permutations: int = 10000,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[float, float]:
    """Permutation test for whether the modal successor differs by predecessor.

    A weaker alternative to full independence: tests whether the most-likely
    next state is the same regardless of the predecessor.

    Test statistic = number of predecessor rows whose column-mode differs
    from the overall column-mode.

    Returns ``(observed_statistic, p_value)``.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    overall_mode = int(table.sum(axis=0).argmax())
    row_modes = table.argmax(axis=1)
    observed = float(np.sum(row_modes != overall_mode))

    rows, cols = _expand_table(table)
    n_rows, n_cols = table.shape
    count_ge = 0
    for _ in range(n_permutations):
        rng.shuffle(rows)
        perm_table = np.zeros((n_rows, n_cols))
        np.add.at(perm_table, (rows, cols), 1.0)
        perm_overall = int(perm_table.sum(axis=0).argmax())
        perm_stat = float(np.sum(perm_table.argmax(axis=1) != perm_overall))
        if perm_stat >= observed - 1e-12:
            count_ge += 1
    p_value = (count_ge + 1) / (n_permutations + 1)
    return observed, p_value


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def _build_per_state_results(
    second_order: SecondOrderCounts,
    all_cluster_ids: Set[int],
    exclude_terminals: bool,
    method: str,
    significance_level: float,
    n_permutations: int,
    random_state: Optional[int],
) -> Dict[int, MarkovTestResult]:
    """Run the chosen test on each state's contingency table."""
    rng = np.random.RandomState(random_state) if random_state is not None else None
    min_total = 5 if method == "chi2" else 3

    per_state: Dict[int, MarkovTestResult] = {}

    for cid in all_cluster_ids:
        if cid not in second_order:
            per_state[cid] = MarkovTestResult(
                state=cid,
                testable=False,
                reason="no_interior_transitions" if exclude_terminals else "no_transitions",
            )

    for current_state, prev_to_next in second_order.items():
        if len(prev_to_next) < 2:
            per_state[current_state] = MarkovTestResult(
                state=current_state, testable=False, reason="only_one_predecessor",
            )
            continue

        all_prev = sorted(prev_to_next.keys())
        all_next: Set[int] = set()
        for nc in prev_to_next.values():
            all_next.update(nc.keys())
        all_next_sorted = sorted(all_next)

        if len(all_next_sorted) < 2:
            per_state[current_state] = MarkovTestResult(
                state=current_state, testable=False, reason="only_one_successor",
            )
            continue

        table = np.zeros((len(all_prev), len(all_next_sorted)), dtype=int)
        for i, prev in enumerate(all_prev):
            for j, nxt in enumerate(all_next_sorted):
                table[i, j] = prev_to_next[prev].get(nxt, 0)

        if table.sum() < min_total:
            per_state[current_state] = MarkovTestResult(
                state=current_state, testable=False, reason="insufficient_data",
            )
            continue

        stat: Optional[float] = None
        p_value: Optional[float] = None
        dof: Optional[int] = None
        try:
            if method == "chi2":
                stat, p_value, dof, _ = chi2_contingency(table)
            elif method == "exact":
                stat, p_value = _permutation_independence_test(table, n_permutations, rng)
            elif method == "modal":
                stat, p_value = _permutation_modal_test(table, n_permutations, rng)
            else:
                raise ValueError(f"Unknown method: {method!r}")
        except ValueError:
            per_state[current_state] = MarkovTestResult(
                state=current_state, testable=False, reason="test_failed",
            )
            continue

        per_state[current_state] = MarkovTestResult(
            state=current_state,
            testable=True,
            chi2=stat,
            p_value=p_value,
            dof=dof,
            markov_holds=p_value >= significance_level,
            contingency_table=table,
            previous_states=all_prev,
            next_states=all_next_sorted,
        )

    return per_state


def _summarize(per_state: Dict[int, MarkovTestResult], significance_level: float) -> Dict:
    testable = {k: v for k, v in per_state.items() if v.testable}
    overall = all(v.markov_holds for v in testable.values()) if testable else None
    return {
        "markov_holds": overall,
        "significance_level": significance_level,
        "num_states_tested": len(testable),
        "num_states_untestable": len(per_state) - len(testable),
        "per_state": per_state,
    }


def test_markov_property(
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    level: str = "rollout",
    significance_level: float = 0.05,
    exclude_terminals: bool = False,
    method: str = "chi2",
    n_permutations: int = 10000,
    random_state: Optional[int] = None,
) -> Dict:
    """Test whether transitions in the behavior graph satisfy the Markov property.

    For each behavior state *s*, builds a contingency table of
    (previous_state, next_state) counts from episodes passing through *s*,
    then tests whether the next-state distribution is independent of the
    predecessor.

    Args:
        cluster_labels: Per-timestep cluster assignments.
        metadata: Per-timestep metadata dicts (must contain episode index and
            timestep keys).
        level: ``"rollout"`` or ``"demo"`` — selects the episode key.
        significance_level: Threshold for rejecting the null (Markov) hypothesis.
        exclude_terminals: If True, exclude START and terminal nodes from
            predecessor/successor roles to isolate behavioral dynamics from
            episode-boundary effects.
        method: Statistical test to use:

            - ``"chi2"`` — asymptotic chi-squared test (default; needs ≥ 5
              observations per table).
            - ``"exact"`` — Monte Carlo permutation test using the chi-squared
              statistic.  Valid for small cell counts where the asymptotic
              approximation breaks down.
            - ``"modal"`` — permutation test on whether the most-likely
              successor changes with the predecessor.  Tests a weaker property
              than full independence, requiring less data.
        n_permutations: Number of permutation resamples (``"exact"`` and
            ``"modal"`` methods only).
        random_state: Seed for the permutation RNG (reproducibility).

    Returns:
        Dict with keys ``markov_holds``, ``significance_level``,
        ``num_states_tested``, ``num_states_untestable``, ``per_state``.
    """
    second_order, all_ids = _compute_second_order_counts(
        cluster_labels, metadata, level, exclude_terminals,
    )
    per_state = _build_per_state_results(
        second_order, all_ids, exclude_terminals,
        method, significance_level, n_permutations, random_state,
    )
    return _summarize(per_state, significance_level)


def markov_test_result_to_jsonable(result: Dict) -> Dict:
    """Make :func:`test_markov_property` / :func:`test_markov_property_pooled` output JSON-safe."""

    def _json_bool(x: Optional[bool]) -> Optional[bool]:
        if x is None:
            return None
        return bool(x)

    def _state_dict(r: MarkovTestResult) -> Dict:
        ct = r.contingency_table
        return {
            "state": int(r.state),
            "testable": bool(r.testable),
            "chi2": float(r.chi2) if r.chi2 is not None else None,
            "p_value": float(r.p_value) if r.p_value is not None else None,
            "dof": int(r.dof) if r.dof is not None else None,
            "markov_holds": _json_bool(r.markov_holds),
            "contingency_table": ct.tolist() if ct is not None else None,
            "previous_states": list(r.previous_states) if r.previous_states else None,
            "next_states": list(r.next_states) if r.next_states else None,
            "reason": r.reason,
        }

    per_state = result.get("per_state") or {}
    serial_per = {
        str(k): _state_dict(v) if isinstance(v, MarkovTestResult) else v
        for k, v in per_state.items()
    }
    out = {
        "markov_holds": _json_bool(result.get("markov_holds")),
        "significance_level": float(result["significance_level"])
        if result.get("significance_level") is not None
        else None,
        "num_states_tested": int(result["num_states_tested"])
        if result.get("num_states_tested") is not None
        else None,
        "num_states_untestable": int(result["num_states_untestable"])
        if result.get("num_states_untestable") is not None
        else None,
        "per_state": serial_per,
    }
    return out


def test_markov_property_pooled(
    datasets: List[Tuple[np.ndarray, List[Dict]]],
    level: str = "rollout",
    significance_level: float = 0.05,
    exclude_terminals: bool = False,
    method: str = "chi2",
    n_permutations: int = 10000,
    random_state: Optional[int] = None,
) -> Dict:
    """Pool second-order transition counts from multiple datasets, then test.

    Use when the **same clustering** (same cluster IDs = same behaviors) is
    applied to multiple independent sets of episodes, e.g. multiple evaluation
    batches or train/test splits under one clustering run.  Pooling increases
    sample size per state and improves statistical power.

    Args:
        datasets: List of ``(cluster_labels, metadata)`` tuples sharing a
            common set of cluster IDs.
        level, significance_level, exclude_terminals, method, n_permutations,
        random_state: Same as :func:`test_markov_property`.
    """
    all_counts: List[SecondOrderCounts] = []
    all_ids: Set[int] = set()
    for labels, meta in datasets:
        counts, ids = _compute_second_order_counts(labels, meta, level, exclude_terminals)
        all_counts.append(counts)
        all_ids |= ids

    merged = _merge_second_order_counts(*all_counts)
    per_state = _build_per_state_results(
        merged, all_ids, exclude_terminals,
        method, significance_level, n_permutations, random_state,
    )
    return _summarize(per_state, significance_level)
