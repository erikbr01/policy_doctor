"""Metrics + uncertainty-aware distribution tests for graph simplification.

All KL / NLL values are in **bits** (log2). Empirical transition distributions
are smoothed with a symmetric Dirichlet(alpha) prior so zero-count successors
don't blow up KL or NLL.

Key functions:
    kl_bits(p, q)              - KL divergence (asymmetric).
    js_bits(p, q)              - Jensen-Shannon (symmetric, ≤ 1 bit).
    hoeffding_compatible(...)  - Alergia compatibility test for merging.
    chi2_pvalue(...)           - Pearson chi-squared test of homogeneity.
    dirichlet_kl_bits(...)     - KL between two Dirichlet posterior MEANS.
    markov_violation_bits(...) - Conditional MI I(prev; next | s), per-node and total.
    trajectory_nll_bits(...)   - -log2 likelihood of observed transitions under graph.
    compute_metrics(...)       - One-stop dataclass with all scalar metrics.
    bootstrap_metric(...)      - Episode-level bootstrap CI for any metric.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from policy_doctor.behaviors.behavior_graph import (
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
    TERMINAL_NODE_IDS,
    BehaviorGraph,
    _compute_second_order_counts,
    _extract_collapsed_sequences,
)

_LOG2 = float(np.log(2.0))
DEFAULT_ALPHA = 1.0  # Laplace prior


# ---------------------------------------------------------------------------
# Smoothing + divergences
# ---------------------------------------------------------------------------

def smoothed_probs(
    counts: Dict[int, int],
    support: Sequence[int],
    alpha: float = DEFAULT_ALPHA,
) -> Dict[int, float]:
    """Posterior mean of Dirichlet(alpha + counts) over the given support.

    Returns p(s) = (counts[s] + alpha) / (N + alpha * |support|) for s in support.
    """
    n_sym = len(support)
    if n_sym == 0:
        return {}
    total = sum(counts.get(s, 0) for s in support)
    denom = total + alpha * n_sym
    return {s: (counts.get(s, 0) + alpha) / denom for s in support}


def kl_bits(p: Dict[int, float], q: Dict[int, float]) -> float:
    """KL(P || Q) in bits.

    Returns +inf if Q is zero on any symbol where P is positive.
    """
    out = 0.0
    for k, pk in p.items():
        if pk <= 0.0:
            continue
        qk = q.get(k, 0.0)
        if qk <= 0.0:
            return float("inf")
        out += pk * (np.log(pk) - np.log(qk))
    return float(out) / _LOG2


def js_bits(p: Dict[int, float], q: Dict[int, float]) -> float:
    """Jensen-Shannon divergence in bits. Symmetric, bounded by 1."""
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}
    pp = {k: p.get(k, 0.0) for k in keys}
    qq = {k: q.get(k, 0.0) for k in keys}
    return 0.5 * kl_bits(pp, m) + 0.5 * kl_bits(qq, m)


def dirichlet_kl_bits(
    counts1: Dict[int, int],
    counts2: Dict[int, int],
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """KL between two smoothed Dirichlet-posterior means.

    Both rows are smoothed to the same union support so the result is finite.
    """
    support = sorted(set(counts1) | set(counts2))
    if not support:
        return 0.0
    p = smoothed_probs(counts1, support, alpha)
    q = smoothed_probs(counts2, support, alpha)
    return kl_bits(p, q)


def js_distance_bits(
    counts1: Dict[int, int],
    counts2: Dict[int, int],
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """Symmetric JS distance between two smoothed posterior means."""
    support = sorted(set(counts1) | set(counts2))
    p = smoothed_probs(counts1, support, alpha)
    q = smoothed_probs(counts2, support, alpha)
    return js_bits(p, q)


# ---------------------------------------------------------------------------
# Uncertainty-aware compatibility tests
# ---------------------------------------------------------------------------

def hoeffding_compatible(
    counts1: Dict[int, int],
    counts2: Dict[int, int],
    delta: float,
) -> bool:
    """Alergia compatibility test (Carrasco-Oncina 1994).

    Two empirical transition distributions are *compatible* at confidence δ if
    for every symbol σ:
        |p1(σ) - p2(σ)| ≤ sqrt(½ · ln(2/δ)) · (1/√n1 + 1/√n2)

    Smaller δ → larger threshold → MORE permissive merging.
    Larger  δ → smaller threshold → LESS permissive merging.

    A 0-count state on both sides is compatible by definition.
    """
    n1 = sum(counts1.values())
    n2 = sum(counts2.values())
    if n1 == 0 and n2 == 0:
        return True
    if n1 == 0 or n2 == 0:
        return False
    delta = max(min(delta, 0.999999), 1e-12)
    bound = float(np.sqrt(0.5 * np.log(2.0 / delta))) * (
        1.0 / np.sqrt(n1) + 1.0 / np.sqrt(n2)
    )
    all_syms = set(counts1) | set(counts2)
    for s in all_syms:
        p1 = counts1.get(s, 0) / n1
        p2 = counts2.get(s, 0) / n2
        if abs(p1 - p2) > bound:
            return False
    return True


def chi2_pvalue(counts1: Dict[int, int], counts2: Dict[int, int]) -> float:
    """Pearson chi-squared p-value for equality of two empirical distributions.

    Small p → reject equality. Large p → no evidence of difference (compatible).
    Returns 1.0 if the test can't be run.
    """
    from scipy.stats import chi2_contingency  # local import to keep cold path light

    keys = sorted(set(counts1) | set(counts2))
    if not keys:
        return 1.0
    table = np.array(
        [[counts1.get(k, 0) for k in keys], [counts2.get(k, 0) for k in keys]],
        dtype=float,
    )
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2 or table.sum() == 0:
        return 1.0
    try:
        _, p, _, _ = chi2_contingency(table)
        return float(p)
    except ValueError:
        return 1.0


def posterior_overlap_prob(
    counts1: Dict[int, int],
    counts2: Dict[int, int],
    alpha: float = DEFAULT_ALPHA,
    n_samples: int = 1000,
    tol_bits: float = 0.05,
    rng_seed: int = 0,
) -> float:
    """Monte-Carlo estimate of P(JS(p1, p2) ≤ tol) under independent Dirichlet posteriors.

    Useful as a probabilistic "are these distributions essentially the same?"
    score. Returns a value in [0, 1].
    """
    rng = np.random.RandomState(rng_seed)
    support = sorted(set(counts1) | set(counts2))
    if not support:
        return 1.0
    a1 = np.array([counts1.get(k, 0) + alpha for k in support])
    a2 = np.array([counts2.get(k, 0) + alpha for k in support])
    s1 = rng.dirichlet(a1, size=n_samples)
    s2 = rng.dirichlet(a2, size=n_samples)
    m = 0.5 * (s1 + s2)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl1 = np.where(s1 > 0, s1 * (np.log(s1) - np.log(m)), 0.0).sum(axis=1)
        kl2 = np.where(s2 > 0, s2 * (np.log(s2) - np.log(m)), 0.0).sum(axis=1)
    js_nats = 0.5 * (kl1 + kl2)
    js = js_nats / _LOG2
    return float(np.mean(js <= tol_bits))


# ---------------------------------------------------------------------------
# Graph-level metrics
# ---------------------------------------------------------------------------

def graph_node_count(graph: BehaviorGraph, include_special: bool = False) -> int:
    return len(graph.nodes) if include_special else len(graph.cluster_nodes)


def graph_edge_count(graph: BehaviorGraph) -> int:
    return sum(len(t) for t in graph.transition_counts.values())


def trajectory_nll_bits(
    graph: BehaviorGraph,
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    alpha: float = DEFAULT_ALPHA,
) -> Tuple[float, int]:
    """Total -log2 likelihood of observed (run-length-collapsed) transitions
    when the labels passed in are interpreted as the model's own node IDs.

    Returns (total_nll_bits, num_transitions). Transitions never observed in
    the graph are assigned a tiny floor probability (1e-9) so the score
    remains finite.
    """
    collapsed, outcomes = _extract_collapsed_sequences(
        cluster_labels, metadata, graph.level,
    )
    has_outcome = any(v is not None for v in outcomes.values())

    smoothed: Dict[int, Dict[int, float]] = {}
    for src, tgts in graph.transition_counts.items():
        supp = list(tgts.keys())
        smoothed[src] = smoothed_probs(tgts, supp, alpha=alpha)

    nll = 0.0
    nt = 0
    for ep_idx, seq in collapsed.items():
        if not seq:
            continue
        if has_outcome:
            o = outcomes.get(ep_idx)
            terminal = (
                SUCCESS_NODE_ID if o is True
                else (FAILURE_NODE_ID if o is False else END_NODE_ID)
            )
        else:
            terminal = END_NODE_ID
        full = [START_NODE_ID] + seq + [terminal]
        for i in range(len(full) - 1):
            src, dst = full[i], full[i + 1]
            p = smoothed.get(src, {}).get(dst)
            if p is None or p <= 0.0:
                p = 1e-9
            nll += -float(np.log(p)) / _LOG2
            nt += 1
    return nll, nt


def predictive_nll_bits(
    graph: BehaviorGraph,
    original_labels: np.ndarray,
    metadata: List[Dict],
    node_mapping: Dict[int, int],
    alpha: float = DEFAULT_ALPHA,
) -> Tuple[float, int, int]:
    """NLL of *original* transitions evaluated under a possibly-merged graph.

    This is the principled cross-method metric: the data (original
    inter-state transitions of the unsimplified clustering) is FIXED across
    methods, so the per-original-transition NLL is directly comparable.

    Recipe per episode:
      1. Run-length-collapse the original labels → original_seq (length L).
      2. Apply ``node_mapping`` to each label (default = identity).
      3. Run-length-collapse the mapped sequence → mapped_seq (length L' ≤ L).
      4. Score the L'-1 inter-state transitions under ``graph``.
      5. Original transitions absorbed by merging (where two adjacent original
         labels map to the same merged ID) are scored as 0 bits ("free
         compression"). This makes the metric *compressive*: aggressive
         merging lowers it.

    Returns ``(total_nll_bits, num_scored_transitions, num_original_transitions)``.
    """
    from collections import defaultdict

    ep_key = "rollout_idx" if graph.level == "rollout" else "demo_idx"

    # Build smoothed transition probabilities for graph
    smoothed: Dict[int, Dict[int, float]] = {}
    for src, tgts in graph.transition_counts.items():
        supp = list(tgts.keys())
        smoothed[src] = smoothed_probs(tgts, supp, alpha=alpha)

    # Build episodes
    eps: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    outcomes: Dict[int, Optional[bool]] = {}
    for i, m in enumerate(metadata):
        lbl = int(original_labels[i])
        if lbl == -1:
            continue
        sort_key = m.get("timestep", m.get("window_start", 0))
        eps[m[ep_key]].append((sort_key, lbl))
        if "success" in m and m[ep_key] not in outcomes:
            outcomes[m[ep_key]] = m["success"]
    has_outcome = any(v is not None for v in outcomes.values())

    def _map(x: int) -> int:
        # Follow merge chain if mapping is given.
        if x in node_mapping:
            seen = set()
            cur = x
            while cur in node_mapping:
                if cur in seen:
                    return cur
                seen.add(cur)
                cur = node_mapping[cur]
            return cur
        return x

    total_nll = 0.0
    n_scored = 0
    n_original = 0
    for ep_idx, seq in eps.items():
        if not seq:
            continue
        seq.sort(key=lambda x: x[0])
        # Original collapsed sequence
        original_seq = [seq[0][1]]
        for _, lbl in seq[1:]:
            if lbl != original_seq[-1]:
                original_seq.append(lbl)
        # Apply mapping then re-collapse
        mapped_seq = [_map(original_seq[0])]
        for lbl in original_seq[1:]:
            m = _map(lbl)
            if m != mapped_seq[-1]:
                mapped_seq.append(m)

        if has_outcome:
            o = outcomes.get(ep_idx)
            terminal = (
                SUCCESS_NODE_ID if o is True
                else (FAILURE_NODE_ID if o is False else END_NODE_ID)
            )
        else:
            terminal = END_NODE_ID

        # Original transitions count: (length - 1) inter-state + 2 (START / terminal)
        n_original += len(original_seq) + 1

        full = [START_NODE_ID] + mapped_seq + [terminal]
        for i in range(len(full) - 1):
            src, dst = full[i], full[i + 1]
            p = smoothed.get(src, {}).get(dst)
            if p is None or p <= 0.0:
                p = 1e-9
            total_nll += -float(np.log(p)) / _LOG2
            n_scored += 1

    return total_nll, n_scored, n_original


def markov_violation_against_original_bits(
    original_labels: np.ndarray,
    metadata: List[Dict],
    node_mapping: Optional[Dict[int, int]] = None,
    level: str = "rollout",
    alpha: float = DEFAULT_ALPHA,
    order: int = 1,
    current_labels: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[int, float]]:
    """Conditional MI I(original_prev[1..order]; original_next | merged_current) in bits.

    With order=1 (default) this is the classic ``I(P_{t-1}; N_t | X_t)``.
    With order=2 it generalizes to ``I((P_{t-1}, P_{t-2}); N_t | X_t)``,
    treating the length-2 predecessor tuple as a single composite predecessor.
    Higher orders need exponentially more data to estimate reliably — we use
    order=2 only as a diagnostic to flag when the 1st-order metric is
    underestimating memory.

    Two ways to specify the merged-state classifier:
      - ``current_labels`` (preferred when available): per-timestep merged
        state. Works for any method, including ones that introduce new IDs
        via splitting (e.g. ``vomm_split_merge``).
      - ``node_mapping``: applied to ``original_labels`` to derive the
        merged state. Only valid for pure-merging methods where every
        original cluster ID maps deterministically to one merged ID.

    If ``current_labels`` is provided it takes precedence; otherwise
    ``node_mapping`` (default {}) is used.


    THIS is the principled "interpretability vs Markov property" axis:

      - With many merged states (close to original clustering), each merged
        state X equals exactly one original cluster c, so original_prev /
        original_next reduce to the merged-state predecessor/successor.
        I(...) → 0 if the underlying clustering is Markov.
      - With few merged states (aggressive merging), each merged X spans
        multiple original clusters. Knowing the merged state doesn't pin
        down the original cluster, so the original predecessor still carries
        information about the original successor → I > 0.

    The visitation-weighted mean across merged states is the scalar lever
    output. Per-merged-state values are also returned.

    Returns ``(total_bits, per_merged_node_bits)``.
    """
    from collections import defaultdict

    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
    nm = node_mapping or {}
    use_current_labels = current_labels is not None

    def _map(x: int) -> int:
        if x in nm:
            seen = set()
            cur = x
            while cur in nm:
                if cur in seen:
                    return cur
                seen.add(cur)
                cur = nm[cur]
            return cur
        return x

    # Build per-episode (sort_key, sample_idx, original_label) triplets so we
    # can look up the per-timestep current_labels[sample_idx] when computing
    # the merged-state classifier.
    eps: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for i, m in enumerate(metadata):
        lbl = int(original_labels[i])
        if lbl == -1:
            continue
        sort_key = m.get("timestep", m.get("window_start", 0))
        eps[m[ep_key]].append((sort_key, i, lbl))

    # For each MERGED current state, collect (composite_prev, orig_next) pairs.
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    min_seq_len = order + 2
    pairs_per_merged: Dict[int, List[Tuple[Tuple[int, ...], int]]] = defaultdict(list)
    for ep_idx, seq in eps.items():
        if len(seq) < min_seq_len:
            continue
        seq.sort(key=lambda x: x[0])
        # Build collapsed sequence of (original_label, representative_sample_idx).
        # The sample_idx points to *some* timestep in the run so we can read
        # current_labels[idx] for it; using the first idx of each run is fine
        # because per-timestep labels within a run are constant for merging
        # methods (and we explicitly preserve that invariant in splitting).
        collapsed: List[Tuple[int, int]] = [(seq[0][2], seq[0][1])]
        for _, idx, lbl in seq[1:]:
            if lbl != collapsed[-1][0]:
                collapsed.append((lbl, idx))
        for i in range(order, len(collapsed) - 1):
            composite_prev = tuple(collapsed[i - k][0] for k in range(1, order + 1))
            o_cur, idx_cur = collapsed[i]
            o_next = collapsed[i + 1][0]
            if use_current_labels:
                m_cur = int(current_labels[idx_cur])
            else:
                m_cur = _map(o_cur)
            pairs_per_merged[m_cur].append((composite_prev, o_next))

    per_node: Dict[int, float] = {}
    total_weight = 0
    weighted_sum = 0.0

    # Min-pair gate scales with the contingency table size (≈ order²).
    min_pairs = max(4, 4 * order * order)
    for m_cur, pairs in pairs_per_merged.items():
        if len(pairs) < min_pairs:
            per_node[m_cur] = 0.0
            continue

        prev_counts: Dict = defaultdict(int)   # key may be tuple
        next_counts: Dict[int, int] = defaultdict(int)
        joint: Dict[Tuple, int] = defaultdict(int)
        for p, n in pairs:
            prev_counts[p] += 1
            next_counts[n] += 1
            joint[(p, n)] += 1

        if len(prev_counts) < 2 or len(next_counts) < 2:
            per_node[m_cur] = 0.0
            continue

        n_total = sum(prev_counts.values())
        # Plug-in MI with light Dirichlet smoothing on the joint
        prev_supp = sorted(prev_counts.keys())
        next_supp = sorted(next_counts.keys())
        p_prev = smoothed_probs(prev_counts, prev_supp, alpha=alpha)
        p_next = smoothed_probs(next_counts, next_supp, alpha=alpha)
        joint_full: Dict[Tuple[int, int], int] = {}
        for p in prev_supp:
            for n in next_supp:
                joint_full[(p, n)] = joint.get((p, n), 0)
        total_joint = sum(joint_full.values())
        denom = total_joint + alpha * len(prev_supp) * len(next_supp)
        p_joint = {
            k: (v + alpha) / denom
            for k, v in joint_full.items()
        }
        mi = 0.0
        for (p, n), pj in p_joint.items():
            if pj <= 0:
                continue
            pp = p_prev[p]
            pn = p_next[n]
            if pp <= 0 or pn <= 0:
                continue
            mi += pj * np.log(pj / (pp * pn))
        mi = max(0.0, float(mi) / _LOG2)
        per_node[m_cur] = mi
        weighted_sum += n_total * mi
        total_weight += n_total

    total = weighted_sum / total_weight if total_weight > 0 else 0.0
    return total, per_node


def free_params_count(graph: BehaviorGraph) -> int:
    """Number of free transition probabilities in graph (each row has k-1 free)."""
    n = 0
    for tgts in graph.transition_counts.values():
        k = len(tgts)
        if k > 1:
            n += k - 1
    return n


def markov_violation_bits(
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    level: str = "rollout",
    alpha: float = DEFAULT_ALPHA,
    include_terminals: bool = False,
) -> Tuple[float, Dict[int, float]]:
    """Conditional mutual information I(prev; next | current_state) in bits.

    For each interior state s:
        I(prev; next | s) = H(next | s) - H(next | s, prev)
    The total is the visitation-weighted mean across states. If the graph is
    truly Markov, this is 0; larger values indicate the predecessor carries
    information about the successor beyond what the current state captures.

    Returns (total_bits, per_node_bits).
    """
    second_order, _all_ids = _compute_second_order_counts(
        cluster_labels, metadata, level, exclude_terminals=not include_terminals,
    )
    per_node: Dict[int, float] = {}
    total_weight = 0
    weighted_sum = 0.0

    for s, prev_to_next in second_order.items():
        marg_counts: Dict[int, int] = defaultdict(int)
        n_s = 0
        for nxts in prev_to_next.values():
            for nxt, cnt in nxts.items():
                marg_counts[nxt] += cnt
                n_s += cnt
        if n_s == 0 or len(marg_counts) < 2 or len(prev_to_next) < 2:
            per_node[s] = 0.0
            continue
        support = sorted(marg_counts.keys())
        marg_probs = smoothed_probs(marg_counts, support, alpha=alpha)
        h_marg = -sum(p * np.log(p) for p in marg_probs.values() if p > 0) / _LOG2

        h_cond = 0.0
        for prev, nxts in prev_to_next.items():
            n_p = sum(nxts.values())
            if n_p == 0:
                continue
            cond_probs = smoothed_probs(nxts, support, alpha=alpha)
            h_p = -sum(q * np.log(q) for q in cond_probs.values() if q > 0) / _LOG2
            h_cond += (n_p / n_s) * h_p

        mi = max(0.0, h_marg - h_cond)
        per_node[s] = mi
        weighted_sum += n_s * mi
        total_weight += n_s

    total = weighted_sum / total_weight if total_weight > 0 else 0.0
    return total, per_node


@dataclass
class GraphMetrics:
    n_nodes: int
    n_edges: int
    n_free_params: int
    nll_bits: float                       # NLL using merged labels (compressive; comparable within method, not across)
    nll_per_transition_bits: float        # nll_bits / (# transitions after merging)
    nll_per_original_bits: float          # predictive NLL / (# original transitions)  ← FAIR across methods
    markov_violation_bits: float          # I(prev_{t-1}; next_t | merged_curr_t) — 1st-order memory missed
    markov_violation_2nd_bits: float      # I((prev_{t-1}, prev_{t-2}); next_t | merged_curr_t) — 2nd-order diagnostic
    mdl_score: float                      # predictive NLL + (k/2) log2(N_original)  ← model selection criterion

    def as_dict(self) -> Dict[str, float]:
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_free_params": self.n_free_params,
            "nll_bits": self.nll_bits,
            "nll_per_transition_bits": self.nll_per_transition_bits,
            "nll_per_original_bits": self.nll_per_original_bits,
            "markov_violation_bits": self.markov_violation_bits,
            "markov_violation_2nd_bits": self.markov_violation_2nd_bits,
            "mdl_score": self.mdl_score,
        }


def compute_metrics(
    graph: BehaviorGraph,
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    mdl_lambda: float = 1.0,
    alpha: float = DEFAULT_ALPHA,
    original_labels: Optional[np.ndarray] = None,
    node_mapping: Optional[Dict[int, int]] = None,
) -> GraphMetrics:
    """One-shot computation of all scalar metrics for a (graph, labels) pair.

    If ``original_labels`` is provided, also compute the fair, cross-method
    "predictive NLL per original transition" using the given ``node_mapping``.
    Otherwise we approximate by treating cluster_labels as both original and
    merged.
    """
    n_nodes = graph_node_count(graph, include_special=False)
    n_edges = graph_edge_count(graph)
    k_params = free_params_count(graph)
    nll, nt = trajectory_nll_bits(graph, cluster_labels, metadata, alpha=alpha)
    nll_per_trans = nll / nt if nt > 0 else 0.0

    if original_labels is not None and node_mapping is not None:
        pred_nll, _n_scored, n_orig = predictive_nll_bits(
            graph, original_labels, metadata, node_mapping, alpha=alpha,
        )
        # Use the actual per-timestep simplified labels for the merged-state
        # classifier — this is correct for both pure-merge methods (where
        # cluster_labels == map(node_mapping, original_labels)) and for
        # split-producing methods (where cluster_labels contains IDs not in
        # any node_mapping entry).
        mv, _ = markov_violation_against_original_bits(
            original_labels, metadata, node_mapping,
            level=graph.level, alpha=alpha, order=1,
            current_labels=cluster_labels,
        )
        mv2, _ = markov_violation_against_original_bits(
            original_labels, metadata, node_mapping,
            level=graph.level, alpha=alpha, order=2,
            current_labels=cluster_labels,
        )
    else:
        pred_nll, n_orig = nll, nt
        mv, _ = markov_violation_bits(
            cluster_labels, metadata, level=graph.level, alpha=alpha,
        )
        mv2 = 0.0
    nll_per_original = pred_nll / n_orig if n_orig > 0 else 0.0
    # MDL (Rissanen / BIC-like): predictive NLL + (k/2) log₂(N_original) * mdl_lambda
    bic_penalty = 0.5 * float(np.log2(max(n_orig, 2))) * k_params
    mdl = pred_nll + mdl_lambda * bic_penalty
    return GraphMetrics(
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_free_params=k_params,
        nll_bits=nll,
        nll_per_transition_bits=nll_per_trans,
        nll_per_original_bits=nll_per_original,
        markov_violation_bits=mv,
        markov_violation_2nd_bits=mv2,
        mdl_score=mdl,
    )


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_mv_ci(
    original_labels: np.ndarray,
    metadata: List[Dict],
    node_mapping: Optional[Dict[int, int]] = None,
    level: str = "rollout",
    order: int = 1,
    alpha: float = DEFAULT_ALPHA,
    n_bootstrap: int = 100,
    rng_seed: int = 0,
    current_labels: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Episode-level bootstrap CI on Markov violation at fixed node_mapping.

    This is the *data-noise* CI (how much MV could fluctuate if we'd seen a
    different sample of episodes), holding the simplification result fixed.
    Cheap — just resamples episodes and recomputes the contingency table;
    typically <100ms per rep.

    Returns ``(point_estimate, p2.5, p97.5)``.
    """
    from collections import defaultdict as _dd

    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
    eps: Dict[int, List[int]] = _dd(list)
    for i, m in enumerate(metadata):
        eps[m[ep_key]].append(i)
    ep_list = list(eps.keys())
    n_eps = len(ep_list)
    if n_eps == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(rng_seed)

    point, _ = markov_violation_against_original_bits(
        original_labels, metadata, node_mapping,
        level=level, alpha=alpha, order=order,
        current_labels=current_labels,
    )
    samples: List[float] = []
    for _ in range(n_bootstrap):
        boot = rng.choice(ep_list, size=n_eps, replace=True)
        idxs: List[int] = []
        new_meta: List[Dict] = []
        for new_ep, old_ep in enumerate(boot):
            for i in eps[old_ep]:
                idxs.append(i)
                mm = dict(metadata[i])
                mm[ep_key] = int(new_ep)
                new_meta.append(mm)
        idx_arr = np.array(idxs)
        boot_labels = original_labels[idx_arr]
        boot_current = current_labels[idx_arr] if current_labels is not None else None
        v, _ = markov_violation_against_original_bits(
            boot_labels, new_meta, node_mapping,
            level=level, alpha=alpha, order=order,
            current_labels=boot_current,
        )
        samples.append(float(v))
    arr = np.array(samples)
    return float(point), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def episode_index_groups(
    metadata: List[Dict], level: str = "rollout",
) -> Dict[int, List[int]]:
    """Return {episode_idx: [sample_indices]} (no time-ordering applied)."""
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
    eps: Dict[int, List[int]] = defaultdict(list)
    for i, m in enumerate(metadata):
        eps[m[ep_key]].append(i)
    return dict(eps)


def split_by_episode(
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    level: str = "rollout",
    test_episodes: Sequence[int] = (),
) -> Tuple[Tuple[np.ndarray, List[Dict]], Tuple[np.ndarray, List[Dict]]]:
    """Partition (labels, metadata) into (train, test) by episode index."""
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
    test_set = set(int(e) for e in test_episodes)
    train_idx: List[int] = []
    test_idx: List[int] = []
    for i, m in enumerate(metadata):
        (test_idx if int(m[ep_key]) in test_set else train_idx).append(i)
    train_idx_arr = np.array(train_idx, dtype=np.int64)
    test_idx_arr = np.array(test_idx, dtype=np.int64)
    return (
        (cluster_labels[train_idx_arr], [metadata[i] for i in train_idx]),
        (cluster_labels[test_idx_arr], [metadata[i] for i in test_idx]),
    )


def heldout_nll_bits(
    train_graph: BehaviorGraph,
    test_labels: np.ndarray,
    test_metadata: List[Dict],
    alpha: float = DEFAULT_ALPHA,
) -> Tuple[float, int]:
    """NLL of held-out trajectories under the train graph's smoothed transitions.

    Test transitions whose source isn't in the train graph fall back to a
    floor probability — those are honest "overfit" penalties.
    """
    return trajectory_nll_bits(train_graph, test_labels, test_metadata, alpha=alpha)


def kfold_episode_splits(
    metadata: List[Dict],
    level: str = "rollout",
    n_folds: int = 5,
    rng_seed: int = 0,
) -> List[List[int]]:
    """Return n_folds lists of episode indices (held-out per fold)."""
    eps = sorted(episode_index_groups(metadata, level).keys())
    rng = np.random.RandomState(rng_seed)
    eps_shuf = list(eps)
    rng.shuffle(eps_shuf)
    n = len(eps_shuf)
    folds: List[List[int]] = [[] for _ in range(n_folds)]
    for i, e in enumerate(eps_shuf):
        folds[i % n_folds].append(int(e))
    return folds


def bootstrap_metric(
    fn: Callable[[np.ndarray, List[Dict]], float],
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    level: str = "rollout",
    n_bootstrap: int = 100,
    rng_seed: int = 0,
) -> Tuple[float, float, float]:
    """Episode-level bootstrap CI on a metric `fn(labels, metadata) -> float`.

    Resamples episodes with replacement (preserving within-episode time order),
    re-labels them so duplicates don't collide, and re-evaluates `fn`. Returns
    (mean, p2.5, p97.5).
    """
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
    eps: Dict[int, List[int]] = defaultdict(list)
    for i, m in enumerate(metadata):
        eps[m[ep_key]].append(i)
    ep_list = list(eps.keys())
    n_eps = len(ep_list)
    if n_eps == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(rng_seed)

    samples = []
    for _ in range(n_bootstrap):
        boot = rng.choice(ep_list, size=n_eps, replace=True)
        idxs: List[int] = []
        new_meta: List[Dict] = []
        for new_ep, old_ep in enumerate(boot):
            for i in eps[old_ep]:
                idxs.append(i)
                m = dict(metadata[i])
                m[ep_key] = int(new_ep)
                new_meta.append(m)
        boot_labels = cluster_labels[np.array(idxs)]
        samples.append(float(fn(boot_labels, new_meta)))
    arr = np.array(samples)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
