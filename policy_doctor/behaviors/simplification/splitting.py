"""Predecessor-conditioned state splitting (variable-order Markov / VOMM).

When a node's outgoing distribution depends significantly on the predecessor,
split the node by predecessor context. Combined with merging, this gives a
variable-order Markov model: a node is refined only where the extra memory
buys predictive power, and is merged otherwise.

Identifier scheme: split children get fresh cluster IDs starting at
`max_cluster_id + 1`. The new label sequence then re-derives a graph that's
locally Markovian.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from policy_doctor.behaviors.behavior_graph import (
    BehaviorGraph,
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
    _compute_second_order_counts,
    _extract_collapsed_sequences,
)
from policy_doctor.behaviors.simplification.api import (
    SimplificationResult,
    register_method,
)
from policy_doctor.behaviors.simplification.merging import (
    _iterative_merge,
)
from policy_doctor.behaviors.simplification.metrics import (
    chi2_pvalue,
    compute_metrics,
    hoeffding_compatible,
    js_distance_bits,
    markov_violation_bits,
    smoothed_probs,
)


def _split_one_round(
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    level: str,
    tau_bits: float,
    min_count_per_predecessor: int = 5,
) -> Tuple[np.ndarray, int]:
    """Greedy single-round split.

    For the cluster node with the largest predecessor-conditioned KL-divergence
    above τ bits, split timesteps by their predecessor (run-length-collapsed).
    Returns (new_labels, num_splits_made).
    """
    _, per_node = markov_violation_bits(
        cluster_labels, metadata, level=level,
    )
    worst = sorted(per_node.items(), key=lambda x: -x[1])
    if not worst or worst[0][1] <= tau_bits:
        return cluster_labels, 0

    target_node = worst[0][0]
    # Get the predecessor counts at this node
    second, _ = _compute_second_order_counts(
        cluster_labels, metadata, level, exclude_terminals=False,
    )
    if target_node not in second:
        return cluster_labels, 0
    prev_to_next = second[target_node]
    if len(prev_to_next) < 2:
        return cluster_labels, 0

    # Marginal distribution at the target node
    marg: Dict[int, int] = defaultdict(int)
    for nxts in prev_to_next.values():
        for n, c in nxts.items():
            marg[n] += c
    support = sorted(marg.keys())
    marg_probs = smoothed_probs(marg, support)

    # For each predecessor with enough data, compute KL vs marginal
    kls = []
    for prev, nxts in prev_to_next.items():
        n_p = sum(nxts.values())
        if n_p < min_count_per_predecessor:
            continue
        cond_probs = smoothed_probs(nxts, support)
        from policy_doctor.behaviors.simplification.metrics import kl_bits
        d = kl_bits(cond_probs, marg_probs)
        kls.append((prev, n_p, d))
    kls.sort(key=lambda x: -x[2])
    if not kls or kls[0][2] <= tau_bits:
        return cluster_labels, 0

    # Split off the highest-KL predecessor group(s) into a new ID
    new_id = int(np.max(cluster_labels)) + 1
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"

    # Identify timesteps at the target node whose immediately-preceding
    # *collapsed* state matches the split predecessor.
    eps: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for i, m in enumerate(metadata):
        sort_key = m.get("timestep", m.get("window_start", 0))
        eps[m[ep_key]].append((sort_key, i))
    new_labels = cluster_labels.copy()
    split_preds = {p for p, _, kl in kls if kl > tau_bits}

    for ep_idx, seq in eps.items():
        seq.sort(key=lambda x: x[0])
        # Build collapsed segments with their member sample indices
        segs: List[Tuple[int, List[int]]] = []
        for _, idx in seq:
            lbl = int(cluster_labels[idx])
            if lbl == -1:
                continue
            if segs and segs[-1][0] == lbl:
                segs[-1][1].append(idx)
            else:
                segs.append((lbl, [idx]))
        for j, (lbl, idxs) in enumerate(segs):
            if lbl != target_node:
                continue
            prev_lbl = segs[j - 1][0] if j > 0 else START_NODE_ID
            if prev_lbl in split_preds:
                for idx in idxs:
                    new_labels[idx] = new_id

    return new_labels, 1 if not np.array_equal(new_labels, cluster_labels) else 0


@register_method(
    name="vomm_split_merge",
    description=(
        "Variable-Order Markov Model induction: alternate predecessor-splitting "
        "(when KL(p(next|s,prev) || p(next|s)) > τ) with Alergia-Hoeffding "
        "merging (when outgoing distributions agree at confidence δ). One "
        "lever τ controls both — δ scales with τ. Splits until Markovian, "
        "merges until minimal."
    ),
    lever_label="τ (bits of Markov violation tolerated)",
)
def vomm_split_merge(
    graph: BehaviorGraph,
    cluster_labels: np.ndarray,
    metadata: List[Dict],
    lever: Optional[float],
    **kwargs,
) -> SimplificationResult:
    tau = 0.05 if lever is None else float(lever)
    level = graph.level
    original_labels = np.asarray(cluster_labels, dtype=np.int64).copy()
    labels = original_labels.copy()

    # Map τ → δ for Hoeffding merge: smaller τ → tighter Markov →
    # also tighter merge (less compression). Empirically a 1:1 mapping works
    # but the constants matter little for the qualitative shape of the curve.
    delta = float(np.clip(np.exp(-(tau + 0.05) * 10.0), 1e-3, 0.5))

    def compat(c1: Dict[int, int], c2: Dict[int, int]) -> bool:
        return hoeffding_compatible(c1, c2, delta)

    def score(c1: Dict[int, int], c2: Dict[int, int]) -> float:
        return js_distance_bits(c1, c2)

    n_splits = 0
    n_merges = 0
    for _ in range(50):
        labels, did_split = _split_one_round(labels, metadata, level, tau)
        if did_split:
            n_splits += 1
            cur = BehaviorGraph.from_cluster_assignments(labels, metadata, level=level)
            cur, labels, _ = _iterative_merge(cur, labels, metadata, compat, score)
            n_merges += 1
        else:
            break
    # Final consolidation pass
    cur = BehaviorGraph.from_cluster_assignments(labels, metadata, level=level)
    cur, labels, _ = _iterative_merge(cur, labels, metadata, compat, score)
    # Pass original_labels + node_mapping={} so compute_metrics takes the
    # *per-timestep* path (using `labels` as current_labels). Without this it
    # falls through to the legacy `markov_violation_bits`, which on split-
    # derived labels gives a misleading near-zero MV (each split-only ID has
    # one predecessor by construction).
    metrics = compute_metrics(
        cur, labels, metadata,
        original_labels=original_labels, node_mapping={},
    )
    return SimplificationResult(
        method="vomm_split_merge",
        lever_value=tau,
        graph=cur,
        new_labels=labels,
        node_mapping={},
        metrics=metrics,
        extras={"n_splits": float(n_splits), "n_merges": float(n_merges)},
    )
