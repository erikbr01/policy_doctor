"""Automatic hyperparameter selection for behavior-graph creation.

Selects `(representation, window_width, stride, K)` from a coverage-gated
sweep of MV₁ measurements. No magic constants: the pipeline reads cells
directly off the metric and the gate.

The selection logic:

1. **Coverage gate** — discard cells where MV₁ coverage is below the
   threshold (default 0.80) OR K < `K_min` (default 5). These cells are
   measurement artefacts, not signal.

2. **Per-rep knee on the (n_nodes, MV₁) curve** — for each rep, the
   chosen K is the smallest K where MV₁ has reached `gamma` of its
   asymptote (default 0.5). This is the "first useful K": small enough
   to be interpretable, large enough that the chain is no longer
   trivially deterministic.

3. **Pareto-dominance across reps** — a rep is preferred if, at every
   K where both clear the gate, its MV₁ is ≤ the other's, with strict
   inequality somewhere. If neither Pareto-dominates, fall back to the
   one with lower MV₁ at its own knee.

The function accepts a list of `{rep, w, s, K, MV1, cov1}` cells; the
caller is responsible for producing that grid. See
`select_K_from_results_dir` for a wrapper that reads
`docs/k_sweep_results` JSONs.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Cell:
    rep: str
    w: int
    s: int
    K: int
    mv1: float
    cov1: float


@dataclass(frozen=True)
class Pick:
    rep: str
    w: int
    s: int
    K: int
    mv1: float
    cov1: float
    rationale: str
    asymp_mv1: float = 0.0      # the MV_asymp used for the knee
    converged: bool = True       # False if MV at K_max_gated is well below MV_asymp
    notes: Tuple[str, ...] = ()  # warning notes (e.g. non-convergence)


def _gated(cells: Iterable[Cell], cov_min: float, K_min: int) -> List[Cell]:
    return [c for c in cells if c.cov1 >= cov_min and c.K >= K_min]


def _knee_within_setting(
    cells_at_setting: List[Cell],
    gamma: float,
    convergence_ratio: float = 0.9,
) -> Optional[Tuple[Cell, float, bool, Tuple[str, ...]]]:
    """For a fixed (rep, w, s), pick smallest K with MV₁ ≥ γ · MV_asymp.

    Patch A: ``MV_asymp = max over ALL gated K cells`` (not just top-3). This
    is robust to non-monotone MV-vs-K curves (e.g. when very high K
    triggers the min-pairs gate and MV reads artificially low).

    Patch B: returns ``converged=False`` and a warning note when
    ``MV(K_max_gated) < convergence_ratio · MV_asymp`` — i.e. the curve
    is still descending at the largest gated K, so the asymptote is
    likely a peak rather than a plateau.

    Returns ``(knee_cell, asymp, converged, notes)`` or ``None`` if the
    setting has no cells.
    """
    if not cells_at_setting:
        return None
    cells_at_setting = sorted(cells_at_setting, key=lambda c: c.K)
    asymp = max(c.mv1 for c in cells_at_setting)             # Patch A
    K_max_cell = cells_at_setting[-1]
    converged = K_max_cell.mv1 >= convergence_ratio * asymp  # Patch B
    notes: List[str] = []
    if not converged:
        # Locate the peak K so the warning is informative.
        peak = max(cells_at_setting, key=lambda c: c.mv1)
        notes.append(
            f"asymptote NOT converged: MV peaks {asymp:.3f} at K={peak.K} "
            f"but drops to {K_max_cell.mv1:.3f} at K_max={K_max_cell.K} "
            f"(min-pairs gate likely firing at high K). γ-knee may be biased."
        )
    target = gamma * asymp
    for c in cells_at_setting:
        if c.mv1 >= target:
            return c, asymp, converged, tuple(notes)
    return K_max_cell, asymp, converged, tuple(notes)


def _best_setting_per_rep(
    cells: List[Cell],
    gamma: float,
    min_gated_per_setting: int,
) -> Dict[str, Tuple[Cell, List[Cell], float, bool, Tuple[str, ...]]]:
    """For each rep, scan (w, s) settings and return the one whose knee has
    lowest MV₁ — *subject to* having enough gated K cells to be trustworthy.

    A (w, s) setting is *admissible* only if at least `min_gated_per_setting`
    K cells clear the coverage gate (otherwise we'd be picking gate-edge
    cells where MV₁ is artificially low). The chosen setting is the one
    whose knee has lowest MV₁ among admissible (w, s).

    Returns {rep -> (knee_cell, gated_cells_at_chosen_setting)}.
    """
    by_rep_ws: Dict[Tuple[str, int, int], List[Cell]] = {}
    for c in cells:
        by_rep_ws.setdefault((c.rep, c.w, c.s), []).append(c)
    per_rep_best: Dict[str, Tuple[Cell, List[Cell], float, bool, Tuple[str, ...]]] = {}
    for (rep, w, s), setting_cells in by_rep_ws.items():
        if len(setting_cells) < min_gated_per_setting:
            continue  # too few K's clear the gate to trust this (w, s)
        result = _knee_within_setting(setting_cells, gamma=gamma)
        if result is None:
            continue
        knee, asymp, conv, notes = result
        prior = per_rep_best.get(rep)
        if prior is None or knee.mv1 < prior[0].mv1:
            per_rep_best[rep] = (knee, setting_cells, asymp, conv, notes)
    return per_rep_best


def _pareto_dominant_rep(
    per_rep_best: Dict[str, Tuple[Cell, List[Cell], float, bool, Tuple[str, ...]]]
) -> Optional[str]:
    """Returns the rep that Pareto-dominates within its chosen (w, s) setting.

    A rep `a` dominates `b` if for every K where both have cells (in their
    chosen settings) with cov₁ ≥ gate, mv1_a ≤ mv1_b, with strict
    inequality at ≥ 1 K. Returns the dominant rep's name, or None on tie.
    """
    if len(per_rep_best) < 2:
        return next(iter(per_rep_best), None)
    reps = list(per_rep_best.keys())
    if len(reps) > 2:
        # Pairwise — fall back to "best MV₁ at knee" if no chain dominates.
        return min(per_rep_best, key=lambda r: per_rep_best[r][0].mv1)

    a, b = reps
    cells_a = {c.K: c for c in per_rep_best[a][1]}
    cells_b = {c.K: c for c in per_rep_best[b][1]}
    common_K = sorted(set(cells_a) & set(cells_b))
    if not common_K:
        return min(per_rep_best, key=lambda r: per_rep_best[r][0].mv1)

    a_dom_b = all(cells_a[K].mv1 <= cells_b[K].mv1 for K in common_K) and \
              any(cells_a[K].mv1 < cells_b[K].mv1 for K in common_K)
    b_dom_a = all(cells_b[K].mv1 <= cells_a[K].mv1 for K in common_K) and \
              any(cells_b[K].mv1 < cells_a[K].mv1 for K in common_K)
    if a_dom_b and not b_dom_a:
        return a
    if b_dom_a and not a_dom_b:
        return b
    # Neither dominates — fall back to lower-MV-at-knee.
    return min(per_rep_best, key=lambda r: per_rep_best[r][0].mv1)


def select_hyperparams(
    cells: Iterable[Cell],
    cov_min: float = 0.80,
    K_min: int = 5,
    gamma: float = 0.5,
    min_gated_per_setting: int = 3,
) -> Optional[Pick]:
    """Joint auto-selection of (rep, w, s, K).

    A (rep, w, s) setting is admissible only if at least
    ``min_gated_per_setting`` K cells clear the coverage gate — that is,
    we trust a (w, s) only when the metric is measurable across a
    meaningful K range, not at a single gate-edge cell.
    """
    gated = _gated(cells, cov_min=cov_min, K_min=K_min)
    if not gated:
        return None
    per_rep_best = _best_setting_per_rep(
        gated, gamma=gamma, min_gated_per_setting=min_gated_per_setting,
    )
    if not per_rep_best:
        return None
    chosen_rep = _pareto_dominant_rep(per_rep_best)
    if chosen_rep is None:
        return None
    knee, _, asymp, conv, notes = per_rep_best[chosen_rep]
    rationale = (
        f"Pareto-dominant rep among {sorted(per_rep_best.keys())}; "
        f"best (w, s) within rep by knee MV₁ (required ≥"
        f"{min_gated_per_setting} gated K cells); K=knee at γ={gamma} of "
        f"MV_asymp={asymp:.3f} within (w={knee.w}, s={knee.s})."
    )
    return Pick(
        rep=knee.rep, w=knee.w, s=knee.s, K=knee.K,
        mv1=knee.mv1, cov1=knee.cov1, rationale=rationale,
        asymp_mv1=asymp, converged=conv, notes=notes,
    )


def select_K_from_results_dir(
    results_dir: pathlib.Path,
    task: str,
    cov_min: float = 0.80,
    K_min: int = 5,
    gamma: float = 0.5,
) -> Optional[Pick]:
    """Read all `{task}__*__w*_s*__K*.json` cells under `results_dir` and
    apply :func:`select_hyperparams`.
    """
    cells: List[Cell] = []
    for p in sorted(results_dir.glob(f"{task}__*__w*_s*__K*.json")):
        try:
            row = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        cells.append(Cell(
            rep=row["rep"],
            w=int(row["w"]),
            s=int(row["s"]),
            K=int(row["K"]),
            mv1=float(row["mv1_point"]),
            cov1=float(row.get("mv1_coverage_fraction", 1.0)),
        ))
    return select_hyperparams(cells, cov_min=cov_min, K_min=K_min, gamma=gamma)


def rank_candidates(
    cells: Iterable[Cell],
    cov_min: float = 0.80,
    K_min: int = 5,
    gamma: float = 0.5,
    min_gated_per_setting: int = 3,
) -> List[Pick]:
    """Return every (rep, w, s) setting's knee, sorted by ``mv1`` then by
    descending ``cov1``. Useful for showing close calls alongside the
    auto-picked winner.
    """
    gated = _gated(cells, cov_min=cov_min, K_min=K_min)
    if not gated:
        return []
    by_rep_ws: Dict[Tuple[str, int, int], List[Cell]] = {}
    for c in gated:
        by_rep_ws.setdefault((c.rep, c.w, c.s), []).append(c)
    knees: List[Pick] = []
    for (rep, w, s), setting_cells in by_rep_ws.items():
        if len(setting_cells) < min_gated_per_setting:
            continue
        result = _knee_within_setting(setting_cells, gamma=gamma)
        if result is None:
            continue
        knee, asymp, conv, notes = result
        knees.append(Pick(
            rep=knee.rep, w=knee.w, s=knee.s, K=knee.K,
            mv1=knee.mv1, cov1=knee.cov1,
            rationale=f"{len(setting_cells)} gated cells in (w={knee.w}, s={knee.s})",
            asymp_mv1=asymp, converged=conv, notes=notes,
        ))
    # Sort: lower MV₁ wins; tiebreak by higher coverage (more honest measurement).
    knees.sort(key=lambda p: (p.mv1, -p.cov1))
    return knees


def select_K_by_silhouette_gamma(
    K_list: List[int],
    sil_list: List[float],
    gamma: float = 0.9,
    K_min: int = 5,
) -> Optional[int]:
    """Select K from a silhouette-vs-K curve using a gamma fraction of the peak.

    Algorithm:
      1. Find K_peak = argmax(silhouette).
      2. Compute target = gamma * sil[K_peak].
      3. Among K < K_peak with sil[K] >= target and K >= K_min,
         return the SMALLEST such K.

    Using min (not max) mirrors the MV γ-selection semantics:
      - Low γ  → low target → many K qualify → pick the smallest (simplest graph)
      - High γ → high target → only K near the peak qualify → pick the smallest
        of those, which is the first K that reaches near-peak silhouette
      - γ = 1.0 → target = max_sil → only K whose silhouette equals the peak
        qualifies; since we exclude K_peak itself, this typically returns None

    Returns None if the curve is monotone decreasing (peak at smallest K, no
    ascending left side) or if no K meets the threshold.
    """
    if not K_list or not sil_list or len(K_list) != len(sil_list):
        return None
    pairs = sorted(zip(K_list, sil_list), key=lambda x: x[0])
    sorted_K = [p[0] for p in pairs]
    sorted_sil = [p[1] for p in pairs]
    max_sil = max(sorted_sil)
    peak_idx = sorted_sil.index(max_sil)
    if peak_idx == 0:
        return None  # monotone decreasing — peak is at smallest K, no ascending left side
    target = gamma * max_sil
    candidates = [
        sorted_K[i] for i in range(peak_idx)
        if sorted_K[i] >= K_min and sorted_sil[i] >= target
    ]
    return min(candidates) if candidates else None


__all__ = [
    "Cell",
    "Pick",
    "select_hyperparams",
    "select_K_from_results_dir",
    "rank_candidates",
    "select_K_by_silhouette_gamma",
]
