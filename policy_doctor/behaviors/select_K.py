"""Automatic hyperparameter selection for behavior-graph clusterings.

Provides silhouette γ-selection for K (``select_K_by_silhouette_gamma``) and a
helper to pick ``(rep, ordering, w, s, k)`` defaults from a sweep index whose
entries carry ``silhouette_mean`` in ``metrics`` (as written by
``compute_clustering_metrics``).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def select_K_by_silhouette_gamma(
    K_list: Sequence[int],
    sil_list: Sequence[float],
    gamma: float = 0.9,
    K_min: int = 5,
) -> Optional[int]:
    """Select K from a silhouette-vs-K curve using a gamma fraction of the peak.

    Algorithm:
      1. Find K_peak = argmax(silhouette).
      2. Compute target = gamma * sil[K_peak].
      3. Among K < K_peak with sil[K] >= target and K >= K_min,
         return the SMALLEST such K.

    Returns None if the curve is monotone decreasing (peak at smallest K) or if
    no K meets the threshold.
    """
    if not K_list or not sil_list or len(K_list) != len(sil_list):
        return None
    pairs = sorted(zip(K_list, sil_list), key=lambda x: x[0])
    sorted_K = [p[0] for p in pairs]
    sorted_sil = [p[1] for p in pairs]
    max_sil = max(sorted_sil)
    peak_idx = sorted_sil.index(max_sil)
    if peak_idx == 0:
        return None
    target = gamma * max_sil
    candidates = [
        sorted_K[i] for i in range(peak_idx)
        if sorted_K[i] >= K_min and sorted_sil[i] >= target
    ]
    return min(candidates) if candidates else None


def _silhouette_from_entry(entry: Mapping[str, Any]) -> Optional[float]:
    metrics = entry.get("metrics") or {}
    val = metrics.get("silhouette_mean")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def select_clustering_by_silhouette(
    entries: Iterable[Mapping[str, Any]],
    *,
    gamma: float = 0.9,
    K_min: int = 5,
    min_k_per_setting: int = 3,
    fallback: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Pick clustering defaults from sweep index rows via silhouette γ-selection.

    Each *entry* is expected to expose ``rep``, ``ordering``, ``w``, ``s``,
    ``k``, and optional ``metrics["silhouette_mean"]``.

    For every distinct ``(rep, ordering, w, s)`` setting with at least
    ``min_k_per_setting`` K values that have silhouette scores, apply
    :func:`select_K_by_silhouette_gamma`. The winning setting is the one whose
    selected K has the highest silhouette score. If γ-selection yields no valid
    K for any setting, fall back to the single entry with the largest
    ``silhouette_mean``. If no silhouettes are available, return *fallback* or
    the first entry's params.
    """
    rows = list(entries)
    if not rows:
        return dict(fallback or {})

    by_setting: Dict[Tuple[Any, ...], List[Tuple[int, float, Mapping[str, Any]]]] = {}
    best_raw: Optional[Tuple[float, Mapping[str, Any]]] = None

    for entry in rows:
        sil = _silhouette_from_entry(entry)
        if sil is None:
            continue
        key = (
            entry.get("rep"),
            entry.get("ordering"),
            int(entry.get("w", 0) or 0),
            int(entry.get("s", 0) or 0),
        )
        k = int(entry.get("k", 0) or 0)
        by_setting.setdefault(key, []).append((k, sil, entry))
        if best_raw is None or sil > best_raw[0]:
            best_raw = (sil, entry)

    best_gamma: Optional[Tuple[float, Mapping[str, Any]]] = None
    for setting_rows in by_setting.values():
        if len(setting_rows) < min_k_per_setting:
            continue
        Ks = [r[0] for r in setting_rows]
        sils = [r[1] for r in setting_rows]
        k_star = select_K_by_silhouette_gamma(Ks, sils, gamma=gamma, K_min=K_min)
        if k_star is None:
            continue
        for k, sil, entry in setting_rows:
            if k == k_star:
                if best_gamma is None or sil > best_gamma[0]:
                    best_gamma = (sil, entry)
                break

    chosen = (best_gamma or best_raw)
    if chosen is None:
        fb = dict(fallback or {})
        if fb:
            return fb
        e0 = rows[0]
        return {
            "rep": e0.get("rep"),
            "ordering": e0.get("ordering"),
            "w": int(e0.get("w", 5) or 5),
            "s": int(e0.get("s", 2) or 2),
            "k": int(e0.get("k", 8) or 8),
        }

    entry = chosen[1]
    return {
        "rep": entry.get("rep"),
        "ordering": entry.get("ordering"),
        "w": int(entry.get("w", 5) or 5),
        "s": int(entry.get("s", 2) or 2),
        "k": int(entry.get("k", 8) or 8),
    }


__all__ = [
    "select_K_by_silhouette_gamma",
    "select_clustering_by_silhouette",
]
