"""Load eval_save_episodes output directories and run multi-policy statistical comparison.

Two statistical paths, selected by whether max_sample_size_per_policy is set:

  DEFAULT (max_sample_size=None) — Barnard exact, all data:
    scipy.stats.barnard_exact on the full pooled boolean success arrays with Bonferroni
    correction across pairwise comparisons. Uses every episode, runs in milliseconds,
    no synthesis required. The right tool when all rollouts are already collected offline.

  EXPLICIT max_sample_size — Sequential STEP Barnard test:
    compare_success_and_get_cld from the sequentialized_barnard_tests notebook:
    https://gist.github.com/HarukiNishimura-TRI/f4820826e7d93af5a5c9452cc6dd44ce
    Designed for online evaluation: check significance after each rollout and stop
    early when the budget is exhausted. Requires one-time O(n²) STEP policy synthesis;
    only processes the first max_sample_size episodes after shuffling.
    Use when n_max is small (≤500) and already cached, or when you genuinely need the
    sequential stopping guarantee.

Both paths accept boolean success arrays and produce CLD letters in the same format.
shuffle=True is recommended whenever pooling episodes from multiple sources.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EvalRun:
    """One leaf eval dir: a single checkpoint × seed combination."""

    path: Path
    successes: np.ndarray  # bool array, shape (n_episodes,)

    @property
    def n_episodes(self) -> int:
        return len(self.successes)

    @property
    def success_rate(self) -> float:
        return float(self.successes.mean()) if self.n_episodes > 0 else 0.0

    @property
    def name(self) -> str:
        return self.path.name


@dataclass
class PolicyGroupSpec:
    """Specification for one policy condition: a label and a list of paths.

    Each path may be either:
    - A leaf eval dir containing eval_log.json directly (single checkpoint/seed), or
    - A parent dir whose immediate children contain eval_log.json (top-k checkpoints
      from one training run).
    """

    label: str
    dirs: list[Path]


@dataclass
class ComparisonResult:
    group_labels: list[str]
    success_arrays: list[np.ndarray]   # full pooled boolean arrays, one per group
    success_rates: list[float]
    n_episodes: list[int]
    cld_letters: list[str]
    global_confidence_level: float
    method: str          # "barnard_exact" or "barnard_sequential"
    max_sample_size: int | None
    shuffle: bool
    per_group_runs: list[list[EvalRun]]


# ── data loading ──────────────────────────────────────────────────────────────

def _load_eval_log(eval_dir: Path) -> dict[str, Any]:
    log_path = eval_dir / "eval_log.json"
    if not log_path.exists():
        raise FileNotFoundError(f"eval_log.json not found in {eval_dir}")
    return json.loads(log_path.read_text())


def _extract_successes(log_data: dict[str, Any], source: Path) -> np.ndarray:
    pairs = sorted(
        (k, v) for k, v in log_data.items() if k.startswith("test/sim_max_reward_")
    )
    if pairs:
        return np.array([bool(v > 0.5) for _, v in pairs], dtype=bool)
    warnings.warn(
        f"No test/sim_max_reward_* keys found in {source}/eval_log.json; skipping run.",
        stacklevel=3,
    )
    return np.array([], dtype=bool)


def _score_from_ckpt_name(name: str) -> float:
    """Parse test_mean_score from a checkpoint dir name like 'epoch=0300-test_mean_score=0.640'."""
    import re
    m = re.search(r"test_mean_score=([0-9.]+)", name)
    return float(m.group(1)) if m else 0.0


def _collect_leaf_dirs(path: Path, top_k: int | None = None) -> list[Path]:
    """Return leaf eval dirs (directly containing eval_log.json) under path.

    When top_k is set and path contains checkpoint subdirs, only the top_k
    subdirs by test_mean_score (parsed from dirname) are returned. This matches
    the standard eval methodology where only the top-5 checkpoints are evaluated.
    """
    path = path.resolve()
    if (path / "eval_log.json").exists():
        return [path]
    leaves = sorted(p.parent for p in path.glob("*/eval_log.json"))
    if not leaves:
        warnings.warn(f"No eval_log.json found under {path}", stacklevel=3)
        return []
    if top_k is not None and len(leaves) > top_k:
        leaves = sorted(leaves, key=lambda p: _score_from_ckpt_name(p.name), reverse=True)[:top_k]
    return leaves


def load_eval_run(eval_dir: Path) -> EvalRun | None:
    log = _load_eval_log(eval_dir)
    successes = _extract_successes(log, eval_dir)
    if len(successes) == 0:
        return None
    return EvalRun(path=eval_dir, successes=successes)


def load_policy_group(
    spec: PolicyGroupSpec,
    top_k_checkpoints: int | None = 5,
) -> tuple[list[EvalRun], np.ndarray]:
    """Load all runs for a group, capping each run at top_k_checkpoints by test_mean_score.

    Defaults to top_k_checkpoints=5 to match the standard evaluation methodology
    (top-5 checkpoint evaluation). Pass None to use all available checkpoints.
    """
    runs: list[EvalRun] = []
    for d in spec.dirs:
        for leaf in _collect_leaf_dirs(Path(d), top_k=top_k_checkpoints):
            try:
                run = load_eval_run(leaf)
            except FileNotFoundError as exc:
                warnings.warn(str(exc), stacklevel=2)
                continue
            if run is not None:
                runs.append(run)
    if not runs:
        raise ValueError(f"No usable eval data found for group '{spec.label}'")
    return runs, np.concatenate([r.successes for r in runs])


# ── statistical tests ─────────────────────────────────────────────────────────

def _cld_from_pairs(
    significant_pairs: list[tuple[str, str]],
    labels: list[str],
    rates: list[float],
) -> dict[str, str]:
    from sequentialized_barnard_tests.tools.plotting import compact_letter_display

    sorted_labels = [lbl for _, lbl in sorted(zip(rates, labels), reverse=True)]
    letters = compact_letter_display(significant_pairs, sorted_labels)
    return {lbl: letter for lbl, letter in zip(sorted_labels, letters)}


# scipy.stats.barnard_exact enumerates O(n²) tables — only viable for small n.
# For large n, chi2_contingency (Pearson chi-squared) is asymptotically equivalent
# and runs in microseconds regardless of sample size.
_BARNARD_EXACT_MAX_N = 500


def _pvalue_for_pair(ki: int, ni: int, kj: int, nj: int) -> float:
    """Two-sided p-value for two independent proportions.

    Uses Barnard exact test for ni, nj ≤ _BARNARD_EXACT_MAX_N (small-sample accuracy),
    and Pearson chi-squared for larger n (asymptotically equivalent, runs in µs).
    """
    if ni <= _BARNARD_EXACT_MAX_N and nj <= _BARNARD_EXACT_MAX_N:
        from scipy.stats import barnard_exact
        return float(barnard_exact([[ki, ni - ki], [kj, nj - kj]], alternative="two-sided").pvalue)
    else:
        from scipy.stats import chi2_contingency
        table = [[ki, ni - ki], [kj, nj - kj]]
        chi2, p, *_ = chi2_contingency(table, correction=False)
        return float(p)


def _barnard_exact(
    success_arrays: list[np.ndarray],
    labels: list[str],
    global_confidence_level: float,
    verbose: bool,
) -> dict[str, str]:
    """Pairwise proportion tests on all data with Bonferroni correction (offline default).

    Uses Barnard exact for n ≤ 500 per group, chi-squared for larger n.
    """
    n = len(labels)
    n_comparisons = n * (n - 1) // 2
    per_alpha = (1 - global_confidence_level) / max(n_comparisons, 1)
    counts = [(int(a.sum()), len(a)) for a in success_arrays]
    rates = [k / m for k, m in counts]
    max_n = max(m for _, m in counts)
    test_name = "Barnard exact" if max_n <= _BARNARD_EXACT_MAX_N else "chi-squared (large n)"

    if verbose:
        print("Statistical Test Specs:")
        print(f"  Method: {test_name} (fixed-n, all data)")
        print(f"  Global Confidence: {global_confidence_level:.5f}")
        print(f"    ({1 - per_alpha:.5f} per comparison, Bonferroni)")
        print(f"  N per group: {[m for _, m in counts]}\n")

    sig_pairs: list[tuple[str, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            ki, ni = counts[i]
            kj, nj = counts[j]
            pval = _pvalue_for_pair(ki, ni, kj, nj)
            sig = pval < per_alpha
            if verbose:
                tag = "SIGNIFICANT" if sig else "n.s."
                print(f"  {labels[i]} vs {labels[j]}: p={pval:.4f}  [{tag}]")
            if sig:
                sig_pairs.append((labels[i], labels[j]))

    cld = _cld_from_pairs(sig_pairs, labels, rates)
    if verbose:
        print("\nStatistical Test Results (Compact Letter Display):")
        for lbl in sorted(labels, key=lambda l: -rates[labels.index(l)]):
            k, m = counts[labels.index(lbl)]
            print(f"  CLD for {lbl}: {cld[lbl]}\n    Success Rate {k} / {m} = {k/m:.3f}")
        print()
    return cld


def _barnard_sequential(
    success_arrays: list[np.ndarray],
    labels: list[str],
    global_confidence_level: float,
    max_sample_size: int,
    shuffle: bool,
    rng: np.random.Generator,
    verbose: bool,
) -> dict[str, str]:
    """Sequential STEP Barnard — for online evaluation with early stopping."""
    from sequentialized_barnard_tests.tools.plotting import compare_success_and_get_cld

    if not shuffle and any(len(runs) > 1 for runs in success_arrays):
        warnings.warn(
            "shuffle=False with pooled multi-source data violates i.i.d. Set shuffle=True.",
            stacklevel=3,
        )
    any_exceeds = any(len(a) > max_sample_size for a in success_arrays)
    with warnings.catch_warnings():
        if any_exceeds:
            warnings.filterwarnings("ignore", message=".*exceeded.*evals.*", category=UserWarning)
        return compare_success_and_get_cld(
            labels, success_arrays, global_confidence_level,
            max_sample_size, shuffle, rng=rng, verbose=verbose,
        )


# ── public API ────────────────────────────────────────────────────────────────

METHODS = ("barnard_exact", "step")


def run_comparison(
    group_specs: list[PolicyGroupSpec],
    global_confidence_level: float = 0.95,
    method: str = "barnard_exact",
    max_sample_size: int | None = None,
    top_k_checkpoints: int | None = 5,
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
    verbose: bool = True,
) -> ComparisonResult:
    """Compare policy groups statistically and produce CLD letters.

    Args:
        group_specs: Policy groups to compare (each pools multiple eval dirs).
        global_confidence_level: Familywise confidence level across all pairwise tests.
        method: Statistical test to use. One of:
            "barnard_exact" (default) — scipy.stats.barnard_exact on all pooled data
                with Bonferroni correction. Uses every episode, runs in milliseconds,
                no synthesis. The right choice when all rollouts are already collected.
            "step" — Sequential STEP Barnard test (sequentialized_barnard_tests).
                Requires max_sample_size and one-time O(n²) policy synthesis. Only
                appropriate for small n (≤500 already cached) or when the sequential
                stopping guarantee is needed.
        max_sample_size: Required when method="step". Episodes beyond this are dropped
            after shuffle. Ignored when method="barnard_exact".
        top_k_checkpoints: Per-run checkpoint cap selected by test_mean_score descending.
            Defaults to 5 to match the standard top-5 evaluation methodology. Pass None
            to use all available checkpoints.
        shuffle: Shuffle episodes before testing. Only used by method="step" to restore
            exchangeability when pooling from multiple sources.
        rng: RNG for step-method shuffling; defaults to seed=42.
        verbose: Print test progress to stdout.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if rng is None:
        rng = np.random.default_rng(42)

    per_group_runs: list[list[EvalRun]] = []
    success_arrays: list[np.ndarray] = []
    for spec in group_specs:
        runs, pooled = load_policy_group(spec, top_k_checkpoints=top_k_checkpoints)
        per_group_runs.append(runs)
        success_arrays.append(pooled)

    labels = [s.label for s in group_specs]

    if method == "barnard_exact":
        cld = _barnard_exact(success_arrays, labels, global_confidence_level, verbose)
    else:  # step
        if max_sample_size is None:
            max_sample_size = max(len(a) for a in success_arrays)
            if verbose:
                print(
                    f"[compare_policies] method=step but max_sample_size not set; "
                    f"using {max_sample_size} (total pooled). "
                    f"Synthesis time scales O(n²) — set max_sample_size_per_policy "
                    "explicitly if this is large."
                )
        elif verbose:
            total = max(len(a) for a in success_arrays)
            if total > max_sample_size:
                print(
                    f"[compare_policies] method=step: max_sample_size={max_sample_size}, "
                    f"total pooled={total}. Episodes beyond {max_sample_size} dropped after "
                    "shuffle. Use method=barnard_exact to use all data."
                )
        cld = _barnard_sequential(
            success_arrays, labels, global_confidence_level,
            max_sample_size, shuffle, rng, verbose,
        )

    return ComparisonResult(
        group_labels=labels,
        success_arrays=success_arrays,
        success_rates=[float(a.mean()) for a in success_arrays],
        n_episodes=[len(a) for a in success_arrays],
        cld_letters=[cld[lbl] for lbl in labels],
        global_confidence_level=global_confidence_level,
        method=method,
        max_sample_size=max_sample_size,  # None for barnard_exact
        shuffle=shuffle,
        per_group_runs=per_group_runs,
    )


def print_comparison_table(result: ComparisonResult) -> None:
    width = max(len(lbl) for lbl in result.group_labels) + 2
    print()
    print("=" * 65)
    print(f"  method={result.method}  confidence={result.global_confidence_level:.0%}")
    print("-" * 65)
    print(f"  {'Policy':<{width}} {'N':>7}  {'Rate':>7}  {'CLD':>4}")
    print("-" * 65)
    for lbl, n, rate, cld in zip(
        result.group_labels, result.n_episodes, result.success_rates, result.cld_letters
    ):
        print(f"  {lbl:<{width}} {n:>7}  {rate:>7.1%}  {cld:>4}")
    print("=" * 65)
    print("  Groups sharing a CLD letter are NOT significantly different.\n")
