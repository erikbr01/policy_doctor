"""Benchmark: IV vs policy_doctor UMAP code paths.

Replicates the exact UMAP call from each codebase on identical random data
and compares wall-clock time across multiple dataset sizes.

Run from monorepo root:
    python policy_doctor/tests/behaviors/test_umap_iv_vs_pd_bench.py
"""

import os
import time

if "NUMBA_THREADING_LAYER" not in os.environ:
    os.environ["NUMBA_THREADING_LAYER"] = "omp"

import numpy as np
from sklearn.preprocessing import StandardScaler


def umap_iv_style(data: np.ndarray, n_components: int) -> np.ndarray:
    """Exact IV code path from render_clustering._apply_dimensionality_reduction."""
    import umap

    scaled = StandardScaler().fit_transform(data)
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        n_jobs=32,
        low_memory=False,
        verbose=False,
    )
    return reducer.fit_transform(scaled)


def umap_pd_style(data: np.ndarray, n_components: int) -> np.ndarray:
    """PD code path AFTER low_memory fix (current clustering.reduce_dimensions)."""
    import umap

    scaled = StandardScaler().fit_transform(data)
    reducer = umap.UMAP(
        n_components=n_components,
        n_jobs=32,
        low_memory=False,
    )
    return reducer.fit_transform(scaled)


def umap_pd_style_old(data: np.ndarray, n_components: int) -> np.ndarray:
    """PD code path BEFORE low_memory fix (low_memory defaults to True)."""
    import umap

    scaled = StandardScaler().fit_transform(data)
    reducer = umap.UMAP(
        n_components=n_components,
        n_jobs=32,
    )
    return reducer.fit_transform(scaled)


def umap_pd_style_explicit(data: np.ndarray, n_components: int) -> np.ndarray:
    """PD path but with all IV params made explicit (isolates param defaults vs code)."""
    import umap

    scaled = StandardScaler().fit_transform(data)
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        n_jobs=32,
        low_memory=False,
        verbose=False,
    )
    return reducer.fit_transform(scaled)


def time_fn(fn, *args, **kwargs) -> float:
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - t0


def warmup():
    """Run a tiny UMAP to trigger Numba JIT compilation before benchmarking."""
    import umap

    tiny = np.random.randn(50, 10).astype(np.float32)
    umap.UMAP(n_components=2, n_jobs=1).fit_transform(tiny)


def run_benchmark():
    print("=" * 80)
    print("UMAP Benchmark: IV vs Policy Doctor code paths")
    print("=" * 80)

    print("\nWarming up (Numba JIT)...")
    warmup()
    print("Done.\n")

    sizes = [
        (500,   200),
        (1000,  500),
        (2000,  1000),
        (5000,  1000),
        (5000,  5000),
        (10000, 1000),
    ]
    n_comp_iv = 50
    n_comp_pd = 100

    rng = np.random.default_rng(42)

    header = (
        f"{'n_samples':>9s} x {'n_feat':>6s} | "
        f"{'IV(50)':>8s}  {'PD_old(100)':>11s}  {'PD_new(100)':>11s}  "
        f"{'PD_new(50)':>10s}  {'PD_explicit(100)':>16s} | "
        f"{'old/IV':>6s}  {'new/IV':>6s}  {'new50/IV':>8s}  {'old/new':>7s}"
    )
    print(header)
    print("-" * len(header))

    for n_samples, n_features in sizes:
        data = rng.standard_normal((n_samples, n_features)).astype(np.float32)

        t_iv = time_fn(umap_iv_style, data, n_comp_iv)
        t_pd_old = time_fn(umap_pd_style_old, data, n_comp_pd)
        t_pd_new = time_fn(umap_pd_style, data, n_comp_pd)
        t_pd_new50 = time_fn(umap_pd_style, data, n_comp_iv)
        t_pd_explicit = time_fn(umap_pd_style_explicit, data, n_comp_pd)

        row = (
            f"{n_samples:>9d} x {n_features:>6d} | "
            f"{t_iv:>7.2f}s  {t_pd_old:>10.2f}s  {t_pd_new:>10.2f}s  "
            f"{t_pd_new50:>9.2f}s  {t_pd_explicit:>15.2f}s | "
            f"{t_pd_old / t_iv:>5.2f}x  {t_pd_new / t_iv:>5.2f}x  "
            f"{t_pd_new50 / t_iv:>7.2f}x  {t_pd_old / t_pd_new:>6.2f}x"
        )
        print(row)

    print()
    print("Legend:")
    print("  IV(50)            = IV code path: low_memory=False, all params explicit, n_comp=50")
    print("  PD_old(100)       = PD BEFORE fix: low_memory=True (default), n_comp=100")
    print("  PD_new(100)       = PD AFTER fix: low_memory=False, n_comp=100")
    print("  PD_new(50)        = PD AFTER fix: low_memory=False, n_comp=50 (apples-to-apples)")
    print("  PD_explicit(100)  = PD with all IV params explicit, n_comp=100")
    print()
    print("  old/IV  = slowdown of original PD vs IV")
    print("  new/IV  = slowdown of fixed PD vs IV")
    print("  new50/IV= fixed PD at n_comp=50 vs IV (pure code path diff)")
    print("  old/new = speedup from the low_memory fix alone")


if __name__ == "__main__":
    run_benchmark()
