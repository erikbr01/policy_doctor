"""Quick side-by-side comparison of a single UMAP config across environments.

Run in each env (from monorepo root):
    conda activate cupid && python policy_doctor/tests/behaviors/test_umap_env_compare.py
    conda activate cupid_clustering && python policy_doctor/tests/behaviors/test_umap_env_compare.py
"""

import os
import sys
import time

if "NUMBA_THREADING_LAYER" not in os.environ:
    os.environ["NUMBA_THREADING_LAYER"] = "omp"

import numpy as np
import numba
import llvmlite
from sklearn.preprocessing import StandardScaler


def warmup():
    import umap
    tiny = np.random.randn(50, 10).astype(np.float32)
    umap.UMAP(n_components=2, n_jobs=1).fit_transform(tiny)


def bench(data, n_components):
    import umap
    scaled = StandardScaler().fit_transform(data)
    t0 = time.perf_counter()
    umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        n_jobs=32,
        low_memory=False,
        verbose=False,
    ).fit_transform(scaled)
    return time.perf_counter() - t0


def main():
    print(f"Python {sys.version}")
    print(f"numba {numba.__version__}, llvmlite {llvmlite.__version__}")
    print(f"numpy {np.__version__}")
    print(f"Threading env: {os.environ.get('NUMBA_THREADING_LAYER', 'not set')}")
    print()

    warmup()
    print(f"Active threading layer: {numba.threading_layer()}")
    print(f"Threads: {numba.get_num_threads()}")
    print()

    shapes = [
        (2000, 1000),
        (5000, 1000),
        (5000, 5000),
    ]
    n_comps = [50, 100]

    print(f"{'shape':>16s} | ", end="")
    for nc in n_comps:
        print(f"{'n_comp=' + str(nc):>12s}", end="  ")
    print(f" | {'100/50 ratio':>12s}")
    print("-" * 65)

    for n_samples, n_features in shapes:
        data = np.random.default_rng(42).standard_normal(
            (n_samples, n_features)
        ).astype(np.float32)

        times = {}
        for nc in n_comps:
            times[nc] = bench(data, nc)

        ratio = times[100] / times[50]
        print(
            f"{n_samples:>7d} x {n_features:<5d} | "
            f"{times[50]:>10.2f}s  {times[100]:>10.2f}s  | "
            f"{ratio:>11.2f}x"
        )


if __name__ == "__main__":
    main()
