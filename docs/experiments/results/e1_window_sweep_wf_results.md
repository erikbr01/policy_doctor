# E1 Window-Parameter Sweep Results — Window-First Architecture

**Architecture**: window/aggregate first → UMAP (100D) → kmeans. Matches F1–F10 methodology.  
Compare to `e1_window_sweep_results.md` (timestep_first, 50D UMAP).

**Status**: COMPLETE — 48/48. Last updated 2026-05-05.

**Headlines:**
- `(w=3, s=2)` is the best window at K=5 (0.556), K=10 (0.469), and K=20 (0.281).
- `(w=5, s=5)` wins at K=15 (0.356, 5.3×).
- `(w=1, s=1)` single-timestep windows are consistently worst at K=5/10.
- The original baseline `(w=5, s=2)` — not tested here — gave 0.481 at K=10; `(3,2)` at 0.469 is the closest match in this sweep.

UMAP fitted once per representation on per-timestep features.  Window config and K are applied post-UMAP.

Representations: infembed  
Window (w, s): [(1, 1), (2, 2), (3, 2), (5, 5)]  
K: [5, 10, 15, 20]  
Seeds: [42, 43, 44]  
Protocol: n_example=3, n_query=3, n_reps=3, composite=512², global_episode_disjoint

Clean accuracy = tier1_global queries. Values are mean ± std across seeds.

## K = 5  (chance = 0.200)

| Rep | (w, s) | Mean clean | ±std | Ratio above chance | n_seeds |
|---|---|---|---|---|---|
| infembed | (3, 2) | 0.556 | ±0.031 | 2.8× | 3/3 |
| infembed | (5, 5) | 0.511 | ±0.063 | 2.6× | 3/3 |
| infembed | (1, 1) | 0.422 | ±0.083 | 2.1× | 3/3 |
| infembed | (2, 2) | 0.422 | ±0.031 | 2.1× | 3/3 |

## K = 10  (chance = 0.100)

| Rep | (w, s) | Mean clean | ±std | Ratio above chance | n_seeds |
|---|---|---|---|---|---|
| infembed | (3, 2) | 0.469 | ±0.046 | 4.7× | 3/3 |
| infembed | (5, 5) | 0.411 | ±0.016 | 4.1× | 3/3 |
| infembed | (2, 2) | 0.370 | ±0.052 | 3.7× | 3/3 |
| infembed | (1, 1) | 0.244 | ±0.016 | 2.4× | 3/3 |

## K = 15  (chance = 0.067)

| Rep | (w, s) | Mean clean | ±std | Ratio above chance | n_seeds |
|---|---|---|---|---|---|
| infembed | (5, 5) | 0.356 | ±0.018 | 5.3× | 3/3 |
| infembed | (1, 1) | 0.310 | ±0.019 | 4.6× | 3/3 |
| infembed | (2, 2) | 0.310 | ±0.085 | 4.6× | 3/3 |
| infembed | (3, 2) | 0.294 | ±0.022 | 4.4× | 3/3 |

## K = 20  (chance = 0.050)

| Rep | (w, s) | Mean clean | ±std | Ratio above chance | n_seeds |
|---|---|---|---|---|---|
| infembed | (3, 2) | 0.281 | ±0.014 | 5.6× | 3/3 |
| infembed | (5, 5) | 0.250 | ±0.014 | 5.0× | 3/3 |
| infembed | (1, 1) | 0.222 | ±0.022 | 4.4× | 3/3 |
| infembed | (2, 2) | 0.211 | ±0.025 | 4.2× | 3/3 |

## Best window config per (rep, K)

| Rep | K | Best (w,s) | Mean clean | Ratio |
|---|---|---|---|---|
| infembed | 5 | (3, 2) | 0.556 | 2.8× |
| infembed | 10 | (3, 2) | 0.469 | 4.7× |
| infembed | 15 | (5, 5) | 0.356 | 5.3× |
| infembed | 20 | (3, 2) | 0.281 | 5.6× |
