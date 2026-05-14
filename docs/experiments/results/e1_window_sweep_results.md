# E1 Window-Parameter Sweep Results

**Status**: In progress — infembed complete (48/48), state partial, state_action pending. Last updated 2026-05-05.

## Architecture note

This sweep uses a restructured pipeline: UMAP is fitted **once per representation on per-timestep features** (trunk), and window aggregation + kmeans are applied **after** UMAP (branches). This differs from all prior E1 results (F1–F10), where windowing happened *before* UMAP. The two architectures are not directly comparable.

Old pipeline (F1–F10): per-timestep features → **window/aggregate** → UMAP → kmeans  
New pipeline (this sweep): per-timestep features → **UMAP** → window/aggregate → kmeans

UMAP dimensionality: 50 (capped at feature_dim − 1; old pipeline used 100).  
Window config (w, s) now controls how UMAP-reduced timestep embeddings are aggregated into slices, not what UMAP sees.

**Key early finding**: InfEmbed K=10 clean acc drops from 0.519 (old pipeline, 768²) to 0.367 (new pipeline, 512²). Some of this is the architecture change and some is the lower composite resolution. The old pipeline accidentally benefited from UMAP seeing temporally-smoothed windowed features; the new pipeline is more principled but loses that smoothing.

UMAP fitted once per representation on per-timestep features.  Window config and K are applied post-UMAP.

Representations: infembed, state, state_action  
Window (w, s): [(1, 1), (2, 2), (3, 2), (5, 5)]  
K: [5, 10, 15, 20]  
Seeds: [42, 43, 44]  
Protocol: n_example=3, n_query=3, n_reps=3, composite=512², global_episode_disjoint

Clean accuracy = tier1_global queries. Values are mean ± std across seeds.

## K = 5  (chance = 0.200)

| Rep | (w, s) | Mean clean | ±std | Ratio above chance | n_seeds |
|---|---|---|---|---|---|
| infembed | (5, 5) | 0.644 | ±0.031 | 3.2× | 3/3 |
| infembed | (1, 1) | 0.600 | ±0.054 | 3.0× | 3/3 |
| state | (1, 1) | 0.489 | ±0.063 | 2.4× | 3/3 |
| infembed | (2, 2) | 0.444 | ±0.083 | 2.2× | 3/3 |
| infembed | (3, 2) | 0.444 | ±0.083 | 2.2× | 3/3 |

## K = 10  (chance = 0.100)

| Rep | (w, s) | Mean clean | ±std | Ratio above chance | n_seeds |
|---|---|---|---|---|---|
| infembed | (5, 5) | 0.367 | ±0.072 | 3.7× | 3/3 |
| state | (1, 1) | 0.367 | ±0.033 | 3.7× | 2/3 |
| infembed | (3, 2) | 0.311 | ±0.057 | 3.1× | 3/3 |
| infembed | (1, 1) | 0.300 | ±0.054 | 3.0× | 3/3 |
| infembed | (2, 2) | 0.300 | ±0.054 | 3.0× | 3/3 |

## K = 15  (chance = 0.067)

| Rep | (w, s) | Mean clean | ±std | Ratio above chance | n_seeds |
|---|---|---|---|---|---|
| infembed | (1, 1) | 0.310 | ±0.051 | 4.6× | 3/3 |
| infembed | (5, 5) | 0.302 | ±0.056 | 4.5× | 3/3 |
| infembed | (2, 2) | 0.286 | ±0.019 | 4.3× | 3/3 |
| infembed | (3, 2) | 0.206 | ±0.030 | 3.1× | 3/3 |

## K = 20  (chance = 0.050)

| Rep | (w, s) | Mean clean | ±std | Ratio above chance | n_seeds |
|---|---|---|---|---|---|
| infembed | (5, 5) | 0.298 | ±0.089 | 6.0× | 3/3 |
| infembed | (2, 2) | 0.263 | ±0.029 | 5.3× | 3/3 |
| infembed | (1, 1) | 0.246 | ±0.014 | 4.9× | 3/3 |
| infembed | (3, 2) | 0.216 | ±0.022 | 4.3× | 3/3 |

## Best window config per (rep, K)

| Rep | K | Best (w,s) | Mean clean | Ratio |
|---|---|---|---|---|
| infembed | 5 | (5, 5) | 0.644 | 3.2× |
| infembed | 10 | (5, 5) | 0.367 | 3.7× |
| infembed | 15 | (1, 1) | 0.310 | 4.6× |
| infembed | 20 | (5, 5) | 0.298 | 6.0× |
| state | 5 | (1, 1) | 0.489 | 2.4× |
| state | 10 | (1, 1) | 0.367 | 3.7× |
