# Behavior-Graph Data Support

For each node in a policy-embedding behavior graph, measure how well-supported it is by the *training* distribution. The hypothesis: nodes a policy visits only during failed rollouts often correspond to behaviour states the policy was never trained on — they are out-of-distribution drift. Coloring the graph by data support surfaces these clusters as a diagnostic.

The whole thing is scoped to **`influence_source == "policy_emb"` clusterings** — InfEmbed / TRAK / pi05_activations clusterings don't have a natural shared-space-with-demos interpretation and the pipeline step no-ops on them.

---

## Architecture overview

```
OFFLINE — one time per (task, seed, layer)
──────────────────────────────────────────
  Policy checkpoint + training HDF5
    └─ compute_policy_embeddings_demos     ← mimicgen_torch2 env, GPU
          (mirrors compute_policy_embeddings.py, but iterates the
           diffusion_policy training dataset instead of rollout pickles)
       →  <train_dir>/policy_embeddings_demos/<layer>.npz
            "demo_embeddings"  (N_train_samples, D)  float32
            "episode_lengths"  per *training* demo

  Per clustering (rollout-side, already exists):
    └─ <eval_dir>/policy_embeddings/<layer>.npz
            "rollout_embeddings"  (N_rollout_timesteps, D)  float32

OFFLINE — per clustering
────────────────────────
  policy_emb clustering + rollout npz + demo npz
    └─ compute_data_support                ← policy_doctor env
          1. window-aggregate both sides with the manifest's
             (window_width, stride, aggregation)
          2. fit a JOINT UMAP on concat(demo_windows, rollout_windows)
          3. one BallTree on demo-side projections
          4. evaluate every selected metric in a single pass
          5. bucket per-slice values by cluster, summarise
       →  <clustering_dir>/data_support.json
       →  <clustering_dir>/joint_umap.joblib       (optional)

ONLINE — Streamlit graph demo
─────────────────────────────
  data_support.json + behavior graph
    └─ "Data support" color mode for trees + Markov
    └─ Sidebar selectors: metric + summary statistic
    └─ Distributional view in Node/Transition Inspector
```

---

## What gets measured

The shared space is a **fresh UMAP** fit jointly on aggregated demo + rollout windows (≈ tens of thousands of points; takes ~2 min for the transport_mh full sweep). Reusing the rollout-only UMAP from `clustering_models.pkl` would be cheaper but the demo windows weren't seen during that fit, so any OOD demos would warp into nonsense.

For each rollout window in joint space we evaluate every registered metric against the demo cloud:

| Metric | Returns per slice | Per-cluster reading |
|---|---|---|
| `count_in_radius` | # demos within Euclidean `radius` | median ≈ local density |
| `binary_coverage` | 1 if ≥1 demo within `radius`, else 0 | `mean` = **coverage fraction** |
| `knn_mean_distance` | mean dist to k nearest demos | lower = better-supported |
| `knn_max_distance` | dist to k-th nearest demo | worst-case proximity |
| `kde_log_density` | Gaussian-KDE log p_demo at the slice | higher = better-supported |

The metric registry (`policy_doctor.behaviors.data_support`) makes adding a new metric ≤10 lines — decorate a `MetricContext → np.ndarray` function with `@register_metric("name")` and add the name to `data_support_metrics` in your Hydra config. All metrics share one BallTree fit, so the marginal cost of "compute one more metric" is just one batched query.

For every selected metric we persist:

- `raw`: the per-slice array (length = #slices in the cluster) — enables histograms / violins later
- `median`, `mean`, `q10`, `q90`, `n_slices`

…per cluster ID. Terminal nodes (SUCCESS / FAILURE / START / END) are excluded.

---

## File contract

```jsonc
// <clustering_dir>/data_support.json
{
  "_config": {
    "radius": 1.0,
    "metrics": ["count_in_radius", "binary_coverage", "knn_mean_distance", ...],
    "knn_k": 10,
    "kde_bandwidth": "scott",
    "umap_n_components": 10,
    "umap_n_neighbors": 15,
    "umap_random_state": 0,
    "umap_normalize": "standard",
    "umap_refit": true,
    "demo_embeddings_path": "...",
    "rollout_embeddings_path": "...",
    "layer": "bottleneck_plan_t0",
    "n_demo_windows": 60096,
    "n_rollout_windows": 3836
  },
  "metrics": {
    "count_in_radius": {
      "0": {"raw": [...], "median": 27.5, "mean": 31.2, "q10": 8.0, "q90": 53.0, "n_slices": 510},
      "8": {"raw": [...], "median": 0.0,  "mean": 0.0,  "q10": 0.0, "q90": 0.0,  "n_slices": 33},
      ...
    },
    "binary_coverage": {...},
    ...
  }
}
```

The Streamlit demo loads this lazily — if the file is missing or malformed the graph still works, just without the "Data support" color option. Co-located `joint_umap.joblib` is only written when `data_support_save_joint_umap=true` (off by default — UMAP models are large and rarely useful after the metrics are computed).

---

## Running it

```bash
# (1) Demo embeddings — once per (train_dir, layer)
conda run -n mimicgen_torch2 python third_party/cupid/compute_policy_embeddings_demos.py \
  --train_dir=/path/to/train_dir \
  --train_ckpt=latest \
  --layer=bottleneck_plan_t0 \
  --batch_size=128 --device=cuda:0

# (2) Data support for a single clustering — one-off trial path
conda activate policy_doctor && python scripts/trial_data_support.py \
  --clustering_dir=data/demo_sweep/transport_mh_jan28/.../policy_emb_bottleneck_plan_t0_seed0_kmeans_k10 \
  --radius=1.0

# (2') Full pipeline path (every policy_emb clustering in a run)
python -m policy_doctor.scripts.run_pipeline \
  steps=[compute_policy_embeddings_demos,compute_data_support] \
  data_support_radius=1.0
```

The pipeline step honours the standard `skip_if_done` semantics — re-running won't re-extract embeddings unless `--overwrite` is set.

---

## Hydra knobs

| Key | Default | Notes |
|---|---|---|
| `data_support_radius` | `0.5` | Euclidean radius in *joint* UMAP space. Sensitive to UMAP scale; values 0.5–2.0 are typical for `umap_n_components=10`. |
| `data_support_metrics` | all five | Subset list to skip metrics you don't need. |
| `data_support_knn_k` | `10` | k for kNN distance metrics. |
| `data_support_kde_bandwidth` | `"scott"` | Float, or `"scott"` / `"silverman"` rules-of-thumb computed on the demo cloud. |
| `data_support_umap_n_components` | `10` | Joint UMAP output dim. Stay above 2 so density isn't crushed to a line. |
| `data_support_umap_n_neighbors` | `15` | UMAP local connectivity. |
| `data_support_umap_random_state` | `0` | Reproducibility. UMAP with a fixed seed forces single-threaded fit (umap-learn warning is expected). |
| `data_support_umap_normalize` | `"standard"` | Pre-UMAP scaling: `"standard"`, `"l2"`, or `"none"`. |
| `data_support_save_joint_umap` | `false` | Persist the fitted UMAP next to `data_support.json`. |
| `policy_emb_demos_batch_size` | `128` | Demo-extraction batch size. |
| `policy_emb_demos_include_holdout` | `false` | When `true`, the demo extractor also embeds holdout demos and concatenates. Default off because "data support" usually means "support from the policy's *training* distribution". |

---

## Streamlit demo (`Graph Demo` page)

When a clustering's `data_support.json` is present:

- The **Color nodes by** dropdown gains a "**Data support (training-demo density)**" option.
- A **Metric** + **Summary stat** pair of selectors appears below the color dropdown — flip between `count_in_radius` / `knn_mean_distance` / etc. and between `median` / `mean` / `q10` / `q90` without re-running anything.
- Sequential palette: ColorBrewer YlGn by default (pale = under-supported), Viridis when the colorblind toggle is on.
- Distance metrics (`knn_mean_distance`, `knn_max_distance`) are **inverted** before colouring so that "saturated = better-supported" stays consistent across all metrics.
- The Node / Transition Inspector adds a per-slice **distribution view** when a node is selected — see how broad the training-data support is around that cluster, not just the median.

If the file is absent (e.g. InfEmbed clustering, or a policy_emb run that pre-dates this feature), the "Data support" option simply isn't shown.

---

## Critical caveats

1. **Mask-aware demo selection.** The training dataset's `dataset_mask_kwargs.train_ratio=0.64` + `uniform_quality=True` splits the HDF5 into train/val/holdout. `compute_policy_embeddings_demos` reads the diffusion_policy `SequenceSampler.indices` to attribute samples back to demos via `np.searchsorted(replay_buffer.episode_ends, sampler.indices[:, 0])` — so the `episode_lengths` array holds **only the training demos** that contributed any samples. This is what the policy actually learned from. Holdout demos are *not* counted.

2. **Window count must match.** The step asserts `len(rollout_windows) == len(cluster_labels)` and refuses to write `data_support.json` if they diverge. The most common cause is the clustering manifest's `(window_width, stride, aggregation)` not matching the cfg you're running with — `run_clustering` now persists those fields in the manifest, so post-step clusterings stay self-describing.

3. **Joint UMAP cost.** ~115 s for ~66k demo+rollout windows in 512D → 10D on a workstation. Cache the result by leaving `skip_if_done=True` (default).

4. **Out-of-scope (follow-ups).**
   - Per-tree-path scoring (`get_rollout_slices_for_paths` with full prefix matching) — well-defined but ~5× the implementation.
   - Multi-radius pre-computation with an interactive slider — currently a single configured radius per run.
   - `pi05_activations` source — needs a separate demo extractor that hooks the pi0.5 backbone.

---

## Files

| File | Role |
|---|---|
| `policy_doctor/behaviors/data_support.py` | Joint-UMAP fit, metric registry, per-cluster aggregation. Pure Python, no Streamlit / sim deps. |
| `third_party/cupid/compute_policy_embeddings_demos.py` | Demo-side activation extraction (mimicgen_torch2 env). Sibling of `compute_policy_embeddings.py`. |
| `policy_doctor/curation_pipeline/steps/compute_policy_embeddings_demos.py` | Pipeline step wrapper. |
| `policy_doctor/curation_pipeline/steps/compute_data_support.py` | Pipeline step that loads the upstream artifacts and writes `data_support.json`. |
| `scripts/trial_data_support.py` | One-off script bypassing pipeline orchestration — useful for ad-hoc clusterings outside a run dir. |
| `tests/behaviors/test_data_support.py` | Per-metric synthetic tests + end-to-end joint-UMAP sanity check. |
| `policy_doctor/streamlit_app/demo_app/_pages/3_graph_demo.py` | Loads the JSON, exposes the color mode + selectors, drives the Markov branch. |
| `policy_doctor/streamlit_app/components/trajectory_tree_view.py` | `color_mode="data_support"` branch for the native-SVG trajectory tree. |
| `policy_doctor/plotting/plotly/trajectory_tree.py` | Sequential-colormap kwargs for sunburst / icicle / treemap. |
