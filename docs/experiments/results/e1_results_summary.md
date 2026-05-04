# E1 Results Summary — `transport_mh` r512 / seed0

This document captures findings, methodological caveats, and queued follow-ups for Experiment E1 on the transport_mh r512x512 policy/rollouts/clustering. It is **separate from `experiment_e1_cluster_coherence.md`**, which describes how to run the experiment; this doc describes what the runs produced.

Last updated: 2026-05-04

---

## Headline numbers

All runs use Qwen3-VL-8B-Instruct (local, GPU-1 or GPU-0, bf16, greedy decoding) on the InfEmbed clustering of `transport_mh_seed0_r512` (8086 slices, 5-frame sliding windows, stride 2, 100D UMAP, kmeans).

The E1 protocol classifies held-out rollout slices into K opaque cluster labels using K · n_example example storyboards in one prompt.

| Run | K | n_example | n_query | n_reps | Examples | Global disjoint | Headline acc | Clean (tier1_global) acc | Clean p (vs 1/K) |
|---|---|---|---|---|---|---|---|---|---|
| K=20 v1 | 20 | 3 | 3 | 1 | random | no | 33.3% (20/60) | 28.6% (12/42) | 6.6e-7 |
| K=20 v2 | 20 | 3 | 3 | 3 | centroid-proximal | yes | 31.7% (19/60) | 24.1% (13/54) | 1.9e-6 |
| K=15 v2 | 15 | 3 | 3 | 3 | centroid-proximal | yes | 37.8% (17/45) | 30.8% (12/39) | 5.5e-6 |
| **K=10 v2** | **10** | 3 | 3 | 3 | centroid-proximal | yes | **53.3% (16/30)** | **48.1% (13/27)** | **5.2e-7** |

"Clean" = queries in the `tier1_global` bucket — episode appears in NO cluster's example pool. This is the proper test; episode-confounded queries are excluded.

---

## Findings

### F1 — K is the dominant factor; cheap fixes were a wash at K=20

Holding the UMAP embedding fixed and only varying K (kmeans refit on the same 100D space), clean accuracy nearly **doubles** going from K=20 → K=10 (24% → 48%). The fraction of well-resolved clusters (≥ 2/3 correct) doubles too: K=20 = 25%, K=10 = 60%.

The cheap fixes — centroid-proximal example selection, n_repetitions=3 with majority vote, and the global episode-disjoint planner — combined did **not** raise the clean-bucket number at fixed K=20 (28.6% v1 → 24.1% v2, within noise). The improvement at K=10 came from reducing K, not from the fixes.

Implication for the paper: K=20 over-clusters this rollout pool. K=10 is closer to the visually-recoverable granularity. Downstream methods that depend on per-cluster crispness should use K ≤ 10 or be robust to over-splitting.

### F2 — Episode-level visual cues are a major confound

A first-pass disjointness check that only excluded queries whose episode appeared in the *same cluster's* examples missed cross-cluster contamination. With the global check (`tier1_global`) on the K=20 v1 data:
- 5/5 queries from the within_only bucket (episode in own examples only): 100% correct
- 0/9 queries from the cross_only bucket (episode in other clusters' examples): 0% correct
- 12/42 queries from the clean bucket: 28.6% correct

Both extremes (100% on within_only, 0% on cross_only) confirm the VLM exploits episode-level visual cues whenever possible. The corrected planner (`global_episode_disjoint=True`) blocks this in subsequent runs.

### F3 — Over-clustering signature is concentrated in K=20

At K=20 v2, 7 clusters are "concentrated" (0% correct, ≥ 2/3 misses on a single dominant target):
- Cluster 2 → cluster 6 (3/3)
- Cluster 18 → cluster 17 (3/3)
- Plus 5 others with strong but not unanimous targets

Plus 3 reciprocal pairs (3↔6, 4↔8, 10↔13). These are the over-splits that K=10 collapses.

### F4 — Same-episode "perfect" clusters are inflated

Across both runs, every "perfect" cluster (3/3 correct) was either same_episode-confounded or, at K=10 v2, the single-episode trapped cluster 7. The single clean perfect cluster across all v2 runs is K=20 v2 cluster 5. 5-frame visual ambiguity makes "perfect" outcomes rare on episode-disjoint data.

---

## Open methodological questions (informing the next sweep batch)

The remaining gap from K=10's 48% clean → ~80% (a reasonable ceiling for cluster coherence) likely reflects multiple compounding factors:

1. **Slice length.** 5-frame windows ≈ 0.5 s of motion. Brief visual context.
2. **VLM capacity.** Qwen3-VL-8B with 30–60 example storyboards per prompt. Attention budget is real.
3. **Image budget per storyboard.** 4 frames in a 2×2 grid; default `image_max_pixels=1024×1024`.
4. **Clustering hyperparameters.** Window width, stride, aggregation, prescale, UMAP dim. Only K has been swept.
5. **Representation choice.** All measurements so far use InfEmbed clustering. State-only and state-action baselines are necessary to compare.
6. **Stronger VLM.** Gemini 3 Flash, Claude Sonnet 4.6 — pending API setup.

---

## Queued / planned

The next implementation phase (in flight, see "Architecture" below) targets questions 1, 4, 5 — and lays the foundation for 2 + 6 by keeping the E1 evaluation interface stable.

### Q1 — Extended visual context

Add `view_window_extension: int` (default 0) to `_load_storyboard`. When > 0, the frames passed to the VLM come from a wider window than the cluster window (extended symmetrically), without changing the clustering or sample plan. Tests whether longer visual context recovers more accuracy at fixed K.

Suggested values: `view_window_extension ∈ {0, 5, 10, 15}`. Run at K=10 (highest signal-to-noise) and one other K for comparison.

### Q4 — Clustering hyperparameter sweeps

Beyond K, sweep:
- `window_width ∈ {3, 5, 10, 15}`
- `stride ∈ {1, 2, 5}`
- `aggregation ∈ {sum, mean, concat}`
- `prescale ∈ {none, standard, l2}`
- `umap_n_components ∈ {25, 50, 100, 200}`

This is a lot of clusterings (each a few minutes of CPU). The sweep harness lets us run them in batch and evaluate any subset through E1.

### Q5 — Baseline representations

Two new representations cluster the same rollout slices using non-influence features:

- **State** — concatenated proprioceptive observations across the window (sum/mean/concat aggregation).
- **State+action** — same as state, with action vectors concatenated to obs at each timestep.

If InfEmbed-clustered slices are more visually-recoverable than state-clustered slices at matched K, that's evidence influence captures behaviorally meaningful structure beyond what raw observations encode.

---

## Architecture (implemented)

The new `SliceRepresentation` abstraction decouples *what* feature each slice is from *how* it's clustered:

```
policy_doctor/data/slice_representations.py
├── SliceWindowParams                    (dataclass: window_width, stride, aggregation)
├── SliceRepresentation (ABC)
│   ├── name: str
│   ├── extract(eval_dir, params, **kwargs) -> (features: ndarray, metadata: List[dict])
│   └── describe(params, **kwargs)       (JSON-able fingerprint for manifests)
├── InfEmbedRepresentation               (wraps extract_infembed_slice_windows)
├── StateRepresentation                  (proprioceptive obs from episode pickles)
│   └── kwargs: obs_strategy ∈ {current, full_history}
├── StateActionRepresentation            (obs + action concat per timestep)
│   └── kwargs: obs_strategy, action_strategy ∈ {executed, full_plan}
└── registry: get_slice_representation(name)
```

All three representations share the same window-aggregation engine (`build_windows_from_rollout_timestep_embeddings`). The output `(features, metadata)` contract is identical, so the downstream `normalize → prescale → reduce → cluster → save` pipeline is unchanged.

The output directory layout (`cluster_labels.npy`, `metadata.json`, `manifest.yaml`, `embeddings_reduced.npy`, `clustering_models.pkl`) matches what `RunClusteringStep` produces, so the existing E1 evaluation pipeline (runner script, Hydra step, analysis script) transparently consumes alternative clusterings without modification.

### Scripts

- **`scripts/build_alt_clustering.py`** — single-config CLI. Pick `--representation`, supply hyperparams (`--window_width`, `--stride`, `--aggregation`, `--prescale`, `--umap_n_components`, `--n_clusters`, etc.), get one clustering dir.
- **`scripts/run_clustering_sweep.py`** — cartesian product over a YAML/JSON spec. Filters rep-specific kwargs (e.g. `obs_strategy` only varied for state/state_action), de-duplicates, runs each combo as a subprocess. Skips combos whose `manifest.yaml` already exists unless `--force`. Writes a per-combo `build.log`.
- **`sweep_specs/transport_r512_alt_clustering.yaml`** — example spec: 108 unique combos across {infembed, state, state_action} × {window_width 3,5,10} × {aggregation sum,mean} × {umap_n_components 50,100} × {K 10,15,20}.

### Pipeline integration (unchanged for legacy users)

- `RunClusteringStep` (the Hydra step) is **not** modified yet — alternative clusterings are produced via the standalone runner. The sample plan / E1 step `validate_cluster_coherence_vlm` reads the new `view_window_extension` knob.
- The existing `vlm_cluster_classification.global_episode_disjoint` and the new `vlm_cluster_classification.view_window_extension` are both configurable via `policy_doctor/configs/experiment/e1_cluster_coherence_vlm.yaml`.

### Tests

- `tests/data/test_slice_representations.py` (13 tests): registry, helper flatteners (current/full_history, executed/full_plan), shape/metadata contracts for each concrete rep.
- `tests/data/test_build_alt_clustering.py` (2 tests): end-to-end smoke on a synthetic eval dir confirming `build_alt_clustering.py` produces a dir the existing E1 planner can consume in `global_episode_disjoint=True` mode.

---

## How-to (no runs yet — GPUs reserved for E2)

When ready, the canonical loop is:

```bash
# 1) Build a non-InfEmbed clustering for transport_mh r512 seed0
python scripts/build_alt_clustering.py \
  --representation state_action \
  --eval_dir /mnt/ssdB/erik/cupid_data/outputs/eval_save_episodes/mar27/mar27_train_diffusion_unet_lowdim_transport_mh_0_r512x512/latest \
  --window_width 5 --stride 2 --aggregation mean \
  --prescale standard --umap_n_components 100 --n_clusters 10 \
  --obs_strategy current --action_strategy executed \
  --seed 42 \
  --out_dir /tmp/clustering_sweeps/sa_k10_mean

# 2) Evaluate any clustering dir via E1 (with optional extended visual context)
CUDA_VISIBLE_DEVICES=1 python scripts/run_e1_transport_r512_qwen.py \
  --clustering_dir /tmp/clustering_sweeps/sa_k10_mean \
  --max_clusters 10 --n_example 3 --n_query 3 --n_repetitions 3 \
  --global_episode_disjoint \
  --view_window_extension 10 \
  --out_dir experiments/e1_state_action_k10_view10

# 3) Or run a whole sweep at once
python scripts/run_clustering_sweep.py \
  --spec sweep_specs/transport_r512_alt_clustering.yaml \
  --dry_run     # remove --dry_run to actually execute (CPU-only, slow)
```

The sweep harness is CPU-only (UMAP + KMeans). It does not touch the GPU. The E1 evaluation step is GPU-bound; that's where coordination with E2 matters.

### Suggested initial evaluation matrix (when GPUs free up)

Most informative single-axis sweeps:

1. **Representation comparison at K=10** — `build_alt_clustering` for {infembed, state, state_action} with default hyperparams, then E1 on each. Tests whether influence captures behavioral structure beyond raw observations.
2. **Extended visual context at K=10** — same InfEmbed clustering, vary `--view_window_extension ∈ {0, 5, 10, 15}`. Isolates the slice-length confound.
3. **Window width sweep** — re-cluster InfEmbed at `window_width ∈ {3, 5, 10, 15}`, K=10, then E1. The clustering window and the visual window co-vary here, so this is a different question than (2).

The pre-built sweep spec covers (3) plus the cross-product with state/state_action. That's 108 clusterings total — likely 60–90 min of CPU on this machine.

---

## Findings to write up after the next batch

- Does state-only clustering recover *any* visually-coherent structure at K=10? (If clean acc near chance, that's evidence influence carries unique signal.)
- Does state_action clustering match or beat InfEmbed at K=10?
- Is the K=10 → 48% → ceiling gap closed by extended visual context? By how much?
- Does window_width=10 with the InfEmbed representation give cleaner clusters than width=5?
