# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run tests by conda env (from project root)
conda activate policy_doctor  && python run_tests.py --suite policy_doctor
conda activate cupid_torch2   && python run_tests.py --suite cupid  # integration tests, real artifacts

# Run a single test file
conda activate policy_doctor
python -m unittest tests.monitoring.test_trajectory_classifier -v

# Run all monitoring tests (policy_doctor env only, no GPU needed)
python -m unittest \
  tests.monitoring.test_trajectory_classifier \
  tests.monitoring.test_monitored_policy \
  tests.monitoring.test_pipeline_integration \
  tests.monitoring.test_fitted_model_assigner \
  tests.monitoring.test_graph_assigner \
  tests.monitoring.test_stream_monitor -v

# Offline behavior classification (cupid_torch2 env)
python scripts/monitor_offline.py \
  --episode <pkl> --checkpoint <ckpt> \
  --infembed_fit <pt> --infembed_npz <npz> --clustering_dir <dir>

# Online eval with monitoring (run from third_party/cupid/, cupid_torch2 env)
# Pass --episodes_dir when clustering was done at the "rollout" (window) level (default)
python ../../scripts/monitor_online.py \
  --output_dir /tmp/out --train_dir <dir> --train_ckpt best \
  --infembed_fit <pt> --infembed_npz <npz> --clustering_dir <dir> \
  --episodes_dir <eval_dir>/episodes

# Benchmark monitor latency and artifact sizes (cupid_torch2 env)
python -m policy_doctor.monitoring.benchmark \
  --checkpoint <ckpt> --infembed_fit <pt> --infembed_npz <npz> \
  --clustering_dir <dir> --obs_dim 20 --action_dim 14 --n_samples 50

# Curation pipeline
python -m policy_doctor.scripts.run_pipeline \
  steps=[run_clustering] task_config=<task> +experiment=<name>
```

## Architecture

### Conda environments

Three environments, each with a hard scope:

| Env | Used for |
|-----|---------|
| `policy_doctor` | Orchestration, curation pipeline, clustering, plotting, monitoring (non-scorer parts), all unit tests |
| `cupid` | Diffusion policy training/eval (`third_party/cupid`), TRAK attribution |
| `cupid_torch2` | InfEmbed attribution, `StreamScorer` classes, monitoring scripts (requires `torch.func` — not in `cupid`'s torch 1.12) |

The `policy_doctor` package can be imported without `diffusion_policy` — heavy scorer imports inside `policy_doctor/monitoring/__init__.py` are wrapped in `try/except ImportError`.

### Path roots

`policy_doctor.paths.REPO_ROOT` resolves to `third_party/cupid` when present. Paths like `data/outputs/train/...` in task YAMLs are relative to this root, not the project root. `iv_task_configs_base()` points to `third_party/influence_visualizer/configs`.

### Curation pipeline (`policy_doctor/curation_pipeline/`)

Hydra-driven pipeline: `python -m policy_doctor.scripts.run_pipeline steps=[...] ...`. Steps are classes under `curation_pipeline/steps/`; each writes a `done` sentinel and `result.json`. The ordered step list is `curation_pipeline.pipeline.ALL_STEPS`. Most steps just call into `behaviors/`, `data/`, or third-party scripts.

**Key step**: `run_clustering` (`steps/run_clustering.py`) — normalizes InfEmbed embeddings, runs UMAP, clusters with KMeans, saves `cluster_labels.npy` + `clustering_models.pkl` (fitted sklearn/UMAP objects, needed by `FittedModelAssigner`).

### Behavior graph (`policy_doctor/behaviors/`)

`BehaviorGraph` is built from cluster assignments via `BehaviorGraph.from_cluster_assignments(labels, metadata, level)`. Nodes are `BehaviorNode` dataclasses; edges carry transition probabilities. `compute_values()` solves Bellman equations for success probability. Level is `"rollout"` (one node per rollout) or `"demo"` (per demo).

### Monitoring pipeline (`policy_doctor/monitoring/`)

Runtime monitor that assigns each timestep to a behavior graph node. Full documentation in `docs/monitoring.md`.

**Component layers (bottom to top):**

1. **`StreamScorer`** (`trak_scorer.py`, `infembed_scorer.py`) — gradient + projection → `(proj_dim,)` embedding. Requires `cupid_torch2` env. `InfEmbedStreamScorer` is preferred (100-dim, faster scoring, no random seed reconstruction).

2. **`GraphAssigner`** (`graph_assigner.py`) — maps embedding to cluster. `FittedModelAssigner` applies the exact clustering pipeline (needs `clustering_models.pkl`); `NearestCentroidAssigner` uses raw-space centroids as a fallback.

3. **`StreamMonitor`** (`stream_monitor.py`) — ties scorer + assigner, returns `MonitorResult` with per-stage timing.

4. **`TrajectoryClassifier`** (`trajectory_classifier.py`) — builds batch windows, handles rotation transforms for demo-mode HDF5 data. `from_checkpoint()` reads all config from the `.ckpt` file.

5. **`MonitoredPolicy`** (`monitored_policy.py`) — wraps any `BaseLowdimPolicy`, intercepts `predict_action()`, accumulates results. Plugs into `RobomimicLowdimRunner` unmodified.

**Input modes for `TrajectoryClassifier`:**
- `mode="rollout"`: data from env or pkl — no transforms needed (policy output is already `rotation_6d`).
- `mode="demo"`: data from HDF5 — applies `RotationTransformer(axis_angle → rotation_6d)` to actions when `abs_action=True`.

### Data layer (`policy_doctor/data/`)

- `clustering_loader.py`: `load_clustering_result_from_path` loads `manifest.yaml + cluster_labels.npy + metadata.json`; `save/load_clustering_models` handles the `ClusteringModels` pkl (normalizer, prescaler, UMAP, KMeans).
- `influence_loader.py`: wraps `GlobalInfluenceMatrix` (accessed via `get_slice()`, not direct indexing).
- `clustering_embeddings.py`: `extract_infembed_slice_windows` builds sliding-window aggregated embeddings from the `rollout_embeddings` array for use as clustering input.

### Tests

Unit tests live under `tests/` and mirror the package structure. New monitoring tests under `tests/monitoring/`. Run with `python run_tests.py --suite policy_doctor` or directly with `python -m unittest`. All monitoring tests run without GPU or real checkpoints (scorer is mocked). See `tests/monitoring/test_pipeline_integration.py` for an end-to-end test using only sklearn + numpy.
