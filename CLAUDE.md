# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands run from the project root (the directory containing `pyproject.toml`) via the `uv_env.sh` dispatcher — there is no `conda activate` step.

```bash
# One-time: provision a uv-managed venv per extra (analysis | cupid | mimicgen | robocasa)
./scripts/uv_env.sh analysis --setup

# Tests — the canonical 40-test golden / experiment / env-dispatch suite
./scripts/uv_env.sh analysis pytest tests/golden/ tests/experiment/ tests/test_env_dispatch.py

# Other test slices
./scripts/uv_env.sh analysis pytest tests/                  # full analysis-side suite
./scripts/uv_env.sh cupid    pytest tests/cupid/            # diffusion_policy integration
./scripts/uv_env.sh mimicgen pytest tests/mimicgen/         # MimicGen seeds + sim

# Pipeline (Hydra entry point)
./scripts/uv_env.sh analysis python -m policy_doctor.scripts.run_pipeline \
    steps=[run_clustering] task_config=transport_mh_jan28
./scripts/uv_env.sh analysis python -m policy_doctor.scripts.run_pipeline \
    +experiment=trak_filtering_mar13_p96 \
    steps=[run_clustering,run_curation_config,train_curated,eval_curated]

# Experiments — create a self-contained on-disk experiment dir
./scripts/uv_env.sh analysis python -m policy_doctor.scripts.experiment_init <name> \
    [--baseline-from <other>]
./scripts/uv_env.sh analysis python -m policy_doctor.scripts.experiment_bundle <name> \
    --out /tmp/<name>.tar.gz

# MimicGen full experiment (select seed -> generate -> train -> eval for all arms)
./scripts/uv_env.sh mimicgen python -m policy_doctor.scripts.run_pipeline \
    data_source=mimicgen_square \
    experiment=mimicgen_square_pipeline_apr23

# Training (bypasses pipeline)
./scripts/uv_env.sh cupid    python third_party/cupid/train.py --config-name=...
./scripts/uv_env.sh mimicgen python third_party/cupid/train.py --config-name=...
# Wrapper scripts forward extra args as Hydra overrides:
./scripts/experiments/train_robomimic_square.sh --compile --tf32
./scripts/experiments/train_mimicgen_square.sh  --compile training.device=cuda:1
./scripts/experiments/train_robocasa_atomic.sh  OpenCabinet

# Runtime monitoring — classify policy behavior per-timestep
./scripts/uv_env.sh analysis python scripts/experiments/monitor_online.py \
    --output_dir /tmp/out --train_dir <dir> --train_ckpt best \
    --infembed_fit <pt> --infembed_npz <npz> --clustering_dir <dir> \
    --episodes_dir <eval_dir>/episodes        # required when clustering level is "rollout"

# Streamlit apps
./scripts/uv_env.sh analysis streamlit run policy_doctor/streamlit_app/app.py
./scripts/uv_env.sh analysis streamlit run policy_doctor/streamlit_app/demo_app/Home.py
./scripts/uv_env.sh analysis streamlit run policy_doctor/streamlit_app/user_study/Home.py
```

## Architecture

### uv workspace + per-extra venvs

The repo is a single uv workspace (`pyproject.toml` + `uv.lock`). Each extra under `[project.optional-dependencies]` maps to one venv under `.venvs/<extra>/`:

| Extra | Role |
|-------|------|
| `analysis` | Default env. Orchestration, clustering, behavior graph, Streamlit, InfEmbed post-processing, runtime monitoring, TRAK featurization. No simulator deps. |
| `cupid` | robomimic 0.2.0 + robosuite 1.2.0 + mujoco 3.2.6. For diffusion-policy training, `eval_save_episodes`, TRAK / InfEmbed online steps. |
| `mimicgen` | robosuite 1.4.1 + robomimic 0.3.0 + mimicgen 1.0.0 + mujoco 3.3.7. For MimicGen generation and MimicGen-trained policies. |
| `robocasa` | robosuite 1.5.2 + robocasa for kitchen-sim tasks. |
| `dev` | pytest + tooling. Added implicitly by `uv_env.sh --setup`. |

`scripts/uv_env.sh <extra> <cmd...>` exports `UV_PROJECT_ENVIRONMENT=.venvs/<extra>` and execs `uv run --no-sync <cmd>`. First-use auto-syncs; pass `--setup` to (re)create the env without running anything. The legacy `scripts/setup/uv_env.sh` is the canonical path; `scripts/uv_env.sh` is a symlink.

Pipeline steps that need a different env (e.g. `train_curated` shells out to `cupid`, `generate_mimicgen_demos` to `mimicgen`) dispatch through `policy_doctor/_env.py::run_in_env`, which builds the right `uv run --extra <name>` command line.

### Three-package layout

`third_party/cupid` is the only remaining editable workspace member; `third_party/influence_visualizer` was absorbed into `policy_doctor.influence` in Phase 3.

- **`policy_doctor/`** — attribution analysis, data structures, clustering, behavior graph, curation logic, Hydra pipeline orchestration, experiment layer, Streamlit UI, influence-data loaders.
- **`third_party/cupid/`** — diffusion policy training workspace (`train.py`), TRAK (`train_trak_diffusion.py`), InfEmbed (`compute_infembed_embeddings.py`), `eval_save_episodes.py`. Packaged as `cupid-workspace`.

### Experiment layer (`policy_doctor.experiment`)

The experiment-centric artifact layout is the new canonical on-disk shape. Each named experiment is a self-contained directory under `<repo>/data/experiments/<name>/` (override via `$POLICY_DOCTOR_DATA`):

```
data/experiments/<name>/
    manifest.yaml          # name, created_at, baseline_from, free-form keys
    config/                # snapshot_<utc>.yaml + canonical.yaml symlink to the first one
    shared/                # baseline_ckpt, source datasets (hard-copied on bundle)
    artifacts/<step>/      # one dir per pipeline step
        seed_<seed>/       # one dir per training seed
            <ckpt>/        # one dir per eval checkpoint
    logs/<label>_<utc>.log # per-invocation logs
```

CLI surface:

- `python -m policy_doctor.scripts.experiment_init <name> [--baseline-from <other>]` creates the skeleton and hard-copies an upstream baseline checkpoint.
- `python -m policy_doctor.scripts.experiment_bundle <name> --out <tar.gz>` dereferences symlinks under `shared/` and tarballs the dir for cross-machine transfer.

`Experiment.step_dir`, `seed_dir`, `ckpt_dir` are the canonical path helpers. `append_config_snapshot` records the resolved Hydra config per invocation. The `CurationPipeline` bridge (`policy_doctor/curation_pipeline/`) accepts an `Experiment` and routes its artifacts into `artifacts/`.

**Deferred:** pipeline steps still construct internal paths from `train_date`/`eval_date` in the legacy `data/outputs/train/<date>/...` layout. The experiment layer is in place but not yet wired into every step — the migration is the #1 follow-up after this refactor lands.

### Path resolution

`policy_doctor.paths` exposes:

- `PACKAGE_ROOT` = `policy_doctor/policy_doctor/`
- `PROJECT_ROOT` = repo root
- `REPO_ROOT` = `third_party/cupid` when that directory exists — training / eval / attribution still resolve `data/outputs/...` from here
- `DATA_SOURCE_ROOT` = `<PROJECT_ROOT>/data/source/`
- `data_root()` (in `policy_doctor.experiment.paths`) = `$POLICY_DOCTOR_DATA` or `<PROJECT_ROOT>/data/`; `experiment_dir(name)` returns `data_root()/experiments/<name>/`.

### Attribution methods

Both TRAK and InfEmbed compute pairwise influence between rollout (test) samples and training demonstrations.

**TRAK** (`train_trak_diffusion.py`): `traker.featurize()` projects per-sample grads via Johnson-Lindenstrauss to `proj_dim` (default 2048); `finalize_features()` approximates the regularized Hessian; `score()` dots test-sample projections against the cached Hessian-inverse-weighted train projections. Output: `(num_train_samples, num_test_samples)` influence matrix.

**InfEmbed** (`compute_infembed_embeddings.py`): `ArnoldiEmbedder.fit()` finds the `arnoldi_dim` (default 200) dominant eigenvectors of the Gauss-Newton Hessian — this approximates `H^{-1/2}` and is the cacheable factor. `predict()` projects gradients into the `projection_dim` (default 100) space. Influence ≈ dot product of embeddings.

Both fits are disk-cached; only `predict`/`score` needs to run for new samples. `infembed` is installed as a workspace member (`third_party/cupid/third_party/infembed/`).

### Influence matrix and data structures

`GlobalInfluenceMatrix` (`policy_doctor/data/structures.py`) is a `(num_rollout_samples, num_demo_samples)` float32 matrix. Access via `get_slice(r_lo, r_hi, d_lo, d_hi)` — never index directly, as the backing store may be memmapped. `get_local_matrix(rollout_idx, demo_idx)` returns a `LocalInfluenceMatrix` for one trajectory pair.

Episodes are `EpisodeInfo(index, num_samples, sample_start_idx, sample_end_idx, success)`. Global indices in the matrix are contiguous; per-episode offsets are stored in `sample_start_idx` / `sample_end_idx`.

### Behavior graph and clustering

`policy_doctor/computations/embeddings.py` extracts per-timestep embeddings by summing the influence matrix along one axis. `policy_doctor/behaviors/clustering.py::reduce_dimensions()` runs UMAP (default) or PCA; `run_clustering()` runs HDBSCAN / GMM / KMeans. `policy_doctor/behaviors/behavior_graph.py::BehaviorGraph.from_cluster_assignments()` builds a Markov chain from run-length-collapsed cluster sequences per episode. Nodes: behavioral clusters + START/SUCCESS/FAILURE/END. `compute_values()` solves linear Bellman equations to assign V-values.

### Influence package (`policy_doctor.influence`)

Absorbed from the former `third_party/influence_visualizer` in Phase 3. Loads HDF5 demos and TRAK results, persists clusterings, and supplies the Streamlit-side data helpers. Public surface: `loader.py`, `clustering_io.py`, `annotations.py`, `frames.py`, `lazy_hdf5.py`, `path_helpers.py`.

### Pipeline step system

`policy_doctor/curation_pipeline/pipeline.py` orchestrates steps in `ALL_STEPS` order. Each `PipelineStep`:

- Writes `<run_dir>/<step_name>/done` as a completion sentinel.
- Writes `<run_dir>/<step_name>/result.json` with its outputs.
- Respects `skip_if_done=true` (default) to resume from the last incomplete step.
- Shells out via `policy_doctor._env.run_in_env(extra, cmd)` for steps that need a different uv env (e.g. `train_curated` in `cupid`, `generate_mimicgen_demos` in `mimicgen`).

`CompositeStep` groups a fixed sub-step sequence under one namespace (sub-steps land in `<run_dir>/<composite_name>/<sub_step>/`). Sub-steps still read sibling top-level results via `parent_run_dir`.

### MimicGen trajectory generation pipeline

Tests whether behavior-graph-guided seed selection improves generated data quality. Three heuristics share upstream `run_clustering`:

- `BehaviorGraphPathHeuristic` (proposed) — ranks paths to SUCCESS by probability.
- `DiversitySelectionHeuristic` — one rollout per path before moving to the next.
- `RandomSelectionHeuristic` (baseline) — uniform from eligible (successful) rollouts.

Sub-step sequence per arm (`policy_doctor/curation_pipeline/steps/mimicgen_arm.py`): `select_mimicgen_seed_from_graph` → `generate_mimicgen_demos` → `train_on_combined_data` → `eval_mimicgen_combined`. Multiple arms share a run by writing to `mimicgen_<heuristic>/` under the same run dir. Generation knobs live in `policy_doctor/configs/mimicgen/<task>.yaml`.

### Streamlit / visualization separation

Strict separation, enforced throughout:

- `render_*.py` files contain all Streamlit UI (`st.*`) and data preprocessing.
- `policy_doctor/plotting/` contains pure functions that accept preprocessed data and return `plotly.Figure` objects — no Streamlit imports.
- Plotting functions are exported from `policy_doctor/plotting/__init__.py`.

Streamlit is the only viz surface that ships with the repo. Three apps live under `policy_doctor/streamlit_app/`: `app.py` (main analysis UI), `demo_app/Home.py` (researcher graph explorer), `user_study/Home.py` and `survey_app/Home.py` (participant survey app). The non-Streamlit viz scripts were removed in Phase 4.

### Data support (`policy_doctor/behaviors/data_support.py`)

Per-cluster diagnostic that measures how well each behavior-graph node is supported by the training distribution. Joint UMAP fit on demo + rollout windows, BallTree over demo points, metric registry (count-in-radius, kNN distance, KDE log-density, binary coverage). Scoped to `influence_source == "policy_emb"` clusterings. Writes `data_support.json` next to `cluster_labels.npy`. Full docs: `docs/data_support.md`.

### Runtime monitoring (`policy_doctor/monitoring/`)

Assigns each policy timestep to a behavior graph node in real time. Layers: `InfEmbedStreamScorer` → `FittedModelAssigner` / `NearestCentroidAssigner` → `StreamMonitor` → `TrajectoryClassifier` → `MonitoredPolicy`. `FittedModelAssigner` requires `clustering_models.pkl` in the clustering dir; otherwise falls back to `NearestCentroidAssigner`. When clustering level is `"rollout"`, pass `--episodes_dir` so window-mean embeddings can be computed. Full docs: `docs/monitoring.md`.

### Hydra config layout

- Base: `policy_doctor/configs/config.yaml` — sets defaults for `data_source`, `pipeline/config`, `vlm/defaults`, `experiment: null`.
- `data_source` group (`cupid_robomimic`, `mimicgen_square`, `robocasa_layout`).
- `mimicgen` group (`square_d0`, `square_d1`, `coffee`, `threading`) — load with `mimicgen=square_d1`.
- Robomimic-specific slices: `policy_doctor/configs/robomimic/` (tasks, baseline, evaluation, attribution, curation_filtering, curation_selection).
- Experiment presets: `policy_doctor/configs/experiment/` — select with `+experiment=name`.
- Attribution `tf32` / `compile` flags live in YAML (`attribution.tf32`, `attribution.compile`), not CLI flags like training scripts.
