# policy_doctor

Influence-based policy analysis and curation. This repo refactors **influence_visualizer** with a clear split between data, computation, and visualization.

**Standalone layout:** training and eval live in vendored **cupid** (`third_party/cupid/`). Task YAMLs and clustering trees used by the app often live in **influence_visualizer** (`third_party/influence_visualizer/`). Hydra configs for the pipeline ship inside the **`policy_doctor`** package (`policy_doctor/configs/`).

`policy_doctor.paths.REPO_ROOT` points at `third_party/cupid` when that directory exists, so paths such as `data/outputs/train/...` in task YAMLs are resolved from the cupid tree. IV-style task configs are resolved via `iv_task_configs_base()` (typically `third_party/influence_visualizer/configs`).

## Quick start

1. Create the conda env and install the three editable packages (requires `third_party/cupid` and `third_party/influence_visualizer`):

   ```bash
   conda env create -f environment_policy_doctor.yaml
   ./scripts/install_policy_doctor_env.sh
   conda activate policy_doctor
   ```

2. Run tests from the **policy_doctor project root** (directory that contains `pyproject.toml`):

   ```bash
   # Full discover in one env (needs all optional deps — often impractical):
   python run_tests.py --suite all

   # Recommended: stack-matched conda envs
   ./scripts/run_tests_policy_doctor.sh   # orchestration (data, curation, plotting, VLM, …)
   ./scripts/run_tests_cupid.sh           # mar27 transport / diffusion_policy integration
   ./scripts/run_tests_mimicgen.sh        # MimicGen seeds (+ MIMICGEN_E2E=1 for HF + sim)
   ```

   Same suites without shell wrappers: `conda activate <env>` then `python run_tests.py --suite policy_doctor|cupid|mimicgen`.

3. Training, rollouts, TRAK, and InfEmbed expect the separate **`cupid`** sim stack (see [Conda environments](#conda-environments)):

   ```bash
   ./scripts/install_cupid_env.sh
   conda activate cupid
   ```

## Curation pipeline (Hydra)

The single entry point is `policy_doctor.scripts.run_pipeline`. It uses **Hydra** (`policy_doctor/configs/config.yaml` plus optional `+experiment=...`). There is **no** positional `STEP ENV STATE TASK` CLI anymore.

**Defaults:** `steps: []` in the base config is treated as “run the full ordered sequence” (see [Pipeline step order](#pipeline-step-order)). To run a subset, pass `steps=[...]` on the command line or in an experiment YAML under `policy_doctor/configs/experiment/`.

**Run directory:** Unless you set an absolute `run_dir`, it defaults to `<REPO_ROOT>/data/pipeline_runs/<run_name>/`. With vendored cupid, `REPO_ROOT` is `third_party/cupid`, so run metadata and per-step folders sit beside `data/outputs/`. Each completed step writes `<step_name>/done` and usually `<step_name>/result.json`. With `skip_if_done=true` (default), re-running resumes from the last incomplete step.

**Common flags (Hydra overrides):**

| Override | Meaning |
|----------|---------|
| `+experiment=name` | Merge `policy_doctor/configs/experiment/<name>.yaml` (note leading `+` when adding a new defaults group) |
| `steps=[a,b]` | Run only those steps, in order |
| `dry_run=true` | Print planned work; no heavy compute |
| `skip_if_done=false` | Ignore `done` sentinels and re-execute |
| `run_name=myrun` | Stable name for `run_dir` (default is a timestamp) |
| `run_dir=/abs/path` | Fixed run folder (absolute path) |

**Examples** (from the policy_doctor project root, with the appropriate conda env active — usually **`cupid`** for anything that trains or hits the sim):

```bash
# Full pipeline (all steps, default order) with explicit dates
python -m policy_doctor.scripts.run_pipeline train_date=jan28 eval_date=jan28

# Same as many experiment YAMLs comment blocks: clustering → curation configs → curated train/eval
python -m policy_doctor.scripts.run_pipeline +experiment=trak_filtering_mar13_p96 \
  steps=[run_clustering,run_curation_config,train_curated,eval_curated]

# Resume later (re-uses completed steps under the same run_dir)
python -m policy_doctor.scripts.run_pipeline +experiment=trak_filtering_mar13_p96 \
  run_dir=third_party/cupid/data/pipeline_runs/myrun \
  steps=[train_curated,eval_curated]

# Dry-run one step
python -m policy_doctor.scripts.run_pipeline steps=[eval_policies] dry_run=true train_date=jan28 eval_date=jan28
```

Switching **task** (e.g. lift vs transport) requires consistent Hydra groups: see `policy_doctor/configs/config.yaml` defaults (`baseline`, `evaluation`, `attribution`, `curation_filtering`, `curation_selection`) and override them together with top-level `task` and `task_config` (the YAML stem under `influence_visualizer/configs` or `policy_doctor/configs`, e.g. `transport_mh_jan28`).

### Pipeline step order

Ordered list (from `policy_doctor.curation_pipeline.pipeline.ALL_STEPS`):

1. `train_baseline` → `eval_policies` → `train_attribution` → `finalize_attribution` → `compute_demonstration_scores` → `compute_infembed` → `run_clustering` → `export_markov_report` → `annotate_slices_vlm` → `summarize_behaviors_vlm` → `evaluate_cluster_coherency_vlm` → `run_curation_config` → `train_curated` → `eval_curated` → `compare`
2. **MimicGen sub-pipeline** (run as a subset — see [MimicGen trajectory generation pipeline](#mimicgen-trajectory-generation-pipeline)): `select_mimicgen_seed` → `generate_mimicgen_demos` → `train_on_combined_data`

Optional VLM steps are no-ops unless configured; `finalize_attribution` skips when `attribution.num_ckpts <= 1`. `export_markov_report` reads `run_clustering/result.json`; `evaluate_cluster_coherency_vlm` requires `annotate_slices_vlm` outputs in the same `run_dir`.

<details>
<summary><strong>Per-step: how to run and how to verify</strong></summary>

Unless noted, run from the project root with `python -m policy_doctor.scripts.run_pipeline steps=[<step>] ...` plus your `train_date`, `eval_date`, `task_config`, `+experiment`, etc. **Artifacts under `data/outputs/`** are relative to **`REPO_ROOT`** (vendored cupid). **Pipeline metadata** is under `<run_dir>/<step_name>/` (see [Curation pipeline](#curation-pipeline-hydra)).

| Step | Typical conda env | Run (example pattern) | Verify |
|------|-------------------|------------------------|--------|
| `train_baseline` | `cupid` | `steps=[train_baseline] train_date=jan28` | Checkpoints under `third_party/cupid/data/outputs/train/<train_date>/<train_date>_train_<policy>_<task>_<seed>/` (layout from `get_train_dir` in `curation_pipeline/paths.py`). |
| `eval_policies` | `cupid` | `steps=[eval_policies] train_date=jan28 eval_date=jan28` | Rollout dir: `.../data/outputs/eval_save_episodes/<eval_date>/<name>/<train_ckpt>/` with episode pickles / logs (used by later steps). |
| `train_attribution` | `cupid` | `steps=[train_attribution] train_date=jan28 eval_date=jan28` | TRAK / attribution outputs under the eval rollout tree per cupid conventions (see `attribution` config). |
| `finalize_attribution` | `cupid` | `steps=[finalize_attribution] ...` | Only runs when `attribution.num_ckpts > 1`; otherwise prints skipped. |
| `compute_demonstration_scores` | `cupid` | `steps=[compute_demonstration_scores] ...` | Demonstration score artifacts under eval/train paths per `eval_demonstration_scores` (see attribution config `result_date`, `exp_name`). |
| `compute_infembed` | `cupid` | `steps=[compute_infembed] ...` | InfEmbed outputs next to eval rollouts / checkpoints per `compute_infembed_embeddings`. |
| `run_clustering` | `policy_doctor` or `cupid` | `steps=[run_clustering] task_config=transport_mh_jan28` (+ clustering hyperparameters or `+experiment`) | `run_clustering/result.json` maps each seed to a directory under `third_party/influence_visualizer/configs/<task_config>/clustering/<slug>/` with `manifest.yaml`, `cluster_labels.npy`, `metadata.json`. |
| `export_markov_report` | `policy_doctor` | `steps=[export_markov_report] ...` | Reads clustering from `run_clustering`; writes `<run_dir>/export_markov_report/markov_report_seed*.json`. Tuning: `markov_export.*` in `configs/pipeline/config.yaml`. |
| `annotate_slices_vlm` | `policy_doctor` | `steps=[annotate_slices_vlm] ...` | Configure `policy_doctor/configs/vlm/`. Outputs under `<run_dir>/annotate_slices_vlm/` (e.g. `annotations_seed*.jsonl`). |
| `summarize_behaviors_vlm` | `policy_doctor` | `steps=[summarize_behaviors_vlm] ...` | Uses VLM defaults; outputs under `<run_dir>/summarize_behaviors_vlm/`. |
| `evaluate_cluster_coherency_vlm` | `policy_doctor` | `steps=[evaluate_cluster_coherency_vlm] ...` | After `annotate_slices_vlm`; per-cluster JSON judgments under `<run_dir>/evaluate_cluster_coherency_vlm/`. Config: `vlm_coherency_eval` in `configs/vlm/defaults.yaml`. |
| `run_curation_config` | `policy_doctor` | `steps=[run_curation_config] ...` (often after `run_clustering`) | `run_curation_config/result.json` lists generated YAML paths (per seed). If `clustering_dir` is unset, paths from the prior `run_clustering` step in the same `run_dir` are used. |
| `train_curated` | `cupid` | `steps=[train_curated] ...` | If `curation_config_path` unset, loads paths from `run_curation_config` in the same run. New training dirs under `data/outputs/train/` with curated naming from Hydra workspace. |
| `eval_curated` | `cupid` | `steps=[eval_curated] ...` | New eval dirs under `data/outputs/eval_save_episodes/`. |
| `compare` | `policy_doctor` | `steps=[compare] ...` | Reads `eval_log.json` for baseline vs curated runs; result in `compare/result.json` and printed table. |
| `select_mimicgen_seed` | `policy_doctor` | `steps=[select_mimicgen_seed] mimicgen_datagen.seed_selection_heuristic=behavior_graph` | Requires `run_clustering` result. Writes `select_mimicgen_seed/seed.hdf5`. Result JSON includes `seed_hdf5_path`, `rollout_idx`, `heuristic`, `selection_info`. |
| `generate_mimicgen_demos` | `mimicgen` (subprocess) | `steps=[generate_mimicgen_demos] mimicgen_datagen.episode_budget=50` | Requires `select_mimicgen_seed` or `select_mimicgen_seed_from_graph` (auto-wired); otherwise falls back to `mimicgen_datagen.source_dataset_path`. Writes `generated_hdf5_path` and EEF data in result JSON. |
| `train_on_combined_data` | `cupid` | `steps=[train_on_combined_data] ...` | Requires `generate_mimicgen_demos`. Merges original + generated HDF5, trains a new policy. Run name encodes the heuristic (e.g. `...-mimicgen_combined-behavior_graph`). Result: `combined_hdf5_path`, `train_dirs`. |

</details>

## MimicGen trajectory generation pipeline

This pipeline extends data-driven curation with **MimicGen-based augmentation**: instead of only selecting which original demonstrations to train on, it also generates new demonstrations and retrains on the combined dataset.

### Motivation

The central hypothesis is that the *choice of seed trajectory* given to MimicGen determines the quality of generated data. A seed that follows the highest-probability behavioral path to task success should yield more successful generated demonstrations than a randomly-chosen seed. This pipeline makes that comparison concrete and reproducible.

### Pipeline overview

```
run_clustering
  → select_mimicgen_seed      (heuristic-based: behavior_graph or random)
  → generate_mimicgen_demos   (MimicGen, episode_budget trials)
  → train_on_combined_data    (train on original + generated demos)
```

### Seed selection heuristics

Two heuristics are available via `mimicgen_datagen.seed_selection_heuristic`:

| Heuristic | Key | Description |
|-----------|-----|-------------|
| **Behavior graph path** (proposed) | `behavior_graph` | Builds a behavior graph from the clustering result. Ranks paths to the SUCCESS node by probability. Selects the first rollout whose collapsed cluster-label sequence exactly matches the highest-probability path. |
| **Random** (baseline) | `random` | Picks a successful rollout uniformly at random. Used to isolate the benefit of informed seed selection from simply running MimicGen at all. |

Both heuristics are implemented as subclasses of `TrajectorySelectionHeuristic` (`policy_doctor/mimicgen/heuristics.py`). New heuristics can be added by subclassing and registering in `build_heuristic()`.

### Config reference (`mimicgen_datagen`)

All keys live under `mimicgen_datagen` in `policy_doctor/configs/pipeline/config.yaml` or in any experiment YAML.

| Key | Default | Description |
|-----|---------|-------------|
| `seed_selection_heuristic` | `behavior_graph` | `"behavior_graph"` or `"random"` |
| `top_k_paths` | `5` | (`behavior_graph` only) Number of candidate paths to try before giving up |
| `min_path_probability` | `0.0` | (`behavior_graph` only) Discard paths below this probability |
| `success_only` | `true` | Both heuristics: only draw from successful rollouts |
| `random_seed` | `null` | (`random` only) Integer RNG seed for reproducibility |
| `policy_seed` | `null` | Which policy seed's clustering result to use (null = first available) |
| `episode_budget` | `50` | Number of MimicGen generation trials |
| `output_dir` | `data/outputs/mimicgen_datagen` | Base directory for generated HDF5s |
| `task_name` | `square` | MimicGen task name |
| `env_interface_name` | `MG_Square` | MimicGen env interface class |
| `env_interface_type` | `robosuite` | MimicGen env interface type |
| `source_dataset_path` | `null` | Fallback source HDF5 when no seed-selection step has run |

### Running the experiment

Both runs must share the same `run_dir` so `select_mimicgen_seed` can read the `run_clustering` result. The heuristic name is embedded in the training run directory, so the two training runs land in separate output directories automatically.

```bash
# Step 0: clustering (shared between both runs)
python -m policy_doctor.scripts.run_pipeline \
  run_dir=data/pipeline_runs/mimicgen_experiment \
  steps=[run_clustering]

# Proposed method — behavior graph seed selection
python -m policy_doctor.scripts.run_pipeline \
  run_dir=data/pipeline_runs/mimicgen_experiment \
  mimicgen_datagen.seed_selection_heuristic=behavior_graph \
  steps=[select_mimicgen_seed,generate_mimicgen_demos,train_on_combined_data]

# Baseline — random seed selection
python -m policy_doctor.scripts.run_pipeline \
  run_dir=data/pipeline_runs/mimicgen_experiment_random \
  mimicgen_datagen.seed_selection_heuristic=random \
  steps=[select_mimicgen_seed,generate_mimicgen_demos,train_on_combined_data]
```

Because different heuristics write to different step result files (`select_mimicgen_seed/result.json`), you can also run both in separate `run_dir`s that each reuse the same clustering result via `clustering_dir=<path>`.

Alternatively, use the pre-configured experiment YAMLs (both require `data_source=mimicgen_square`):

```bash
# Proposed method (behavior graph)
python -m policy_doctor.scripts.run_pipeline \
  data_source=mimicgen_square \
  experiment=mimicgen_square_bg_pipeline

# Baseline (random selection)
python -m policy_doctor.scripts.run_pipeline \
  data_source=mimicgen_square \
  experiment=mimicgen_square_random_pipeline
```

### Output artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| `seed.hdf5` | `<run_dir>/select_mimicgen_seed/` | Materialised seed trajectory for MimicGen |
| `select_mimicgen_seed/result.json` | `<run_dir>/` | Rollout index, heuristic name, selection info (path/prob for graph; eligible count for random) |
| `generate_mimicgen_demos/result.json` | `<run_dir>/` | Generated HDF5 path, EEF trajectories, MimicGen stats |
| `combined.hdf5` | `<run_dir>/train_on_combined_data/` | Merged original + generated dataset used for training |
| Training run | `data/outputs/train/<date>/<name>-mimicgen_combined-<heuristic>/` | Policy checkpoint trained on combined data |

### Implementation notes

- **`policy_doctor/mimicgen/heuristics.py`** — `TrajectorySelectionHeuristic` ABC, `BehaviorGraphPathHeuristic`, `RandomSelectionHeuristic`, `build_heuristic()` factory.
- **`policy_doctor/mimicgen/combine_datasets.py`** — `combine_hdf5_datasets()`: copies original HDF5, appends generated demos as `demo_N`, `demo_{N+1}`, … Returns the total demo count.
- **`policy_doctor/curation_pipeline/steps/select_mimicgen_seed.py`** — `SelectMimicgenSeedStep`: reads clustering, resolves rollout HDF5 via the same eval-dir chain as `SelectMimicgenSeedFromGraphStep`, dispatches to the configured heuristic, materialises `seed.hdf5`.
- **`policy_doctor/curation_pipeline/steps/train_on_combined_data.py`** — `TrainOnCombinedDataStep`: merges datasets, trains via `++task.dataset.dataset_path=<combined>` Hydra override (same pattern as `TrainCuratedStep`).
- **`generate_mimicgen_demos`** auto-wires with `SelectMimicgenSeedStep` (checked first) or the legacy `SelectMimicgenSeedFromGraphStep` (fallback).

---

## Diffusion policy training

Three scripts under `scripts/experiments/` launch full diffusion policy training runs directly (bypassing the curation pipeline). Each wraps `third_party/cupid/train.py` with the right conda env, config directory, and data paths. Any extra arguments are passed through as Hydra overrides.

### Data sources and environments

| Script | Conda env | Data source | Policy type |
|--------|-----------|-------------|-------------|
| `train_robomimic_square.sh` | `cupid` | Robomimic Square MH (`low_dim_abs.hdf5`) | Transformer, low-dim |
| `train_mimicgen_square.sh` | `cupid` | MimicGen Square D1 (`demo.hdf5`) | CNN, low-dim |
| `train_robocasa_atomic.sh` | `robocasa` | RoboCasa LeRobot v2 (no HDF5) | Transformer, image/hybrid |

The **robocasa** env uses robosuite 1.5.2 and evaluates entirely via live rollouts — no eval HDF5 is needed or used.

### Quick start

```bash
# Robomimic Square MH — full training run
./scripts/experiments/train_robomimic_square.sh

# MimicGen Square D1 — full training run
./scripts/experiments/train_mimicgen_square.sh

# RoboCasa atomic — PickPlaceCounterToCabinet (default)
./scripts/experiments/train_robocasa_atomic.sh

# RoboCasa atomic — different task (dataset auto-discovered from data/source/robocasa/)
./scripts/experiments/train_robocasa_atomic.sh OpenCabinet
```

### Overriding Hydra parameters

All extra positional arguments are forwarded as Hydra overrides:

```bash
# Custom output directory and W&B project
./scripts/experiments/train_robomimic_square.sh \
  multi_run.run_dir=/data/outputs/my_run \
  logging.project=corl_experiments

# Shorter run for debugging
./scripts/experiments/train_mimicgen_square.sh \
  training.num_epochs=10 \
  training.device=cuda:1

# RoboCasa with explicit dataset path (overrides auto-discovery)
./scripts/experiments/train_robocasa_atomic.sh PickPlaceCounterToCabinet \
  "task.dataset.dataset_path=/mnt/ssdB/erik/robocasa_data/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot"
```

Common Hydra keys across all three scripts:

| Key | Default | Meaning |
|-----|---------|---------|
| `training.num_epochs` | 800 | Total training epochs |
| `training.device` | `cuda:0` | PyTorch device |
| `multi_run.run_dir` | `data/outputs/...` (relative to `third_party/cupid`) | Output directory |
| `logging.project` | `diffusion_policy_debug` | W&B project name |
| `logging.mode` | `online` | Set to `offline` to disable W&B upload |
| `task.dataset.dataset_path` | set by script | Dataset path (HDF5 or LeRobot root) |
| `task.dataset_path` | set by script | Top-level alias (also set by script) |
| `task.env_runner.dataset_path` | set by script | Eval HDF5 path (robomimic/mimicgen only) |
| `task.env_runner.n_test` | 50 (varies) | Number of eval rollout episodes |

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `MUJOCO_GL` | `egl` | MuJoCo rendering backend (`egl`, `osmesa`, `glfw`) |
| `WANDB_MODE` | `online` | Set to `offline` to skip W&B cloud sync |

### Data paths

Scripts resolve datasets from **`data/source/`** at the project root (symlinks to `/mnt/ssdB/erik/`):

| Script | Dataset path |
|--------|-------------|
| `train_robomimic_square.sh` | `data/source/robomimic/datasets/square/mh/low_dim_abs.hdf5` |
| `train_mimicgen_square.sh` | `data/source/mimicgen/core_datasets/square/demo_src_square_task_D1/demo.hdf5` |
| `train_robocasa_atomic.sh` | `data/source/robocasa/v1.0/target/atomic/<TASK>/<latest-date>/lerobot/` (auto-discovered) |

---

## Shell scripts (this repository)

Scripts below live under **`scripts/`** at the **policy_doctor project root** (next to `pyproject.toml`).

| Script | Purpose |
|--------|---------|
| `scripts/install_policy_doctor_env.sh` | After `conda env create -f environment_policy_doctor.yaml`, installs `requirements_policy_doctor.txt` and `pip install -e` for `third_party/cupid`, `third_party/influence_visualizer`, and `.` |
| `scripts/install_cupid_env.sh` | Creates or updates conda env **`cupid`** from `third_party/cupid/conda_environment.yaml` (or sibling `../cupid`). Same role as `third_party/cupid/scripts/install_cupid_env.sh`; use whichever path you prefer. Flags: `--update` / `-u` |
| `scripts/install_mimicgen_env.sh` | Creates conda env **`mimicgen`** from `environment_mimicgen.yaml`, then pins MimicGen / robosuite / robomimic per NVlabs docs |

**Related (not in `scripts/`):**

- `python scripts/mimicgen_headless_smoke_test.py` — headless MimicGen / MuJoCo smoke check (`MUJOCO_GL=egl` or `osmesa`).

<details>
<summary><strong>Repository layout</strong></summary>

- **/** — Project root: `pyproject.toml`, `README.md`, `run_tests.py`, `tests/`, `scripts/`, `environment_*.yaml`, `third_party/`
- **policy_doctor/** — Importable package: attribution, `data/`, `computations/`, `behaviors/`, `curation/`, `plotting/`, `streamlit_app/`, `curation_pipeline/`, pipeline **`configs/`** (no bundled diffusion_policy / train workspaces)
- **third_party/cupid/** — Training stack (diffusion_policy, `eval_save_episodes.py`, robomimic Hydra configs); packaged as editable **`cupid-workspace`**
- **third_party/influence_visualizer/** — Data loading, clustering persistence, Streamlit-oriented helpers
- **third_party/mimicgen/** — Optional NVlabs MimicGen submodule; **mimicgen** conda env (older MuJoCo / robosuite pin)
- **third_party/robocasa/** — RoboCasa kitchen sim submodule (`git submodule update --init third_party/robocasa`)
Source package map (under `policy_doctor/`): **data** (trajectories, influence matrices), **computations**, **behaviors** (clustering, behavior graph), **curation**, **plotting** (Plotly + Pyvis), **streamlit_app** (UI only), **scripts/run_pipeline.py** (Hydra CLI).

</details>

## Conda environments

| Env | Install | Role |
|-----|---------|------|
| **`policy_doctor`** | `environment_policy_doctor.yaml` + `./scripts/install_policy_doctor_env.sh` | Package dev, tests, Streamlit, pipeline orchestration, clustering/curation steps that only need Python deps |
| **`cupid`** | `./scripts/install_cupid_env.sh` | Diffusion training, `eval_save_episodes`, TRAK, InfEmbed, curated retraining (Py 3.9, pinned sim stack) |
| **`mimicgen`** | `./scripts/install_mimicgen_env.sh` | MimicGen data generation (Py 3.8, MuJoCo 2.3.2, pinned robosuite / robomimic) |

Constants: `policy_doctor.paths.CUPID_CONDA_ENV_NAME`, `MIMICGEN_CONDA_ENV_NAME`, `POLICY_DOCTOR_CONDA_ENV_NAME`.

**Paths:** training and eval use `policy_doctor.paths.REPO_ROOT` (typically `third_party/cupid`). TRAK / InfEmbed settings live under Hydra `cfg.attribution` (`configs/robomimic/attribution/`) and pipeline steps `compute_demonstration_scores`, `compute_infembed`, `finalize_attribution`.

<details>
<summary><strong>Install details (system packages, CUDA, submodule)</strong></summary>

**Cupid**

```bash
./scripts/install_cupid_env.sh
./scripts/install_cupid_env.sh --update
```

System packages for `free-mujoco-py` (see cupid `conda_environment.yaml`): e.g. Debian/Ubuntu `libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf`.

**MimicGen**

```bash
./scripts/install_mimicgen_env.sh
conda activate mimicgen
```

Optional CUDA PyTorch: set `TORCH_INDEX` (see script header). Submodule: `git submodule update --init third_party/mimicgen`.

Headless: `MUJOCO_GL=egl python scripts/mimicgen_headless_smoke_test.py` (or `MUJOCO_GL=osmesa`).

Optional E2E (HF + sim): `MIMICGEN_E2E=1 python -m unittest tests.integration.test_mimicgen_square_e2e -v`.

**policy_doctor**

- **CUDA PyTorch:** adjust `requirements_policy_doctor.txt` (drop CPU index, install GPU wheels), then re-run editable installs.
- **PyTorch3D:** needed when loading real diffusion checkpoints in Streamlit / some tests; install to match your torch build (e.g. `conda install pytorch3d -c pytorch3d`).
- **Monorepo fallback:** if `third_party/cupid` is missing, `REPO_ROOT` can fall back to a parent tree that contains sibling `influence_visualizer/` (legacy layout).

</details>

## Streamlit app

From the project root, with `policy_doctor` env active:

```bash
streamlit run policy_doctor/streamlit_app/app.py
```

The Clustering tab and plotting stack expect `pyvis` for the interactive behavior graph when enabled.

## Config layout

Hydra defaults: **`policy_doctor/configs/config.yaml`**. **Data source** (which simulator / HDF5 family): Hydra group **`data_source`** in **`policy_doctor/configs/data_source/`** — default **`cupid_robomimic`** (transport MH + cupid); switch with `data_source=mimicgen_square` or `data_source=robocasa_layout`. Canonical diffusion / datagen YAMLs stay under **`third_party/cupid`** and **`third_party/mimicgen`**; policy_doctor composes them via `baseline.config_dir` and optional `baseline.diffusion_dataset_path` / `baseline.diffusion_compose_overrides`.

Robomimic task slices: **`policy_doctor/configs/robomimic/`** (`tasks/`, `baseline/`, `evaluation/`, `attribution/`, `curation_filtering/`, `curation_selection/`). Pipeline slice search / curation defaults: **`policy_doctor/configs/pipeline/config.yaml`**. Experiment presets (Hydra group **`experiment`**): **`policy_doctor/configs/experiment/`** — select with `experiment=name` or `+experiment=name` (e.g. `trak_filtering_mar13_p96`, `auto_pipeline_test_mar13`).

**Experiment shell wrappers** (conda `run_pipeline` with the right `data_source`): **`scripts/experiments/`** — e.g. `./scripts/experiments/run_cupid_robomimic_transport.sh cupid steps=[run_clustering] dry_run=true`. Direct diffusion training scripts (bypassing the pipeline) are also in `scripts/experiments/` — see [Diffusion policy training](#diffusion-policy-training).

**RoboCasa submodule:** `git submodule update --init third_party/robocasa` — path constant `policy_doctor.paths.ROBOCASA_ROOT`.

**Local data (not committed):** put HDF5 / exports under **`data/source/robomimic`**, **`data/source/robocasa`**, **`data/source/mimicgen`** at the **project root** (next to `pyproject.toml`). Constant `policy_doctor.paths.DATA_SOURCE_ROOT` points there. Diffusion and eval still read **`data/...` relative to `REPO_ROOT`** (vendored **`third_party/cupid`**), so either symlink (e.g. `third_party/cupid/data/source` → `../../../data/source` or per-dataset links into `cupid/data/robomimic/...`), or set **`baseline.diffusion_dataset_path`** / Hydra `++task.dataset.dataset_path=...` to an **absolute** path under `data/source/...`.

---

*If anything here disagrees with the code, treat the implementation (`run_pipeline.py`, `curation_pipeline/pipeline.py`, `configs/config.yaml`) as canonical.*
