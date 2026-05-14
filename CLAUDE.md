# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Tests — run from project root with matching conda env active
python run_tests.py --suite policy_doctor   # orchestration, data, behaviors, VLM (policy_doctor env)
python run_tests.py --suite cupid           # diffusion_policy integration (cupid env)
python run_tests.py --suite mimicgen        # MimicGen seeds (mimicgen env)

# Shell wrappers (handle conda activation automatically):
./scripts/run_tests_policy_doctor.sh
./scripts/run_tests_cupid.sh
./scripts/run_tests_mimicgen.sh

# Run a single test module
conda activate policy_doctor && python -m unittest tests.behaviors.test_behavior_graph -v

# Pipeline (Hydra entry point)
python -m policy_doctor.scripts.run_pipeline steps=[run_clustering] task_config=transport_mh_jan28
python -m policy_doctor.scripts.run_pipeline +experiment=trak_filtering_mar13_p96 \
  steps=[run_clustering,run_curation_config,train_curated,eval_curated]
python -m policy_doctor.scripts.run_pipeline steps=[eval_policies] dry_run=true train_date=jan28 eval_date=jan28

# MimicGen full experiment (select seed → generate → train → eval for all arms)
python -m policy_doctor.scripts.run_pipeline \
  data_source=mimicgen_square \
  experiment=mimicgen_square_pipeline_apr23

# MimicGen ablations reusing an existing run_dir (run in same run_dir as main experiment)
python -m policy_doctor.scripts.run_pipeline \
  data_source=mimicgen_square \
  experiment=mimicgen_square_ablations_apr23 \
  steps=[mimicgen_random_20,mimicgen_behavior_graph_20]

# MimicGen variance sweep (standalone, no pipeline — ablates generation knobs on D0)
./scripts/run_variance_sweep.sh       # generates to /tmp/mimicgen_variance_sweep/
./scripts/run_variance_sweep_finish.sh  # re-runs last arm + plots all

# Plot EEF trajectories from a variance run result.json
conda activate policy_doctor && python scripts/plot_mimicgen_eef_from_result.py \
  --result /tmp/mimicgen_variance_sweep/A_baseline/result.json \
  --out_dir /tmp/out/A_baseline

# Training (bypasses pipeline, wraps cupid/train.py)
./scripts/experiments/train_robomimic_square.sh --compile --tf32
./scripts/experiments/train_robomimic_square.sh --num-gpus 2 --compile --tf32
./scripts/experiments/train_mimicgen_square.sh --compile training.device=cuda:1
./scripts/experiments/train_robocasa_atomic.sh OpenCabinet

# Attribution
python -m policy_doctor.scripts.run_pipeline steps=[train_attribution] \
  attribution.tf32=false attribution.compile=false train_date=jan18 eval_date=jan28

# Runtime monitoring — classify policy behavior per-timestep (cupid_torch2 env)
# Run from third_party/cupid/ (diffusion_policy must be on PYTHONPATH)
python ../../scripts/monitor_online.py \
  --output_dir /tmp/out --train_dir <dir> --train_ckpt best \
  --infembed_fit <pt> --infembed_npz <npz> --clustering_dir <dir> \
  --episodes_dir <eval_dir>/episodes   # required when clustering level is "rollout"

# Offline monitoring of a saved rollout pkl or HDF5 demo
python scripts/monitor_offline.py \
  --episode <pkl> --checkpoint <ckpt> \
  --infembed_fit <pt> --infembed_npz <npz> --clustering_dir <dir>

# Streamlit app
conda activate policy_doctor && streamlit run policy_doctor/streamlit_app/app.py
```

## Architecture

### Conda environment split

The codebase is split across four conda environments because `cupid` requires PyTorch 1.12 + old robosuite/robomimic pinned for MuJoCo compatibility:

| Env | Use for |
|-----|---------|
| `policy_doctor` | Analysis, clustering, pipeline orchestration, Streamlit, most unit tests |
| `cupid` | Training, eval rollouts, TRAK attribution — anything that imports `diffusion_policy` |
| `cupid_torch2` | InfEmbed attribution, runtime monitoring scripts (requires `torch.func` — absent in `cupid`'s torch 1.12); also `--compile` / `torch.compile` |
| `cupid_torch25` | **torch 2.5.1+cu124** upgrade of `cupid_torch2`; fixes the torch 2.4.x TensorAlias AOT-autograd bug so `obs_encoder` can be compiled — full `torch.compile` gives **1.19× fwd+bwd** vs eager on the droid image policy. See `scripts/create_cupid_torch25.sh` to recreate. |
| `mimicgen` | Legacy MimicGen env (Py 3.8, MuJoCo 2.3.2, cpu-only torch 1.12) — superseded by `mimicgen_torch2` |
| `mimicgen_torch2` | **Primary training/eval/MimicGen env**: clone of `cupid_torch2` with robosuite 1.4.1 + robomimic 0.3.0 + mimicgen (all compatible, correct `is_success()`) |

### Three-package layout

`third_party/cupid` and `third_party/influence_visualizer` are editable installs alongside the main `policy_doctor` package:

- **`policy_doctor/`** — attribution analysis, data structures, clustering, behavior graph, curation logic, Hydra pipeline orchestration, Streamlit UI
- **`third_party/cupid/`** — diffusion policy training workspace (`train.py`), TRAK (`train_trak_diffusion.py`), InfEmbed (`compute_infembed_embeddings.py`), `eval_save_episodes.py`; packaged as `cupid-workspace`
- **`third_party/influence_visualizer/`** — data loading from disk (HDF5, TRAK results), clustering persistence, Streamlit render helpers

### Path resolution: REPO_ROOT vs PROJECT_ROOT

`policy_doctor.paths` exposes several roots:

- `PACKAGE_ROOT` = `policy_doctor/policy_doctor/` (where configs, curation_pipeline live)
- `PROJECT_ROOT` = `policy_doctor/` (pyproject.toml, tests/, scripts/)
- `REPO_ROOT` = `third_party/cupid` when that directory exists — **training, eval, and attribution scripts resolve `data/outputs/...` from here**, not `PROJECT_ROOT`
- `DATA_SOURCE_ROOT` = `PROJECT_ROOT/data/source/` — local HDF5 datasets (gitignored; symlink or pass absolute path to `baseline.diffusion_dataset_path`)

Consequence: pipeline run directories default to `third_party/cupid/data/pipeline_runs/<run_name>/` and checkpoints under `third_party/cupid/data/outputs/train/...`.

Training artifact path pattern: `data/outputs/train/<train_date>/<train_date>_train_<policy>_<task>_<seed>/` (see `curation_pipeline/paths.py:get_train_dir`).

### Attribution methods

Both TRAK and InfEmbed compute pairwise influence between rollout (test) samples and training demonstrations:

**TRAK** (`train_trak_diffusion.py`):
1. Offline: `traker.featurize()` computes per-sample gradients and projects them to `proj_dim` (default 2048) via random Johnson-Lindenstrauss projection — this is the cacheable low-dim projection
2. Offline: `traker.finalize_features()` approximates the (regularized) Hessian from the projected gradients
3. Online: `traker.score()` computes projected gradients for test samples and dots them against the cached Hessian-inverse-weighted train projections
4. Output: `(num_train_samples, num_test_samples)` influence matrix; called "TRAK scores"

**InfEmbed** (`compute_infembed_embeddings.py`):
1. Offline: `ArnoldiEmbedder.fit()` uses Arnoldi iteration to find the `arnoldi_dim` (default 200) dominant eigenvectors of the Gauss-Newton Hessian of the training loss — this approximates `H^{-1/2}` and is the cacheable Hessian factor
2. Online: `ArnoldiEmbedder.predict()` projects gradients of any sample (train or rollout) into the `projection_dim` (default 100) embedding space using the fitted Hessian factor
3. Output: per-sample embeddings in R^100; influence between two samples ≈ dot product of their embeddings
4. Both use JL-style dimensionality reduction — TRAK at the projection step; InfEmbed implicitly via the Arnoldi truncation

The InfEmbed fit (`infembed_fit.pt`) and TRAK projections are both disk-cached — only the predict step needs to run for new samples.

### Influence matrix and data structures

The core data structure is `GlobalInfluenceMatrix` (`policy_doctor/data/structures.py`): a `(num_rollout_samples, num_demo_samples)` float32 matrix where each cell is the scalar influence of one training demo sample on one rollout sample. Access via `get_slice(r_lo, r_hi, d_lo, d_hi)` — never index directly, as the backing store may be a memmap.

Episodes are represented as `EpisodeInfo` (index, num_samples, sample_start_idx, sample_end_idx, success). Global indices in the matrix are contiguous across episodes; per-episode offsets are stored in `sample_start_idx`/`sample_end_idx`.

`GlobalInfluenceMatrix.get_local_matrix(rollout_idx, demo_idx)` returns a `LocalInfluenceMatrix` for one trajectory pair.

### Behavior graph and clustering

The clustering → behavior graph pipeline:

1. **Embeddings**: `policy_doctor/computations/embeddings.py` extracts per-timestep embeddings from the influence matrix by summing along one axis. The result is `(num_rollout_samples, num_demo_samples)` — each row is a rollout timestep embedded in "demo influence space"
2. **Dimensionality reduction**: `policy_doctor/behaviors/clustering.py` → `reduce_dimensions()` (UMAP or PCA, default UMAP to 2D)
3. **Clustering**: `run_clustering()` runs HDBSCAN / GMM / KMeans on the (optionally normalized) embeddings; returns per-timestep `cluster_labels` and 2D `coords`
4. **Behavior graph**: `policy_doctor/behaviors/behavior_graph.py` → `BehaviorGraph.from_cluster_assignments()` builds a Markov chain from run-length-collapsed cluster sequences per episode. Nodes: behavioral clusters + START/SUCCESS/FAILURE/END. Edges: transition counts and probabilities. `compute_values()` solves linear Bellman equations to assign V-values to each node

The graph is used for two things: visualization (Streamlit, Pyvis) and slice search — `get_rollout_slices_for_paths()` extracts the raw timestep spans from episodes that follow a given path through the graph.

### Pipeline step system

`policy_doctor/curation_pipeline/pipeline.py` orchestrates steps in `ALL_STEPS` order. Each step:
- Inherits from `PipelineStep` (`base_step.py`)
- Writes `<run_dir>/<step_name>/done` as a sentinel on completion
- Writes `<run_dir>/<step_name>/result.json` with its outputs (e.g. clustering dir paths)
- `skip_if_done=true` (default) means re-running resumes from the last incomplete step
- Steps that invoke the sim stack (`train_attribution`, `compute_infembed`, `train_curated`, etc.) shell out to `conda run -n cupid` to use the correct environment

**`CompositeStep`** (`base_step.py`) groups a fixed sub-step sequence under one namespace:
- Sub-steps write to `<run_dir>/<composite_name>/<sub_step_name>/`; resumability is per-sub-step
- Sub-steps can still read sibling top-level step results (e.g. `run_clustering/`) via `parent_run_dir`, which is transparently set to the top-level run root
- Subclasses declare `sub_step_classes` (ordered list) and `cfg_overrides` (dotpath→value applied before any sub-step)

### MimicGen trajectory generation pipeline

The MimicGen experiment tests whether behavior-graph-guided seed selection improves generated data quality. The pipeline compares three heuristics, all sharing upstream `run_clustering` results:

**Seed selection heuristics** (`policy_doctor/mimicgen/heuristics.py`):
- `BehaviorGraphPathHeuristic` — ranks paths to SUCCESS by probability, returns rollouts matching the highest-probability path (proposed method)
- `DiversitySelectionHeuristic` — takes one rollout per path before moving to the next, maximizing behavioral diversity across seeds
- `RandomSelectionHeuristic` — uniform random from eligible (successful) rollouts (baseline)

**Sub-step sequence per arm** (implemented as `CompositeStep` in `steps/mimicgen_arm.py`):
1. `select_mimicgen_seed_from_graph` — uses behavior graph to pick N seed rollouts; materializes `seed.hdf5`
2. `generate_mimicgen_demos` — runs MimicGen in `mimicgen_torch2` env; auto-wires to `seed.hdf5` from prior step
3. `train_on_combined_data` — merges original + generated HDF5, trains policy in `mimicgen_torch2` env
4. `eval_mimicgen_combined` — evaluates the retrained policy

**Run directory layout** with multiple arms sharing a run:
```
<run_dir>/
    run_clustering/               # shared upstream result
    mimicgen_random/              # CompositeStep arm: random heuristic
        select_mimicgen_seed_from_graph/
        generate_mimicgen_demos/
        train_on_combined_data/
    mimicgen_behavior_graph/      # CompositeStep arm: BG heuristic
    mimicgen_diversity/           # CompositeStep arm: diversity heuristic
```

**Adding a new arm**: subclass `CompositeStep`, set `name`, `sub_step_classes = _SUB_STEPS`, and `cfg_overrides`, then register in `pipeline.py`'s `ALL_STEPS` and `_build_step_registry()`.

**MimicGen generation parameters** live in `policy_doctor/configs/mimicgen/<task>.yaml` (e.g. `square_d0.yaml`, `square_d1.yaml`). Key variance knobs: `action_noise`, `subtask_term_offset_range`, `nn_k`, `interpolate_from_last_target_pose`, `fix_initial_object_poses`, `object_pose_ranges` (per-object, per-axis offset from seed pose).

### Streamlit / visualization separation

Strict separation (enforced in `third_party/cupid/CLAUDE.md` and followed throughout):
- **`render_*.py`** files contain all Streamlit UI (buttons, sliders, `st.*`) and data preprocessing
- **`policy_doctor/plotting/`** contains pure functions that accept preprocessed data and return `plotly.Figure` objects — no Streamlit imports
- Plotting functions are exported from `policy_doctor/plotting/__init__.py`
- Never import Streamlit in plotting modules; never create Plotly figures inside render functions

### Runtime monitoring (`policy_doctor/monitoring/`)

Assigns each policy timestep to a behavior graph node in real time. Full documentation in `docs/monitoring.md`.

Component layers: `InfEmbedStreamScorer` → `FittedModelAssigner` / `NearestCentroidAssigner` → `StreamMonitor` → `TrajectoryClassifier` → `MonitoredPolicy`. The scorer requires `cupid_torch2`; the assigner and graph are pure numpy and run in `policy_doctor`.

`FittedModelAssigner` requires `clustering_models.pkl` in the clustering directory (saved by `run_clustering` step). If absent, `NearestCentroidAssigner` is used as a fallback (nearest centroid in raw embedding space). When clustering level is `"rollout"` (the default), pass `--episodes_dir` to both online/offline scripts so window-mean embeddings can be computed from `metadata.yaml`.

`infembed` is installed as a package in `cupid_torch2` via `pip install -e third_party/cupid/third_party/infembed`.

### Hydra config layout

- Base config: `policy_doctor/configs/config.yaml` — sets defaults for `data_source`, `pipeline/config`, `vlm/defaults`, and `experiment: null`
- Task/simulator profile: `data_source` group (`cupid_robomimic`, `mimicgen_square`, `robocasa_layout`)
- MimicGen generation params: `mimicgen` group (`square_d0`, `square_d1`, `coffee`, `threading`) — load with `mimicgen=square_d1` or inline under `mimicgen_datagen:` in experiment YAML
- Robomimic-specific slices: `policy_doctor/configs/robomimic/` (tasks, baseline, evaluation, attribution, curation_filtering, curation_selection)
- Experiment presets: `policy_doctor/configs/experiment/` — selected with `+experiment=name` (the `+` is needed when adding a new defaults group not present in the base config)
- Attribution compile/tf32 flags live in YAML (`attribution.tf32`, `attribution.compile`), not CLI flags like training scripts
