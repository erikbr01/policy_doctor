# policy_doctor

Influence-based policy analysis and curation. Refactor of the influence_visualizer package with strict separation of data, computation, and visualization.

This tree is a **standalone repository**. Training, eval scripts, and Hydra configs come from vendored **cupid** under `third_party/cupid/`. Task YAMLs, clustering artifacts, and `influence_visualizer` live under `third_party/influence_visualizer/`. `policy_doctor.paths.REPO_ROOT` points at `third_party/cupid` when that directory exists (data paths such as `data/outputs/...` in configs are relative to that root). IV task configs are resolved via `iv_task_configs_base()` (`third_party/influence_visualizer/configs` in this layout).

## Layout

- **/** (this directory) — Repository root: `pyproject.toml`, `README.md`, `tests/`, `third_party/`
- **policy_doctor/** — Importable Python package `policy_doctor` (code + bundled `configs/`)
- **third_party/cupid/** — Vendored cupid (training stack, `eval_save_episodes.py`, `configs/` for Robomimic, etc.); packaged as editable **`cupid-workspace`** (`pyproject.toml`: namespace `diffusion_policy/*`, `influence_embeddings`, and root script modules). Excludes nested `policy_doctor` / `influence_visualizer`
- **third_party/influence_visualizer/** — Vendored influence_visualizer (data loading, clustering persistence, Streamlit helpers)

Source tree (high level):

- **data/** — Data structures (Trajectory, Segment, Sample), influence matrix wrappers (GlobalInfluenceMatrix, LocalInfluenceMatrix), backing store, loaders, aggregation
- **computations/** — Slice influence, embeddings, trajectory-level explanations
- **behaviors/** — Clustering (KNN, HDBSCAN, Gaussian Mixture; UMAP, PCA), behavior graph, slice values
- **curation/** — Curation config (save/load, fingerprint), attribution (slice search)
- **plotting/** — Pure Plotly figures (heatmaps, clusters, behavior graph, frames) and **Pyvis** interactive behavior graph (vis.js; returns HTML for st.components.v1.html). No Streamlit; used by streamlit_app and scripts. Ported from influence_visualizer.
- **streamlit_app/** — Streamlit UI (orchestration only): sidebar config, tabs (Clustering, Behavior graph, Annotation, Learning). Uses data/, computations/, behaviors/, curation/, and **plotting/** for figures.
- **scripts/** — `run_pipeline.py`: single CLI entry point for all pipeline steps (training, eval, attribution, infembed, clustering, curation, comparison)
- **curation_pipeline/** — Config loaders, path resolution, and step runners for the full curation flow (train baseline → eval → attribution → curation config → train curated)
- **configs/** — Task YAMLs, pipeline defaults, and **configs/robomimic/** for env-specific configs (baseline, evaluation, attribution, curation_filtering, curation_selection)
- **tests/** (under project root) — Unit tests for data, computations, behaviors, curation
- **third_party/mimicgen/** — Git submodule: NVlabs MimicGen (data-generation code). Used with a **dedicated** conda env (see below), not mixed into the cupid training stack.
- **policy_doctor/datagen/mimicgen/** — Scaffolding to turn rollouts into robomimic-shaped source HDF5 and call MimicGen scripts; safe to develop against the **mimicgen** env. Attribution / curation code is unchanged until we explicitly wire it to MimicGen outputs.

## Conda environments (three stacks)

We deliberately use **separate conda envs** for three concerns. Training data and rollouts are tied to a **specific MuJoCo + robosuite + robomimic** combination; NVlabs’ released MimicGen assets target an **older** stack (MuJoCo 2.3.2 and pinned robosuite/robomimic). Putting everything in one env would force either broken dataset replay or abandoned MimicGen compatibility.

| Env name | File / entrypoint | Role |
|----------|-------------------|------|
| **`policy_doctor`** | `environment_policy_doctor.yaml`, `scripts/install_policy_doctor_env.sh` | Package dev: tests, Streamlit, pipeline CLI, editable `policy_doctor` + vendored cupid + influence_visualizer. Python 3.10; PyTorch from `requirements_policy_doctor.txt` (CPU index by default). **Not** the full cupid sim pin set. |
| **`cupid`** | `third_party/cupid/conda_environment.yaml` (or top-level `cupid/`), `scripts/install_cupid_env.sh` | **Diffusion training and robomimic experiments** (e.g. transport, `eval_save_episodes`): Python 3.9, PyTorch 1.12 / CUDA 11.6, robosuite fork, robomimic 0.2, MuJoCo 3.2.6, editable `diffusion_policy` via the yaml `pip:` section. |
| **`mimicgen`** | `environment_mimicgen.yaml`, `scripts/install_mimicgen_env.sh` | **MimicGen data generation** aligned with [their install docs](https://mimicgen.github.io/docs/introduction/installation.html): Python 3.8, `mujoco==2.3.2`, robosuite + robomimic at the commits they specify, then `pip install -e third_party/mimicgen`. |

Constants for tooling: `policy_doctor.paths.CUPID_CONDA_ENV_NAME` / `MIMICGEN_CONDA_ENV_NAME`.

**Workflow in practice**

- Day-to-day curation and UI: `conda activate policy_doctor` (and install PyTorch3D if you load real diffusion checkpoints; see below).
- Train or eval policies, reproduce mar27-style transport runs: `conda activate cupid` after `./scripts/install_cupid_env.sh` (from this repo) or `./scripts/install_cupid_env.sh` from the cupid checkout.
- Run `prepare_src_dataset` / `generate_dataset` on NVlabs-style data: `conda activate mimicgen`. Future work: run attribution against artifacts produced here without merging the sim stacks.

**Cupid install (create or update)**

```bash
# From policy_doctor (discovers third_party/cupid or ../cupid)
./scripts/install_cupid_env.sh
./scripts/install_cupid_env.sh --update   # conda env update --prune

# Or from the cupid repo root
./scripts/install_cupid_env.sh
```

System packages for `free-mujoco-py` (see `conda_environment.yaml`): e.g. Debian/Ubuntu `libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf`.

**Mimicgen install**

```bash
./scripts/install_mimicgen_env.sh   # creates conda env mimicgen if missing, then pip pins
conda activate mimicgen
```

If you only changed `environment_mimicgen.yaml` and the env exists: `conda env update -f environment_mimicgen.yaml -n mimicgen`, then re-run the install script for pip steps.

Optional CUDA PyTorch for mimicgen: set `TORCH_INDEX` (see script header). Submodule: `git submodule update --init third_party/mimicgen`.

Headless / no `DISPLAY`: `MUJOCO_GL=egl python scripts/mimicgen_headless_smoke_test.py` (or `MUJOCO_GL=osmesa`). The upstream `demo_random_action.py` needs X11 because it imports keyboard utilities.

The install script copies robosuite `macros_private.py` and installs `robosuite_task_zoo` (pinned, `--no-deps`) so `import mimicgen` is quiet. If you created the env earlier, re-run `./scripts/install_mimicgen_env.sh` once to pick that up.

**MimicGen E2E tests** (optional, Hugging Face + sim): `MIMICGEN_E2E=1 python -m unittest tests.integration.test_mimicgen_square_e2e -v`.

### Conda environment `policy_doctor` (analysis / dev)

Editable installs for all three **code** trees — no `PYTHONPATH` or `sys.path` injection for normal development.

```bash
conda env create -f environment_policy_doctor.yaml
./scripts/install_policy_doctor_env.sh   # uses .../envs/policy_doctor/bin/python -m pip
conda activate policy_doctor
```

The script installs, in order: `requirements_policy_doctor.txt` (PyTorch CPU wheels by default), then `pip install -e third_party/cupid`, `pip install -e third_party/influence_visualizer`, `pip install -e .`.

- **CUDA PyTorch:** edit `requirements_policy_doctor.txt` — remove the `--extra-index-url …/cpu` line and install `torch` / `torchvision` the way you usually do for your GPU stack, then re-run the three `pip install -e` lines (or the whole script after commenting the duplicate torch install).

- **PyTorch3D:** `load_influence_data` pulls in diffusion_policy modules that import `pytorch3d`. For integration tests and Streamlit on real checkpoints, install it the usual way for your platform (e.g. `conda install pytorch3d -c pytorch3d`, matching your CUDA PyTorch build).

- **Monorepo fallback:** if `third_party/cupid` is missing, `REPO_ROOT` in `policy_doctor.paths` falls back to the parent directory when it contains a sibling `influence_visualizer/` (legacy cupid layout). You still need `influence_visualizer` and `diffusion_policy` importable (e.g. editable installs from those repos).

## Running tests

From the repo root with `conda activate policy_doctor`:

```bash
python run_tests.py
```

Or with pytest:

```bash
pytest tests/ -v
```

## Streamlit app

From the repo root:

```bash
streamlit run policy_doctor.streamlit_app.app
```

Select a task config from the sidebar (e.g. `transport_mh_jan28`). Data loading uses `policy_doctor.data.influence_loader` (requires influence_visualizer when not using a native loader). The Clustering tab shows an influence heatmap preview via `policy_doctor.plotting`. Static figures use Plotly; the interactive draggable behavior graph uses **Pyvis** (vis.js) — same as influence_visualizer. Install with `pip install pyvis` for the interactive graph view.

## Config layout and curation pipeline

Configs are under **configs/robomimic/** (and optionally other envs later):

- **tasks/** — Task-level defaults (dataset_path, obs_dim, action_dim) for lift_mh, square_mh, transport_mh
- **baseline/** — Baseline training (no curation): one YAML per state/task (e.g. low_dim/lift_mh.yaml) with method, seeds, epochs, dataset splits
- **evaluation/** — Params for eval_save_episodes (train_date, eval_date, num_episodes, device, etc.)
- **attribution/** — Params for train_trak, finalize_trak, eval_demonstration_scores (TRAK and scoring flags)
- **curation_filtering/** — Retrain with filtered train set; references baseline + optional curation_config_path
- **curation_selection/** — Retrain with train + selected holdout; references baseline + curation_config_path (e.g. from run_pipeline)
- **configs/pipeline/** — Defaults for run_pipeline (clustering → behavior graph → slice search → save curation YAML)

Override any key via the CLI when running the pipeline (e.g. `train_date=jan18`).

### Running one step or the full pipeline

From the **repo root**:

```bash
# Single step (positional: step, env, state, task; then KEY=VALUE overrides)
python -m policy_doctor.scripts.run_pipeline train_baseline robomimic low_dim lift_mh train_date=jan18
python -m policy_doctor.scripts.run_pipeline eval_policies robomimic low_dim lift_mh train_date=jan18
python -m policy_doctor.scripts.run_pipeline train_attribution robomimic low_dim lift_mh train_date=jan18 eval_date=jan18
python -m policy_doctor.scripts.run_pipeline compute_infembed robomimic low_dim transport_mh seeds=[1,2] train_date=jan28 eval_date=jan28
python -m policy_doctor.scripts.run_pipeline compute_demonstration_scores robomimic low_dim lift_mh train_date=jan18
# Curation config (requires task_config and clustering_dir in overrides)
python -m policy_doctor.scripts.run_pipeline run_curation_config robomimic low_dim transport_mh task_config=transport_mh_jan28 clustering_dir=../influence_visualizer/configs/transport_mh_jan28/clustering/sliding_window_rollout_kmeans_k15_2026_03_05
# Curated training (set curation_config_path in config or overrides); paths are relative to third_party/cupid when using the default repo_root
python -m policy_doctor.scripts.run_pipeline train_curated robomimic low_dim transport_mh train_date=jan18 curation_config_path=../influence_visualizer/configs/transport_mh_jan28/curation/test_advantage_selection.yaml

# Full pipeline (all steps in order)
python -m policy_doctor.scripts.run_pipeline full robomimic low_dim transport_mh train_date=jan18 eval_date=jan28
```

Steps: `train_baseline` → `eval_policies` → `train_attribution` → `finalize_attribution` (if multi-ckpt) → `compute_demonstration_scores` → `compute_infembed` → `run_clustering` → `run_curation_config` → `train_curated` → `eval_curated` → `compare`. Use `--dry-run` to print commands without running.
