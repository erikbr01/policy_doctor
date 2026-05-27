# policy_doctor

Influence-based policy analysis and curation. The repo is structured as a single uv workspace with one editable third-party member (`third_party/cupid`) for the diffusion-policy training stack. Hydra configs for the pipeline ship inside the `policy_doctor` package (`policy_doctor/configs/`).

`policy_doctor.paths.REPO_ROOT` points at `third_party/cupid` when that directory exists, so legacy paths such as `data/outputs/train/...` resolve from the cupid tree. The new experiment layer (`policy_doctor.experiment`) writes to `<repo>/data/experiments/<name>/` and is the canonical place for new artifact layouts.

## Quick start

```bash
# One-time: create the analysis venv at .venvs/analysis/
./scripts/uv_env.sh analysis --setup

# Run the canonical golden / experiment / env-dispatch suite (40 tests)
./scripts/uv_env.sh analysis pytest tests/golden/ tests/experiment/ tests/test_env_dispatch.py
```

Other extras (`cupid`, `mimicgen`, `robocasa`) are created the same way:

```bash
./scripts/uv_env.sh cupid    --setup
./scripts/uv_env.sh mimicgen --setup
./scripts/uv_env.sh robocasa --setup
```

The `cupid` / `mimicgen` / `robocasa` extras pull in heavy simulator stacks. They are validated on Linux x86_64 + Python 3.10; macOS / Apple Silicon installs of the sim extras may fail on legacy transitive deps (free-mujoco-py, etc.). The `analysis` env works everywhere and covers all orchestration / clustering / Streamlit code paths.

## Experiments

The experiment layer creates a self-contained on-disk directory per named run, under `<repo>/data/experiments/<name>/` (override via `$POLICY_DOCTOR_DATA`).

```bash
# Create a new experiment skeleton
./scripts/uv_env.sh analysis python -m policy_doctor.scripts.experiment_init my_experiment

# Or copy a baseline checkpoint from an existing one
./scripts/uv_env.sh analysis python -m policy_doctor.scripts.experiment_init my_experiment \
    --baseline-from prior_experiment

# Bundle for cross-machine transfer (dereferences symlinks under shared/)
./scripts/uv_env.sh analysis python -m policy_doctor.scripts.experiment_bundle my_experiment \
    --out /tmp/my_experiment.tar.gz
```

Directory layout:

```
data/experiments/<name>/
    manifest.yaml          # name, created_at, baseline_from, free-form keys
    config/                # snapshot_<utc>.yaml per invocation + canonical.yaml symlink
    shared/                # baseline_ckpt, source datasets
    artifacts/<step>/      # one dir per pipeline step
        seed_<seed>/<ckpt>/
    logs/<label>_<utc>.log
```

`Experiment.step_dir(step_name)`, `seed_dir(step_name, seed)`, and `ckpt_dir(step_name, seed, ckpt)` are the canonical path helpers.

## Curation pipeline (Hydra)

Entry point: `policy_doctor.scripts.run_pipeline`. Uses Hydra (`policy_doctor/configs/config.yaml` plus optional `+experiment=...`).

`steps: []` in the base config means "run the full ordered sequence" (see [Pipeline step order](#pipeline-step-order)). Override with `steps=[...]` on the command line or via an experiment YAML under `policy_doctor/configs/experiment/`.

**Run directory:** unless you pass an absolute `run_dir`, it defaults to `<REPO_ROOT>/data/pipeline_runs/<run_name>/`. With vendored cupid, `REPO_ROOT` is `third_party/cupid`. Each completed step writes `<step_name>/done` and (usually) `<step_name>/result.json`. With `skip_if_done=true` (default), re-running resumes from the last incomplete step.

**Common Hydra overrides:**

| Override | Meaning |
|----------|---------|
| `+experiment=name` | Merge `policy_doctor/configs/experiment/<name>.yaml` |
| `steps=[a,b]` | Run only those steps, in order |
| `dry_run=true` | Print planned work; no heavy compute |
| `skip_if_done=false` | Ignore `done` sentinels and re-execute |
| `run_name=myrun` | Stable name for `run_dir` (default is a timestamp) |
| `run_dir=/abs/path` | Fixed run folder (absolute path) |

**Examples:**

```bash
# Full pipeline
./scripts/uv_env.sh analysis python -m policy_doctor.scripts.run_pipeline \
    train_date=jan28 eval_date=jan28

# Clustering -> curation configs -> curated train/eval (training steps shell out to cupid)
./scripts/uv_env.sh analysis python -m policy_doctor.scripts.run_pipeline \
    +experiment=trak_filtering_mar13_p96 \
    steps=[run_clustering,run_curation_config,train_curated,eval_curated]

# Resume in the same run_dir
./scripts/uv_env.sh analysis python -m policy_doctor.scripts.run_pipeline \
    +experiment=trak_filtering_mar13_p96 \
    run_dir=third_party/cupid/data/pipeline_runs/myrun \
    steps=[train_curated,eval_curated]

# Dry-run one step
./scripts/uv_env.sh analysis python -m policy_doctor.scripts.run_pipeline \
    steps=[eval_policies] dry_run=true train_date=jan28 eval_date=jan28
```

Pipeline steps that need a sim stack (training, eval, MimicGen generation) dispatch via `policy_doctor._env.run_in_env(extra, ...)` — no `conda run -n` shell-outs anywhere in the tree.

### Pipeline step order

From `policy_doctor.curation_pipeline.pipeline.ALL_STEPS`:

1. `train_baseline` → `eval_policies` → `train_attribution` → `finalize_attribution` → `compute_demonstration_scores` → `compute_infembed` → `run_clustering` → `export_markov_report` → `annotate_slices_vlm` → `summarize_behaviors_vlm` → `evaluate_cluster_coherency_vlm` → `run_curation_config` → `train_curated` → `eval_curated` → `compare`
2. **MimicGen sub-pipeline:** `select_mimicgen_seed` → `generate_mimicgen_demos` → `train_on_combined_data` → `eval_mimicgen_combined`

VLM steps are no-ops unless configured; `finalize_attribution` skips when `attribution.num_ckpts <= 1`. `evaluate_cluster_coherency_vlm` requires `annotate_slices_vlm` outputs in the same `run_dir`.

## MimicGen trajectory generation pipeline

Compares seed-selection heuristics for MimicGen-based data augmentation. Three heuristics share an upstream `run_clustering` step:

| Heuristic | Key | Description |
|-----------|-----|-------------|
| Behavior graph (proposed) | `behavior_graph` | Ranks paths to SUCCESS by probability; picks the first rollout matching the highest-probability path |
| Diversity | `diversity` | One rollout per path before moving to the next |
| Random (baseline) | `random` | Uniform over eligible successful rollouts |

Both arms below run end-to-end (seed select → generate → train → eval), with each arm in its own sub-directory under the shared `run_dir`:

```bash
# Step 0: clustering, shared between arms
./scripts/uv_env.sh analysis python -m policy_doctor.scripts.run_pipeline \
    run_dir=data/pipeline_runs/mimicgen_experiment \
    steps=[run_clustering]

# Proposed method
./scripts/uv_env.sh mimicgen python -m policy_doctor.scripts.run_pipeline \
    data_source=mimicgen_square \
    experiment=mimicgen_square_pipeline_apr23 \
    run_dir=data/pipeline_runs/mimicgen_experiment

# Ablations reusing an existing run_dir
./scripts/uv_env.sh mimicgen python -m policy_doctor.scripts.run_pipeline \
    data_source=mimicgen_square \
    experiment=mimicgen_square_ablations_apr23 \
    steps=[mimicgen_random_20,mimicgen_behavior_graph_20]
```

Generation knobs live in `policy_doctor/configs/mimicgen/<task>.yaml` (e.g. `square_d0.yaml`, `square_d1.yaml`).

## Direct training (bypasses pipeline)

Three shell wrappers under `scripts/experiments/` launch diffusion-policy training directly against `third_party/cupid/train.py`:

| Script | Env extra | Data source | Policy type |
|--------|-----------|-------------|-------------|
| `train_robomimic_square.sh` | `cupid` | Robomimic Square MH (`low_dim_abs.hdf5`) | Transformer, low-dim |
| `train_mimicgen_square.sh` | `mimicgen` | MimicGen Square D1 (`demo.hdf5`) | CNN, low-dim |
| `train_robocasa_atomic.sh` | `robocasa` | RoboCasa LeRobot v2 (no HDF5) | Transformer, image |

Flags before any Hydra overrides:

| Flag | Effect |
|------|--------|
| `--compile` | Wrap with `torch.compile` |
| `--no-compile` | Disable compilation |
| `--tf32` | Enable TF32 matmul |
| `--no-tf32` | Disable TF32 |
| `--num-gpus N` | Use N GPUs via `torchrun` (DDP); default 1 |

```bash
./scripts/experiments/train_robomimic_square.sh --compile --tf32
./scripts/experiments/train_robomimic_square.sh --num-gpus 2 --compile --tf32
./scripts/experiments/train_mimicgen_square.sh   --compile training.device=cuda:1
./scripts/experiments/train_robocasa_atomic.sh   OpenCabinet
```

`torch.compile` notes: `fullgraph=True, dynamic=False` for the diffusion backbones; EMA reads via `ddp_util.unwrap_model` (strips both `DDP.module` and `._orig_mod`). Multi-GPU uses `torchrun` + DDP; rank 0 handles W&B + checkpoints.

**`training.num_steps`** is an alternative to `num_epochs`: when set, the workspace computes `num_epochs = ceil(num_steps / steps_per_epoch)` and breaks out of the inner batch loop once `global_step >= num_steps`. Mutually exclusive with `num_epochs`. With DDP + `DistributedSampler`, `num_steps` keeps the gradient-update count constant across GPU counts; `num_epochs` doubles the effective batch size with 2 GPUs (scale `training.lr` accordingly).

### Attribution performance flags (TRAK / InfEmbed)

YAML keys in `policy_doctor/configs/robomimic/attribution/low_dim/*.yaml`:

```yaml
tf32: true     # torch.backends.cuda.matmul.allow_tf32 + cudnn.allow_tf32
compile: true  # torch.compile the policy / DiffusionLossWrapper before attribution
```

Both default to `true`. TRAK compiles with `dynamic=True, fullgraph=False` (because `torch.func.grad` + `vmap` introduce control flow); InfEmbed compiles `DiffusionLossWrapper` with `fullgraph=False` (because `functional_call` receives an `nn.Module` argument that dynamo can't trace through with `fullgraph=True`). Numerical equivalence vs. eager is verified to `atol=1e-5` in `tests/attribution/test_attribution_flags.py`.

Override on the command line:

```bash
./scripts/uv_env.sh cupid python -m policy_doctor.scripts.run_pipeline \
    steps=[train_attribution] \
    attribution.tf32=false attribution.compile=false \
    train_date=jan18 eval_date=jan28
```

## Streamlit apps

```bash
# Main analysis UI
./scripts/uv_env.sh analysis streamlit run policy_doctor/streamlit_app/app.py

# Researcher graph explorer
./scripts/uv_env.sh analysis streamlit run policy_doctor/streamlit_app/demo_app/Home.py

# Participant survey (Group A/B)
./scripts/uv_env.sh analysis streamlit run policy_doctor/streamlit_app/user_study/Home.py
```

The Clustering tab and plotting stack use `pyvis` for the interactive behavior graph. All three apps are pure Python — no separate visualization daemon. The legacy non-Streamlit viz scripts were removed in Phase 4 of the refactor.

## Repository layout

```
/                            pyproject.toml, README.md, run_tests.py, tests/, scripts/, .venvs/, third_party/
policy_doctor/               importable package: attribution, data/, computations/, behaviors/,
                             curation/, plotting/, streamlit_app/, curation_pipeline/, experiment/,
                             influence/, monitoring/, scripts/, configs/
third_party/cupid/           training stack (diffusion_policy, eval_save_episodes, robomimic Hydra configs);
                             editable workspace member as cupid-workspace
third_party/cupid/third_party/infembed/  InfEmbed library, editable as infembed
third_party/mimicgen/        optional NVlabs MimicGen submodule (mimicgen extra pulls source from git)
third_party/robocasa/        RoboCasa kitchen sim submodule
scripts/                     uv_env.sh + setup/ (env install scripts), experiments/ (training + sweep wrappers),
                             dev/ (one-off conversion / debug tools), benchmarks/
data/                        gitignored. source/ (local HDF5 datasets), experiments/ (experiment dirs),
                             pipeline_runs/ (legacy)
```

## Config layout

Hydra base: `policy_doctor/configs/config.yaml`.

**Data source** (which simulator / HDF5 family): Hydra group `data_source` in `policy_doctor/configs/data_source/` — default `cupid_robomimic`; switch with `data_source=mimicgen_square` or `data_source=robocasa_layout`.

Canonical diffusion / datagen YAMLs stay under `third_party/cupid` and `third_party/mimicgen`; `policy_doctor` composes them via `baseline.config_dir` and optional `baseline.diffusion_dataset_path` / `baseline.diffusion_compose_overrides`.

Robomimic task slices: `policy_doctor/configs/robomimic/`. Pipeline slice search / curation defaults: `policy_doctor/configs/pipeline/config.yaml`. Experiment presets (Hydra group `experiment`): `policy_doctor/configs/experiment/` — select with `+experiment=name`.

**RoboCasa submodule:** `git submodule update --init third_party/robocasa` — path constant `policy_doctor.paths.ROBOCASA_ROOT`.

**Local data:** put HDF5 / exports under `data/source/robomimic`, `data/source/robocasa`, `data/source/mimicgen` at the project root. Constant `policy_doctor.paths.DATA_SOURCE_ROOT` points there. Diffusion and eval read `data/...` relative to `REPO_ROOT` (`third_party/cupid`), so either symlink (e.g. `third_party/cupid/data/source` → `../../../data/source`) or set `baseline.diffusion_dataset_path` / `++task.dataset.dataset_path=...` to an absolute path.

---

*If anything here disagrees with the code, treat the implementation (`run_pipeline.py`, `curation_pipeline/pipeline.py`, `configs/config.yaml`, `_env.py`) as canonical.*
