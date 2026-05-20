# How to Run a MimicGen Budget × Rep Sweep

This guide documents the end-to-end process for launching a MimicGen augmentation
experiment on a new task. It is derived from the **CoffeePreparation D1 May 18 2026**
experiment and generalizes the pattern.

The experiment tests whether MimicGen-generated demonstrations improve policy success
rate, and whether the seed-selection heuristic (random vs. behavior-graph) matters.
It produces a matrix of trained policies across:

- **Heuristics**: `random`, `behavior_graph` (diversity not yet implemented)
- **Budgets**: e.g. `[100, 300, 500]` generated demos appended to the baseline
- **Reps**: 3 fixed selection seeds to estimate variance

---

## Prerequisites

### Conda environments

| Env | Purpose |
|-----|---------|
| `mimicgen` | D1 pool generation (MuJoCo 2.3.2, Python 3.8) |
| `mimicgen_torch2` | Training, eval, InfEmbed attribution, datagen |
| `policy_doctor` | Pipeline orchestration, clustering, Streamlit |

### Data

The official MimicGen D1 release datasets (square, coffee_preparation, etc.) live at:
```
data/source/mimicgen/core_datasets/<task>/demo.hdf5
```
These have **1000 demos** (or ~10 source demos for tasks requiring D1 pool generation
— see Phase 0 below).

---

## Step 1: Create configs (7 files)

For a new task `<task>`, you need to create or copy-adapt the following files.
Replace `<task>` with e.g. `coffee_preparation`, `square`, `lift`.

### 1a. Data source config

**`policy_doctor/configs/data_source/mimicgen_<task>.yaml`**

Copy from `mimicgen_coffee_preparation.yaml` and update:
```yaml
task: <task>_mimicgen          # e.g. coffee_preparation_mimicgen
task_config: <task>_<date>     # matches the IV task config name

data_source:
  id: mimicgen_<task>
  conda_env_train: mimicgen_torch2
  conda_env_datagen: mimicgen_torch2
  mimicgen_submodule: true
```

Defaults list must point to the correct robomimic sub-configs:
```yaml
defaults:
  - /robomimic/baseline/low_dim@baseline: <task>_mimicgen
  - /robomimic/attribution/low_dim@attribution: <task>_mimicgen
  - /robomimic/evaluation/low_dim@evaluation: <task>_mimicgen
```

### 1b. Experiment config

**`policy_doctor/configs/experiment/mimicgen_<task>_<date>_d100_<constraint>.yaml`**

This is the main sweep config. Key sections:

```yaml
# @package _global_
task_config: <task>_<date>
experiment_name: mimicgen_<task>_<date>_d100_<constraint>

run_dir: data/pipeline_runs/mimicgen_<task>_<date>_seed1_d100_<constraint>

seeds: [1]
reference_seed: 1
train_date: <date>_<task>_d100_<constraint>
device: cuda:0
project: mimicgen_<task>
wandb_tags: [<task>, mimicgen, <date>, budget_sweep, rep_sweep, <constraint>]

baseline:
  max_train_episodes: 100    # demos from D1 pool for baseline policy
  checkpoint_topk: 5

evaluation:
  train_date: <date>_<task>_d100_<constraint>
  eval_date:  <date>_<task>_d100_<constraint>
  overwrite: false
  num_episodes: 500

attribution:
  train_date: <date>_<task>_d100_<constraint>
  eval_date:  <date>_<task>_d100_<constraint>

# After run_clustering completes, fill these in manually:
clustering_dir: <absolute_path_to_clustering_result_dir>
clustering_run_dir: data/pipeline_runs/mimicgen_<task>_<date>_seed1_d100_<constraint>

# Clustering params (same as square apr26)
clustering_influence_source: infembed
clustering_level: rollout
clustering_demo_split: train
clustering_window_width: 5
clustering_stride: 2
clustering_umap_n_components: 100
clustering_n_clusters: 15
clustering_normalize: none
clustering_aggregation: sum

mimicgen_datagen:
  task_name: <task>                    # e.g. coffee_preparation (NOT square by default!)
  env_interface_name: MG_<TaskClass>   # e.g. MG_CoffeePreparation
  num_seeds: 10
  output_dir: data/outputs/mimicgen_datagen
  policy_seed: 1
  success_only: true
  top_k_paths: 20
  min_path_probability: 0.0
  fix_initial_object_poses: true
  object_pose_ranges:
    <object>:
      x: [-0.04, 0.04]       # ±40mm tight constraint
      y: [-0.04, 0.04]
      z_rot: [-0.524, 0.524] # ±30°
    <other_objects>:          # leave unconstrained:
      x: null
      y: null
      z_rot: null

mimicgen_budget_rep_sweep:
  heuristics: [random, behavior_graph]   # diversity not implemented
  budgets: [100, 300, 500]
  rep_seeds: [1, 2, 3]
  devices: [cuda:0, cuda:0, cuda:0]      # 3 concurrent slots on one GPU

steps: [mimicgen_budget_rep_sweep]
```

**Critical:** `task_name` and `env_interface_name` must match the MimicGen task, not
default to `square`. Find the correct `env_interface_name` in
`third_party/mimicgen/mimicgen/env_interfaces/robosuite.py` — look for
`class MG_<TaskClass>`.

### 1c. Robomimic sub-configs (3 files)

Copy from the existing coffee_preparation versions and update `task:` and `train_date:`.

**`policy_doctor/configs/robomimic/baseline/low_dim/<task>_mimicgen.yaml`**
```yaml
# Copy from coffee_preparation_mimicgen.yaml; update dataset_path
```

**`policy_doctor/configs/robomimic/attribution/low_dim/<task>_mimicgen.yaml`**
```yaml
task: <task>_mimicgen
# ...
featurize_holdout: false    # IMPORTANT: set false unless you have a true holdout set
compile: true               # but set false if using torch.func double backward
tf32: true
```

**`policy_doctor/configs/robomimic/evaluation/low_dim/<task>_mimicgen.yaml`**
```yaml
# Copy from coffee_preparation_mimicgen.yaml; update task name
```

### 1d. Cupid diffusion policy config

**`third_party/cupid/configs/low_dim/<task>_mimicgen_lowdim/diffusion_policy_cnn/config.yaml`**

Copy from `coffee_preparation_mimicgen_lowdim/`. Update:
- `task.dataset.dataset_path` → path to the D1 source HDF5
- `task.env_runner.n_test` → 50 (for eval)
- `task.env_runner.n_test_vis: 0` (required — video rendering crashes without MuJoCo display)
- `task.env_runner.n_train_vis: 0`
- `task.env_runner.n_envs: 10` (parallel rollout envs; use 1 if `save_episodes=True`)

### 1e. Influence Visualizer task config

**`third_party/influence_visualizer/configs/<task>_<date>.yaml`**

```yaml
task: robomimic_<task>
name: "<Task> D1 (<date> sweep, demos=100, <constraint>)"

eval_dir: "data/outputs/eval_save_episodes/<train_date>/<run_name>/latest"
train_dir: "data/outputs/train/<train_date>/<run_name>"
train_ckpt: "latest"
exp_date: "default"
seeds: ["1"]
wandb_project: "mimicgen_<task>"

state_labels: [...]   # copy from coffee_preparation or enumerate obs dims
action_labels: ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "gripper"]
```

**Note:** `eval_dir` and `train_dir` must use the **seed-1** run name suffix `_1`
(e.g. `..._mimicgen_1`), not `_0`. Check the actual training output dir name after
baseline training completes.

---

## Step 2: Phase 0 — D1 Pool Generation (if needed)

Some MimicGen tasks ship only ~10 source demos in the D1 release dataset.
Coffee preparation is one of them. If the D1 dataset already has 1000+ demos, skip
this step.

```bash
conda run -n mimicgen python scripts/generate_<task>_d1_pool.py \
  --output_dir data/source/mimicgen/core_datasets/<task>_d1 \
  --n_success 300
```

Output: `data/source/mimicgen/core_datasets/<task>_d1/demo.hdf5`

Update the cupid config to point `task.dataset.dataset_path` at this file.

---

## Step 3: Run train_baseline + eval_baseline + compute_infembed + run_clustering

These are the prerequisite pipeline steps that produce the clustering result the sweep
arms need for seed selection.

```bash
# From WORKTREE root:
nohup conda run -n policy_doctor --no-capture-output \
  python -m policy_doctor.scripts.run_pipeline \
    data_source=mimicgen_<task> \
    +experiment=mimicgen_<task>_<date>_d100_<constraint> \
    steps=[train_baseline,eval_baseline,compute_infembed,run_clustering] \
  > logs/<task>_<date>_pipeline.log 2>&1 &
```

**Important env and path notes:**
- Run from the **worktree root** (not main checkout), or prepend worktree to
  `PYTHONPATH` so `import policy_doctor` resolves to the right checkout.
- Data outputs go to `third_party/cupid/data/`. If that lives on the boot drive,
  symlink it to SSD first:
  ```bash
  mv third_party/cupid/data /mnt/ssdB/<your_path>/data
  ln -s /mnt/ssdB/<your_path>/data third_party/cupid/data
  ```
- Similarly symlink `outputs/` (Hydra working dirs) if needed.

**Step-by-step expected timeline (single GPU, 100 demos):**

| Step | Approx time | Notes |
|------|-------------|-------|
| `train_baseline` | ~38h | 1751 epochs, checkpoint every 50 |
| `eval_baseline` | ~8h | 500 episodes (run with `n_envs=28` via env_runner override) |
| `compute_infembed` | ~4h | ArnoldiEmbedder, 200 iterations, proj_dim=100; run in `cupid_torch2` |
| `run_clustering` | ~30min | UMAP to 100D then 2D, KMeans k=15 |

**Known issues and fixes:**

- **`compute_infembed` must run in `cupid_torch2`** (not `policy_doctor`) — it needs
  `torch.func`. The pipeline step handles this automatically via `conda run`.
- **`featurize_holdout: false`** in the attribution config — the baseline has no
  holdout set; setting this true wastes hours featurizing the full dataset unused.
- **Compiled checkpoints**: if training used `torch.compile`, the eval script strips
  `_orig_mod.` prefixes from state dict keys automatically (fix in `eval_save_episodes.py`).
- **`n_test_vis: 0`** in the cupid env_runner config — video recording requires an
  offscreen render context that crashes without a display. Always set this to 0.
- **`save_episodes=True`** requires `n_envs=1` (sequential) — the parallel runner
  doesn't support HDF5 episode saving. Eval without `save_episodes` can use `n_envs=28`.
- **`attribution.compile: false`, `attribution.tf32: false`** — `torch.compile` with
  `aot_autograd` doesn't support double backward (used by InfEmbed). Set both false.
- **`val_ratio: 0.0`** in combined-data training — non-divisible dataset sizes cause
  `AssertionError('Remainder demos!')` with any non-zero val_ratio.

---

## Step 4: Note the clustering dir path

After `run_clustering` completes, find the clustering dir:

```bash
cat data/pipeline_runs/<run_dir>/run_clustering/result.json | python3 -m json.tool
```

Copy the clustering dir path into the experiment config:

```yaml
clustering_dir: /absolute/path/to/clustering/<experiment>_seed1_kmeans_k15
clustering_run_dir: data/pipeline_runs/<run_dir>
```

Also create `data/pipeline_runs/<run_dir>/mimicgen_budget_rep_sweep/run_clustering/result.json`
manually (needed for the sweep arms' seed-selection step):

```json
{"clustering_dirs": {"1": "/absolute/path/to/clustering/..."}}
```

Note: use the **same seed key** as `mimicgen_datagen.policy_seed` (default `"1"`).

---

## Step 5: Run the sweep

```bash
nohup conda run -n policy_doctor --no-capture-output \
  python -m policy_doctor.scripts.run_pipeline \
    data_source=mimicgen_<task> \
    +experiment=mimicgen_<task>_<date>_d100_<constraint> \
    steps=[mimicgen_budget_rep_sweep] \
  > logs/<task>_<date>_sweep.log 2>&1 &
```

The sweep runs **3 arms concurrently** (one per device slot). Each arm:
1. `select_mimicgen_seed` — picks a rollout from the behavior graph
2. `generate_mimicgen_demos` — runs MimicGen in `mimicgen_torch2` env
3. `train_on_combined_data` — trains on original + generated demos
4. `eval_mimicgen_combined` — evaluates 500 episodes

**Seed selection uses rollouts.hdf5** — the eval step must have saved rollout HDF5s with
full MuJoCo states (not just obs/actions). If the baseline eval ran without
`save_episodes=True`, use the D1 source demos as a proxy:

```bash
# Create rollouts.hdf5 symlink from source demos
ln -s /path/to/d1/demo.hdf5 \
  data/pipeline_runs/<run_dir>/eval_baseline/<eval_subdir>/episodes/rollouts.hdf5
```

This uses training demo diversity for seed selection instead of rollout behavioral
diversity — a known limitation, acknowledged in the experiment design.

**Timeline per arm batch** (3 arms concurrent on 1 GPU):
- `select + generate`: 1–2h (MimicGen sim generation)
- `train`: ~38h (1751 epochs, budget100) up to ~60h (budget500, larger dataset)
- `eval`: ~8h (500 episodes)

Total for 9 random arms (3 batches): ~5–7 days.
Total for behavior_graph arms: add another ~5–7 days.

---

## Step 6: Monitor

The pipeline is resilient — failed arms log errors and continue. Check:

```bash
# Overall arm status
ls data/pipeline_runs/<run_dir>/mimicgen_budget_rep_sweep/

# Step completion per arm
for arm in data/pipeline_runs/<run_dir>/mimicgen_budget_rep_sweep/mimicgen_*/; do
  done=$(ls "$arm"/*/done 2>/dev/null | xargs -I{} dirname {} | xargs -I{} basename {} | tr '\n' ' ')
  echo "$(basename $arm): ${done:-(none)}"
done

# Training epoch progress (check Hydra output dirs)
for d in data/outputs/<hydra_outputs_dir>/*/; do
  ls "$d/checkpoints/" 2>/dev/null | grep -v latest | sort -V | tail -1
done
```

Disk usage: training checkpoints are ~22 MB each × topk(5) × num_arms. 
At 18 arms × 5 ckpts × 22 MB = ~2 GB for checkpoints. Eval HDF5s are larger
(~500 MB per arm). Keep data on SSD, not the boot drive.

---

## Gotchas checklist

- [ ] `task_name` and `env_interface_name` set correctly in experiment config (not `square`)
- [ ] `featurize_holdout: false` in attribution config
- [ ] `n_test_vis: 0` and `n_train_vis: 0` in cupid env_runner config
- [ ] `val_ratio: 0.0` — handled automatically by `train_on_combined_data` step
- [ ] `attribution.compile: false` and `attribution.tf32: false`
- [ ] SSD symlinks in place before launching (check `df -h /`)
- [ ] `run_clustering/result.json` manually created in the sweep step dir with correct seed key
- [ ] `clustering_dir` and `clustering_run_dir` filled into experiment config after clustering
- [ ] `eval_dir` in IV task config uses `_1` suffix (seed 1 run), not `_0`
- [ ] `conda_env_train: mimicgen_torch2` in data_source config
- [ ] Pool generation complete before running pipeline (if task needs D1 pool)
