# MimicGen Coffee Preparation D1 — May 18 2026

**Status:** Setting up experiment  
**Goal:** Replicate the apr26 square tight-constraint sweep for coffee preparation D1 with mug pose constrained.

---

## Experiment Design

### Task: CoffeePreparation_D1

Coffee preparation is a 5-subtask task:
1. Grasp mug (relative to mug)
2. Place mug on coffee machine + open lid (relative to coffee machine)
3. Open drawer (relative to drawer)
4. Grasp coffee pod (relative to coffee pod)
5. Insert pod + close lid (relative to coffee machine)

**D1 randomization bounds (from `CoffeePreparation_D1._get_initial_placement_bounds`):**

| Object         | x range         | y range          | z_rot range              |
|----------------|-----------------|------------------|--------------------------|
| drawer         | 0.15 (fixed)    | -0.35 (fixed)    | π (fixed)                |
| coffee_machine | [-0.25, -0.15]  | [-0.30, -0.25]   | [-π/6, π/6] (60°)       |
| mug            | [-0.15, 0.20]   | [0.05, 0.25]     | [0, 2π] (full rotation!) |
| coffee_pod     | [-0.03, 0.03]   | [-0.05, 0.03]    | 0 (fixed)                |

### Mug Pose Constraint

Analogous to the square nut tight constraint (±40mm x/y, ±30° z_rot):
- `mug.x: [-0.04, 0.04]` — ±40 mm around seed pose
- `mug.y: [-0.04, 0.04]` — ±40 mm around seed pose
- `mug.z_rot: [-0.524, 0.524]` — ±30° around seed pose

All other objects (drawer, coffee_machine, coffee_pod): **unconstrained** (null = use D1 env range).

### Selection Seeds

Three fixed seeds (no null): `rep_seeds: [1, 2, 3]` via `mimicgen_budget_rep_sweep`.

### Baseline

- `max_train_episodes: 100` — 100 demos from D1-generated pool.
- The D1 pool must be generated first (Phase 0) since only 10 source demos exist.

### Generation Budgets

`budgets: [100, 300, 500]`

### Device

`cuda:0` only — 3 slots.

### Clustering

Same parameters as square apr26:
- `clustering_influence_source: infembed`
- `clustering_level: rollout`
- `clustering_demo_split: train`
- `clustering_window_width: 5`
- `clustering_stride: 2`
- `clustering_umap_n_components: 100`
- `clustering_n_clusters: 15`
- `clustering_normalize: none`
- `clustering_aggregation: sum`

---

## Phase 0: D1 Pool Generation

**Prerequisite:** The coffee preparation source dataset has only 10 demos (`source/coffee_preparation.hdf5`). We need 100+ D1-difficulty demos for baseline training.

**Script:** `scripts/generate_coffee_prep_d1_pool.py`

**Target:** 300 successful D1 demos (to have buffer for 100-demo baseline + seed selection).

**Environment:** `mimicgen` conda env, uses `CoffeePreparation_D1` env.

**Output:** `data/source/mimicgen/core_datasets/coffee_preparation_d1/demo.hdf5`

### Success Rate Probe

Probed 20 trials per source demo with full D1 (mug z_rot ∈ [0, 2π]):

| Source Demo | Success Rate | Notes |
|-------------|-------------|-------|
| demo_0 | 5% (1/20) | Poor — mug position/approach not retargetable |
| demo_1 | 65% (13/20) | Excellent |
| demo_2+ | TBD | Probe killed early to start pool generation |

**Key findings:**
- Full D1 generation works for most demos (~65% success rate for good seeds)
- demo_0 appears to be a poor seed (5%) — will contribute ~7 demos from 150 trials
- Average across good demos: ~40-65%, enough for 200 demos from 10 demos × 150 trials
- API compat shim needed: robomimic 0.3.0 in `mimicgen_torch2` missing `env_class` kwarg

**Technical fix applied:**
- Added `_patch_robomimic_api_compat()` to both `generate_coffee_prep_d1_pool.py` and `run_mimicgen_generate.py`
- Patches `EnvUtils.create_env_for_data_processing` to drop unsupported kwargs at runtime

### Pool Generation Status

Started at 01:15 UTC. Processing all 10 source demos, 150 trials each.

| Parameter | Value |
|-----------|-------|
| n_trials_per_demo | 150 |
| target_n_success | 200 |
| guarantee | False |
| Expected demos | 150-200+ |
| Est. wall-clock | 2-3 hours |

---

## Phase 1: Pipeline

### Steps (in order)

1. `train_baseline` — train on 100 D1 pool demos, seed=1
2. `eval_baseline` — 500 episodes at test_start_seed=100000
3. `compute_infembed` — InfEmbed attribution
4. `run_clustering` — cluster with square parameters
5. `mimicgen_budget_rep_sweep` — 3 seeds × 3 budgets × 3 heuristics = 27 arms

### Run Configuration

| Parameter | Value |
|-----------|-------|
| experiment_name | `mimicgen_coffee_prep_d1_may18_d100_mug_constrained` |
| run_dir | `data/pipeline_runs/mimicgen_coffee_prep_d1_may18_seed1_d100_mug_constrained` |
| train_date | `may18_coffee_prep_d1_d100_mug_constrained` |
| seeds | [1] |
| project | `mimicgen_coffee_preparation` |

---

## Observation Dimensions

Coffee preparation uses a 95-dim low-dim observation:
- `object`: 86 dims (4 objects: mug, coffee_machine, drawer, coffee_pod — with relative poses)
- `robot0_eef_pos`: 3 dims
- `robot0_eef_quat`: 4 dims
- `robot0_gripper_qpos`: 2 dims
- **Total: 95 dims**

With `n_obs_steps=2`: `global_cond_dim = 190`

Training config: `configs/low_dim/coffee_preparation_mimicgen_lowdim/diffusion_policy_cnn/config.yaml`

---

## Code Changes

### `scripts/run_mimicgen_generate.py`

Modified to support non-square tasks. When `task_name != "square"`:
- Uses the template subtask spec (from `config_factory`) instead of hardcoded square subtasks
- Still overrides dynamic parameters (action_noise, num_interpolation_steps, nn_k for nn strategies)

### New Files

| File | Purpose |
|------|---------|
| `scripts/generate_coffee_prep_d1_pool.py` | Generates D1 pool from source |
| `policy_doctor/configs/mimicgen/coffee_preparation_d1.yaml` | MimicGen datagen config |
| `policy_doctor/configs/data_source/mimicgen_coffee_preparation.yaml` | Data source config |
| `policy_doctor/configs/robomimic/baseline/low_dim/coffee_preparation_mimicgen.yaml` | Baseline train config |
| `policy_doctor/configs/robomimic/evaluation/low_dim/coffee_preparation_mimicgen.yaml` | Eval config |
| `policy_doctor/configs/robomimic/attribution/low_dim/coffee_preparation_mimicgen.yaml` | Attribution config |
| `policy_doctor/configs/experiment/mimicgen_coffee_prep_d1_may18_d100_mug_constrained.yaml` | Experiment config |
| `third_party/cupid/diffusion_policy/config/task/coffee_preparation_mimicgen_lowdim.yaml` | Cupid task config |
| `third_party/cupid/configs/low_dim/coffee_preparation_mimicgen_lowdim/diffusion_policy_cnn/config.yaml` | Cupid training config |
| `third_party/influence_visualizer/configs/coffee_prep_d1_may18.yaml` | Visualizer config |
| `scripts/run_coffee_prep_d1_may18_sweep.sh` | Launch script |

---

## Timeline

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0: D1 pool generation | Not started | Need probe result first |
| Configs creation | In progress | Creating all configs |
| Phase 1: Baseline training | Pending pool | ~2-4 hrs |
| Phase 1: Eval + attribution | Pending baseline | ~2-3 hrs |
| Phase 1: Budget rep sweep | Pending clustering | ~6-12 hrs (27 arms on cuda:0 × 3 slots) |

---

## Concerns / Risks

1. **Success rate:** Coffee preparation D1 has 5 subtasks. MimicGen success rate may be lower than square. Probe result will determine if timeline is feasible.
2. **D1 vs D0 env:** The source HDF5 has `env_name=CoffeePreparation_D0`. Pool generation patches this to D1.
3. **Observation dim:** 86-dim object key is large. Verified from source HDF5. Training config uses obs_dim=95.
4. **max_steps for eval:** Coffee prep demos average ~600-700 timesteps (demo_0 has 676). Using max_steps=800 (vs 500 for square) to avoid timeout.
