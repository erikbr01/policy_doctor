# MimicGen Coffee Preparation D1 — May 18 2026

**Status:** Re-launch in progress (May 24) after invalidation of the May 20 run.
**Goal:** Replicate the apr26 square tight-constraint sweep for coffee preparation D1 with mug pose constrained.

---

## Setup Cross-Check (May 24 v15 — verified against the four most recent commits)

Before launching, confirmed the pipeline is consistent with the most recent fixes
on `feat/mimicgen-trajectory-pipeline`:

| Concern | Commit | Status in v15 |
|---|---|---|
| `seed_object_poses` were silently read from the SQUARE source HDF5, so coffee/kitchen/threading constraints fell through (`square_nut`/`square_peg` ≠ `mug`/`drawer`/...) and the mug ran free at full D1 | `9568637 fix: read seed_object_poses from prepared seed.hdf5 directly` + `b8a0210 docs+fix: round-3 seed_object_poses source-dataset bug` | ✅ v15 log shows `read seed_object_poses from prepared seed.hdf5: ['coffee_machine', 'coffee_pod', 'drawer', 'mug']` and per-seed `mug(x=[…], y=[…], z_rot=[…])` constraint windows — the ±40 mm / ±30° spec is actually being enforced. All earlier coffee_prep runs (v8 and prior) generated data with **no mug constraint**; that data has been deleted. |
| Combined-arm trainer was loading the full source dataset (1000 demos) + N generated, defeating the few-shot baseline-vs-augmented comparison | `3e6f2f0 fix: combined arm trains on baseline subset + generated, not full source` (this branch — adds `max_original_demos` to `combine_hdf5_datasets`, wired from `baseline.max_train_episodes`) | ✅ v15's `combined.hdf5` should be `100 baseline + 100 generated = 200` demos. Verify per arm via `find $RD/mimicgen_budget_rep_sweep -name combined.hdf5 -exec h5ls -d {}/data \; \| wc -l`. (v14 incorrectly produced 1100/1029-demo files due to stale `__pycache__` from a prior pipeline launch — pyc cache nuked before v15.) |
| Per-arm mimicgen datagen was writing to a shared output path, so every arm trained on essentially the same generated demos | `e6f3f18 fix: per-arm mimicgen datagen dir + --guarantee passthrough` (later obsoleted by EEF's `step_dir/output` isolation in `f2e42c9 restore: generate_mimicgen_demos + run_mimicgen_generate from EEF worktree`) | ✅ Each arm writes under `$RD/mimicgen_budget_rep_sweep/<arm_name>/generate_mimicgen_demos/output/seed_<i>/`. |
| `task_spec` for non-square tasks was hardcoded to SQUARE (`subtask_term_signal="grasp"`) → `KeyError: 'grasp'` against coffee's `{mug_grasp, mug_place, drawer_open, pod_grasp}` signals | `7ea2bb9 fix: non-square task_spec + source_dataset_path for coffee_prep` (this branch) | ✅ v15 uses the env-interface template subtask spec for non-square tasks; first arm's log shows `signals=['mug_grasp', 'mug_place', 'drawer_open', 'pod_grasp', None]`. |

**Invariant for the v15 run:** every arm trains on a 200-demo combined dataset
(100 baseline + 100 mug-constrained generated). All earlier coffee_prep
generation data has been deleted before this launch.

---

## Run Commands (May 24 v13 — budget=100 only, random vs BG)

```bash
# From the worktree root: /home/erbauer/refactor_cupid/policy_doctor/.claude/worktrees/feat+mimicgen-traj-pipeline

# 0. ONE-TIME: pre-prepare the coffee_prep_d1 source dataset (adds datagen_info via prepare_src_dataset).
# Skip if third_party/cupid/data/source/mimicgen/core_datasets/coffee_preparation_d1_prepared/demo.hdf5 already exists.
cp third_party/cupid/data/source/mimicgen/core_datasets/coffee_preparation_d1_official/core/coffee_preparation_d1.hdf5 \
   third_party/cupid/data/source/mimicgen/core_datasets/coffee_preparation_d1_prepared/demo.hdf5
conda run -n mimicgen_torch2 --no-capture-output python -c "
import sys; sys.path.insert(0, 'third_party/mimicgen'); sys.path.insert(0, '.')
from scripts.run_mimicgen_generate import _ensure_mimicgen_on_path, _apply_robomimic_base_env_shim
_ensure_mimicgen_on_path(); _apply_robomimic_base_env_shim()
from mimicgen.scripts.prepare_src_dataset import prepare_src_dataset
prepare_src_dataset(
    dataset_path='third_party/cupid/data/source/mimicgen/core_datasets/coffee_preparation_d1_prepared/demo.hdf5',
    env_interface_name='MG_CoffeePreparation', env_interface_type='robosuite',
    filter_key=None, n=None, output_path=None,
)
"

# 1. Clean stale per-arm state from prior failed attempts.
RD=third_party/cupid/data/pipeline_runs/mimicgen_coffee_prep_d1_may18_seed1_d100_mug_constrained
SSD_RD=/mnt/ssdB/erik/cupid_data/worktree_data/pipeline_runs/mimicgen_coffee_prep_d1_may18_seed1_d100_mug_constrained
rm -rf "$RD/mimicgen_budget_rep_sweep/mimicgen_"*
rm -rf "$SSD_RD/mimicgen_budget_rep_sweep/mimicgen_"* 2>/dev/null
rm -f "$RD/mimicgen_budget_rep_sweep/done" "$RD/mimicgen_budget_rep_sweep/result.json"
rm -f "$SSD_RD/mimicgen_budget_rep_sweep/done" "$SSD_RD/mimicgen_budget_rep_sweep/result.json" 2>/dev/null

# 2. Launch the pipeline (cuda:0 only, 6 concurrent arms — gen is MuJoCo/CPU-bound).
mkdir -p logs
nohup conda run -n policy_doctor --no-capture-output python -m policy_doctor.scripts.run_pipeline \
  data_source=mimicgen_coffee_preparation \
  experiment=mimicgen_coffee_prep_d1_may18_d100_mug_constrained \
  >> logs/coffee_prep_d1_may18_pipeline_v2.log 2>&1 &
echo $! > logs/coffee_prep_d1_may18_pipeline_v2.pid
disown

# 3. Monitor.
tail -F logs/coffee_prep_d1_may18_pipeline_v2.log

# 4. Per-arm generation rate (rolled-up across passes).
for arm_dir in $RD/mimicgen_budget_rep_sweep/mimicgen_*; do
  arm=$(basename "$arm_dir")
  stats_files=$(find "$arm_dir/generate_mimicgen_demos" -name "stats.json" 2>/dev/null)
  [ -z "$stats_files" ] && continue
  python3 -c "
import json, sys
s=a=0
for f in '''$stats_files'''.split():
    d=json.load(open(f)); s+=d.get('num_success',0); a+=d.get('num_attempts',0)
if a>0: print(f'  $arm: {s}/{a} = {s/a*100:.1f}%')
"
done

# 5. Per-arm final eval (after train_on_combined_data + eval_mimicgen_combined complete).
for d in $RD/mimicgen_budget_rep_sweep/mimicgen_*/eval_mimicgen_combined/result.json; do
  arm=$(basename "$(dirname "$(dirname "$d")")")
  python3 -c "
import json
r = json.load(open('$d'))
print(f'  {\"$arm\"}: mean={r[\"mean_success_rate\"]:.3f}  best={r[\"best_success_rate\"]:.3f}')
"
done

# 6. Kill if needed (process tree).
PIPE=$(cat logs/coffee_prep_d1_may18_pipeline_v2.pid)
pkill -KILL -P "$PIPE"; kill -KILL "$PIPE"; pkill -KILL -f "run_mimicgen_generate"
```

### Changing the budget set

Two configs and one Hydra override knob control the budget × parallelism shape.
To change the sweep (e.g. add budget=300/500 back, or split across more GPUs):

**A. Edit the experiment yaml** —
`policy_doctor/configs/experiment/mimicgen_coffee_prep_d1_may18_d100_mug_constrained.yaml`,
block `mimicgen_budget_rep_sweep`:

```yaml
mimicgen_budget_rep_sweep:
  heuristics: [random, behavior_graph]   # add 'diversity' if it stops failing
  budgets: [100]                          # ← edit this; e.g. [100, 300, 500]
  rep_seeds: [1, 2, 3]                    # variance replicates per (heuristic, budget)
  devices: [cuda:0, cuda:0, cuda:0,       # ← one entry per concurrent worker
            cuda:0, cuda:0, cuda:0]       #   total arms = len(heuristics)*len(budgets)*len(rep_seeds)
```

The sweep dispatches arms in **budget-outer order**
(`for budget in budgets: for heuristic in heuristics: for rep_seed in rep_seeds`)
so adding budgets later doesn't push the budget=100 results back. See
`policy_doctor/curation_pipeline/steps/mimicgen_budget_sweep.py:213` (the
`arms = [...]` list comprehension) if you want to change that order.

**B. CLI Hydra override** — same change without editing the yaml:

```bash
... +mimicgen_budget_rep_sweep.budgets='[100,300]' \
... +mimicgen_budget_rep_sweep.devices='[cuda:0,cuda:0,cuda:0,cuda:0,cuda:0,cuda:0]'
```

**C. Per-arm generation budget mechanics** — at the arm level, `budget` becomes
`mimicgen_datagen.success_budget` (read in
`policy_doctor/curation_pipeline/steps/generate_mimicgen_demos.py:282`). The
adaptive retry loop (`while total_successes < success_budget and total_trials < max_total_trials`)
keeps issuing trials in per-seed passes until either the budget's worth of
successes lands, or `20 × success_budget` total trials have been attempted —
whichever first. At ~30% gen success that's roughly:

| budget | min trials | max trials | wall (1 arm, sequential) | wall (6 in parallel) |
|--------|-----------|-----------|--------------------------|----------------------|
| 100    | ~333      | 2000      | ~1.5–2h                  | ~1.5–2h              |
| 300    | ~1000     | 6000      | ~4.5–6h                  | ~4.5–6h              |
| 500    | ~1666     | 10000     | ~7.5–10h                 | ~7.5–10h             |

The hard cap (`max_total_trials = success_budget * 20`) is set in
`generate_mimicgen_demos.py:550` if you ever need to push it.

**D. Stale state to clean before relaunching a different budget set** — the
sweep step caches each arm's per-step `done` sentinels. Add or remove arms
without re-running already-completed ones by leaving those step_dirs intact,
but always nuke the **sweep-level** sentinel so the orchestrator re-scans:

```bash
RD=third_party/cupid/data/pipeline_runs/mimicgen_coffee_prep_d1_may18_seed1_d100_mug_constrained
rm -f "$RD/mimicgen_budget_rep_sweep/done" "$RD/mimicgen_budget_rep_sweep/result.json"
```

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

### Heuristics

`heuristics: [random, behavior_graph]`. The `diversity` heuristic was DROPPED
on May 24 — in the May 20 run all 9 diversity arms failed with
`DiversitySelectionHeuristic: no rollout matched any path to SUCCESS`. Whether
the new policy_emb-based clustering fixes that is an open question; we'll add
diversity back if early budget-100 results look clean.

### Budgets (May 24 update)

Scoped down to `budgets: [100]` only for the first lap — clean random vs BG
comparison at the smallest budget gets us a result in ~3-4h instead of
~20h+. Larger budgets (300, 500) are intentionally deferred until we've
seen budget=100 outcomes and decided they justify the additional compute.

### Selection Seeds

Three fixed seeds (no null): `rep_seeds: [1, 2, 3]` via `mimicgen_budget_rep_sweep`.

### Baseline

- `max_train_episodes: 100` — 100 demos from D1-generated pool.
- The D1 pool must be generated first (Phase 0) since only 10 source demos exist.

### Generation Budgets

`budgets: [100, 300, 500]`

### Device

6 concurrent arms split across two GPUs:
`devices: [cuda:0, cuda:0, cuda:0, cuda:1, cuda:1, cuda:1]`.
Cuda:1 slots are released ~2.5h after pipeline launch (GPU 1 reserved for other
work). In-flight arms on cuda:1 get killed; remaining work continues on cuda:0.

### Clustering

Switched from `infembed` to **policy embeddings** on May 24 (matches the may20
sweep configs for kitchen / threading / three_piece_assembly):

- `clustering_influence_source: policy_emb`
- `clustering_policy_emb_layer: bottleneck_plan_t0`
  (U-Net mid-block activation at plan t=0; the documented "best known" layer
  per `compute_policy_embeddings.py`. The regex
  `^(?P<hook>bottleneck|decoder|encoder)(?:_(?P<action>plan8|plan|exec))?(?:_t(?P<t>\d+))?$`
  rejects the `plan_bottleneck` form some old configs used — those are broken.)
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

### Correction: Official Dataset Used

**Correction (10:39 UTC):** Initial training used 278 self-generated demos (wrong). Stopped and restarted on
the official MimicGen data release (`amandlek/mimicgen_datasets`, `coffee_preparation_d1`):
- **1000 demos** official, training on 100 (as intended)
- env_name: CoffeePreparation_D1, object: 86 dims ✓
- WandB run: `f118vzfc` (fresh start at 10:39 UTC)

Downloaded to: `data/source/mimicgen/core_datasets/coffee_preparation_d1_official/core/coffee_preparation_d1.hdf5`
Symlinked to standard path: `data/source/mimicgen/core_datasets/coffee_preparation_d1/demo.hdf5`

### Pool Generation Status (self-generated, discarded)

Started at 01:15 UTC. Completed at 03:16 UTC with 278 demos. **This data was deleted at 10:39 UTC**
in favor of the official 1000-demo dataset.

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

1. `train_baseline` — train on 100 official D1 demos, seed=1
2. `eval_baseline` — 500 episodes at test_start_seed=100000 (n_envs=28, save_episodes=True)
3. `compute_policy_embeddings` — extract U-Net bottleneck activations on the eval rollouts
4. `run_clustering` — k-means k=15 on UMAP(100) of the policy_emb representation
5. `mimicgen_budget_rep_sweep` — 3 reps × 3 budgets × 2 heuristics = **18 arms** (budget-outer ordering)

### Sweep arm ordering (May 24 update)

Arms iterate **budget outer**: all 6 budget-100 arms (random×3 + behavior_graph×3)
complete before budget-300 starts, so we get early signal on the smallest budget
first. (Previously: heuristic-outer; small-budget arms were interleaved across
the whole sweep.)

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
| Phase 0: D1 pool generation | **Superseded** | Official 1000-demo dataset used instead |
| Configs creation | **Done** | All configs created |
| May 20 sweep run | **INVALIDATED** | Shared-path bug (see below). 60h compute lost. |
| May 24 relaunch — baseline train | **Done** | Reused via symlink, 1h saved |
| May 24 relaunch — eval_baseline | **In progress** | Parallelized to n_envs=28 (was 1) |
| May 24 relaunch — policy_emb + clustering | Pending | |
| May 24 relaunch — sweep | Pending | 18 arms, budget-outer order |

---

## May 20 Run: Why It Was Invalidated

All 18 arms produced suspiciously uniform success rates (~58–69%). Root cause:
**`generate_mimicgen_demos.py` wrote every arm's generated demos to the same
path** — `data/outputs/mimicgen_datagen/demo_0/demo.hdf5`. Since the sweep
runs arms concurrently with `skip_if_done=true`, the first arm to finish
generation became the de-facto dataset for ALL subsequent arms regardless of
heuristic, budget, or seed. The 58–69% spread was just training-time
nondeterminism (tf32 + compile) on essentially identical data.

Fixed in `e6f3f18`: each arm now writes to
`<step_dir>/datagen/<seed_demo_key>/demo.hdf5`, isolated per arm.

The diversity heuristic separately failed all 9 of its arms with
`DiversitySelectionHeuristic: no rollout matched any path to SUCCESS` — a
clustering-quality issue (no rollout's cluster sequence matched any
enumerated path through the behavior graph to a SUCCESS terminal).

---

## May 24 Relaunch

**Pipeline launched at 12:58 PDT, log: `logs/coffee_prep_d1_may18_pipeline_v2.log`.**

### Issues hit and fixed this session

1. **Hydra `+training.tf32` conflict** — the arch yaml now declares `tf32`/`compile`
   as keys, so `+` (append-only) fails. Fixed: use `++` prefix in
   `train_baseline.py` and `train_curated.py`.
2. **`AsyncVectorEnv.reset_wait` AssertionError** — commit `f0adcdc` flipped
   `shared_memory=False`, exposing a latent positional-arg bug in the local
   `async_vector_env.py`: it called `concatenate(space, results, observations)`
   but gym 0.21 expects `concatenate(items, out, space)`. Fixed both call
   sites (lines 233, 296).
3. **`train_baseline` checkpoint dir mismatch** — Hydra writes ckpts to
   `outputs/<timestamp>/checkpoints/` but `eval_baseline` looks under
   `multi_run.run_dir/checkpoints`. The fix existed in `train_on_combined_data.py`
   (find latest `.hydra/overrides.yaml` matching `logging.name`, symlink
   parent's `checkpoints` dir) but was missing from `train_baseline.py`.
   Ported the symlink logic.
4. **`eval_baseline` was sequential (n_envs=1)** — would have taken ~5h.
   Commit `f0adcdc` lifted the `n_envs==1` restriction on `save_episodes=True`,
   so eval_baseline now passes `--n_envs=28 --save_episodes=True`, completing
   in ~10–15min.

### Sweep config (relaunch)

- 6 concurrent arms across cuda:0 (×3) and cuda:1 (×3)
- Budget-outer ordering: all 6 budget-100 arms finish before budget-300 starts
- Cuda:1 cutoff watcher running (PID logged in `logs/cuda1_cutoff.pid`),
  fires 2.5h after launch (~15:24 PDT). In-flight cuda:1 arms get SIGTERM'd.

### Open question

**Timing of the cuda:1 cutoff:** Cutoff fires at ~15:24 but the sweep itself
doesn't start until ~14:25 (after baseline train+eval+embed+cluster). That
gives only ~1h of dual-GPU sweep time before cuda:1 work gets cancelled.
Each sweep arm trains for 1.5–2h, so most cuda:1 progress would be wasted.
Decision pending on whether to push the cutoff out.

---

## Concerns / Risks

1. **Success rate:** Coffee preparation D1 has 5 subtasks. MimicGen success rate may be lower than square. Probe result will determine if timeline is feasible.
2. **D1 vs D0 env:** The source HDF5 has `env_name=CoffeePreparation_D0`. Pool generation patches this to D1.
3. **Observation dim:** 86-dim object key is large. Verified from source HDF5. Training config uses obs_dim=95.
4. **max_steps for eval:** Coffee prep demos average ~600-700 timesteps (demo_0 has 676). Using max_steps=800 (vs 500 for square) to avoid timeout.
5. **Rollout eval slows with high success rate:** At 60-70% success, episodes run all 800 steps → rollout eval takes 30-90 min per checkpoint interval instead of ~10 min. Each 50-epoch cycle takes 60-98 min instead of the initial 65 min.
