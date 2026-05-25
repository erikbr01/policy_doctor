# MimicGen Kitchen + Threading + ThreePieceAssembly D1 — May 20 2026 (re-run May 24)

**Status (2026-05-24, 21:40 UTC):** Kitchen budget=100 sweep running — 9 arms in MimicGen generate, ~94% CPU each.
**Goal:** Failure-targeted budget × rep × heuristic sweep on Kitchen D1, Threading D1, ThreePieceAssembly D1, using policy-embedding clustering (`bottleneck_plan_t0`) for behavior-graph + diversity heuristics.

---

## Experiment Design

### Three tasks, one constrained-pose object per task

| Task | Constrained object | x | y | z_rot |
|---|---|---|---|---|
| Kitchen D1 | bread | [-0.04, 0.04] | [-0.04, 0.04] | null *(was [-0.524, 0.524])* |
| Threading D1 | needle | [-0.04, 0.04] | [-0.04, 0.04] | null *(was [-0.524, 0.524])* |
| ThreePieceAssembly D1 | piece_1 | [-0.04, 0.04] | [-0.04, 0.04] | null *(was [-0.524, 0.524])* |

All other objects unconstrained (`null` = use D1 env range). The z_rot=null change is documented under "Lessons learned" below.

### Sweep dimensions per task

- **Heuristics:** `random`, `behavior_graph`, `diversity` (3)
- **Budgets:** `[100, 300, 500]` — additional generated demos on top of the 100-demo baseline
- **Reps:** `rep_seeds: [1, 2, 3]` (3)
- **Total arms per task:** 3 × 3 × 3 = **27**

### Baseline

- `max_train_episodes: 100` — 100 demos from the D1 source dataset
- All 3 baselines already trained, evaluated (500 episodes), and have `metadata.yaml` + `eval_log.json` (kitchen 9 ckpts, threading 14 ckpts, TPA 6 ckpts)

### Clustering for behavior_graph + diversity

- `clustering_influence_source: policy_emb`
- `clustering_policy_emb_layer: bottleneck_plan_t0` — UNet mid-block under planned-action conditioning at the clean denoise step (t=0). Pulled from `main`'s canonical `compute_policy_embeddings.py` (commit 904e845).
- `clustering_level: rollout`, `window_width: 5`, `stride: 2`, `n_clusters: 15` (kmeans), `umap_n_components: 100`

### Device pool (kitchen budget=100, current run)

`devices: [cuda:0, cuda:1, cuda:2, cuda:3, cuda:0, cuda:1, cuda:2, cuda:3, cuda:0]` — 9 slots = all 9 arms in parallel.

---

## How to Run This Pipeline (Full Reference)

### Configs to set per experiment

For each task there are **3 yaml files** to align before running. Listed for kitchen; analogous files exist for threading (`mimicgen_threading_d1_may20_d100_needle_constrained.yaml`, target=`needle`) and TPA (`mimicgen_three_piece_assembly_d1_may20_d100_piece1_constrained.yaml`, target=`piece_1`).

**1. Experiment yaml** — `policy_doctor/configs/experiment/mimicgen_kitchen_d1_may20_d100_bread_constrained.yaml`

Key fields (current working values):

```yaml
task_config: kitchen_d1_may20
experiment_name: mimicgen_kitchen_d1_may20_d100_bread_constrained
run_dir: data/pipeline_runs/mimicgen_kitchen_d1_may20_seed1_d100_bread_constrained

seeds: [1]
reference_seed: 1
train_date: may20_kitchen_d1_d100_bread_constrained
device: cuda:0
project: mimicgen_kitchen

baseline:
  max_train_episodes: 100
  checkpoint_topk: 5
  num_epochs: 750

evaluation:
  train_date: may20_kitchen_d1_d100_bread_constrained
  eval_date:  may20_kitchen_d1_d100_bread_constrained
  overwrite: false
  num_episodes: 500

attribution:
  train_date: may20_kitchen_d1_d100_bread_constrained
  eval_date:  may20_kitchen_d1_d100_bread_constrained

# Clustering — MUST be policy_emb with bottleneck_plan_t0
clustering_influence_source: policy_emb
clustering_policy_emb_layer: bottleneck_plan_t0
clustering_level: rollout
clustering_demo_split: train
clustering_window_width: 5
clustering_stride: 2
clustering_umap_n_components: 100
clustering_n_clusters: 15
clustering_normalize: none
clustering_aggregation: sum

# MimicGen generation: target the bread initial pose only; pin every other object at seed.
# OMIT z_rot from bread → defaults to [0,0] offset = pin at seed (otherwise the
# place_bread_in_pot subtask fails 47% of the time per the round-2 diagnosis).
mimicgen_datagen:
  task_name: kitchen
  env_interface_name: MG_Kitchen
  num_seeds: 10
  output_dir: data/outputs/mimicgen_datagen     # ignored — actual writes go to step_dir/datagen
  policy_seed: 1
  success_only: true
  top_k_paths: 20
  min_path_probability: 0.0
  num_trials: 500            # initial pass size; adaptive loop overrides via success_budget
  guarantee: true
  fix_initial_object_poses: true
  object_pose_ranges:
    bread:
      x: [-0.04, 0.04]
      y: [-0.04, 0.04]
      # z_rot intentionally omitted → pin at seed
    pot: null                # whole-object null = pin at seed (NOT per-axis null)
    stove: null
    button: null
    serving_region: null

# Sweep — heuristic × budget × rep
mimicgen_budget_rep_sweep:
  heuristics: [random, behavior_graph, diversity]
  budgets: [100, 300, 500]
  rep_seeds: [1, 2, 3]
  devices: [cuda:0, cuda:0, cuda:0, cuda:1, cuda:1, cuda:1]   # baseline device pool; override on CLI

# clustering_dir intentionally unset — let run_clustering produce a fresh result
# clustering_dir: ...
clustering_run_dir: data/pipeline_runs/mimicgen_kitchen_d1_may20_seed1_d100_bread_constrained

steps: [mimicgen_budget_rep_sweep]   # default; the wave commands below override `steps=` on CLI
```

**Key YAML semantics gotchas:**
- `pot: null` (whole-object null) → pin all axes at seed pose. ✓ what we want for non-target objects.
- `pot: {x: null, y: null, z_rot: null}` (per-axis null) → randomize each axis over D1 env range. ✗ silently breaks the experiment.
- Omitting an axis from a target object's dict (e.g. just `x: …, y: …`) → defaults to `[0,0]` offset = pin at seed.

**2. Task IV config** — `third_party/influence_visualizer/configs/kitchen_d1_may20.yaml`
Resolves the eval_dir for `run_clustering` and `select_mimicgen_seed`. Don't usually need to touch.

**3. Cupid arch config** — `third_party/cupid/configs/low_dim/kitchen_mimicgen_lowdim/diffusion_policy_cnn/config.yaml`
Diffusion policy hyperparameters. Already tuned.

### Defaults that must be in code (not config)

These live in pipeline-step code and override config defaults if the YAML doesn't set them. Confirm before running:

- `run_clustering.py` defaults: `clustering_influence_source = "policy_emb"`, `clustering_policy_emb_layer = "bottleneck_plan_t0"` (commit `8629537`).
- `compute_policy_embeddings.py` (cupid) is main's canonical version: regex-parsed layer grammar including `bottleneck_plan_t0` (restored in commit `d546a2d`).
- `run_mimicgen_generate.py` — gates the square_nut/square_peg hardcode behind `if task_name == "square"` (commit `7ea2bb9`); applies CLI variance knobs to every subtask for non-square tasks.
- `generate_mimicgen_demos.py` — has the adaptive retry loop (restored in `f2e42c9`): keeps passing until `success_budget` is met or `success_budget * 20` total trials.

### Pre-flight checklist

```bash
# 1. Branch + remote in sync
cd /home/erbauer/policy_doctor
git log --oneline -1   # should be one of: 3b17e4c, 7ea2bb9 or later
git status --short    # tracked changes empty

# 2. No leftover pipeline/mimicgen processes
ps -eo cmd | grep -E "run_mimicgen_generate|policy_doctor.scripts.run_pipeline|conda run -n mimicgen_torch2" | grep -v grep | wc -l   # must be 0

# 3. Baseline state per task (D D . . . means train_baseline + eval_baseline done, rest pending)
for run in mimicgen_kitchen_d1_may20_seed1_d100_bread_constrained \
           mimicgen_threading_d1_may20_seed1_d100_needle_constrained \
           mimicgen_three_piece_assembly_d1_may20_seed1_d100_piece1_constrained; do
  echo "$run"
  for sub in train_baseline eval_baseline compute_policy_embeddings run_clustering mimicgen_budget_rep_sweep; do
    [ -f "third_party/cupid/data/pipeline_runs/$run/$sub/done" ] && echo "  ✓ $sub" || echo "  · $sub"
  done
done

# 4. Disk + GPU + RAM
df -h /home/erbauer | head -3
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
free -h | awk '/Mem/{print "RAM used="$3" avail="$7}'
```

### Pipeline commands (wave-by-wave)

**Wave 0 (one-time per task, only if not already done):** train baseline + eval_baseline. Both are `D D` in the checklist above for all three tasks — skip unless you reset.

**Wave 1 — kitchen, budget=100, 9 arms across all 4 GPUs (currently running):**

```bash
mkdir -p logs
nohup conda run -n policy_doctor --no-capture-output python -m policy_doctor.scripts.run_pipeline \
  experiment=mimicgen_kitchen_d1_may20_d100_bread_constrained \
  data_source=mimicgen_kitchen \
  steps=[compute_policy_embeddings,run_clustering,mimicgen_budget_rep_sweep] \
  mimicgen_budget_rep_sweep.budgets='[100]' \
  'mimicgen_budget_rep_sweep.devices=[cuda:0,cuda:1,cuda:2,cuda:3,cuda:0,cuda:1,cuda:2,cuda:3,cuda:0]' \
  > logs/kitchen_b100.log 2>&1 &
```

Note: `experiment=...` (NOT `+experiment=...`) because the base config already declares `experiment: null`.

**Wave 2 — kitchen, budget=300:** same command, change `budgets='[100]'` to `budgets='[300]'` and log to `logs/kitchen_b300.log`. (compute_policy_embeddings + run_clustering will auto-skip since they're already done.)

**Wave 3 — kitchen, budget=500:** `budgets='[500]'`, log to `logs/kitchen_b500.log`.

**Waves 4-6 — threading:** swap `experiment=` and `data_source=` to threading variants. Per-task device pool — kitchen + TPA share `cuda:0/1` per their yaml; threading owns `cuda:2/3`. Either let kitchen finish first or override `devices=` on CLI to spread.

```bash
nohup conda run -n policy_doctor --no-capture-output python -m policy_doctor.scripts.run_pipeline \
  experiment=mimicgen_threading_d1_may20_d100_needle_constrained \
  data_source=mimicgen_threading \
  steps=[compute_policy_embeddings,run_clustering,mimicgen_budget_rep_sweep] \
  mimicgen_budget_rep_sweep.budgets='[100]' \
  'mimicgen_budget_rep_sweep.devices=[cuda:0,cuda:1,cuda:2,cuda:3,cuda:0,cuda:1,cuda:2,cuda:3,cuda:0]' \
  > logs/threading_b100.log 2>&1 &
```

**Waves 7-9 — TPA:** analogous.

```bash
nohup conda run -n policy_doctor --no-capture-output python -m policy_doctor.scripts.run_pipeline \
  experiment=mimicgen_three_piece_assembly_d1_may20_d100_piece1_constrained \
  data_source=mimicgen_three_piece_assembly \
  steps=[compute_policy_embeddings,run_clustering,mimicgen_budget_rep_sweep] \
  mimicgen_budget_rep_sweep.budgets='[100]' \
  'mimicgen_budget_rep_sweep.devices=[cuda:0,cuda:1,cuda:2,cuda:3,cuda:0,cuda:1,cuda:2,cuda:3,cuda:0]' \
  > logs/tpa_b100.log 2>&1 &
```

### Live monitoring

```bash
# Active arm processes
ps -eo cmd | grep "python /home/erbauer/policy_doctor/scripts/run_mimicgen_generate" | grep -v grep | wc -l

# Per-arm progress (replace TASK with kitchen / threading / three_piece_assembly)
TASK=kitchen; CONSTRAINT=bread
SWEEP="third_party/cupid/data/pipeline_runs/mimicgen_${TASK}_d1_may20_seed1_d100_${CONSTRAINT}_constrained/mimicgen_budget_rep_sweep"
for arm in "$SWEEP"/mimicgen_*/; do
  name=$(basename "$arm")
  states=""
  [ -f "$arm/select_mimicgen_seed/done" ] && states+="SEL "
  [ -f "$arm/generate_mimicgen_demos/done" ] && states+="GEN "
  [ -f "$arm/train_on_combined_data/done" ] && states+="TRAIN "
  [ -f "$arm/eval_mimicgen_combined/done" ] && states+="EVAL "
  printf "  %-50s %s\n" "$name" "$states"
done

# Per-arm MimicGen success rate
for s in $(find "$SWEEP" -name "stats.json" 2>/dev/null); do
  ns=$(grep -oE '"num_success"[: ]+[0-9]+' "$s" | head -1 | awk '{print $NF}')
  na=$(grep -oE '"num_attempts"[: ]+[0-9]+' "$s" | head -1 | awk '{print $NF}')
  arm=$(echo "$s" | grep -oE 'mimicgen_[^/]+_budget[0-9]+_rep[0-9]')
  echo "  $arm: $ns/$na"
done
```

### Stop + clean (between waves or after a failed run)

```bash
# Kill all pipeline + mimicgen processes (use this — don't rely on pkill alone)
ps -eo pid,cmd | grep -E "run_mimicgen_generate|policy_doctor.scripts.run_pipeline|conda run -n (mimicgen_torch2|policy_doctor)" \
  | grep -v grep | awk '{print $1}' | xargs -r kill -9
sleep 3

# Clean a failed kitchen sweep (replace TASK / CONSTRAINT for other tasks)
TASK=kitchen; CONSTRAINT=bread
SWEEP="third_party/cupid/data/pipeline_runs/mimicgen_${TASK}_d1_may20_seed1_d100_${CONSTRAINT}_constrained/mimicgen_budget_rep_sweep"
rm -rf "$SWEEP"/mimicgen_*
rm -f "$SWEEP/done" "$SWEEP/result.json"

# Truncate log to start fresh
> logs/${TASK}_b100.log

# Optional: also drop the clustering + policy_embeddings if you want to re-cluster
# (only do this if you changed clustering config or the embedding layer)
# rm -rf third_party/cupid/data/pipeline_runs/mimicgen_${TASK}_d1_may20_seed1_d100_${CONSTRAINT}_constrained/{run_clustering,compute_policy_embeddings}
# rm -rf third_party/influence_visualizer/configs/${TASK}_d1_may20/clustering
# find third_party/cupid/data/outputs/eval_save_episodes -name "bottleneck_plan_t0.npz" -delete
```

### After all 3 waves complete — comparison

```bash
# Aggregate eval scores across (task, heuristic, budget, rep) and run statistical comparison
# (compare_policies.py expects all eval_log.json files in place)
python -m policy_doctor.scripts.compare_policies \
  --task_config kitchen_d1_may20 \
  --train_date may20_kitchen_d1_d100_bread_constrained \
  --output_dir reports/may20_sweep
```

---

## Data Flow Audit — `generate_mimicgen_demos.py` (post round-3)

Walking the pipeline end-to-end to verify objects, subtasks, constraints, and timing are all correct now.

### 1. Object names match across env ↔ yaml ↔ seed.hdf5

Verified `_get_initial_placement_bounds()` returns object names that match our yaml `object_pose_ranges` keys, which match the names in the prepared seed.hdf5's `datagen_info/object_poses` group:

| Task | Env objects | Yaml keys | seed.hdf5 keys |
|---|---|---|---|
| Kitchen | bread, pot, stove, button, serving_region | bread, pot, stove, button, serving_region | bread, button, pot, serving_region, stove |
| Threading | needle, tripod | needle, tripod | needle, tripod |
| TPA | piece_1, piece_2, base | piece_1, piece_2, base | piece_1, piece_2, base |

All matches confirmed. The `_constrained_bounds` fuzzy-match (exact-or-substring) on `bounds_key ↔ seed_poses key` succeeds.

### 2. Subtask specs come from `config_factory(task_name)` template

| Task | # Subtasks | Subtask chain (object_ref / term_signal) |
|---|---|---|
| Kitchen | **7** | button/stove_on → pot/grasp_pot → stove/place_pot_on_stove → bread/grasp_bread → pot/place_bread_in_pot → serving_region/serve → button/(end) |
| Threading | **2** | needle/grasp → tripod/(end) |
| TPA | **4** | piece_1/grasp_1 → base/insert_1 → piece_2/grasp_2 → piece_1/(end) |

All subtasks use `selection_strategy='random'` for these 3 tasks (none use `nearest_neighbor_object`, so `--nn_k` is a no-op).

`run_mimicgen_generate.py` correctly:
- Square branch (when `task_name == 'square'`): hardcodes `square_nut`/`square_peg` subtask_1/2
- Else branch: keeps `config_factory` template intact, applies CLI variance knobs (`action_noise`, `num_interpolation_steps`, `num_fixed_steps`) to every subtask in the template

Live log from the current kitchen run confirms:
```
[run_mimicgen_generate] using template subtask spec for task_name='kitchen' (7 subtasks);
  signals=['stove_on', 'grasp_pot', 'place_pot_on_stove', 'grasp_bread',
           'place_bread_in_pot', 'serve', None]
```

### 3. Constraint application — actually applied now (round-3 fix)

Previously broken: pipeline step pre-computed `seed_object_poses` from the SOURCE dataset (default hardcoded to square) → kitchen runs got `{square_nut, square_peg}` poses → no match against `{bread, pot, ...}` env objects → no constraint applied → bread sampled from full D1 range → ~0% MimicGen success.

Now (post commits `6012904` + `092b524`): `run_mimicgen_generate.py` reads `seed_object_poses` directly from the prepared seed.hdf5's `datagen_info/object_poses` *after* `prepare_src_dataset` runs.

Live log confirms correct constraint application:
```
[run_mimicgen_generate] read seed_object_poses from prepared seed.hdf5:
  ['bread', 'button', 'pot', 'serving_region', 'stove']
[run_mimicgen_generate] constrained object poses on Kitchen_D1:
  bread(x=[-0.304,-0.224], y=[-0.244,-0.164], z_rot=0.012(pinned));
  button(x=-0.169(pinned), y=0.110(pinned), z_rot=3.142(pinned));
  pot(x=-0.051(pinned), y=-0.071(pinned), z_rot=0.218(pinned));
  serving_region(x=0.145(pinned), y=-0.178(pinned), z_rot=0.000(pinned));
  stove(x=-0.018(pinned), y=0.186(pinned), z_rot=0.000(pinned))
```

Reading:
- bread x range is `[seed_x − 0.04, seed_x + 0.04]` (±4cm around the rollout's bread x), same for y
- bread z_rot pinned at 0.012 rad (seed value)
- All other objects pinned exactly at seed poses

### 4. Constraint timing — applied at the right moment

Kitchen's `_get_placement_initializer()` is called once at env `__init__` and bakes bounds into `placement_initializer` (concrete x_range / y_range / rotation values). The monkey-patch order in `run_mimicgen_generate.py`:

1. Line ~280: `env_cls._get_initial_placement_bounds = _constrained_bounds`
2. Line ~351: `generate_dataset(cfg, ...)` internally calls `robosuite.make()` → `env_cls.__init__()`
3. `__init__` → `_get_placement_initializer()` → `self._get_initial_placement_bounds()` resolves to the patched method
4. `placement_initializer` is built with our constrained ranges
5. Every subsequent `env.reset()` samples from those baked ranges

Patch must run BEFORE env creation — and it does. Verified.

### 5. Per-seed adaptive retry — correct

`generate_mimicgen_demos.py` flow when `success_budget` is set and seed.hdf5 has multiple demos:

1. Extract each demo into `seed_<i>/seed.hdf5` (single-demo file)
2. Adaptive loop: while `total_successes < success_budget` and `total_trials < success_budget * 20`:
   - Estimate `trials_per_seed = ceil(remaining_needed / n_seeds / observed_rate * 1.2)` (with 5% effective-rate floor to prevent explosion)
   - For each seed: launch subprocess with `seed_<i>/seed.hdf5` and `trials_per_seed`
   - Subprocess writes pass-stats.json; loop reads it and updates totals
3. Merge all pass outputs; if merged > success_budget, subsample to exactly `success_budget` demos
4. Final stats.json reports per-seed and aggregate counts

This guarantees: equal data quantity per arm (`success_budget` demos), variable compute (bad seeds spend more passes), hard 20× cap on compute waste.

### 6. Does the design fundamentally make sense? Yes, with one open empirical question.

**Yes — the pipeline is now coherent end-to-end:**

- `select_mimicgen_seed` picks rollouts via heuristic and materializes them as a multi-demo `seed.hdf5`
- `generate_mimicgen_demos` extracts each seed individually, then for each: `prepare_src_dataset` annotates with `datagen_info` (subtask signals, object poses by replaying in the env), then `generate_dataset` adapts the seed trajectory to new initial conditions sampled within our constraint
- Constraint correctly limits object initial poses around the seed's own pose
- Adaptive loop guarantees `success_budget` demos per arm or 20× compute cap
- Aggregated demos feed into `train_on_combined_data` (baseline 100 + generated success_budget)

**Open empirical question — kitchen specifically.** Even with the constraint now correctly applied, kitchen's 7-subtask chain involves bread being grasped at sub_4 then placed-in-pot at sub_5. A perturbed bread x/y at grasp time changes the in-gripper bread pose; sub_5's place trajectory (recorded with the seed's grasp) may drop the bread off-pot. We previously observed `place_bread_in_pot` failing 47% in a (mis-)configured run; with the real constraint now applied, the actual rate is TBD.

Threading (2 subtasks) and TPA (4 subtasks) have shorter chains and should be more robust. The methodology should validate cleanly on at least one of those even if kitchen needs further per-task tuning.

### 7. Latent design issues not blocking the current run

- `_DEFAULT_SOURCE_DATASET = "data/source/mimicgen/core_datasets/square/demo_src_square_task_D1/demo.hdf5"` is still hardcoded in `generate_mimicgen_demos.py:85`. Only reachable when `SelectMimicgenSeedStep` hasn't run (a path none of our experiments take). Worth cleaning eventually.
- `--seed_object_poses` CLI arg in `run_mimicgen_generate.py` is now a legacy fallback (with WARNING). Could be removed entirely once no callers depend on it.

---

## Session-2026-05-24 Re-run — Lessons Learned

This is a redo of an earlier failed sweep. Documenting the failure modes so future runs avoid them.

### Round-1 failure: wrong policy embedding layer

The earlier sweep clustered on a simplified embedding (`plan_bottleneck`: UNet mid-block at t=0 with a **zero action** input, averaged over horizon). This captures mostly *what the policy sees*, not *what it decides*. Behavior-graph clusters therefore split by observation phase rather than behavioral phase, so `behavior_graph` and `diversity` heuristics had no real signal advantage over `random` — preliminary results showed `random` ≥ `behavior_graph` ≈ `diversity` across all 3 tasks.

**Fix:** Use main's canonical `compute_policy_embeddings.py` (commit `904e845`), which supports `bottleneck_plan_t0`: hook = UNet bottleneck, action = planned (full denoise run then condition on its output), t_single = 0. Pulled into branch `feat/mimicgen-trajectory-pipeline` in commit `d546a2d`.

### Round-1 cleanup

- Archived per-arm sweep work for all 3 tasks (training, eval, per-arm pipeline_runs) — about 308 GB total, kept locally only briefly then **deleted** to free disk (we needed disk for the re-run, no external archive destination available)
- Kept baselines: trained checkpoints + 500-episode eval rollouts + `metadata.yaml` + `eval_log.json` for all 3 tasks
- Deleted bad `policy_embeddings/plan_bottleneck.npz` files from baseline eval dirs
- Deleted stale clustering dirs in `third_party/influence_visualizer/configs/<task>_d1_may20/clustering/`
- Deleted `~/data/robocasa_data` (190 GB, unused) to make room
- Disk after cleanup: 548 GB free

### Round-2 launch — bug chain

Launch attempts revealed several issues:

1. **Hydra `+experiment=` vs `experiment=`** — base config already declares `experiment: null`, so `+experiment=...` errors with "Multiple values for experiment". Use `experiment=...` (no `+`).
2. **`OmegaConf.select` fallback to plain dict** crashes downstream selects — `ComputePolicyEmbeddingsStep` used `OmegaConf.select(cfg, "policy_emb") or {}`. Fixed: `or OmegaConf.create({})`. Commit `615704c`.
3. **Kitchen eval path mismatch** — kitchen baseline eval was at `eval_save_episodes/<train_dir>/latest/` (flat) while `get_eval_dir` expects `eval_save_episodes/<eval_date>/<train_dir>/latest/` (nested). Worked around with a symlink.
4. **CompositeStep `parent_run_dir` wasn't propagating** — every sweep arm raised "No clustering directories found" because `SelectMimicgenSeedStep` was looking under `<top>/mimicgen_budget_rep_sweep/run_clustering/` instead of `<top>/run_clustering/`. The composite arm's sub-steps were getting `parent_run_dir = self.run_dir` (the sweep step_dir) instead of `self.parent_run_dir` (the top-level pipeline run_dir). Fixed in commit `51a700b`.
5. **z_rot constraint causes ~0% MimicGen success** — `bread.z_rot: [-0.524, 0.524]` (±π/6) caused all 9 arms to retry trials with 0 successes after 55+ min. MimicGen replays the seed trajectory adapted to a new sampled initial pose, but Cartesian interpolation of the end-effector trajectory doesn't rotate the gripper to match the new object orientation, so every trial fails the grasp. Memory note `project_ic_bugs_fixed` already documents this fix. Resolved by setting `z_rot: null` (use the seed's own rotation; only ±4cm x/y perturbation applied). Commit `cec6149`.
6. **Mid-debug regression**: at one point I "loosened" z_rot to `[-1.047, 1.047]` (±π/3). That's the wrong direction — wider range = harder MimicGen replay, worse success. Reverted.
7. **HDF5 file-lock + b-tree corruption from concurrent test runs on shared `seed.hdf5`** — investigating z_rot's effect by running multiple `prepare_src_dataset` calls in parallel on the same seed file corrupted it (b-tree "duplicate key", "negative link count"). Resolved by clearing `select_mimicgen_seed/` for every arm so seeds regenerate cleanly; also exported `HDF5_USE_FILE_LOCKING=FALSE` in test scripts.

### Round-3 (2026-05-25): the `seed_object_poses` source-dataset bug

After resolving every subtask/config issue in round-2, MimicGen still hit ~0% success on kitchen with bread perturbed even ±4cm x/y. Diagnosis (via inspecting CLI args of running mimicgen subprocesses):

```
[run_mimicgen_generate] constrained object poses on Kitchen_D1:
  square_nut(x=-0.024(pinned), y=-0.001(pinned), z_rot=2.786(pinned));
  square_peg(x=0.100(pinned), y=-0.082(pinned), z_rot=0.000(pinned))
```

Kitchen has no `square_nut`/`square_peg` — those are the square task's objects. The script was silently constraining **non-existent objects** in the kitchen env, so the constraint did nothing and bread sampled across the full D1 random range (`x ∈ (-0.2, 0.0), y ∈ (-0.25, -0.05), z_rot ∈ (-π/2, π/2)`) — guaranteeing ~0% replay success.

**Root cause** (3 layers):

1. **Hardcoded square default**: `_DEFAULT_SOURCE_DATASET = "data/source/mimicgen/core_datasets/square/demo_src_square_task_D1/demo.hdf5"` in `generate_mimicgen_demos.py:85`. None of the kitchen/threading/TPA yamls override `mimicgen_datagen.source_dataset_path` → this default kicks in.

2. **Wrong fallback target**: when `seed.hdf5` (from `select_mimicgen_seed`) lacks `datagen_info`, the pipeline step falls back to reading object poses from the SOURCE dataset. But source object poses are from a SOURCE demo, not the rollout being seeded. For square's source happened to share object names with square's env, so the bug was invisible there.

3. **`prepare_src_dataset` is run by the subprocess ANYWAY**: line 192 of `run_mimicgen_generate.py` calls `prepare_src_dataset(seed_hdf5, …)` before constraint application. After this, `seed.hdf5` HAS `datagen_info/object_poses` — those are the correct rollout-state poses. The pre-computation in the pipeline step was unnecessary AND wrong.

**Fix (commits `6f79de3` + `6012904` + this one):**

- Set `mimicgen_datagen.source_dataset_path: data/source/mimicgen/core_datasets/<task>_d1/demo.hdf5` in all three may20 experiment yamls (defensive: future code paths that read from source get the right file).
- `run_mimicgen_generate.py`: read `seed_object_poses` directly from prepared `seed.hdf5` (after the subprocess's own `prepare_src_dataset` runs). CLI `--seed_object_poses` kept as a fallback for legacy callers, with a WARNING.
- `generate_mimicgen_demos.py`: stop pre-computing `seed_object_poses` from a source dataset. Pass only `--object_pose_ranges` to the subprocess.

**Aborted detour**: tried to fix by running `prepare_src_dataset` once on the kitchen source HDF5 (1000 demos × ~2s each = ~33 min). Killed mid-way once we realized source-dataset poses are not the same as rollout-seed poses — fixing the read-path was the right fix, not running prepare on the source.

Aside: the killed `prepare_src_dataset` left `kitchen_d1/demo.hdf5` in a corrupted state (`Unable to open object (len not positive after adjustment for EOA)`). The new code doesn't read from this file, but if any other downstream code does, the file would need to be restored.

### Code fixes committed during the re-run (branch `feat/mimicgen-trajectory-pipeline`)

| Commit | Description |
|---|---|
| `e6f3f18` | fix: per-arm mimicgen datagen dir + --guarantee passthrough |
| `f0adcdc` | fix: rollouts.hdf5 + AsyncVectorEnv shared-memory crash |
| `516666e` | fix: strip `_orig_mod.` prefix from torch.compile state_dicts |
| `9238c1d` | fix: auto-inject model_file when seed HDF5 lacks it |
| `68ce938` | feat: kitchen/threading/three_piece_assembly D1 may20 sweep configs |
| `d546a2d` | revert + restore: use main's policy_emb implementation (`bottleneck_plan_t0`) |
| `8629537` | feat: pipeline-step wrapper for compute_policy_embeddings + bottleneck_plan_t0 default |
| `78b5c2f` | config: switch may20 sweep experiments to bottleneck_plan_t0 layer |
| `f2f5c14` | fix: misc pipeline + training fixes needed by the re-run (hydra.run.dir, wandb allow_val_change) |
| `615704c` | fix: OmegaConf.select needs DictConfig not plain dict for fallback |
| `5a75016` | fix: unset stale clustering_dir in threading experiment |
| `51a700b` | fix: composite step propagates parent_run_dir to its sub-steps |
| `cec6149` | revert: z_rot=null (documented fix; ±π/3 was wrong direction) |

---

## Current State (2026-05-24, 21:40 UTC)

- Kitchen budget=100: 9 arms in MimicGen generate, all worker pythons at ~94% CPU
- compute_policy_embeddings ✓ done (50000 timesteps × 128-dim at `bottleneck_plan_t0.npz`)
- run_clustering ✓ done (output at `third_party/influence_visualizer/configs/kitchen_d1_may20/clustering/...kmeans_k15`)
- No results yet — first stats.json from any arm expected within ~20-30 min

### Planned waves after kitchen budget=100 completes

| Wave | Task(s) | Budget | Notes |
|---|---|---|---|
| 1 (current) | kitchen | 100 | 9 arms in 9 GPU slots |
| 2 | kitchen | 300 | reuses compute_policy_embeddings + run_clustering |
| 3 | kitchen | 500 | same |
| 4-6 | threading | 100/300/500 | needs separate launches; threading config still uses x/y±4cm, z_rot=null |
| 7-9 | TPA | 100/300/500 | same |

After all 3 tasks × 3 budgets have completed evals, run `compare_policies.py` for proper Wilson CIs / Beta posteriors / CLD letters.

---

## Open Risks

- **z_rot=null gives up the rotational-failure-targeting dimension.** Once we have data with z_rot=null, we may want to investigate whether a tighter range (e.g. ±0.05 rad ≈ ±3°) is small enough that MimicGen's nearest-neighbor source selection can find a matching seed. The z_rot test sweep this session was aborted due to seed.hdf5 corruption from concurrent runs — would need a clean retry on a single arm to test.
- **Behavior_graph + diversity heuristics still depend on cluster quality.** Even with the correct embeddings, if the kitchen task's behavior space is shallow (few distinguishable phases), we may not see large gaps between heuristics. The earlier session's `random ≥ all` result is suggestive — but it was confounded by bad embeddings. Need fresh data to draw any conclusion.
- **TPA has only 6 baseline checkpoints** (vs kitchen 9, threading 14). Smaller checkpoint pool means eval estimates have wider variance.
- **Threading needle constraint may still be too hard** even with z_rot=null. The previous run had 7/9 threading arms with 0 demos despite the looser constraint; needle is structurally hard for MimicGen replay. May produce baseline-only training for many threading arms.
