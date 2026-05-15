# MimicGen Budget Sweep — Apr 25 2026

Identifier: `apr25_sweep`

This document describes the experimental design, config layout, policy inventory,
and run instructions for the first full budget sweep over the MimicGen trajectory
generation pipeline.

---

## Overview

The experiment asks: **how does augmentation budget (number of successfully generated
MimicGen demos) interact with seed-selection heuristic?**

We sweep `success_budget ∈ {20, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}`
across all three seed-selection heuristics (`random`, `behavior_graph`, `diversity`)
for three independently trained baseline policies (dataset seeds 0, 1, 2).

This gives a full **3 heuristics × 11 budgets × 3 seeds = 99 augmented policies**,
each evaluated at 500 episodes × 5 top-k checkpoints = 2 500 rollouts per policy.

---

## Experimental Design

### Baseline seeds (Phase 1 — "outer loop")

The outer loop varies the **dataset seed** passed to the baseline policy trainer
(`task.dataset.seed=<seed>`). This produces three independent baseline policies
trained on different orderings of the 60 Square D1 source demos. This is the same
seed mechanism used in `third_party/cupid/scripts/train/train_policies.sh`.

| Seed | Run directory |
|------|--------------|
| 0 | `data/pipeline_runs/mimicgen_square_apr25_sweep_seed0/` |
| 1 | `data/pipeline_runs/mimicgen_square_apr25_sweep_seed1/` |
| 2 | `data/pipeline_runs/mimicgen_square_apr25_sweep_seed2/` |

Each seed gets its own complete Phase 1 (train → eval → attribution → clustering)
before Phase 2 (budget sweep) begins for that seed.

### Heuristics

| Name | Class | Description |
|------|-------|-------------|
| `random` | `RandomSelectionHeuristic` | Uniformly samples a successful rollout as seed trajectory |
| `behavior_graph` | `BehaviorGraphPathHeuristic` | Selects the rollout that best matches the highest-probability success path in the behavior graph |
| `diversity` | `DiversitySelectionHeuristic` | Picks one rollout per distinct success path, maximising behavioral coverage |

### Budgets

`[20, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]`

`success_budget` is the number of **successfully generated** MimicGen demonstrations.
MimicGen attempts more trajectories internally and discards failures; the budget
controls how many successes land in the output HDF5.

### Policies trained per seed

For each baseline seed, the sweep trains **33 augmented policies** (3 heuristics ×
11 budgets), each on `original (60 demos) + generated (budget demos)`.
Additionally, the **baseline policy itself** (60 demos, no augmentation) is
evaluated as a reference condition (`eval_baseline` step).

Total across all three seeds: **3 × (1 baseline + 33 augmented) = 102 policies**.

---

## Phase breakdown

### Phase 1 — Upstream (once per seed)

| Step | Description | Output |
|------|-------------|--------|
| `train_baseline` | Train diffusion policy on 60 source demos | `data/outputs/train/apr25_sweep/...` |
| `eval_policies` | 500-episode eval of top-5 checkpoints | `data/outputs/eval_save_episodes/apr25_sweep/...` |
| `train_attribution` | TRAK attribution on source demos vs rollouts | `data/outputs/trak/...` |
| `finalize_attribution` | Assemble full influence matrix | `<run_dir>/finalize_attribution/` |
| `compute_infembed` | InfEmbed embeddings for all rollout timesteps | `<run_dir>/compute_infembed/` |
| `run_clustering` | UMAP → KMeans → behavior graph | `<run_dir>/run_clustering/` |

Phase 1 always re-runs from scratch (`skip_if_done=false`) to ensure clean state.

### Phase 2 — Budget sweep (per seed, concurrent)

A single `mimicgen_budget_sweep` pipeline step spawns all 33 arms concurrently
using a device pool of **4 slots (2 per GPU)**.

Each arm runs:

| Sub-step | Description |
|----------|-------------|
| `select_mimicgen_seed` | Apply heuristic to clustering result → materialize seed HDF5 |
| `generate_mimicgen_demos` | Run MimicGen to produce `success_budget` demos |
| `train_on_combined_data` | Train on original + generated combined HDF5 |
| `eval_mimicgen_combined` | 500-episode eval × top-5 checkpoints |

Arm results land under `<run_dir>/mimicgen_<heuristic>_budget<N>/`.

Phase 2 uses `skip_if_done=true` (default) so individual arms can resume if
the process is interrupted.

---

## Config layout

### Experiment YAML

`policy_doctor/configs/experiment/mimicgen_square_sweep_apr25.yaml`

Contains all task-agnostic sweep parameters:

```yaml
baseline:
  max_train_episodes: 60
  checkpoint_topk: 5          # top-5 checkpoints saved and evaluated
  n_test_rollouts: 100        # in-training eval episodes per checkpoint

evaluation:
  num_episodes: 500

mimicgen_datagen:
  success_budget: 200         # default; overridden per arm
  num_seeds: 10               # seed trajectories evaluated per arm
  policy_seed: 0
  success_only: true
  top_k_paths: 20
  min_path_probability: 0.0
  random_seed: null

mimicgen_budget_sweep:
  heuristics: [random, behavior_graph, diversity]
  budgets: [20, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
  devices: [cuda:0, cuda:0, cuda:1, cuda:1]   # 2 concurrent arms per GPU
```

### Data source YAML

`policy_doctor/configs/data_source/mimicgen_square.yaml`

Contains task-specific MimicGen parameters (Square D1):

```yaml
mimicgen_datagen:
  original_dataset_path: data/mimicgen/square_d1_60.hdf5
  task_name: square
  env_interface_name: MG_Square
  env_interface_type: robosuite
  action_noise: 0.05
  subtask_term_offset_range: [10, 20]
  nn_k: 3
  interpolate_from_last_target_pose: true
  ...

attribution:
  dataset_path: data/source/mimicgen/core_datasets/square/demo_src_square_task_D1/demo.hdf5
```

To run the sweep for a different task (e.g. `transport`), create
`data_source/mimicgen_transport.yaml` with the appropriate values and pass
`data_source=mimicgen_transport`. The experiment YAML stays untouched.

---

## Running the sweep

### Full run (recommended)

```bash
# Default: Square, seeds 0–2, cuda:0 for Phase 1
./scripts/run_mimicgen_budget_sweep.sh

# Override task, seeds, or Phase 1 device
TASK=square SEEDS="0 1 2" DEVICE=cuda:0 ./scripts/run_mimicgen_budget_sweep.sh

# Background with logging
nohup bash -lc "
  source ~/miniforge3/etc/profile.d/conda.sh
  cd /path/to/worktree
  TASK=square SEEDS='0 1 2' DEVICE=cuda:0 ./scripts/run_mimicgen_budget_sweep.sh
" > /tmp/budget_sweep_apr25.log 2>&1 &
echo "PID=$!"
```

### Manual (single seed, two phases)

```bash
# Phase 1 — upstream (always re-runs from scratch)
python -m policy_doctor.scripts.run_pipeline \
  data_source=mimicgen_square \
  experiment=mimicgen_square_sweep_apr25 \
  run_dir=data/pipeline_runs/mimicgen_square_apr25_sweep_seed0 \
  seeds=[0] \
  device=cuda:0 \
  skip_if_done=false \
  steps=[train_baseline,eval_policies,train_attribution,finalize_attribution,\
         compute_infembed,run_clustering]

# Phase 2 — budget sweep (resumes if interrupted)
python -m policy_doctor.scripts.run_pipeline \
  data_source=mimicgen_square \
  experiment=mimicgen_square_sweep_apr25 \
  run_dir=data/pipeline_runs/mimicgen_square_apr25_sweep_seed0 \
  seeds=[0] \
  steps=[mimicgen_budget_sweep]
```

### Resuming an interrupted Phase 2

Phase 2 arms write a `done` sentinel on completion. Re-running the Phase 2
command will skip completed arms and continue from where it left off.

To force a specific arm to re-run, delete its sentinel:

```bash
rm data/pipeline_runs/mimicgen_square_apr25_sweep_seed0/mimicgen_behavior_graph_budget200/done
```

### Adding a new heuristic to the sweep

1. Implement the heuristic class in `policy_doctor/mimicgen/heuristics.py`
2. Register it in `build_heuristic()` in that file
3. Add its name to the `heuristics` list in the experiment YAML (or pass as CLI override):

```bash
python -m policy_doctor.scripts.run_pipeline \
  experiment=mimicgen_square_sweep_apr25 \
  "mimicgen_budget_sweep.heuristics=[my_new_heuristic]" \
  steps=[mimicgen_budget_sweep]
```

---

## Output structure

```
data/pipeline_runs/mimicgen_square_apr25_sweep_seed{0,1,2}/
├── pipeline_config.yaml              # full Hydra config snapshot
├── train_baseline/
│   └── done
├── eval_policies/
│   └── done
├── train_attribution/
│   └── done
├── finalize_attribution/
│   └── done
├── compute_infembed/
│   └── done
├── run_clustering/
│   └── done
└── mimicgen_budget_sweep/
    ├── mimicgen_random_budget20/
    │   ├── select_mimicgen_seed/
    │   ├── generate_mimicgen_demos/
    │   ├── train_on_combined_data/
    │   ├── eval_mimicgen_combined/
    │   └── done
    ├── mimicgen_random_budget100/
    │   └── ...
    ├── mimicgen_behavior_graph_budget20/
    │   └── ...
    ├── mimicgen_diversity_budget1000/
    │   └── ...
    └── ...   (33 arms total)
```

Training checkpoints and eval episodes are written to:

```
third_party/cupid/data/outputs/train/apr25_sweep/...
third_party/cupid/data/outputs/eval_save_episodes/apr25_sweep/...
```

---

## Evaluation summary

| Quantity | Value |
|----------|-------|
| Baseline policies | 3 (one per dataset seed) |
| Augmented policies | 33 per seed × 3 seeds = 99 |
| Total policies | 102 |
| Checkpoints evaluated per policy | 5 (top-k) |
| Eval episodes per checkpoint | 500 |
| Total eval rollouts | 102 × 5 × 500 = 255 000 |
| Concurrent arms during Phase 2 | 4 (2 per RTX 4090) |

---

## Relationship to prior runs

| Run | Identifier | Description |
|-----|-----------|-------------|
| `mimicgen_square_pipeline_apr23` | `apr23` | First sign-of-life: 3 heuristics × budget=200, 3 replicates each + budget=20 ablation |
| `mimicgen_square_apr25_sweep_seed{0,1,2}` | `apr25_sweep` | Full budget sweep: 3 heuristics × 11 budgets × 3 baseline seeds |

The apr23 runs used a single fixed baseline policy (seed 0) and varied `random_seed`
for replication. The apr25 sweep instead varies the **baseline training seed**, which
is a stronger form of statistical independence (different data ordering → different
policy weights → different behavior graph → different seed selection).
