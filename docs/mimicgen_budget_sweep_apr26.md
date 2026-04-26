# MimicGen Budget Sweep — Apr 26 2026

Identifier: `apr26_sweep`

This document describes the experimental design, config layout, policy inventory,
and run instructions for the second full budget sweep. It extends the apr25 sweep
by adding a **baseline training data size** dimension (n_demos), enabling a joint
study of how source data quantity and augmentation strategy interact.

---

## Research Questions

1. **Does behavior-graph seed selection outperform random at each augmentation budget?**
   (fixed n_demos, vary heuristic × budget — same question as apr25)

2. **How does source data quantity interact with MimicGen augmentation?**
   (fixed heuristic × budget, vary n_demos)
   — Does augmentation provide more value when baseline data is scarce?
   — Does the behavior graph remain useful when only 60 source demos are available?

3. **Where is the crossover point between more source data and more augmentation?**
   At what (n_demos, budget) combination does performance saturate?

---

## Experimental Design

### Factor 1 — Baseline training data size (n_demos)

Each baseline policy is trained on a different-sized subset of the Square D1 source
demonstrations. This is controlled via `baseline.max_train_episodes=<N>`.

| n_demos | Train date tag | Notes |
|---------|---------------|-------|
| 60 | `apr26_sweep_demos60` | Same as apr25 baseline (full D1 set) |
| 100 | `apr26_sweep_demos100` | ~1.7× more source data |
| 300 | `apr26_sweep_demos300` | 5× more source data |

The Square D1 dataset has 1000 demos total; all three conditions are well within
the available data.

### Factor 2 — Dataset seed (outer loop)

Three independently trained baseline policies per (n_demos) condition, varying
`task.dataset.seed ∈ {0, 1, 2}`. This gives statistically independent replicates
with different data orderings → different policy weights → different behavior
graphs → different seed selections.

### Factor 3 — Seed-selection heuristic

| Name | Description |
|------|-------------|
| `random` | Uniformly samples a successful rollout as seed trajectory (baseline condition) |
| `behavior_graph` | Selects the rollout best matching the highest-probability success path |
| `diversity` | Picks one rollout per distinct success path, maximising behavioral coverage |

### Factor 4 — Augmentation budget

`success_budget ∈ {20, 100, 500, 1000}`

Compared to apr25 (11 values: 20–1000 step 100), apr26 uses 4 representative
points that span the range while reducing compute by ~3×. The intermediate
points (200–400) showed smooth monotonic trends in apr25 and are not essential
for the main story.

---

## Policy Inventory

**Per (seed, n_demos) combination:**
- 1 baseline policy (no augmentation, `n_demos` source demos only)
- 3 heuristics × 4 budgets = **12 augmented policies**
- Subtotal: **13 policies per combination**

**Full matrix: 3 seeds × 3 n_demos × 13 = 117 total policies**

| Condition | Count |
|-----------|-------|
| Baseline policies | 9 (3 seeds × 3 n_demos) |
| Augmented policies (random) | 36 (3 seeds × 3 n_demos × 4 budgets) |
| Augmented policies (behavior_graph) | 36 |
| Augmented policies (diversity) | 36 |
| **Total** | **117** |

Each policy is evaluated at 500 episodes × 5 top-k checkpoints = 2 500 rollouts.
**Total eval rollouts: 117 × 2 500 = 292 500**

---

## Changes from Apr 25

| Aspect | Apr 25 | Apr 26 |
|--------|--------|--------|
| n_demos dimension | Fixed (60) | Swept [60, 100, 300] |
| Budgets | [20, 100, …, 1000] (11 values) | [20, 100, 500, 1000] (4 values) |
| Phase 1 concurrency | Sequential (1 job at a time) | Parallel (4 concurrent, 2 per GPU) |
| Phase 1 device | Single `DEVICE` env var | `phase1_devices` list in YAML |
| Demo count config | Shell env vars `N_DEMOS` / `N_DEMOS_START` | `n_demos_sweep.demo_counts` in YAML |
| Train date per run | Single `apr25_sweep` | `apr26_sweep_demos<N>` per n_demos arm |
| Episode images saved | No (offscreen renderer disabled) | **Yes** (EEF pipeline fix applied) |
| Total policies | 102 | 117 |
| Total eval rollouts | ~255 000 | ~292 500 |

---

## Phase Breakdown

### Phase 1 — Upstream steps (once per (seed, n_demos))

All 9 (seed, n_demos) combinations run concurrently, up to 4 at a time
(controlled by `phase1_devices` in the experiment YAML).

| Step | Description | Output path |
|------|-------------|-------------|
| `train_baseline` | Train diffusion policy on n_demos source demos | `data/outputs/train/<train_date>/...` |
| `eval_policies` | 500-episode eval of top-5 checkpoints (with images saved) | `data/outputs/eval_save_episodes/<train_date>/...` |
| `train_attribution` | TRAK attribution on source demos vs rollouts | `data/outputs/trak/<train_date>/...` |
| `finalize_attribution` | Assemble full influence matrix | `<run_dir>/finalize_attribution/` |
| `compute_infembed` | InfEmbed embeddings for all rollout timesteps | `<run_dir>/compute_infembed/` |
| `run_clustering` | UMAP → KMeans → behavior graph | `<run_dir>/run_clustering/` |

Expected wall time per (seed, n_demos) combination: ~5–8 hours.
With 4 concurrent jobs: all 9 Phase 1 combinations complete in ~12–18 hours.

### Phase 2 — Budget sweep (per (seed, n_demos))

Phase 2 for each combination runs after **all** Phase 1 jobs complete.
Within each combination, 12 arms (3 heuristics × 4 budgets) run concurrently
via the `mimicgen_budget_sweep.devices` device pool.

| Sub-step | Description |
|----------|-------------|
| `select_mimicgen_seed` | Apply heuristic → materialize seed HDF5 |
| `generate_mimicgen_demos` | MimicGen generation → `success_budget` demos |
| `train_on_combined_data` | Train on original + generated combined HDF5 |
| `eval_mimicgen_combined` | 500-episode eval × top-5 checkpoints |

Expected wall time per combination: ~6–12 hours.
Combinations run sequentially (Phase 2 already uses all 4 GPU slots internally).

---

## Config Layout

### Experiment YAML

`policy_doctor/configs/experiment/mimicgen_square_sweep_apr26.yaml`

Key sections:

```yaml
train_date: apr26_sweep          # base; shell script appends _demos<N> per arm

baseline:
  max_train_episodes: 60         # default; overridden per n_demos arm

evaluation:
  num_episodes: 500

n_demos_sweep:
  demo_counts: [60, 100, 300]    # or: start/stop/step for a regular grid

phase1_devices: [cuda:0, cuda:0, cuda:1, cuda:1]   # 4 concurrent Phase 1 slots

mimicgen_budget_sweep:
  heuristics: [random, behavior_graph, diversity]
  budgets: [20, 100, 500, 1000]
  devices: [cuda:0, cuda:0, cuda:1, cuda:1]         # 2 concurrent arms per GPU
```

Because `evaluation.train_date` and `attribution.train_date` are **not** set
explicitly in the YAML, the shell script's per-arm `train_date=apr26_sweep_demos<N>`
CLI override propagates through all Phase 1 steps automatically. This ensures
each (seed, n_demos) combination writes to a unique checkpoint directory.

### Data source YAML

`policy_doctor/configs/data_source/mimicgen_square.yaml` — unchanged from apr25.

---

## Run Instructions

### Full run (recommended)

```bash
# Default: Square, seeds 0–2, everything from YAML
TASK=square SEEDS='0 1 2' DATE=apr26 ./scripts/run_mimicgen_budget_sweep.sh

# Background with logging
nohup bash -lc "
  source ~/miniforge3/etc/profile.d/conda.sh
  cd /path/to/worktree
  TASK=square SEEDS='0 1 2' DATE=apr26 ./scripts/run_mimicgen_budget_sweep.sh
" > /tmp/mimicgen_apr26_sweep.log 2>&1 &
echo "PID=$!"
```

### Adjusting the sweep from YAML (no shell env vars needed)

```yaml
# In mimicgen_square_sweep_apr26.yaml:

# Change demo counts:
n_demos_sweep:
  demo_counts: [60, 300]       # run only the extreme points

# Change budgets:
mimicgen_budget_sweep:
  budgets: [20, 500]           # quick ablation

# Add more Phase 1 concurrency (e.g. 6 slots across 3 GPUs):
phase1_devices: [cuda:0, cuda:0, cuda:1, cuda:1, cuda:2, cuda:2]
```

### Manual (single seed + n_demos)

```bash
# Phase 1
python -m policy_doctor.scripts.run_pipeline \
  data_source=mimicgen_square \
  experiment=mimicgen_square_sweep_apr26 \
  run_dir=data/pipeline_runs/mimicgen_square_apr26_sweep_seed0_demos60 \
  seeds=[0] \
  train_date=apr26_sweep_demos60 \
  baseline.max_train_episodes=60 \
  skip_if_done=true \
  steps=[train_baseline,eval_policies,train_attribution,finalize_attribution,\
         compute_infembed,run_clustering]

# Phase 2
python -m policy_doctor.scripts.run_pipeline \
  data_source=mimicgen_square \
  experiment=mimicgen_square_sweep_apr26 \
  run_dir=data/pipeline_runs/mimicgen_square_apr26_sweep_seed0_demos60 \
  seeds=[0] \
  train_date=apr26_sweep_demos60 \
  baseline.max_train_episodes=60 \
  steps=[mimicgen_budget_sweep]
```

### Resuming an interrupted Phase 2 arm

```bash
# Delete the arm's done sentinel to force re-run:
rm data/pipeline_runs/mimicgen_square_apr26_sweep_seed0_demos60/\
   mimicgen_behavior_graph_budget500/done
```

---

## Output Structure

```
third_party/cupid/data/pipeline_runs/
└── mimicgen_square_apr26_sweep_seed{0,1,2}_demos{60,100,300}/   (9 dirs)
    ├── pipeline_config.yaml
    ├── train_baseline/done
    ├── eval_policies/done
    ├── train_attribution/done
    ├── finalize_attribution/done
    ├── compute_infembed/done
    ├── run_clustering/done
    └── mimicgen_budget_sweep/
        ├── mimicgen_random_budget20/done
        ├── mimicgen_random_budget100/done
        ├── mimicgen_random_budget500/done
        ├── mimicgen_random_budget1000/done
        ├── mimicgen_behavior_graph_budget20/done
        ├── ...                                   (12 arms total)
        └── mimicgen_diversity_budget1000/done

third_party/cupid/data/outputs/train/
├── apr26_sweep_demos60/
│   ├── apr26_sweep_demos60_train_diffusion_unet_lowdim_square_mh_mimicgen_0/
│   ├── apr26_sweep_demos60_train_diffusion_unet_lowdim_square_mh_mimicgen_1/
│   └── apr26_sweep_demos60_train_diffusion_unet_lowdim_square_mh_mimicgen_2/
├── apr26_sweep_demos100/
│   └── ...
└── apr26_sweep_demos300/
    └── ...

third_party/cupid/data/outputs/eval_save_episodes/
├── apr26_sweep_demos60/...    (with per-timestep images saved)
├── apr26_sweep_demos100/...
└── apr26_sweep_demos300/...
```

---

## Evaluation Summary

| Quantity | Value |
|----------|-------|
| n_demos conditions | 3 (60, 100, 300) |
| Baseline seeds per condition | 3 |
| Heuristics | 3 (random, behavior_graph, diversity) |
| Budgets | 4 (20, 100, 500, 1000) |
| Baseline policies | 9 |
| Augmented policies | 108 |
| **Total policies** | **117** |
| Checkpoints evaluated per policy | 5 (top-k) |
| Eval episodes per checkpoint | 500 |
| **Total eval rollouts** | **292 500** |
| Phase 1 concurrent slots | 4 (2 per RTX 4090) |
| Phase 2 concurrent arms | 4 (2 per RTX 4090) |
| Estimated total wall time | ~3–4 days |

---

## Technical Notes

### Image saving in eval rollouts

The `eval_policies` step now saves per-timestep RGB frames alongside the low-dim
observations in each episode's pickle file. This required fixing the offscreen
renderer path in `mimicgen_lowdim_runner.py`:

- `create_env()` now accepts `render_offscreen=save_episodes` so the robosuite
  environment initializes EGL when episode saving is requested.
- A `dummy_env_fn` with `has_offscreen_renderer=False` is used for the
  `AsyncVectorEnv` dummy environment to avoid EGL context conflicts in forked
  worker processes.
- The `media_dir` guard was fixed to handle the case where `n_test_vis=0`
  (no video recording) leaves the media directory uncreated.

Images are stored as `(128, 128, 3)` uint8 arrays in the `img` column of each
episode pickle, enabling EEF trajectory visualization and qualitative analysis.

### Per-run train_date prevents checkpoint conflicts

When sweeping n_demos, naive parallelism would cause all demo-count arms for the
same seed to write to the same `data/outputs/train/apr26_sweep/...` path. This is
prevented by passing `train_date=apr26_sweep_demos<N>` as a CLI override per arm,
which propagates automatically through all Phase 1 steps (train, eval, attribution)
because the experiment YAML does not set explicit sub-section dates.

### WandB service stagger

Launching 4 training jobs simultaneously causes WandB service socket conflicts
(WandB 0.13.3 binds a per-user service port on startup). Phase 1 background jobs
are staggered by 30 seconds each to allow each WandB service to bind before the
next process starts.

---

## Relationship to Prior Runs

| Run | Identifier | Description |
|-----|-----------|-------------|
| `mimicgen_square_pipeline_apr23` | `apr23` | Sign-of-life: 3 heuristics × budget=200, fixed baseline seed |
| `mimicgen_square_apr25_sweep_seed{0,1,2}` | `apr25_sweep` | Budget sweep: 3 heuristics × 11 budgets × 3 seeds, fixed n_demos=60 |
| `mimicgen_square_apr26_sweep_seed{0,1,2}_demos{60,100,300}` | `apr26_sweep` | **This run.** Adds n_demos dimension: 3 heuristics × 4 budgets × 3 seeds × 3 n_demos |

The apr26 budgets [20, 100, 500, 1000] were chosen based on apr25 results, which
showed smooth monotonic trends in the intermediate range (200–400). The coarser
grid captures the same story at 3× lower compute cost.
