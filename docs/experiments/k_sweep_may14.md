# K Robustness Sweep — May 14 2026

**Goal:** Show that `behavior_graph` (few-modes) seed selection is robust to K-means K.  
**Design:** D60 baseline, tight nut constraint (±40 mm x/y, ±30° z_rot), budget=300, n=3 reps, K∈{5,10,15,20,25}.  
UMAP fit once (seed=1, 12978 windows × 100d); K-means re-fit per K.  

## Setup

| Key | Value |
|-----|-------|
| Script | `scripts/run_k_sweep_tight.sh` |
| Experiment config | `mimicgen_square_rep_sweep_apr26_d60_budget300_nut_constrained_tight` |
| Baseline data | `~/data/cupid_data/outputs/eval_save_episodes/apr26_sweep_demos60/` |
| Run dirs | `third_party/cupid/data/pipeline_runs/mimicgen_square_apr26_seed1_d60_budget300_nut_constrained_tight_k{K}/` |
| Heuristics | random, behavior_graph, diversity |
| Reps | Phase A (rep-1, random_seed=null) + Phase B (rep-2/3, random_seed=1/2) |
| Eval envs | 5 per run (n_envs=5) |
| GPU slots | 3 × cuda:0 |
| Dataloader | num_workers=4, persistent_workers=True |

## Status

| K | Phase A (rep-1) | Phase B (rep-2) | Phase B (rep-3) |
|---|-----------------|-----------------|-----------------|
| 5  | 🏋 training (BG+manual), gen active (rand/div) | 🔄 generating | ⏳ pending |
| 10 | ⏳ pending | ⏳ pending | ⏳ pending |
| 15 | ⏳ pending | ⏳ pending | ⏳ pending |
| 20 | ⏳ pending | ⏳ pending | ⏳ pending |
| 25 | ⏳ pending | ⏳ pending | ⏳ pending |

Legend: ⏳ pending · 🔄 generating · 🏋 training · 📊 eval · ✅ done · ❌ failed

## Results

*(filled in as arms complete)*

| K | heuristic | best (n=3) | top5_mean (n=3) |
|---|-----------|------------|-----------------|
| 5  | random | — | — |
| 5  | behavior_graph | — | — |
| 5  | diversity | — | — |
| 10 | random | — | — |
| 10 | behavior_graph | — | — |
| 10 | diversity | — | — |
| 15 | random | — | — |
| 15 | behavior_graph | — | — |
| 15 | diversity | — | — |
| 20 | random | — | — |
| 20 | behavior_graph | — | — |
| 20 | diversity | — | — |
| 25 | random | — | — |
| 25 | behavior_graph | — | — |
| 25 | diversity | — | — |

## Run log

- **2026-05-14 22:26**: `run_k_sweep_tight.sh` launched (PID 14213).
- **2026-05-14 22:26**: Step 1 complete — UMAP fit (12978×100 → 100d, n_jobs=-1, ~1 min). K-means run for K=5,10,15,20,25 using seed=1 infembed embeddings. All 5 clustering dirs written.
- **2026-05-14 22:29**: K=5 Phase A generation started. 3 arms (random, behavior_graph, diversity) each running 3 `run_mimicgen_generate.py` workers in `mimicgen_torch2`. Tight constraint: nut ±40mm x/y, ±30° z_rot. 900 trials per arm (10 seeds × 90 trials/pass).
- **2026-05-14 22:38**: Launched Phase B for K=5 in parallel with Phase A (generation is CPU-only, independent). Also launched Phase A+B for K=10,15,20,25 all in parallel — 80 concurrent generation workers total.
- **2026-05-15 01:10**: **CRASH** — K=5 behavior_graph Phase A training failed immediately with `IndentationError: expected an indented block` at `mimicgen_lowdim_runner.py:332`. Root cause: merge conflict resolution left an empty `if self.save_episodes:` block with no body. Fix: deleted the empty `if` block. All other arms continued generating. behavior_graph training relaunched manually (PID 39083, wandb run `qbt3ap1t`).
- **2026-05-15 01:17**: **GPU training started** — K=5 behavior_graph Phase A epoch 0 running on cuda:0. 236 MiB GPU memory. ~2.5h expected to completion (1751 epochs).
- **2026-05-15 01:18**: **CRASH 2** — EMA shape mismatch: `RuntimeError: tensor a (128) must match tensor b (5) at non-singleton dimension 2` in `ema_model.py`. Root cause: `torch.compile(policy.model)` replaces the UNet sub-module with `OptimizedModule`, which adds an extra level to `modules()` traversal. `EMAModel.step()` uses `zip(model.modules(), ema_model.modules())` which then misaligns module pairs → shape mismatch on the very first EMA update. Fix: `compile: true → false` in `square_mh_mimicgen.yaml`. Also fixed `dummy_env_fn=self.dummy_env_fn` (attribute never set in `__init__`) in runner.
- **2026-05-15 01:18**: All live pipeline orchestration processes already loaded `compile: true` from config at startup. Strategy: let them fail gracefully at training → write composite `done` → restart with new config (skips gen, runs training with compile=false). behavior_graph training relaunched manually (compile=false, PID 45716).
- **2026-05-15 03:06**: Pulled `ema_safe_model()` from main, applied to lowdim workspace. Re-enabled `compile: true`. Tested OK.
- **2026-05-15 03:13**: Relaunched 5 training jobs (k5_a/rep1/rep2, k10_a/rep2). Machine OOMed shortly after — k5_rep2 and k10_rep2 died with `BrokenPipeError`, k5_a was at epoch 53 with checkpoint saved. Cause: 5 training jobs × MuJoCo eval subprocesses spiked RAM/CPU simultaneously.
- **2026-05-15 03:26**: Restarted with `run_k_sweep_tight.sh` (sequential K values, 3 device slots). Set `num_workers=4, persistent_workers=True`. Pipeline running: clustering skipped, K=5 Phase A in progress with 3 concurrent training arms. GPU 97%, eval at 3.4 it/s (fast with compile=true).
- **2026-05-15 03:27**: **K=5 Phase A behavior_graph FAILED** — training crashed at startup (exit code 1, no wandb run created). Root cause: training output dir contaminated by k10 training run from 03:13 parallel launch; `latest.ckpt` pointed to k10's epoch-50 checkpoint (score 0.040), and simultaneous 3-arm startup OOMed before training began. Fixed: deleted k10 checkpoint, `latest.ckpt` now → k5 epoch-50 (score 0.060). Needs manual retry after Phase A composite step completes.
- **2026-05-15 04:18**: K=5 Phase A random and diversity training healthy — epochs 200-206 at ~25 it/s with compile=true. Three arms training concurrently. Random generation proceeding (25/50 trial successes at 50% rate). Pipeline sequential, behavior_graph Phase A FAILED but random/diversity will complete Phase A and pipeline will proceed to Phase B.

## Bugs fixed during launch

All found and fixed 2026-05-14 during initial launch attempts:

| File | Bug | Fix |
|------|-----|-----|
| `clustering_results.py:145` | `Path \| None` type hint breaks Python 3.9 | → `Optional[Path]` |
| `select_mimicgen_seed.py` | `OmegaConf.select(plain_dict, ...)` crashes when `failure_analysis` key absent (fallback `or {}` returns plain Python dict) | → `OmegaConf.create({})` as empty fallback |
| `run_k_sweep_tight.sh` | `+experiment=` caused "Multiple values for experiment" | → `experiment=` |
| `run_k_sweep_tight.sh` | `clustering_n_clusters_sweep` / `clustering_run_dir` not in struct | → `+` prefix |
| `run_k_sweep_tight.sh` | `evaluation.eval_output_dir` can't be overridden directly | → `~evaluation.eval_output_dir` delete + `+evaluation.eval_output_dir=` re-add |
| `run_k_sweep_tight.sh` | `run_name=` ignored when config already sets `run_dir` | → `run_dir=data/pipeline_runs/${RUN_NAME}` |
| `mimicgen_lowdim_runner.py:332` | Empty `if self.save_episodes:` block (no body) after merge resolution → `IndentationError` on import in `mimicgen_torch2`, crashes all training | → deleted the empty `if` block |
| `square_mh_mimicgen.yaml` | `compile: true` → `torch.compile(policy.model)` makes `OptimizedModule` appear as extra module in `modules()`, misaligning EMA zip → shape mismatch crash on first EMA update | → `compile: false` (workaround); **properly fixed** by pulling `ema_safe_model()` from main and applying to lowdim workspace |
| `mimicgen_lowdim_runner.py:307` | `dummy_env_fn=self.dummy_env_fn` passed to `AsyncVectorEnv` but attribute never set in `__init__` (merge artifact from HEAD branch) → `AttributeError` on env creation | → removed `dummy_env_fn` arg |

## Notes

- `robosuite_task_zoo` import warning on every generation worker is non-fatal (module not installed in `mimicgen_torch2`; not needed for Square task).
- `"keep_failed": true` in log is MimicGen JSON config output, not an error.
- Generation success rate starts at 0% for initial trials; accumulates over 900 trials per arm. Low early success rate is normal for tight constraint.
- Step 2 runs K values sequentially (K=5 → K=10 → … → K=25). Each K runs Phase A then Phase B before moving to next K. Total arms: 5 K × 3 heuristics × 3 reps = 45 arms.
