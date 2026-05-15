# MimicGen pipeline — speedup & resource-utilisation plan

**Date:** 2026-04-29
**Worktree:** `worktree-feat+mimicgen-eef-pipeline` (commit `72535a7`)
**Reference experiment:** `mimicgen_square_sweep_apr26` — 3 heuristics × 4 budgets × 3 seeds × 3 demo counts = **108 arms**, each with up to 5 trained checkpoints.

---

## TL;DR — biggest wins, ordered by ROI

| # | Change | Where | Estimated saving |
|---|---|---|---|
| 1 | **Drop `train_attribution` + `finalize_attribution` from Phase 1 steps** — they compute TRAK, but apr26 sets `clustering_influence_source: infembed` and the result is unused | `mimicgen_square_sweep_apr26.yaml`, `run_mimicgen_budget_sweep.sh` | **20-30% of Phase 1 wall time** (no behavioural risk) |
| 2 | **Two-pass eval (race short → confirm long)** for `eval_mimicgen_combined` | `eval_mimicgen_combined.py` | **40-60% of eval wall time** (no result-quality loss if done right) |
| 3 | **Stage-aware scheduler** in `mimicgen_budget_sweep` — separate pools for generate (CPU), train (GPU-light), eval (GPU+RAM) | `mimicgen_budget_sweep.py` | **30-50% of Phase 2 wall time** |
| 4 | **Decouple training from eval**: train phase saves all top-k ckpts; a separate "eval queue" service walks them later with hard RAM concurrency limit | new module + reorder `_SUB_STEPS` | **Eliminates eval-saturation idle time on training GPUs** |
| 5 | **Parallelise per-seed MimicGen generation** (currently serial in the per-seed retry loop) | `generate_mimicgen_demos.py:597-648` | **2-4× speedup on `generate_demos`** when seeds > 1 |

Combined, these are realistic to take the apr26 sweep from "weeks of wall clock" to "days." Quick wins (1, 2, 5) are <1 day of code; the architectural ones (3, 4) are 1-3 days.

---

## 1. Where the time actually goes (rough budgets)

Per (seed, demo-count) combo, with 4 budgets × 3 heuristics = 12 arms:

| Stage | Per arm | Per combo (12 arms) | Per sweep (9 combos) | Notes |
|---|---|---|---|---|
| `train_baseline` (Phase 1) | — | 1 × ~1 h | 9 h serial | 1 baseline per combo (shared by all arms) |
| `eval_policies` (Phase 1) | — | 1 × ~10 min | 1.5 h | 1 ckpt × 100 eps |
| `train_attribution` + `finalize_attribution` (Phase 1) | — | 1 × ~1 h | 9 h | **Currently unused — see #1 below** |
| `compute_infembed` (Phase 1) | — | 1 × ~30-45 min | 5 h | Sequential; `arnoldi_dim=200`, `proj_dim=100` |
| `run_clustering` (Phase 1) | — | 1 × ~10-15 min | 2 h | Mostly CPU (UMAP) |
| `select_mimicgen_seed` | <1 min | 12 min | 2 h | h5py only, fast |
| `generate_mimicgen_demos` | 15-40 min | 3-8 h | 30-70 h | CPU-bound MuJoCo; **per-seed loop is serial** |
| `train_on_combined_data` | 1-2 h (260-1300 demos × 1751 epochs) | 12-24 h | 100-200 h | GPU-bound |
| `eval_mimicgen_combined` | **2-3 h** (5 ckpts × 500 eps) | 24-36 h | **200-300 h** | **dominant cost** |

**Total wall: ~400 GPU-hours.** Eval is ~half. Phase 1 attribution waste is ~20 GPU-hours that produce nothing used.

Numbers above are rough orders-of-magnitude; replace with measured values from your `result.json`s when planning real budgets.

---

## 2. Stage-by-stage analysis & recommendations

### 2.1 Phase 1 — upstream steps

Currently `run_mimicgen_budget_sweep.sh` runs:

```
[train_baseline, eval_policies, train_attribution, finalize_attribution, compute_infembed, run_clustering]
```

inside **one pipeline subprocess** per (seed, demos) combo, on **one GPU**. The 4-slot device pool only parallelises across combos, not within them.

#### W1.1  ✅ Drop TRAK steps when `clustering_influence_source: infembed`

`policy_doctor/curation_pipeline/steps/run_clustering.py:104-119` selects between `extract_trak_slice_windows` and `extract_infembed_slice_windows` based on `clustering_influence_source`. The apr26 yaml sets it to `infembed`, so TRAK output is never read. `train_attribution` + `finalize_attribution` together are likely **~1 hour per combo of pure waste**.

**Action:** edit `mimicgen_square_sweep_apr26.yaml:127-135` to:
```yaml
steps: [train_baseline, eval_policies, compute_infembed, run_clustering, mimicgen_budget_sweep]
```
and update `UPSTREAM_STEPS` in `run_mimicgen_budget_sweep.sh:68`.

(Keep TRAK in non-mimicgen experiments that actually use it.)

#### W1.2  Pipeline run_clustering off the GPU pool

`run_clustering` is mostly UMAP + KMeans on CPU. While it runs, the assigned `cuda:N` slot is idle but tied to the combo. Two options:

- (a) Split Phase 1 into two sub-phases inside the shell: 1A = GPU steps (`train_baseline, eval_policies, compute_infembed`) using the GPU pool; 1B = `run_clustering` using a separate CPU pool (more concurrent slots, no GPU).
- (b) Quicker hack: lower `clustering_umap_n_jobs` impact by *speeding up clustering itself* — set `clustering_umap_n_components: 50` (down from 100). UMAP scales O(n_components²); halving cuts time ~4×. Validate clustering quality stays acceptable on a single seed first.

#### W1.3  Increase `compute_infembed` batch size

`compute_infembed.py:63` defaults `infembed_batch_size=32`. The Arnoldi iteration is GPU-bound and easily scales to whatever fits in GPU memory. On a typical 24 GB GPU you can push to 128-256 with no behavioural change. Add `attribution.infembed_batch_size: 128` to the experiment yaml. Likely **2-3× faster infembed**.

(Verify by running once: timing & match the resulting `infembed_fit.pt` against a small-batch reference.)

#### W1.4  Phase 1 parallelism is "wide × shallow" — make it "wide × wide"

Currently 4 slots × {6 sequential steps each on one GPU}. The 6 steps don't all need a GPU at the same time. A finer-grained scheduler would: launch as many `train_baseline` as you have free GPUs; queue `compute_infembed` to start when its `train_baseline` is done; let `run_clustering` run on CPU while another combo's `compute_infembed` uses the GPU. This is workflow-engine territory; if you don't want that complexity, W1.1 + W1.2 alone capture most of the benefit.

---

### 2.2 Phase 2 — `mimicgen_budget_sweep` orchestration

`policy_doctor/curation_pipeline/steps/mimicgen_budget_sweep.py:116-130` uses `concurrent.futures.ThreadPoolExecutor` with `len(devices)` workers. Each worker runs **all four sub-steps serially** for one arm: `select_seed → generate_demos → train_combined → eval_combined`.

This is the wrong abstraction because the four sub-steps have **completely different resource profiles**:

| Sub-step | GPU mem | RAM | CPU | Wall |
|---|---|---|---|---|
| `select_mimicgen_seed` | 0 | low | low | <1 min |
| `generate_mimicgen_demos` | 0 (CPU MuJoCo) | medium | **high** (one per trial) | 15-40 min |
| `train_on_combined_data` | **high** | low (~5 GB) | low | 1-2 h |
| `eval_mimicgen_combined` | medium | **very high** (~10-20 GB per process for 28-env stack × 5 ckpts serial) | medium | 2-3 h |

With 4 thread-pool slots and 12 arms, you get blockages of all four kinds depending on phase alignment:
- All 4 in `generate` → CPU saturated, 0% GPU utilisation
- All 4 in `train` → competing for GPU memory
- All 4 in `eval` → **RAM blowout** (~80 GB)

#### W2.1 (BIG) Stage-aware queues — replace ThreadPoolExecutor

Refactor `MimicgenBudgetSweepStep.compute()` to: (a) build the list of (arm, sub-step) tasks, (b) maintain three independent worker pools sized by resource, (c) move each arm's task through pools as its predecessor completes.

```
generate_pool   = N_CPUS // 2          # MuJoCo trials are CPU-thread-hungry
train_pool      = len(GPUs) × 2        # diffusion training fits ≥2 per GPU at compile=true
eval_pool       = max(1, RAM_GB // 16) # ~16 GB per parallel eval process
```

Suggested implementation: `asyncio` with three `asyncio.Semaphore` per pool, or three `ThreadPoolExecutor`s and an arm-state machine. Either is ~150 lines. Big win on overall throughput because train and eval no longer block each other; CPU-only `generate_demos` runs alongside GPU-only training.

#### W2.2 (BIGGEST) Decouple train from eval — defer eval to a queue

The dominant cost is eval. Today's design makes every arm run `eval_mimicgen_combined` immediately after `train_on_combined_data`, gated by the same thread-pool slot. This means an arm's GPU slot is held during the entire eval (multiple hours) even though training is done.

**Better design:**
1. `TrainOnCombinedDataStep` writes all top-k checkpoints and a sentinel.
2. A new step `QueueForEvalStep` (or just a list dropped into a JSON queue file) records (arm_name, ckpt_path) tuples.
3. A separate process / final pipeline step `RunEvalQueueStep` walks the queue with **dedicated eval concurrency** — typically 1-2 processes per machine due to RAM. It's the only step constrained by RAM.

Benefits:
- Training pool saturates GPUs; eval pool runs at RAM-safe concurrency.
- A crashed eval doesn't block downstream train arms.
- You can re-run eval (e.g. after fixing checkpoint loading) without re-running training.

#### W2.3 Save MORE checkpoints during training, not fewer

The user's intuition is right. With training cheap relative to eval, save many checkpoints (every 50 epochs → ~35 over 1751 epochs) and let eval be the selection mechanism on the de-coupled queue. Today `checkpoint_topk=5` already saves a few; you could keep 10 and pay the disk cost (each ~50 MB for diffusion-unet-lowdim → ~500 MB per arm × 108 = 54 GB; fine).

Caveat: cupid currently picks `topk` by *val_loss* during training. For mimicgen, **val_loss is not the right metric** — success rate is. Keeping more checkpoints lets you re-rank by success rate post-hoc.

---

### 2.3 Eval (the dominant cost)

`policy_doctor/curation_pipeline/steps/eval_mimicgen_combined.py:108-176` evaluates each top-k checkpoint by spawning a *separate* `eval_save_episodes.py` subprocess. Each subprocess: conda init (~3 s) + Hydra init (~5 s) + MuJoCo env stack (~30-60 s) + 500 episodes × 28 parallel envs (~30-60 min).

#### W3.1 (BIG) Two-pass / racing eval

Allocate the eval budget non-uniformly. Per arm with 5 checkpoints:

```
Pass 1: 5 ckpts × 50 episodes  →  rank by mean
Pass 2: top-2 ckpts × 200 episodes → confirm
```

Total = 5×50 + 2×200 = **650 episodes** vs current **2500 episodes** (~75% saving), with negligible loss in the chosen-best estimate provided the Pass 1 pool is well-calibrated. Standard successive halving result.

For the **headline number** (mean across checkpoints, per-arm reporting), do a single 200-episode pass on every checkpoint after the race. Total ≈ 1650 vs 2500 (~34% saving) and the same statistical power on per-checkpoint estimates.

Implementation: extend `EvalMimicgenCombinedStep` with an `eval_strategy: "uniform" | "racing"` config knob. The racing branch calls `_call_eval_subprocess` twice with different `--num_episodes`.

#### W3.2 (BIG) Multi-checkpoint eval in one subprocess

Each `eval_save_episodes.py` invocation spends ~40 s on init *before* any episode runs. With 5 ckpts → 200 s of pure startup × 108 arms = ~6 hours just paid in spawn overhead.

Patch `cupid/eval_save_episodes.py` to accept a list of checkpoints (`--train_ckpts=ckpt_a,ckpt_b,...`) and reuse the env stack. The runner would:
1. Build `AsyncVectorEnv` once.
2. For each ckpt: load weights into the existing policy module, run rollouts.
3. Write per-ckpt `eval_log.json` to its own subdir.

Caller side: `eval_mimicgen_combined` issues *one* subprocess per arm, not five.

Saves ~3-4 minutes per arm. ~5-7 hours over the sweep. Cheap and high-value.

#### W3.3 Trim `n_envs` to fit RAM, not maximize parallelism

`square_mimicgen_lowdim.yaml:45` sets `n_envs=28`, with `n_train=6, n_test=50`. With 28 parallel MuJoCo envs (~300-500 MB each), one eval process is ~10-15 GB. If the new W2.2 eval queue runs 2 in parallel that's 30 GB — tight on a 64 GB box.

If you stay with the current model, consider lowering `n_envs` to 16 (saves ~5 GB per process, lets you fit 3 evals concurrently). The wall-clock cost is sub-linear because GPU inference is already the bottleneck inside one process.

#### W3.4 Pin `test_start_seed` per-arm to allow honest cross-arm variance

The current code uses one `test_start_seed=100000` for all arms. Eval episodes are seeded `[100000, 100001, …, 100499]`. This is *common across arms*, which is fine for variance reduction (paired test) but means cross-arm differences aren't IID. **Document this** in your stats writeup; otherwise OK.

#### W3.5 `evaluation.overwrite: false` (apr26 yaml line 64) + cleanup logic at line 150-156

Cached eval is skipped when `output_dir/eval_log.json` exists. Good for resumability. But: if the eval crashed mid-write, the partial dir is **rm -rf**'d (line 156). Confirmed behaviour, but be aware: if eval crashes due to a transient GPU issue, you lose all in-progress episodes.

Safer: write each episode result incrementally to a per-episode jsonl, then assemble `eval_log.json` at end. Only delete failed episode files. This makes eval *resumable mid-process* — huge if a long eval run gets killed.

---

### 2.4 MimicGen generation (`generate_mimicgen_demos.py`)

#### W4.1 (BIG) Parallelise the per-seed retry loop

`generate_mimicgen_demos.py:597-648`: when `success_budget` is set and `num_seeds > 1`, the per-seed mode iterates seeds **sequentially** in the inner loop, even though each seed's MimicGen subprocess is independent and CPU-bound (no GPU contention).

```python
for seed_i, seed_i_dir, seed_i_hdf5 in seed_hdf5s:
    ...
    res = subprocess.run(_build_cmd(seed_i_hdf5, pass_out_dir, trials_per_seed))
```

Replace with a `ThreadPoolExecutor(max_workers=N_PARALLEL_GEN)` where `N_PARALLEL_GEN = min(n_seeds_in_hdf5, n_cpus // mujoco_threads_per_trial)`. Each subprocess is a separate process so GIL is moot. With 10 seeds and 8 cores, expect ~3-4× speedup on `generate_demos`.

Caveat: the adaptive retry loop measures success rate against *total* trials. Parallel execution means stats get reported per-pass after all seeds finish — fine, but the inner `effective_rate` computation needs to use post-pass aggregated rate, not running rate.

#### W4.2 Drop `failed_eef_xyz` extraction when EEF analysis isn't used

Lines 711-713 read failed-rollout EEF trajectories. For the budget sweep, these aren't analysed. Skipping saves a bit of h5py read time per arm. Marginal.

#### W4.3 Early termination when success_rate is far below floor

The loop runs up to `max_total_trials = success_budget × 20`. If observed rate after pass 1 is, say, 5% (vs 40% expected), `effective_rate` adjusts but the wall-clock impact is severe. Add a heuristic early-fail: if after `pass_num=2` the success rate is below `min_acceptable_rate` (e.g. 10%), raise instead of grinding for an hour. Better to fail loud than silently produce a degenerate dataset.

---

### 2.5 Training (`train_on_combined_data.py`)

Already has `tf32: true, compile: true` enabled in `square_mh_mimicgen.yaml`. Good.

#### W5.1 Use `num_steps` not `num_epochs` across demo-count arms

`mimicgen_square_sweep_apr26.yaml`'s `num_epochs: 1751` (via baseline yaml) means a 60-demo arm gets ~1751 × ~10 batches = 17K steps; a 300-demo arm gets ~87K steps. **Five times more gradient updates per arm** confounds the demo-count comparison. Set `baseline.num_steps=20000` (or similar) in apr26 yaml so all demo-count arms see the same optimisation budget.

This is **also a correctness issue** (called out in the audit), but it's mentioned here because it reduces *unnecessary* training time on high-demo arms.

#### W5.2 num_workers for dataloader

`train_robomimic_lowdim_workspace.yaml` has `num_workers: 1`. For the small lowdim dataset this is fine; data loading is not the bottleneck. **No action.**

#### W5.3 Multi-GPU per training is already supported but rarely used

`train_on_combined_data.py:211-214` supports `torchrun --nproc_per_node=N`. For the small lowdim model, single GPU + compile is already efficient. **No action**, but worth knowing for image-obs experiments.

#### W5.4 Save fewer intermediate checkpoints during training

`baseline.checkpoint_every: 50` writes a checkpoint every 50 epochs → ~35 writes over 1751 epochs. Each is ~50 MB. Disk write isn't the bottleneck, but the val-loss eval inside the training loop *is* expensive at every checkpoint (rolls a few episodes). If `n_train=6, n_test=22` per checkpoint, that's 35×28 = ~1000 in-training rollouts. Increasing `checkpoint_every` to 100 halves that — small but free.

---

### 2.6 Subprocess startup overhead

Every `conda run -n <env>` invocation pays ~1-2 s. Every Hydra startup pays ~3-5 s. Every MuJoCo/robosuite import pays ~10-30 s.

In Phase 2, per arm:
- 1× select (1 conda call)
- N× generate (1 conda call per per-seed-pass; could be 10-30 calls)
- 1× train (1 conda call)
- 5× eval (5 conda calls)

≈ 20-40 conda starts per arm × 108 arms = 2000-4000 starts × ~30 s with full env init = **15-30 hours of pure startup cost**.

#### W6.1 Persistent eval worker

W3.2 (multi-checkpoint eval per subprocess) covers most of this. Going further: a long-running eval daemon reading from a queue, never restarting MuJoCo. Saves another ~20 s per ckpt eval. Higher engineering effort; lower leverage now.

#### W6.2 Skip `conda run` when in-process is safe

Several pipeline steps (`select_mimicgen_seed`, `combine_datasets`, etc.) run pure Python in the orchestrator's env. Verify no spurious `conda run` indirections. (Already mostly done — `select_mimicgen_seed` runs in-process. Audit to confirm.)

---

### 2.7 Disk / HDF5 I/O

#### W7.1 Symlink-share original HDF5 across arms

`combine_datasets.py:57` does `shutil.copy2(original_path, output_path)` — full byte copy of the original (60 demos ~50 MB; 1000 demos ~1 GB). Per arm. With 12 arms × 9 combos = 108 copies. For 1000-demo case, **108 GB of needless I/O**.

Better: open the output file in write mode and *copy demos one by one* from the original (same pattern as `_copy_group`). Or use `h5py`'s external links to point at the original. Saves disk + I/O wall time.

#### W7.2 Compress generated HDF5

The `_copy_group_standalone` helper preserves chunks/compression but doesn't add compression if absent. Adding `compression="gzip", compression_opts=4` to writes shrinks files 3-5× with ~5 % CPU overhead. Useful when scaling out to many arms.

---

## 3. Proposed orchestration redesign (sketch)

If you have time for one architectural change, this is it. Pseudocode:

```python
# Replaces both run_mimicgen_budget_sweep.sh phases AND
# MimicgenBudgetSweepStep's ThreadPoolExecutor.

class StagedScheduler:
    def __init__(self, devices, ram_gb, n_cpus):
        self.gen_pool   = ThreadPoolExecutor(max_workers=n_cpus // 4)   # MuJoCo
        self.train_pool = ThreadPoolExecutor(max_workers=len(devices))  # one per GPU
        self.eval_pool  = ThreadPoolExecutor(max_workers=max(1, ram_gb // 16))
        self.device_q   = queue.Queue(); [self.device_q.put(d) for d in devices]

    async def run_arm(self, arm):
        await loop.run_in_executor(self.gen_pool,   arm.select)
        await loop.run_in_executor(self.gen_pool,   arm.generate)
        device = self.device_q.get()
        try:
            await loop.run_in_executor(self.train_pool, lambda: arm.train(device))
        finally:
            self.device_q.put(device)
        await loop.run_in_executor(self.eval_pool, arm.eval)  # eval on its own GPU pool
```

Key benefits:
- `generate` (CPU-bound) doesn't hold a GPU slot.
- `train` and `eval` use independent GPU pools so eval doesn't block training.
- `eval` concurrency is bounded by RAM, not GPUs.

Tied to W2.1 + W2.2 above.

---

## 4. Suggested phased rollout

### Quick wins (≤1 day, no architecture change)

| Action | File(s) |
|---|---|
| Drop `train_attribution`, `finalize_attribution` from steps list (W1.1) | apr26 yaml + shell launcher |
| Bump `attribution.infembed_batch_size: 128` (W1.3) | apr26 yaml |
| Bump `clustering_umap_n_components: 50` and verify (W1.2 partial) | apr26 yaml |
| Two-pass eval (W3.1) | `eval_mimicgen_combined.py` |
| Multi-ckpt eval in one subprocess (W3.2) | `eval_save_episodes.py` + `eval_mimicgen_combined.py` |
| Parallelise per-seed generation (W4.1) | `generate_mimicgen_demos.py:597-648` |
| Reduce `n_envs` for eval to 16 (W3.3) | `square_mimicgen_lowdim.yaml:45` |

Combined: ~50 % wall-clock reduction.

### Medium (1-3 days)

| Action | File(s) |
|---|---|
| Stage-aware scheduler in `mimicgen_budget_sweep` (W2.1) | `mimicgen_budget_sweep.py` |
| HDF5 symlink-share or h5py-external links (W7.1) | `combine_datasets.py` |
| Per-episode incremental eval log (W3.5) | `cupid/eval_save_episodes.py` |
| `num_steps` instead of `num_epochs` (W5.1) | apr26 yaml |
| Early-fail in MimicGen retry (W4.3) | `generate_mimicgen_demos.py` |

Combined: another ~25 % on top of quick wins, plus much better mid-run resilience.

### Big (≥3 days)

| Action | File(s) |
|---|---|
| Decoupled eval queue (W2.2 + W2.3) | new `eval_queue.py` step + queue-runner CLI |
| Persistent eval daemon (W6.1) | new module |
| Replace shell launchers with a workflow engine (W1.4) | drop `run_*.sh`, add Snakemake/Prefect |

Each is high-effort; do the quick + medium first and only escalate if needed.

---

## 5. Things I would *not* spend time optimising

- **Dataloader `num_workers`**: lowdim dataset is too small for it to matter.
- **TF32 / `torch.compile`**: already on for training. Adding to InfEmbed is explicitly broken (`compute_infembed.py:91-94`); leave it off.
- **GPU memory tuning**: the diffusion-unet-lowdim training easily fits >1 per GPU. The bottleneck is training time, not memory.
- **Replacing MuJoCo with Genesis or another engine**: out of scope; engineering cost dwarfs the speedup.

---

## 6. Profiling / measurement to do before any of this

Before chasing the % numbers above, measure on your hardware:

1. Per-stage wall time on **one arm** end-to-end. Save to a `.csv` keyed on `(stage, arm, t_start, t_end)`.
2. RAM high-water mark during a single concurrent `eval_mimicgen_combined`. Use `psutil` from a sidecar.
3. GPU utilisation during Phase 1's `run_clustering`. (Likely 0 %.) Confirms W1.2.
4. Per-pass success-rate variability across seeds in `generate_demos`. If >40 % vs expected 40 %, the retry loop is fine; if <20 %, W4.3 (early-fail) becomes mandatory.

These four numbers will turn this plan from "informed guess" into "decided strategy" in <1 day of measurement.

---

## 7. Cross-references with audit findings

Several speedups overlap with correctness fixes from `mimicgen_scale_up_audit.md`. Doing these in the right order matters:

- **B2** (max_train_episodes ignored) + **W5.1** (use num_steps): fix together. Don't tune training time without first fixing the demo-count semantics.
- **B1** (heuristic determinism) + **W3.1** (two-pass eval): re-running base arms is fast on the new eval; do correctness fix first, then re-run, then optimise eval.
- **H1** (silent shortfall) + **W4.3** (early-fail): the same site, same loop. Fix together.
- **H7** (no checkpoint validation in `is_done`) + **W2.3** (decoupled eval queue): the queue dispatcher should refuse arms whose checkpoints aren't actually on disk.

Treat the audit as a precondition: don't speed up an experiment whose results you can't trust.
