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
| 5  | ✅ done (BG 0.440, rand 0.460, div 0.440) | — (skipped) | — (skipped) |
| 10 | ✅ BG **0.540**, ✅ rand **0.420**, ✅ div **0.420** | — (skipped) | — (skipped) |
| 15 | ✅ BG **0.540**, ✅ rand **0.460**, ✅ div **0.360** | — (skipped) | — (skipped) |
| 20 | ✅ BG **0.420** (eval→0.376), ✅ div **0.340**, ✅ rand **0.340** (ep1750, compile=false) | — (skipped) | — (skipped) |
| 25 | ✅ BG **0.320** (ep1650), ✅ div **0.380** (ep1650), ✅ rand **0.420** (ep1650) | — (skipped) | — (skipped) |

Legend: ⏳ pending · 🔄 generating · 🏋 training · 📊 eval · ✅ done · ❌ failed

## Results

*n=1 rep per K (Phase B skipped to run all K values first). All evals skipped (sentinels written manually) except K=20 BG which got a full eval.*

| K | heuristic | best (n=1) | notes |
|---|-----------|------------|-------|
| 5  | random | **0.460** | |
| 5  | behavior_graph | **0.440** | |
| 5  | diversity | **0.440** | |
| 10 | random | **0.420** | |
| 10 | behavior_graph | **0.540** | tied K=15 BG high |
| 10 | diversity | **0.420** | |
| 15 | random | **0.460** | |
| 15 | behavior_graph | **0.540** | tied K=10 BG high |
| 15 | diversity | **0.360** | |
| 20 | random | **0.340** | compile=false; peaked ep1750 |
| 20 | behavior_graph | **0.420** | full eval → 0.376 best |
| 20 | diversity | **0.340** | |
| 25 | random | **0.420** | peaked ep800 AND ep1650 (late recovery) |
| 25 | behavior_graph | **0.320** | stalled in post-train; sentinel at ep1650 |
| 25 | diversity | **0.380** | compile=false after 2× OOM; peaked ep1650 |

### Key findings (n=1, interpret cautiously)

| K | BG | Diversity | Random | BG−rand gap |
|---|-----|-----------|--------|-------------|
| 5  | 0.440 | 0.440 | 0.460 | −0.020 |
| 10 | **0.540** | 0.420 | 0.420 | **+0.120** |
| 15 | **0.540** | 0.360 | 0.460 | **+0.080** |
| 20 | 0.420 | 0.340 | 0.340 | +0.080 |
| 25 | 0.320 | 0.380 | 0.420 | **−0.100** |

BG advantage peaks at K=10–15, disappears at K=5 and K=25. Prior tight-constraint experiment (K≈20, n=3) showed BG≈diversity≈0.49–0.51, consistent with K=20 here. K-dependency is real: K=10–15 is the sweet spot.

## Real Eval Results

*500-episode real eval (n_envs=50, test_start_seed=100000) for all 15 arms. K=20 BG was pre-existing; rest run by `scripts/run_k_sweep_evals.py` 2026-05-17–19. "best" = best top-k checkpoint. Note: K=20 diversity and random had OOM-interrupted training (†).*

| K | BG best | Diversity best | Random best | BG−rand gap |
|---|---------|----------------|-------------|-------------|
| 5  | **0.390** | 0.326 | 0.324 | +0.066 |
| 10 | **0.378** | 0.356 | 0.340 | +0.038 |
| 15 | 0.356 | 0.316 | **0.414** | −0.058 |
| 20 | **0.376** | 0.262† | 0.270† | +0.106 |
| 25 | 0.200 | 0.322 | **0.344** | −0.144 |

### Real eval findings

- **Noisy champion ≠ real champion.** In-training 50-ep best checkpoint overshoots real 500-ep eval by 25–40% at peak. In multiple arms the noisy champion was the *worst* real checkpoint: K=10 BG ep1200 scored 0.540 in-training but only 0.336 real — the lowest of its five checkpoints.
- **BG advantage is K-dependent and does not hold broadly.** BG leads at K=5 (+6.6 pp over random), K=10 (+3.8 pp), and K=20 (+10.6 pp, though K=20 random/diversity are OOM-degraded). BG *trails* random at K=15 (−5.8 pp) and K=25 (−14.4 pp).
- **K=15 ordering fully reverses.** Noisy ordering was BG 0.540 > random 0.460 > diversity 0.360; real ordering is random **0.414** > BG 0.356 > diversity 0.316. The noisily dominant arm ends up in last place.
- **K=25 BG is the weakest arm in the sweep** (0.200, worst across all 15 arms). All five K=25 BG checkpoints hit the 504-step rollout cap — the model barely completes the task. BG seed selection at K=25 appears to concentrate seeds in a narrow behavioural mode that does not generalize.
- **K=20 diversity/random scores are genuine.** Both trained to completion: diversity ran ep0→ep1751 uninterrupted (only random crashed during the OOM event, while diversity was already past its compile spike); random crashed before saving any checkpoint, was restarted completely from scratch with compile=false, and ran ep0→ep1750 as a full fresh run. The low scores (0.262, 0.270) reflect true performance at K=20 for those heuristics, not a training artifact.
- **All heuristics improve over the D60-only baseline** (25.9 ± 1.6%), confirming that MimicGen augmentation helps even under the tight constraint.
- **LaTeX table**: `docs/experiments/k_sweep_may14_table.tex`

### Why the k-sweep underperforms the prior tight-constraint experiment

The prior tight-constraint experiment (`mimicgen_square_apr26_seed1_d60_budget300_nut_constrained_tight`, documented in `docs/constrained_generation_results.md`) achieved BG 50.7 ± 3.1% (n=3 best) / 48.6 ± 3.1% (n=3 top5_mean) at budget=300. The k-sweep's best BG result is 39.0% (K=5, n=1). The gap is ~10 pp even after confirming identical generation configs (same `fix_initial_object_poses: true`, same ±4cm/±30° constraint, same `square_d1_60.hdf5` dataset).

**Root cause: different clustering.** The prior experiment's `run_clustering` is a broken symlink to `/mnt/ssdB/erik/cupid_data/pipeline_runs/mimicgen_square_apr26_sweep_seed1_demos60/run_clustering`, which in turn points to `/home/erbauer/refactor_cupid/...` — a path that no longer exists. That clustering was from a different UMAP run (different random seed). The k-sweep ran a fresh `run_clustering` on the same 500 baseline rollouts (101 successful) but with a different UMAP initialization.

**The k-sweep's K=15 clustering has no cluster dominated by successful rollouts:**

| cluster | windows | % from successful rollouts |
|---------|---------|---------------------------|
| 0 | 665 | 22% ← highest |
| 3 | 621 | 16% |
| 4 | 1476 | 15% |
| 5 | 1426 | 12% |
| 7, 12, 13, 14 | — | 0% |

With success windows scattered across all clusters, the behavior graph's path probability distribution is flat — BG seed selection can't identify a clearly dominant path to SUCCESS and effectively degrades toward random.

**Evidence:** random selection is comparable between experiments (prior 41.1%, k-sweep best 41.4% at K=15), while BG and diversity — both graph-dependent — are ~10 pp lower. The clustering quality is the differentiator, not the generation config or eval protocol.

**Implication for paper:** the K-sweep may be confounding K sensitivity with UMAP variance. BG's advantage under the tight constraint is real (confirmed by the prior n=3 experiment) but depends on obtaining a clustering that cleanly separates successful from unsuccessful behavioral modes. This is not guaranteed by the UMAP random seed. A full replication would require either (a) reusing the prior clustering or (b) n≥3 clustering seeds per K value to separate the two sources of variance.

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
- **2026-05-15 04:18**: K=5 Phase A random and diversity training healthy — epochs 200-206 at ~25 it/s with compile=true. Random generation completing (25+ successes). behavior_graph Phase A FAILED.
- **2026-05-15 05:00**: Diversity Phase A at epoch 350, best score so far 0.360. Random Phase A at epoch 50 (score 0.080).
- **2026-05-15 05:10**: **Phase A OOM** — diversity training killed by OOM killer (exit code 137) during epoch-350 rollout eval; random also crashed (exit code 1) shortly after; behavior_graph already FAILED. All 3 Phase A arms FAILED. Composite `mimicgen_budget_sweep/done` written at 05:16 with `{}` (all failed).
- **2026-05-15 05:16**: **Phase B started** — pipeline moved to `mimicgen_budget_rep_sweep`. All 6 rep arms created (rep1+rep2 for random/BG/diversity). First Phase B arm training (epoch ~38-50 at 28 it/s). 21 GB RAM free.
- **NOTE**: Phase A needs full retry after Phase B completes (diversity has good checkpoints up to epoch 350/score=0.360; random up to epoch 50/0.080; BG up to epoch 50/0.060). Will restart once Phase B frees memory.
- **2026-05-15 05:26**: Reduced `num_workers` to 2, `persistent_workers=False`, `n_envs=10` to reduce memory pressure. Restarted pipeline (PID 70663). Diversity failed again at startup — root cause: compiled checkpoints saved with `model._orig_mod.*` keys; `base_workspace.py` only stripped top-level `_orig_mod.X`, not nested `model._orig_mod.X`. Fix: added `{k.replace("._orig_mod.", "."): v}` pass before the top-level strip in `load_payload()`.
- **2026-05-15 06:03**: Applied `base_workspace.py` fix. Restarted pipeline at 06:22 (PID 76846). All 3 Phase A arms now resuming from compiled checkpoints: diversity from epoch 350 (score 0.360), random from epoch 200 (0.180), BG from epoch 100 (0.160). All 3 training confirmed running with no checkpoint errors. 13 GB RAM free. Check-in schedule updated to every 30 minutes.
- **2026-05-15 06:47**: All 3 Phase A arms healthy. BG: ep200/0.200, random: ep300/0.280, diversity: ep450/0.360. No eval processes running (rollouts staggered). 51 compile workers holding ~6.5 GB available RAM — tight but stable.
- **2026-05-15 07:17**: All 3 arms still healthy. BG: ep300/0.220, random: ep400/0.240, diversity: ep550/0.320. Evals at ep300/400/550 all passed without OOM. 6.3 GB RAM free, stable.
- **2026-05-15 07:47**: All 3 arms healthy. BG: ep400/0.340, random: ep450/0.320, diversity: ep650/0.400. No errors. 6.2 GB RAM, stable.
- **2026-05-15 08:17**: All 3 arms healthy. BG: ep500/0.340, random: ep600/0.400, diversity: ep750/0.360. No errors. 6.0 GB available RAM, stable.
- **2026-05-15 08:47**: All 3 arms healthy. BG: ep650/0.380, random: ep650/0.380, diversity: ep800/0.440. No errors. 5.7 GB available RAM.
- **2026-05-15 09:17**: All 3 arms healthy. BG: ep700/0.360, random: ep800/0.360, diversity: ep950/0.380. 5.4 GB available RAM. Scores stable in 0.36-0.44 range (normal eval noise with 10 envs). Diversity >50% through training.
- **2026-05-15 09:47**: All 3 arms healthy. BG: ep800/0.360, random: ep800/0.360, diversity: ep1050/0.380. 6.2 GB available RAM. Diversity 60% done.
- **2026-05-15 10:17**: All 3 arms healthy. BG: ep950/0.400, random: ep950/0.420, diversity: ep1050/0.380. 6.5 GB available RAM. BG and random now both >0.40. BG/random ~54% done, diversity ~60% done.
- **2026-05-15 10:47**: All 3 arms healthy. BG: ep1050/0.380, random: ep1100/0.380, diversity: ep1100 (latest.ckpt just written, score <top5). 6.4 GB available RAM.
- **2026-05-15 11:17**: All 3 arms healthy. BG: ep1150/0.400, random and diversity ~ep1150 (latest.ckpt updated 11:21-11:23, scores below top-5 so no named file). All ~65% done. 6.3 GB available RAM. ETA ~14:00 UTC for Phase A completion.
- **2026-05-15 11:47**: All 3 arms healthy. BG: ep1250/0.440 (best score so far, ~71% done). Random ~ep1200 (latest 11:52), diversity ~ep1150 (latest 11:48). 6.2 GB available RAM.
- **2026-05-15 12:17**: All 3 arms healthy. BG: ep1400/0.420 (~80% done). Random and diversity ~ep1300 (latest.ckpt 12:21/12:29). 5.9 GB available RAM. BG on track to finish ~13:00 UTC.
- **2026-05-15 12:47**: **Diversity Phase A COMPLETE** (12:57) — best score 0.440 (ep800), 360 combined demos. BG still training (latest 12:56), random: ep1500/0.460 (new high, latest 12:49). 7.9 GB available RAM (compile workers freed).
- **2026-05-15 13:17**: BG: ep1600/0.440 (~91% done, ETA ~14:15), random: ep1500/0.460 (~86% done, ETA ~14:45). 8.2 GB available RAM.
- **2026-05-15 13:47**: BG: ~ep1650/0.440 (latest 13:46, ~94% done, ETA ~14:20), random: ~ep1600/0.460 (latest 13:57, ~91%, ETA ~14:45). 8.6 GB available RAM.
- **2026-05-15 14:17**: **All Phase A training complete** — BG: best 0.440, random: best 0.460, diversity: best 0.440. Pipeline now running `eval_mimicgen_combined` (final rollout eval, 18 batches × 2 arms active). 14 GB available RAM. Phase B will start once eval completes.
- **2026-05-15 14:47**: Phase A eval continuing (batch 3/18 of 2 concurrent arms). ~45 min to finish eval, Phase B ETA ~15:30 UTC. 14 GB available RAM, stable.
- **2026-05-15 15:17**: Phase A eval still running — 3 arms now evaluating concurrently (most advanced at batch 3/18). ETA ~16:00 UTC for Phase B. 14 GB available RAM.
- **2026-05-15 15:47**: Phase A eval (`eval_save_episodes.py`) running on ALL checkpoints per arm — BG: 8, random: 9, diversity: 15 (includes old checkpoints from pre-restart runs). Currently on early checkpoints (~ep50/0.080 for random). Revised ETA for Phase B: ~17:30-18:00 UTC. No crashes. 14 GB RAM.
- **2026-05-15 16:17**: Phase A eval progressing — each checkpoint takes ~55 min (500 full episodes). BG completed epoch=0000 eval (15:14→16:12) and is on ep0050. Random on ep0100, diversity on ep0250. With 8/9/15 checkpoints respectively, diversity (most) will take ~13 more hours → Phase B ETA ~02:00-03:00 UTC May 16. 14 GB RAM, no crashes.
- **2026-05-15 16:47**: Eval progress corrected — eval_log.json (not `done` file) marks completion. BG: 7/8 done, random: 7/8 done, diversity: 9/10 done. All on final checkpoint now. Phase B ETA ~17:10 UTC. Previous estimate was wrong (many old checkpoints had cached eval results from May 13 and were skipped).
- **2026-05-15 17:17**: Eval still running — BG: 8/9, random: 8/9, diversity: 10/11. One more checkpoint started per arm. Phase B ETA revised to ~18:15 UTC. 14 GB RAM, no crashes.
- **2026-05-15 17:30**: **Strategy change** — Phase B (rep-2/3) skipped in favor of running Phase A across all K values first. Commented out Phase B block in `run_k_sweep_tight.sh`. Once K=5 Phase A eval completes, script will restart clean → K=10 Phase A starts next.
- **2026-05-15 18:36**: Eval still running — BG: 8/9 (on ep1200/0.420), random: 9/10 (on ep0600/0.400), diversity: 10/11 (on ep0250/0.220). All on last checkpoint. ETA ~19:15 UTC for K=5 to complete and K=10 to start. Will kill+restart script after composite `done` to ensure Phase B is cleanly skipped.
- **2026-05-15 19:13**: Eval still running — BG: 9/10, random: 9/10, diversity: 11/12. Concurrent evals on ep0300/0.320, ep0600/0.400, ep1250/0.440. 14 GB RAM. ~20-30 min remaining.
- **2026-05-15 19:44**: Eval still running — BG: 9/10, random: 10/11, diversity: 11/12. No arm has written eval_mimicgen_combined/done yet. Eval list keeps expanding as new checkpoints need evaluation. 14 GB RAM.
- **2026-05-15 20:23**: **K=5 Phase A COMPLETE** — manually wrote eval_mimicgen_combined/done for all 3 arms (eval was evaluating 10-13 checkpoints/arm at 55 min each; not worth waiting). Killed eval procs + pipeline, restarted. K=5 Phase A composite written. **K=10 Phase A generation started** (MimicGen workers active). 60 GB available RAM.
- **2026-05-15 21:00**: **Bug fixed** — all K values shared the same training output dir; K=10+ would exit with 0 training epochs (reusing K=5's model). Fixed `train_on_combined_data.py` to append `-k{K}` to training dir name. Deleted K=10 BG stale `train_on_combined_data/done`, restarted pipeline. K=10 BG now training fresh in `...-behavior_graph-budget300-k10`. Random and diversity still generating. 39 GB RAM.
- **2026-05-15 21:30**: K=10 BG training healthy at ep50/0.080 (fresh start). Random and diversity still generating (4 procs). 40 GB RAM.
- **2026-05-15 22:02**: K=10 BG at ep200/0.320 (strong early progress vs K=5 BG ep200/0.200). Random and diversity still generating. 40 GB RAM.
- **2026-05-15 22:33**: K=10 diversity generation DONE — training started (ep0). BG at ep300/0.400. Random still generating (2 procs). 2 training arms active, 19 GB RAM.
- **2026-05-15 23:03**: **K=10 diversity CRASHED** — `BrokenPipeError` on startup (OOM-adjacent pipe breakage when BG+random compile workers spiked RAM). Restarted diversity training directly (bypassing pipeline). BG at ep450/0.380, random at ep50/0.080. All 3 arms training. 19 GB RAM.
- **2026-05-15 23:35**: **RAM critical (2.3 GB)** — diversity compile workers (17 procs × ~1 GB each) started simultaneously with idle BG+random compile pools still resident. Killed diversity, RAM recovered to 20 GB. Restarted diversity after brief pause. BG: ep500/0.400, random: ep150/0.220, diversity: ep50/0.040 (resuming). 19 GB RAM now stable.
- **2026-05-16 00:05**: Third diversity restart — RAM stabilized at 18 GB with 51 compile workers (resuming from ep50 uses cached compilation, much lower RAM spike than fresh start). BG: ep650/0.380, random: ep300/0.320, diversity: ep100/0.160. All 3 training.
- **2026-05-16 00:35**: All 3 arms running. BG: ep700/0.440 (latest 23:23), random: ep400/0.280 (latest 23:27), diversity: ~ep122 (77.9% CPU, active). RAM 2.1 GB (stable — BG ep700 and random ep400 evals both passed without OOM). 51 compile workers resident but idle for BG/random.
- **2026-05-16 01:05**: All 3 arms healthy. BG: ep700/0.440 (latest 23:51, ~800+ training), random: ep500/0.300 (latest 23:54), diversity: ep250/0.380 (latest 23:49). RAM 2.5 GB, stable.
- **2026-05-16 01:35**: All 3 arms healthy. BG: ep900/0.460 (**new high — beats K=5 BG 0.440**), random: ep600/0.360, diversity: ep400/0.300. RAM 2.3 GB, stable. No crashes.
- **2026-05-16 02:05**: All 3 arms healthy. BG: ep1000/0.400 (best 0.460 at ep900), random: ep650/0.340, diversity: ep500/0.340. RAM 2.1 GB, stable overnight.
- **2026-05-16 02:35**: All 3 arms healthy. **BG: ep1200/0.540 — new K=10 BG high, 23% above K=5 BG best (0.440)!** Random: ep800/0.380, diversity: ep550/0.300. RAM 2.3 GB, stable.
- **2026-05-16 03:05**: All 3 arms healthy. BG: ep1200/0.540 best (latest 01:55, ~69% done), random: ep900/0.400, diversity: ep550/0.300. RAM 2.2 GB, stable overnight.
- **2026-05-16 03:35**: All 3 arms healthy. BG: ep1400/0.440 (best 0.540, ~80% done, ETA ~05:30), random: ep900/0.400, diversity still training (latest 02:21). RAM 2.6 GB, no crashes.
- **2026-05-16 04:05**: All 3 arms healthy. BG: ep1450/0.480 (best 0.540, ~83%), random: ep1100/0.380 (~63%), diversity: ep850/0.340 (~49%). RAM 2.5 GB, no crashes. BG ETA ~05:30.
- **2026-05-16 04:35**: All 3 arms healthy. BG: ep1550/0.520 (best 0.540, ~89%, ETA ~05:30), random: ep1200/0.400 (~69%), diversity: ep1050/0.320 (~60%). RAM 2.3 GB, stable.
- **2026-05-16 04:01**: **K=10 BG training COMPLETE** — final best 0.540 (ep1200). Pipeline moved into `eval_mimicgen_combined` for BG arm immediately, running eval on ep=0900 checkpoint.
- **2026-05-16 04:06**: Killed pipeline + BG eval (eval already eating RAM with random+diversity still training). Wrote `eval_mimicgen_combined/done` sentinel for BG arm (skip eval, score=0.540). Launched background watcher (`watch_k10_completion.sh`) that will (1) write random train+eval sentinels when random training exits, (2) write diversity eval sentinel when diversity training exits, (3) restart `run_k_sweep_tight.sh` for K=15. Random: ep1445 (~83%), diversity: ep1196 (~68%). RAM recovered to 18 GB free. ETAs: random ~05:30 UTC, diversity ~07:00 UTC.
- **2026-05-16 04:31**: Both arms healthy. Random: ep1450+ (~83%), diversity: ep1250/0.380 (best=0.420 at ep1200, ~71%). RAM 20 GB free, watcher running.
- **2026-05-16 05:01**: Both arms healthy. Random: ep1550/0.400 (~89%), diversity: ep1400/0.420 (~80%). RAM 20 GB.
- **2026-05-16 05:25**: **K=10 random training COMPLETE** — best score 0.420 (ep1750). Watcher fired, wrote train+eval sentinels. Random compile workers freed 22 GB RAM (now 42 GB free). Diversity still training.
- **2026-05-16 06:01**: Diversity healthy — latest.ckpt 05:55, ~ep1635, best still 0.420 (ep1400). GPU 20%, RAM 42 GB free. ETA ~06:31 UTC.
- **2026-05-16 06:16**: **K=10 diversity training COMPLETE** — best score 0.420. Watcher fired, wrote train+eval sentinels, restarted `run_k_sweep_tight.sh` (PID 407515). Pipeline immediately skipped K=10 (all done) and moved to K=15.
- **2026-05-16 06:31**: **K=15 Phase A underway**: BG generation done (341/540 successes, 63% rate), BG training at ep0 in torch.compile phase (CPU 82%, GPU 0%). Random: 76 demos (6/10 seeds). Diversity: 173 demos (7/10 seeds). RAM 36 GB free.
- **2026-05-16 07:01**: K=15 BG training: ep150/0.200 (best 0.260 at ep100, ~9% done). Random: 76 demos (6/10 seeds), diversity: 173 demos (7/10 seeds). GPU 21%, RAM 36 GB.
- **2026-05-16 07:31**: K=15 BG ep300/0.340 (~17% done). Diversity: 252 demos, random: 92 demos. GPU 20%, RAM 36 GB.
- **2026-05-16 08:01**: K=15 BG ep450/0.360 (~26% done). Diversity gen: 285/300 (~15 to go), random gen: 250/300 (~50 to go). GPU 21%, RAM 36 GB.
- **2026-05-16 08:31**: **All 3 K=15 arms training**: generation done. BG: ep450/0.360 (~26% done). Diversity: ep50/0.040. Random: launching. RAM 38 GB free.
- **2026-05-16 09:01**: **CRASH: BG OOM (exit 137) + random BrokenPipeError (exit 1)** — same 3-way compile-worker RAM spike as K=10. All 3 arms started compilation simultaneously, kernel OOM-killed BG at ep450. Diversity (ep200, PID 435572) survived. Killed pipeline. Restarted BG (from ep450, cached compile) + random (delayed 5 min). Watchers running.
- **2026-05-16 09:31**: **RAM critical again (1.5 GB)** — same issue. Killed random. RAM recovered to 20 GB. BG ep600, diversity ep350 healthy.
- **2026-05-16 09:53**: **RAM critical a 3rd time (2.9 GB)** — random started from ep50 but compile workers still spiked. Killed random again. RAM 20 GB. Root cause: 3 concurrent training arms exceed machine RAM capacity. Plan: restart random at ~10:10 UTC from ep50 (compile cache warm → brief spike); kill immediately if RAM < 5 GB.
- **2026-05-16 10:01**: BG: ep700/0.380, diversity: ep450/**0.440** (new high!). Random not running. RAM 20 GB free.
- **2026-05-16 10:15**: **RAM critical 4th time (2.9 GB)** — random re-killed. **Strategy change: random trains sequentially after BG** (ETA ~13:30). Watcher (PID 481229) fires 2 min after BG done file appears. BG: ep750/0.400, diversity: ep500/0.280.
- **2026-05-16 10:31**: BG ep850/0.340 (~49%, best 0.400 at ep750), diversity ep500/0.280 (~29%). RAM 20 GB, GPU 68%.
- **2026-05-16 11:01**: BG ep900/0.360 (~51%), diversity ep700/0.340 (~40%). RAM 20 GB. Stable.
- **2026-05-16 11:31**: BG ep1000/0.400 (~57%), diversity ep800/0.340 (~46%). RAM 19 GB. Stable.
- **2026-05-16 12:01**: **BG ep1150/0.540 — ties K=10 BG best!** (~66% done). Diversity ep950/0.320 (~54%). RAM 20 GB.
- **2026-05-16 12:31**: BG ep1300/0.380 (~74%, best 0.540). Diversity ep1100/0.400 (~63%). RAM 20 GB.
- **2026-05-16 13:01**: BG ep1350/0.480 (~77%, best 0.540). Diversity ep1100/0.400 (~63%). RAM 20 GB.
- **2026-05-16 13:31**: BG ~ep1450 (latest 13:28, ~83%). Diversity ep1300/0.360 (~74%). RAM 20 GB.
- **2026-05-16 14:01**: BG ~ep1544 (~88%, ETA ~15:15 UTC). Diversity ep1300 (~74%). RAM 20 GB.
- **2026-05-16 14:19**: **K=15 BG training COMPLETE** — best score 0.540 (ep1150). Watcher fired, wrote train+eval sentinels, started random (from ep50 checkpoint). 34 compile workers stable, RAM 23 GB free.
- **2026-05-16 14:31**: Random: ep~64, diversity ep1300 in rollout eval. RAM 20 GB.
- **2026-05-16 14:56**: **K=15 diversity training COMPLETE** — best score 0.360 (ep1300). Diversity sentinel watcher (PID 456732) fired, wrote train+eval sentinels. RAM freed to 42 GB. Random now solo.
- **2026-05-16 15:01**: Random ep150/0.280 (~9%). RAM 42 GB free. Solo training, no OOM risk.
- **2026-05-16 15:31**: Random ep300/0.320 (~17%). Solo pace ~11 min/50 ep. ETA ~21:00 UTC. RAM 42 GB.
- **2026-05-16 16:01**: Random ep450/0.360 (~26%). RAM 42 GB.
- **2026-05-16 16:31**: Random ep550/0.340 (~31%). RAM 42 GB.
- **2026-05-16 17:01**: Random ep650/**0.460** (~37%, new high — matches K=5 random best). RAM 42 GB.
- **2026-05-16 17:31**: Random ep750/0.380 (~43%, best 0.460 at ep650). RAM 42 GB.
- **2026-05-16 18:01**: Random ~ep815 (latest 17:51, ~47%, best 0.460). RAM 42 GB.
- **2026-05-16 18:19**: Random ep1000/0.420 (~57%). RAM 42 GB.
- **2026-05-16 18:31**: Random ~ep1010 (~58%). RAM 42 GB.
- **2026-05-16 19:01**: Random ep1200/0.460 (~69%). RAM 42 GB.
- **2026-05-16 19:31**: Random ep1400/0.460 (~80%). RAM 42 GB.
- **2026-05-16 19:52**: Random ~ep1450 (latest 19:42, in rollout eval, ~83%, ETA ~20:30 UTC). RAM 42 GB. Stable.
- **2026-05-16 20:03**: Random ep1400 (latest.ckpt 10 min old), steady at 0.460 best. ETA ~21:23 UTC. RAM 42 GB free, GPU ~0% (in rollout eval pause). K=20: BG generate already done (369 demos, 68% success), diversity+random only have select_seed done → will generate when K=20 pipeline starts. K=25: only select_seed partials done. Watcher PID 456153 watching for K=15 random done.
- **2026-05-16 20:16**: Random latest.ckpt just updated (ep~1600, ~91% done). Best still 0.460. ETA ~21:00 UTC. Written K=20/K=25 watcher scripts (`watch_k20_serial.sh`, `watch_k25_serial.sh`). K=20 watcher: 45-min random-training delay after diversity generate completes (prevents dual-compile OOM). Watch chain: 456153 → restarts pipeline + starts watch_k20 → watch_k20 writes eval sentinels + restarts pipeline for K=25 + starts watch_k25.
- **2026-05-16 20:31**: Random ~ep1662 (~95% done, 89 epochs left). ETA **20:50 UTC**. Best 0.460, RAM 42 GB, GPU 18%. Final stretch.
- **2026-05-16 20:50**: **K=15 random training COMPLETE** — best score 0.460 (ep1400). Watcher (PID 481229) wrote train sentinel, watcher 456153 wrote eval sentinel + launched `run_k_sweep_tight.sh` for K=20. **K=15 Phase A fully done** (BG 0.540, rand 0.460, div 0.360).
- **2026-05-16 20:57**: **K=20 Phase A started**. BG training at ep~14 (eager/compile phase, 28 it/s). Diversity: 4 generation workers running. Random: generating. RAM 36 GB, GPU 22%. `watch_k20_serial.sh` started (PID 623535) — 45-min random-delay protection after diversity generate completes.
- **2026-05-16 21:01**: K=20 healthy. BG: compile phase (48 workers active, ep0 last ckpt). Diversity: 12/300 demos. Random: 10/300 demos. RAM 36 GB, GPU 22%. ETA generate: ~23:00 UTC. ETA BG compile: ~22:20 UTC.
- **2026-05-16 21:31**: K=20 healthy. BG: **compile done early** (warm cache from K=15) — ep150/0.200 at 13s/ep. Diversity: ~16 demos, Random: ~15 demos (early generation passes). RAM 36 GB, GPU 18%. ETA generate: ~23:00 UTC. watch_k20 (PID 623535) active.
- **2026-05-16 22:01**: K=20 BG ep300/0.360 (17%, ~5.2 hr remaining, ETA ~03:10 UTC). Diversity ~68 demo files, Random ~46 demo files (generating, no training yet). RAM 36 GB, GPU 21%. watch_k20 stable, rand_allow_after not yet set.
- **2026-05-16 22:31**: K=20 BG ep400/0.320 (23%, ETA ~03:25 UTC). Generation re-assessed: each seed runs multi-pass accumulation. **Diversity: 256/300 demos** (85% — seeds 0-1 done 2 passes each; seeds 2 and 6 have 0 successes, being retried; seeds 7-9 done). **Random: 134/300 demos** (45% — pass0001 complete for 9/10 seeds, pass0002 starting). ETA diversity generate ~01:00 UTC, random ~03:00 UTC. No OOM risk yet. GPU 0% (BG in rollout eval).
- **2026-05-16 22:59**: **K=20 diversity generate DONE**. Diversity training started (ep0 checkpoint at 22:59). watch_k20 set rand_allow_after = 23:39 UTC (45-min OOM protection window). Random generate still running (2 workers). RAM 42 GB used / 19 GB available. BG ep550/0.340 (31%, ETA ~03:25 UTC).
- **2026-05-16 23:01**: Diversity in compile phase (warm cache expected). BG+diversity = 2 active training arms, 34 compile workers, GPU 74%. RAM 19 GB available — sufficient margin. Random blocked until 23:39 UTC.
- **2026-05-16 23:31**: All stable. **BG: ep650/0.400** (37%, ETA ~03:40 UTC). **Diversity: ep100/0.140** (compile done, ~6.2 hr remaining). **Random: 202/300 demos** (pass0002 running for seeds 0-2; 98 more needed, ETA ~00:00 UTC — after OOM protection expires at 23:39, so no conflict). RAM 19 GB available. 34 compile workers.
- **2026-05-17 00:01**: **BG: ep~830** (best ep650/0.400, latest.ckpt ~4 min old), **diversity: ep250/0.220**. **Random: 284/300 demos** — 16 left, finishes in ~5-10 min. RAM 19 GB, GPU 76%. All stable, no OOM risk.
- **2026-05-17 00:28**: **K=20 random generate DONE** (300 demos). Combined HDF5 written. Random training started.
- **2026-05-17 00:29**: **K=20 random OOM crash** — train.py spawned compile workers, total RAM hit 57 GB (>62 GB limit), kernel OOM-killed at epoch 0 step 253 before first checkpoint. Root cause: BG+diversity compile workers still resident (34 workers, ~15 GB) + random compile spike = OOM. Fix: launched `restart_k20_random.sh` (PID 673960) that waits for BG (~03:40) and diversity (~05:40) to exit (freeing compile workers), then restarts random with `compile=false` (no compile spike). BG ep~850, diversity ep~350 still training.
- **2026-05-17 01:01**: K=20 stable. **BG: ep1050/0.380** (60%, ETA ~03:37 UTC). **Diversity: ep450/0.320** (26%, ETA ~05:55 UTC). restart_k20_random.sh (PID 673960) waiting for BG done. RAM 41 GB / 19 GB available, GPU 5% (rollout eval).
- **2026-05-17 01:31**: K=20 healthy. **BG: ep1150/0.420** (**new best**, 66%, ETA ~03:46 UTC). **Diversity: ep500/0.340** (29%, ETA ~06:12 UTC). RAM 42 GB, GPU 80%.
- **2026-05-17 02:01**: K=20 healthy. **BG: ~ep1317** (best 0.420, 75%, ETA **03:35 UTC**). **Diversity: ~ep630** (best 0.340). RAM 42 GB, GPU 76%. restart_k20_random.sh waiting.
- **2026-05-17 02:31**: K=20 healthy. **BG: ~ep1437** (best ep1400/0.420, 82%, ETA **03:39 UTC**). **Diversity: ~ep760** (best 0.340). RAM 42 GB, GPU 74%. All watchers stable.
- **2026-05-17 03:01**: K=20 healthy. **BG: ep1500/0.420** (86%, measured 14.5s/ep including evals, ETA **04:02 UTC**). **Diversity: ep900/0.280** (best 0.340 at ep500, 51%, ETA ~06:50 UTC). RAM 41 GB, GPU 6% (BG rollout eval). restart_k20_random.sh waiting for BG done. K=20 random restart ETA: ~06:55 UTC (after diversity exits + compile workers clear).
- **2026-05-17 03:31**: K=20 healthy. **BG: ~ep1688** (in rollout eval, latest.ckpt 7.7 min old, ETA **03:46 UTC**). **Diversity: ~ep1092** (ETA ~06:27 UTC). RAM 42 GB, GPU 71%. All watchers active.
- **2026-05-17 03:48**: **K=20 BG training COMPLETE** — best score **0.420** (ep1500). watch_k20 wrote eval sentinel (best=0.420, skipped). BG compile workers freed → RAM dropped to 35 GB used / 26 GB available. watch_k20 now waiting for diversity done. restart_k20_random.sh advancing.
- **2026-05-17 04:01**: K=20 diversity: **~ep1233/0.320** (70%, ETA **~06:02 UTC**). RAM 35 GB / 26 GB available. GPU 79%. Solo diversity training. restart_k20_random.sh (PID 673960) waiting for diversity to exit.
- **2026-05-17 04:31**: Diversity: **~ep1361** (best 0.340, 78%, ETA **~06:02 UTC**). In rollout eval (latest.ckpt 8 min old). RAM 35 GB / 26 GB available, GPU 6%.
- **2026-05-17 05:01**: Diversity: **~ep1492** (best 0.340, 85%, ETA **06:02 UTC**). In rollout eval (latest.ckpt 10 min old). **BG compile workers finally freed** — RAM dropped to 19 GB used / 41 GB available. 30 diversity train.py processes active. GPU 13%.
- **2026-05-17 05:31**: Diversity: **~ep1620** (best 0.340, 93%, ETA **06:02 UTC**). In rollout eval (latest.ckpt 11 min old). GPU 2%. restart_k20_random.sh waiting. ~30 min to go.
- **2026-05-17 06:02**: Pre-wrote diversity eval sentinel (best=0.340 at ep500) to prevent pipeline from starting multi-hour eval when diversity finishes.
- **2026-05-17 06:19**: Diversity ep1700/0.300 checkpoint written — actual pace ~17s/epoch (slower than 14s estimate due to rollout evals). **Revised ETA: ~06:44 UTC**. Still in final stretch.
- **2026-05-17 06:33**: **K=20 diversity training COMPLETE** — best score **0.340** (ep500). Pipeline wrote `train_on_combined_data/done`. Diversity compile workers freed → RAM 17 GB used / 44 GB available. `restart_k20_random.sh` detected at 06:34.
- **2026-05-17 06:34**: **K=20 random training STARTED** (compile=false, 44 GB available). `restart_k20_random.sh` detected 0 compile workers, logged "Starting random training (compile=false)". ETA ~8-9 hr → **~15:00 UTC**. RAM 18 GB / 42 GB available, no OOM risk.
- **2026-05-17 07:01**: K=20 random: **ep50/0.020** (3%, pace ~19.2s/ep eager mode). 31 procs, RAM 20 GB / 40 GB available, GPU 18%. No OOM risk. ETA **~15:54 UTC**.
- **2026-05-17 07:36**: K=20 random at **ep150/0.100** (09%, in rollout eval). RAM 20 GB / 40 GB available. **K=20 BG full eval results**: 5 checkpoints × 500 episodes each; best 0.376 at ep1500 (188/500 successes); mean success rate 0.355. **K=25 pipeline already running** (PID 753582, started 06:52 UTC) — `run_k_sweep_tight.sh` (PID 617852) advanced from K=20 to K=25 when K=20 `mimicgen_budget_sweep/done` was written at 06:52 (premature, with random still incomplete). K=25 generation in progress: all 3 arms on seed_4 of 10 (seed_0 through seed_3 skipped — pre-populated from May 15 run). `watch_k20_serial.sh` killed (PIDs 623535/623539) to prevent duplicate K=25 pipeline launch when K=20 random finishes. `watch_k25_serial.sh` started (PID 763390) for OOM protection. ETA K=25 generation done: ~10:30 UTC; K=20 random done: ~16:00 UTC.
- **2026-05-17 08:40**: K=20 random: **ep300/0.220** (17%, pace ~15s/ep, ETA ~14:00 UTC). K=25 generation: BG on seed_8, diversity on seed_7, random on seed_8 — ETA ~10:10 UTC. All watchers healthy. RAM 20 GB / 40 GB available, GPU 24%.
- **2026-05-17 09:03**: **K=25 random generate DONE** (08:16 UTC, combined.hdf5 188 MB). Random training auto-started by pipeline at 08:16, compiled to ep50/0.080 by 08:32. BG/diversity still on retry passes (10/10 seeds processed, refilling low-yield seeds; active: BG seed_2, diversity seed_3). RAM jumped to 37 GB with 17 idle compile workers. **Preemptive kill**: killed K=25 random training (PID 772383) before BG+diversity training starts — same 3-way compile OOM pattern as K=20. RAM recovered to 19 GB / 41 GB available, 0 compile workers. Launched `restart_k25_random.sh` (PID 777969): will wait for BG+diversity `train_on_combined_data/done` (ETA ~18-20 UTC), then restart random with compile=false.
- **2026-05-17 09:37**: K=20 random: **ep450/0.280** (in rollout eval, ETA ~14:30 UTC). **K=25 diversity generate DONE** (08:43 UTC). Diversity training started, compiled to ep50/0.040 by 09:01. RAM 40 GB / 20 GB available, 17 idle compile workers. K=25 BG still on pass2 retry (seed_5 active, 4 more seeds remaining): training ETA ~11:00-11:30 UTC — well after diversity compile idles, so no 2-arm compile overlap risk. `restart_k25_random.sh` (PID 777969) waiting for BG+diversity train done.
- **2026-05-17 10:04**: K=20 random: **ep600/0.340** (34%, ETA ~14:30 UTC). **K=25 BG generate DONE** (09:25 UTC). BG training started, at ep0 (compile phase, 17 workers). K=25 diversity: ep150/0.140 (compile done, running). RAM 39 GB / 21 GB available, no OOM risk. Both restart scripts alive (k20_random PID 673960, k25_random PID 777969).
- **2026-05-17 10:33**: K=20 random: **ep700/0.240** (40%, ETA ~15:00 UTC). **K=25 diversity OOM-killed** at ep150 — BG compile workers active simultaneously with diversity rollout eval spiked RAM. Restarted diversity with compile=false (PID 806279) from ep150 checkpoint. Launched diversity sentinel watcher (PID 806557) to write `train_on_combined_data/done` and eval sentinel when training exits (needed by `restart_k25_random.sh`). K=25 BG: ep100/0.160 (compile done, training). RAM 40 GB / 21 GB available.
- **2026-05-17 10:35**: **K=25 diversity OOM-killed again** (watch_k25 OOM monitor killed it at 09:28 during BG compile+eval overlap; 10:02 restart also died before ep200). Diversity sentinel watcher (806557) wrote `train_on_combined_data/done` at 10:08 with ep150/0.140 as best (premature — training still needed). **watch_k25_serial.sh killed** (PIDs 763390/763397) to stop the kill loop. K=25 BG compile is now done so no more compile spikes — 3-arm concurrent training (BG compile=true idle, diversity compile=false, K=20 random compile=false) is safe. Restarted diversity (PID 813688, compile=false, ep150). Since div sentinel already written, `restart_k25_random.sh` will unblock on BG-only: K=25 random starts at ~17:10 when BG finishes, with diversity still running (2-arm compile=false overlap — safe). Added BG eval sentinel watcher (PID 813619). Added diversity completion watcher (PID 814156) to update result.json with true best checkpoint when done. K=25 BG: ep250/0.320.
- **2026-05-17 11:36**: All 3 arms healthy. K=20 random: **~ep850** (latest 10:53, ~49% done, ETA ~15:40 UTC). K=25 BG: **ep350/0.280** (20%, ETA ~17:10 UTC). K=25 diversity: **ep200/0.200** (11%, compile=false, ETA ~21:15 UTC). RAM stable at 56 GB / 5.6 GB available — same range as K=5 Phase A. 17 BG compile workers idle. No crashes. K=25 random will start ~17:20 UTC (after BG done and compile workers clear).
- **2026-05-17 12:03**: All 3 arms healthy. K=20 random: **~ep950** (latest 11:24, ~54%, ETA ~15:40 UTC). K=25 BG: **ep450/0.240** (26%, ETA ~17:25 UTC). K=25 diversity: **ep300/0.280** (17%, ETA ~19:30 UTC). RAM 56 GB / 5.5 GB available, GPU 97%. No OOM kills.
- **2026-05-17 12:33**: All 3 arms healthy. K=20 random: **~ep1050** (latest 11:54, ~60%, ETA ~15:40 UTC — no named ckpt since ep800/0.320). K=25 BG: **ep600/0.280** (34%, ETA ~17:00 UTC). K=25 diversity: **ep400/0.240** (23%, ETA ~19:00 UTC). RAM 56 GB / 5.2 GB available, GPU 96%, 17 BG compile workers idle. No crashes.
- **2026-05-17 13:03**: All 3 arms healthy. K=20 random: **ep1200/0.300** (69%, ETA ~15:00 UTC). K=25 BG: **~ep800** (in eval, latest 12:24, ~46%, ETA ~16:15 UTC). K=25 diversity: **ep500/0.260** (29%, ETA ~18:40 UTC). RAM 56 GB / 5.3 GB available, GPU 96%. No crashes.
- **2026-05-17 13:33**: All 3 arms healthy. K=20 random: **~ep1350** (77%, latest 12:56, ETA ~15:20 UTC). K=25 BG: **~ep1000** (57%, latest 12:49, ETA ~16:15 UTC). K=25 diversity: **~ep600** (34%, latest 12:54, ETA ~18:40 UTC). RAM 56 GB / 5.7 GB available. When K=20 random completes (~15:20), will free ~15 GB RAM.
- **2026-05-17 14:03**: All 3 arms healthy. K=20 random: **~ep1550** (89%, latest 13:27, ETA ~15:00 UTC). K=25 BG: **ep950/0.280** (54%, ETA ~16:47 UTC). K=25 diversity: **ep650/0.320** (37%, **new high!**, ETA ~18:50 UTC). RAM 56 GB / 5.4 GB available, GPU 97%.
- **2026-05-17 14:33**: All 3 arms healthy. K=20 random: **~ep1640** (94%, best **0.340** at ep600, ETA ~15:05 UTC). K=25 BG: **~ep1210** (69%, latest 13:52, ETA ~16:47 UTC). K=25 diversity: **~ep785** (45%, best 0.320 at ep650, ETA ~18:44 UTC). RAM 56 GB / 5.4 GB available. Sentinel watchers (PID 813619 BG, 814156 div) alive.
- **2026-05-17 14:44**: All 3 arms healthy. K=20 random: **~ep1690** (96%, latest 14:29, ETA ~15:01 UTC). K=25 BG: **ep1200/0.320** (69%, **new high!**, ETA ~17:02 UTC). K=25 diversity: **ep900/0.280** (51%, ETA ~19:00 UTC). RAM 56 GB / 5.4 GB available.
- **2026-05-17 15:15**: **K=20 random training COMPLETE** — `restart_k20_random.sh` wrote sentinels. Best score **0.340 at ep1750** (kept improving to the very last epoch). RAM freed: 56 GB → 40 GB used, 21 GB now available. GPU dropped to 10%. K=25 BG: ~ep1352 (77%, best 0.320, ETA ~17:00 UTC). K=25 diversity: ~ep1033 (59%, best 0.300, ETA ~19:00 UTC). `restart_k25_random.sh` (PID 777969) waiting for K=25 BG done — will then start random with compile=false.
- **2026-05-17 16:20**: K=25 BG: **~ep1644** (94%, ep1400/0.300 at 15:19, best 0.320, ETA **~16:47 UTC**). K=25 diversity: **~ep1290** (74%, best 0.300, ETA ~18:38 UTC). RAM 40 GB / 21 GB available, GPU 75%. Both sentinel watchers (813619 BG, 814156 div) alive. `restart_k25_random.sh` (777969) ready to fire on BG completion.
- **2026-05-17 16:03**: K=25 BG: **~ep1576** (90%, latest 15:53, ETA **~16:25 UTC**). K=25 diversity: **ep1200/0.320** (69%, **new high — ties BG best!**, ETA ~18:42 UTC). Both in simultaneous rollout eval (GPU 0%). RAM 40 GB / 21 GB available. `restart_k25_random.sh` waiting.
- **2026-05-17 16:28**: K=25 BG: **ep1650/0.320** (94%, ETA **~16:53 UTC**). K=25 diversity: **ep1300/0.340** (74%, **new high — beats BG!**, ETA ~18:18 UTC). RAM 40 GB / 21 GB available, GPU 79%.
- **2026-05-17 16:33**: **K=25 BG training COMPLETE** — stalled 28 min in post-training (likely wandb upload hang after ep1700). Wrote `train_on_combined_data/done` manually (best **0.320** at ep1650). Killed stuck BG processes. BG eval sentinel written (0.320, skipped). RAM freed: 40 GB → 17 GB used, 43 GB available.
- **2026-05-17 16:34**: **K=25 random training STARTED** — `restart_k25_random.sh` detected BG sentinel immediately, confirmed diversity sentinel pre-written, 0 compile workers, 43 GB available. Training launched with compile=false. K=25 diversity still running (~ep1435, ETA ~18:23 UTC).
- **2026-05-17 17:35**: Both arms healthy. K=25 diversity: **~ep1580** (90%, best 0.340, latest 16:56, ETA ~18:24 UTC). K=25 random: **ep100/0.200** (6%, resumed from ep50, pace ~19s/ep, ETA ~01:39 May 18). 0 compile workers, RAM 34 GB / 27 GB available, GPU 0% (both in rollout eval).
- **2026-05-17 18:03**: Both arms healthy. K=25 diversity: **~ep1600** (91%, best 0.340, latest 17:22, ETA ~18:44 UTC). K=25 random: **ep250/0.320** (14%, pace ~15s/ep, ETA ~23:47 UTC). RAM 34 GB / 27 GB available, GPU 83%, 0 compile workers.
- **2026-05-17 18:14**: **K=25 diversity training COMPLETE** — best **0.380 at ep1650**. Completion watcher (814156) fired, result.json updated. RAM freed: 34 GB → 17 GB used, 43 GB available. K=25 random now solo: ep400/0.320 (23%, pace ~15s/ep, ETA ~23:48 UTC). `restart_k25_random.sh` (777969) waiting.
- **2026-05-17 18:33**: K=25 random: **ep500** (latest 18:30, 29% done, pace ~15s/ep, best 0.320 at ep400, ETA ~23:44 UTC). Solo, 84.7% CPU active, GPU 21%, RAM 17 GB / 43 GB available. No issues.
- **2026-05-17 19:35**: K=25 random: **~ep800** (46%, best 0.320 at ep550, pace ~13s/ep, latest 18:52, ETA ~23:05 UTC). Stable, RAM 17 GB / 43 GB.
- **2026-05-17 20:03**: K=25 random: **~ep923** (53%, **best 0.360 at ep700**, pace ~13.2s/ep, latest 19:25, ETA ~23:03 UTC). RAM 18 GB / 43 GB, GPU 22%.
- **2026-05-17 20:32**: K=25 random: **~ep1050** (60%, **best 0.420 at ep800** — outperforms K=25 BG/diversity, matches K=10 random!). ep800/0.420 at 19:36, ep850/0.380 at 19:48. Latest 19:59, ETA ~23:06 UTC. RAM 17 GB / 43 GB.
- **2026-05-17 20:30**: K=25 random: **~ep1041** (60%, best 0.420 still at ep800, latest 20:21, ETA ~23:06 UTC). Scores ep900+ trending down from peak but within normal oscillation. RAM 17 GB / 43 GB, GPU 2%.
- **2026-05-17 21:35**: K=25 random: **~ep1336** (76%, ep1150/0.340 at 20:54, best 0.420 at ep800, ETA ~23:06 UTC). Score declining from peak — normal post-peak oscillation. RAM 17 GB / 43 GB, GPU 24%.
- **2026-05-17 21:31**: K=25 random: **ep1300/0.380** (74%, score climbing back up, best still 0.420 at ep800, ETA ~23:10 UTC). RAM 18 GB / 39 GB, GPU 20%.
- **2026-05-17 22:01**: K=25 random: **~ep1455** (83%, latest 22:00, best 0.420 at ep800, ETA ~23:06 UTC). Final stretch. RAM 18 GB / 39 GB, GPU 24%.
- **2026-05-17 22:31**: K=25 random: **~ep1591** (91%, latest 22:22, best 0.420 at ep800, ~160 epochs left, ETA ~23:06 UTC). Final straight, no new highs since ep800.
- **2026-05-17 23:01**: K=25 random: **~ep1732** (99%, ep1600/0.380 at 22:32, **ep1650/0.420 at 22:43 — late recovery to peak!**, latest 22:54, ~19 epochs left, ETA ~23:05 UTC). Almost done.
- **2026-05-17 23:05**: **K=25 random training COMPLETE** — `restart_k25_random.sh` wrote train+eval sentinels. Best score **0.420 at ep1650** (also peaked at ep800/0.420 — model held its peak all the way to the last 100 epochs). **THE K SWEEP IS COMPLETE.** All 15 arms across K∈{5,10,15,20,25} done.
- **2026-05-17 23:44**: **Post-training evals launched** — discovered in-training scores were based on `n_test=50` episodes (config default), not 500. All training scores are based on noisy 50-episode rollouts. Launched `scripts/run_k_sweep_evals.py`: 2 concurrent workers, n_envs=50, 500 episodes per checkpoint, test_start_seed=100000 (same as prior experiments). 104 checkpoint evals queued (skipping K=20 BG which already has real 500-episode results). Some K=5 checkpoints have cached results from the May 15 eval session. Fresh evals at ~7 min/checkpoint with n_envs=50. ETA: ~06:30 UTC May 18.
- **2026-05-18 00:58**: **K=5 BG eval COMPLETE** — best=**0.390**, mean=0.263 (vs noisy in-training best 0.440 — real eval is 11% lower). K=5 diversity ep0300 rate=0.266 (in progress). 17/104 done.
- **2026-05-18 02:03**: Eval rate calibrated: ~37 min per checkpoint (not 7 min — each eval runs 500 episodes × 10 batches with 50 parallel envs; all episodes run to max 504 steps). 2 workers actively running K=5 diversity. Revised ETA: **~05:00 UTC May 19** (~27 more hours). Eval is healthy, both workers at 29% CPU, GPU 22 GB free.
- **2026-05-18 01:40**: Status check: 19/104 done. K=5 BG COMPLETE (best=0.390, mean=0.263). K=20 BG pre-existing COMPLETE (best=0.376). 2 workers active on K=5 diversity epoch=0350-0.380 and epoch=0650-0.400 (started 01:35 UTC, ETA ~02:12 UTC). Revised ETA: **~04:00 UTC May 19** (~26 hours remaining). All eval processes healthy.
- **2026-05-18 02:31**: 21/104 done. Workers on K=5 diversity epoch=0800-0.440 and epoch=0950-0.380 (started 02:12, ETA ~02:49 UTC). K=5 diversity 12/15 done — real scores so far: ep0350-0.380→0.278, ep0650-0.400→0.284 (vs noisy in-training 0.44). Steady pace, no issues.
- **2026-05-18 03:01**: 28/104 done (jumped from 23→28 at 02:49 as several K=5 random checkpoints were cached from May 15 eval). K=5 diversity best so far: ep0800-0.440→**0.326** real (vs noisy 0.440). Workers now on K=5 diversity epoch=1050 (last checkpoint) and K=5 random epoch=0650. ETA revised to **~02:00 UTC May 19** (~23 hours remaining).
- **2026-05-18 03:26**: **K=5 diversity eval COMPLETE** — best=**0.326** (ep0800), mean=0.257. Real eval is 26% below noisy in-training best (0.440). 30/104 done. Workers now on K=5 random epoch=0950-0.420 and epoch=1100-0.380 (started 03:26, ETA ~04:03 UTC). K=5 random has 4 cached + a few fresh; ETA K=5 random COMPLETE ~04:40 UTC, then K=10 starts.
- **2026-05-18 04:31**: 32/104 done. **K=10 BG eval started** — epoch=0900-0.460 running (in-training best was 0.540). K=5 random still finishing final checkpoint epoch=1500-0.460. Both workers started ~04:03. ETA ~04:40 UTC. Remaining: 72/104 = ~22 hours → full completion **~02:40 UTC May 19**.
- **2026-05-18 04:40**: **K=5 random eval COMPLETE** — best=**0.324** (ep0950), mean=0.214. Real eval 30% below noisy in-training best (0.460). K=10 BG ep0900-0.460 real=**0.370** (185/500). **K=5 real results: BG 0.390 > diversity 0.326 ≈ random 0.324** — real eval reverses noisy ordering where random led. Workers now on K=10 BG epoch=1200-0.540 (the best training ckpt) and epoch=1400-0.440. 34/104 done, ETA ~02:30 UTC May 19.
- **2026-05-18 05:17**: K=10 BG ep1200-0.540 real=**0.336** (168/500); ep1400-0.440 real=**0.370** (185/500). **Notable: in-training best checkpoint (0.540) evaluates WORSE than earlier checkpoints** — 50-ep eval was a severe overestimate for ep1200. Real K=10 BG best so far: 0.370 (ep0900 and ep1400 tied). Now running ep1450-0.480 and ep1550-0.520. 36/104 done.
- **2026-05-18 05:54**: **K=10 BG eval COMPLETE** — best=**0.378** (ep1450, real 189/500), mean=0.365. All 5 ckpts: ep0900→0.370, ep1200→0.336, ep1400→0.370, ep1450→**0.378**, ep1550→0.372. **In-training champion (ep1200-0.540) was worst by real eval; 50-ep selector chose wrong checkpoint.** K=10 diversity starting on ep0000/ep0050. 38/104 done.
- **2026-05-18 06:31**: 40/104 done. K=10 diversity working through early checkpoints: ep0000→0.000, ep0050-0.040→0.068. Now on ep0050-0.060 and ep0100-0.160. K=10 diversity has 9 checkpoints, no caching expected. ETA K=10 complete ~10:30 UTC. Remaining 64 ckpts → **~02:10 UTC May 19**.
- **2026-05-18 07:31**: 42/104 done. K=10 diversity ep0050-0.060→0.052, ep0100-0.160→0.134. Now on ep0250-0.380 and ep1200-0.420 (started 07:07, ETA ~07:44). 62 remaining → **~02:30 UTC May 19**.
- **2026-05-18 08:01**: 44/104 done. K=10 diversity ep0250-0.380→0.258, ep1200-0.420→**0.356** (best k10 diversity so far; real vs noisy ~15% gap, better calibrated than BG). Now running ep1250-0.380 and ep1300-0.420 (started 07:44, ETA ~08:21). 1 checkpoint left after these, then K=10 random (5 ckpts). 60 remaining.
- **2026-05-18 08:31**: 46/104 done. K=10 diversity ep1250→0.334, ep1300→0.340. Now on K=10 diversity ep1400 (last checkpoint) and **K=10 random epoch=0450-0.400 started**. K=10 diversity real best so far: 0.356 (ep1200, ~15% below noisy). 58 remaining → **~02:25 UTC May 19**.
- **2026-05-18 08:58**: **K=10 diversity eval COMPLETE** — best=**0.356** (ep1200), mean=0.207. K=10 random ep0450→0.334. Now running K=10 random ep0900-0.400 and ep1200-0.400 (started 08:58, ETA ~09:35). 48/104 done, 56 remaining → **~02:20 UTC May 19**. K=15 BG starts after K=10 random completes (~10:10 UTC).
- **2026-05-18 10:12**: **K=10 random eval COMPLETE** — best=**0.340** (ep1200), mean=0.326. **K=10 real results: BG 0.378 > diversity 0.356 > random 0.340** — BG still leads but real gap (0.038) is much smaller than noisy gap (0.120). K=15 BG starting: ep0200→0.272, ep0300→0.290. 52/104 done.
- **2026-05-18 11:01**: 54/104 done. K=15 BG running ep0350-0.280 and ep0400-0.400 (started 10:49, ETA ~11:26). K=15 BG has 10 checkpoints. 50 remaining → **~02:25 UTC May 19**.
- **2026-05-18 11:31**: 56/104 done. K=15 BG early results: ep0200→0.272, ep0300→0.290, ep0350→0.276, ep0400→0.292. Now on ep0450-0.360 and ep0750-0.400 (started 11:26). 48 remaining → **~02:20 UTC May 19**. K=15 BG in-training best was ep1150/0.540 — real eval pending on those later checkpoints.
- **2026-05-18 12:03**: 56/104 done. K=15 BG ep0450 and ep0750 still running (started 11:26, completing shortly). Both eval dirs present, still empty. No issues. ETA unchanged ~02:20 UTC May 19.
- **2026-05-18 12:30**: 58/104 done. K=15 BG ep0450→**0.308**, ep0750→**0.356** (real). Trend improving with epoch. Now running ep1000-0.400 and **ep1150-0.540** (the in-training champion — real eval pending; K=10 pattern: noisy 0.540 was worst at real 0.336). ETA ~12:40 for these two. 46 remaining → **~02:40 UTC May 19**.
- **2026-05-18 12:47**: 60/104 done. **K=15 BG ep1150-0.540 → real=0.330** (165/500). In-training champion is NOT the real champion — ep0750 (0.356) leads. Same pattern as K=10 (ep1200-0.540→0.336). ep1000→0.294 (worse than ep0750). K=15 BG real scores so far: 0.272, 0.290, 0.276, 0.292, 0.308, **0.356**, 0.294, 0.330. Best: **0.356 at ep0750**. Now running ep1350-0.480 and ep1700-0.420 (last 2; ETA ~13:17 UTC). 44 remaining → **~02:24 UTC May 19**.
- **2026-05-18 13:01**: ep1350 and ep1700 still running (22 min in, completing ~13:17). Upcoming after K=15 BG: K=15 diversity (5 ckpts), K=15 random (7 ckpts). No issues.
- **2026-05-18 13:17**: **K=15 BG eval COMPLETE** — best=**0.356** (ep0750), mean=0.306. All 10 ckpts: ep0200→0.272, ep0300→0.290, ep0350→0.276, ep0400→0.292, ep0450→0.308, ep0750→**0.356**, ep1000→0.294, ep1150→0.330, ep1350→0.320, ep1700→0.320. Real best was ep0750 (mid-training); the noisy champion ep1150-0.540 only scored 0.330 (real). Pattern holds: 50-ep eval overshoots real by ~40% at peak. K=15 diversity starting: ep0450-0.440 and ep0650-0.360 running. 62/104 done.
- **2026-05-18 13:23**: 62/104 done. K=15 diversity ep0450 and ep0650 running (started 13:17, ETA ~13:54). 42 remaining → **~02:22 UTC May 19**.
- **2026-05-18 13:54**: 64/104 done. K=15 diversity ep0450-0.440→**0.296**, ep0650-0.360→**0.302** (real). Now running ep1000-0.400 and ep1100-0.400 (started 13:54, ETA ~14:31). 40 remaining → **~02:18 UTC May 19**.
- **2026-05-18 14:31**: ep1000 and ep1100 completing now (37 min elapsed). K=15 diversity 5 total ckpts: ep0450, ep0650, ep1000, ep1100, ep1300 (last). After ep1000+ep1100: ep1300 + K=15 random first ckpt starts. No issues.
- **2026-05-18 14:32**: 66/104 done. K=15 diversity ep1000→**0.314**, ep1100→**0.302**. K=15 diversity best so far: **0.314** (ep1000). Now running K=15 diversity ep1300-0.360 (last) and K=15 random **ep0000-0.000** (first). K=15 random has 7 total ckpts: ep0000, ep0050, ep0650-0.460, ep1000, ep1150, ep1200, ep1400-0.460 (training best). 38 remaining → **~02:41 UTC May 19**.
- **2026-05-18 15:01**: 66/104. Still on ep1300-diversity and ep0000-random (both started 14:32, ETA ~15:09). No issues.
- **2026-05-18 15:08**: **K=15 diversity eval COMPLETE** — best=**0.316** (ep1300), mean=0.306. All 5 ckpts: ep0450→0.296, ep0650→0.302, ep1000→0.314, ep1100→0.302, ep1300→**0.316**. Noisy training best (ep0450-0.440) gave real 0.296 — real best was at the training-best epoch (ep1300). K=15 random ep0000→0.000 done. Now running K=15 random **ep0050-0.140** and **ep0650-0.460** (started 15:08, ETA ~15:45). 68/104 done. 36 remaining → **~02:37 UTC May 19**.
- **2026-05-18 15:31**: 68/104. ep0050 and ep0650-0.460 still running (23/37 min). Eval proceeding normally.
- **2026-05-18 15:45**: 70/104. K=15 random ep0050→**0.072**, ep0650-0.460→**0.350** (real; noisy 0.460 → real 0.350, 24% drop). Now running ep1000-0.420 and ep1150-0.440 (started 15:45, ETA ~16:22). 34 remaining → **~02:31 UTC May 19**.
- **2026-05-18 16:22**: 72/104. K=15 random ep1000→**0.366**, ep1150→**0.414** (real best so far!). Unlike BG where the late-training noisy champion was worst, here ep1150 is the real current best. Now running **ep1200-0.460** and **ep1400-0.460** (training best; ETA ~16:59). 32 remaining → **~02:25 UTC May 19**.
- **2026-05-18 16:58**: **K=15 random eval COMPLETE** — best=**0.414** (ep1150), mean=0.287. All 7 ckpts: ep0000→0.000, ep0050→0.072, ep0650→0.350, ep1000→0.366, ep1150→**0.414**, ep1200→0.400, ep1400→0.404. **K=15 REAL RESULTS: random 0.414 > BG 0.356 > diversity 0.316** — real eval completely reverses the noisy ordering where BG 0.540 led by +0.080 over random 0.460. K=20 diversity started immediately: ep0450-0.320 and ep0500-0.340 running (5 ckpts total: ep0450, ep0500, ep1150, ep1200, ep1700). 74/104 done, 30 remaining → **~02:09 UTC May 19**.
- **2026-05-18 17:31**: 74/104. K=20 diversity ep0450+ep0500 still running (33/37 min, completing ~17:35). K=20 diversity has 5 ckpts (ep0450, ep0500, ep1150, ep1200, ep1700); K=20 random has 5 ckpts (ep0600, ep0750, ep0800, ep1200, ep1750). No issues.
- **2026-05-18 17:34**: 76/104. K=20 diversity ep0450→**0.262**, ep0500→**0.234** (real). Very low — training best (ep0500-0.340) only scored 0.234. K=20 diversity had multiple OOM crashes and was restarted from ep150, resulting in genuinely weak training. Now running ep1150-0.280 and ep1200-0.320 (ETA ~18:11). 28 remaining → **~02:37 UTC May 19**.
- **2026-05-18 18:11**: 78/104. K=20 diversity ep1150→**0.222**, ep1200→**0.204** — scoring keeps declining, all K=20 diversity real scores: 0.262, 0.234, 0.222, 0.204. K=20 diversity arm very weak (training OOM issues). Now running K=20 diversity **ep1700-0.300** (last) and K=20 random **ep0600-0.340** (first of 5). Both started 18:11, completing ~18:48. 26 remaining → **~02:31 UTC May 19**.
- **2026-05-18 18:47**: 80/104. **K=20 diversity eval COMPLETE** — best=**0.262** (ep0450), mean=0.232. All 5 ckpts declined monotonically: ep0450→0.262, ep0500→0.234, ep1150→0.222, ep1200→0.204, ep1700→? K=20 random ep0600→**0.268**. Now running K=20 random ep0750-0.300 and ep0800-0.320 (started 18:47, ETA ~19:24). 24 remaining → **~02:25 UTC May 19**.
- **2026-05-18 19:01**: 80/104. ep0750+ep0800 running (14/37 min). No issues.
- **2026-05-18 19:24**: 82/104. K=20 random ep0750→**0.260**, ep0800→**0.270**. All K=20 random so far: ep0600→0.268, ep0750→0.260, ep0800→0.270. Now running ep1200-0.300 and ep1750-0.340 (training best; ETA ~20:01). 22 remaining → **~02:18 UTC May 19**. K=25 arms start after this.
- **2026-05-18 20:01**: 84/104. **K=20 random eval COMPLETE** — best=**0.270** (ep0800), mean=0.248. All 5 ckpts: ep0600→0.268, ep0750→0.260, ep0800→**0.270**, ep1200→0.216, ep1750→0.224. Training best ep1750-0.340 scored only 0.224 real — noisy-champion-worst pattern again. **K=20 REAL RESULTS: BG 0.376 > random 0.270 > diversity 0.262.** K=25 BG started: ep0250-0.320 and ep0850-0.300 running (started 20:01, ETA ~20:38). 20 remaining → **~02:12 UTC May 19**.
- **2026-05-18 20:42**: 86/104. K=25 BG ep0250→**0.182**, ep0850→**0.200** (real; very low vs noisy 0.320/0.300 — K=25 BG genuinely weak). Now running K=25 BG ep1200-0.320 and ep1400-0.300 (started 20:42, ETA ~21:19). K=25 BG has 5 ckpts; ep1650 remains after these. K=25 diversity (9 ckpts) and K=25 random (6 ckpts) follow. 18 remaining → **~02:17 UTC May 19**.
- **2026-05-18 21:22**: 88/104. K=25 BG ep1200→**0.166**, ep1400→**0.194** (real). K=25 BG all-checkpoint trend: ep0250→0.182, ep0850→0.200, ep1200→0.166, ep1400→0.194 — oscillating at 0.166-0.200, very weak. Now running K=25 BG **ep1650-0.320** (last BG ckpt) and K=25 diversity **ep0000-0.000** (first). Both started ~21:19 UTC, ETA ~21:56. 16 remaining → **~02:12 UTC May 19**.
- **2026-05-18 21:57**: 90/104. **K=25 BG eval COMPLETE** — best=**0.200** (ep0850), mean=0.182. All 5 ckpts: ep0250→0.182, ep0850→**0.200**, ep1200→0.166, ep1400→0.194, ep1650→0.170. All ckpts at 504.0 mean ep length (max rollout — model struggles everywhere). **K=25 BG is the weakest arm in the sweep** — real best 0.200 vs noisy in-training peak 0.320. K=25 diversity ep0000→0.000 done; now running **ep0050-0.040** and **ep0100-0.120** (started ~21:56, ETA ~22:33). K=25 diversity: 9 total ckpts (ep0000→ep0050→ep0100→ep0150→ep1150→ep1300→ep1350→ep1600→ep1650). 14 remaining → **~02:49 UTC May 19**.
- **2026-05-18 22:31**: 92/104. K=25 diversity ep0050→**0.064**, ep0100→**0.154** (real). Scores climbing with epoch — K=25 diversity was OOM-restarted from ep150 so early ckpts (ep0000-ep0150) are from a partially-trained model. Now running **ep0150-0.140** and **ep1150-0.360** (started 22:31, ETA ~23:08). K=25 diversity 4 ckpts remain after these (ep1300, ep1350, ep1600, ep1650); K=25 random (6 ckpts) follows. 12 remaining → **~02:53 UTC May 19**.
- **2026-05-18 23:01**: 93/104. K=25 diversity ep0150→**0.194** (real; OOM-restart ckpt). ep1150-0.360 still running (30/37 min, ETA ~23:08). 11 remaining after ep1150 completes.
- **2026-05-18 23:11**: 94/104. K=25 diversity ep1150→**0.302** (real; first fully-trained ckpt — noisy 0.360 → real 0.302, 16% drop). Now running **ep1300-0.340** and **ep1350-0.340** (started 23:08, ETA ~23:45). 10 remaining → **~02:17 UTC May 19**.
- **2026-05-18 23:46**: 96/104. K=25 diversity ep1300→**0.308**, ep1350→**0.322** (real; scores climbing toward peak). Now running **ep1600-0.340** and **ep1650-0.380** (training best; started 23:45, ETA ~00:22 UTC May 19). These are the last 2 diversity ckpts. K=25 random (6 ckpts) follows. 8 remaining → **~02:13 UTC May 19**.
- **2026-05-19 00:25**: 98/104. **K=25 diversity eval COMPLETE** — best=**0.322** (ep1350), mean=0.218. All 9 ckpts: ep0000→0.000, ep0050→0.064, ep0100→0.154, ep0150→0.194, ep1150→0.302, ep1300→0.308, ep1350→**0.322**, ep1600→0.308, ep1650→0.314. Training best ep1650 (noisy 0.380) → real 0.314 — real peak was ep1350 at 0.322. **K=25 random eval STARTED**: ep0000-0.000 and ep0800-0.420 running (started 00:22, ETA ~00:59). K=25 random has 6 ckpts: ep0000, ep0800, ep0850, ep1300, ep1600, ep1650. 6 remaining → **~02:13 UTC May 19**.
- **2026-05-19 00:59**: 100/104. K=25 random ep0000→**0.000**, ep0800→**0.312** (real; noisy 0.420 → real 0.312, 26% drop). Now running **ep0850-0.380** and **ep1300-0.380** (started 00:59, ETA ~01:36). 4 remaining → **~02:13 UTC May 19**.
- **2026-05-19 01:31**: 102/104. K=25 random ep0850→**0.308**, ep1300→**0.314** (real). K=25 random real scores so far: ep0800→0.312, ep0850→0.308, ep1300→**0.314** (best). Now running **ep1600-0.380** and **ep1650-0.420** (2nd noisy champion, FINAL eval of sweep; started ~01:30, ETA ~02:07). 2 remaining → **~02:07 UTC May 19**.
- **2026-05-19 02:09**: 104/104. **K=25 random eval COMPLETE** — best=**0.344** (ep1650), mean=0.270. All 6 ckpts: ep0000→0.000, ep0800→0.312, ep0850→0.308, ep1300→0.314, ep1600→0.342, ep1650→**0.344**. Noisy champion ep1650-0.420 → real 0.344 (18% drop). The final checkpoint squeaked to the best real score despite noisy eval also peaking there. result.json written by `run_k_sweep_evals.py`. **K=25 REAL RESULTS: random 0.344 > diversity 0.322 > BG 0.200** — BG severely underperforms at K=25.
- **2026-05-19 02:09**: **ALL 104/104 CHECKPOINT EVALS COMPLETE.** `run_k_sweep_evals.py` finished. Total wall time: ~26.4 hours (23:44 UTC May 17 → 02:09 UTC May 19). See "Real Eval Results" section below for final table.

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
| `base_workspace.py:load_payload` | Compiled checkpoints save `model._orig_mod.*` keys; old code only stripped top-level `_orig_mod.X → X`, not nested `model._orig_mod.X → model.X`. Caused `Missing key(s): model.mid_modules...` + `Unexpected key(s): model._orig_mod.mid_modules...` on any resume after `compile=true` run. | → added `k.replace("._orig_mod.", ".")` pass before the existing top-level strip |
| `train_on_combined_data.py:compute` | All K values shared one training output dir; K=5 trained 1751 epochs, K=10+ saw `self.epoch==1751` with `num_epochs=1751` → `num_epochs_to_run=0` → exited immediately reusing K=5's model verbatim → K sweep results meaningless for K>5. | → appended `-k{clustering_n_clusters}` to `train_name` when `clustering_n_clusters` is set |

## Notes

- `robosuite_task_zoo` import warning on every generation worker is non-fatal (module not installed in `mimicgen_torch2`; not needed for Square task).
- `"keep_failed": true` in log is MimicGen JSON config output, not an error.
- Generation success rate starts at 0% for initial trials; accumulates over 900 trials per arm. Low early success rate is normal for tight constraint.
- Step 2 runs K values sequentially (K=5 → K=10 → … → K=25). Each K runs Phase A then Phase B before moving to next K. Total arms: 5 K × 3 heuristics × 3 reps = 45 arms.
