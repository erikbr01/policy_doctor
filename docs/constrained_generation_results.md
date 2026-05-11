# MimicGen Constrained Generation — Results

**Setup:** Nut-only initial pose constraint (±50 mm x, ±200 mm y, ±90° z_rot relative to seed demo pose).
Peg left at full D1 randomisation. Everything else identical to the apr26 unconstrained experiments.

**Constraint config** (added to `mimicgen_datagen`):
```yaml
fix_initial_object_poses: true
object_pose_ranges:
  nut:
    x: [-0.05, 0.05]
    y: [-0.2, 0.2]
    z_rot: [-1.57, 1.57]
  peg:
    x: null   # D1 full range
    y: null
    z_rot: null
```

**Experiment configs:**
- `mimicgen_square_rep_sweep_apr26_d60_nut_constrained` — 60-demo baseline, budget=100, 3 reps
- `mimicgen_square_rep_sweep_apr26_d60_budget300_nut_constrained` — 60-demo baseline, budget=300, 3 reps
- `mimicgen_square_rep_sweep_apr26_d300_nut_constrained` — 300-demo baseline, budgets [100, 500], 3 reps

**Run dirs** (upstream `run_clustering` symlinked from original seed=1 runs):
- `data/pipeline_runs/mimicgen_square_apr26_seed1_d60_nut_constrained/`
- `data/pipeline_runs/mimicgen_square_apr26_seed1_d60_budget300_nut_constrained/`
- `data/pipeline_runs/mimicgen_square_apr26_seed1_d300_nut_constrained/`

---

## D60 Baseline — Budget 100, Nut Constrained

**Launch:** 2026-05-07  
**Log:** `logs/budget_rep_sweep_d60_nut_constrained.log`  
**Run dir:** `mimicgen_square_apr26_seed1_d60_nut_constrained`  
**Arms:** 3 heuristics × 1 budget × 3 reps = 9 arms  
**Devices:** 2 concurrent slots on cuda:0  

### Arm status

| arm | rep | status | gen success% | train | eval (best) | eval (mean) |
|-----|-----|--------|-------------|-------|------------|------------|
| random_budget100 | 1 (null) | **done** | 19.9% (101/508) | done | **0.342** | 0.314 |
| behavior_graph_budget100 | 1 (null) | **done** | 52.9% (127/240) | done | **0.416** | 0.396 |
| diversity_budget100 | 1 (null) | **done** | 32.6% (100/307) | done | **0.396** | 0.374 |
| random_budget100 | 2 (seed=1) | **done** | — | done | **0.316** | 0.296 |
| behavior_graph_budget100 | 2 (seed=1) | **done** | — | done | **0.416** | 0.386 |
| diversity_budget100 | 2 (seed=1) | **done** | — | done | **0.396** | 0.377 |
| random_budget100 | 3 (seed=2) | **done** | — | done | **0.392** | 0.374 |
| behavior_graph_budget100 | 3 (seed=2) | **done** | — | done | **0.434** | 0.399 |
| diversity_budget100 | 3 (seed=2) | **done** | — | done | **0.390** | 0.382 |

### Results (best_success_rate, n=3)

| heuristic | mean ± std | min | max |
|-----------|-----------|-----|-----|
| random | 0.350 ± 0.039 | 0.316 | 0.392 |
| behavior_graph | 0.422 ± 0.010 | 0.416 | 0.434 |
| diversity | 0.394 ± 0.003 | 0.390 | 0.396 |

### Unconstrained d60 b=100 reference (from apr26 rep sweep)

| heuristic | mean ± std (ddof=1) | best |
|-----------|-------------------|------|
| random | 0.533 ± 0.029 | 0.558 |
| behavior_graph | 0.557 ± 0.059 | 0.621 |
| diversity | 0.561 ± 0.083 | 0.636 |

---

## D60 Baseline — Budget 300, Nut Constrained

**Launch:** 2026-05-08 01:33 (PID=2036819)  
**Log:** `logs/budget_rep_sweep_d60_budget300_nut_constrained.log`
**Run dir:** `mimicgen_square_apr26_seed1_d60_budget300_nut_constrained`
**Arms:** 3 heuristics × 1 budget × 3 reps = 9 arms
**Devices:** 2 concurrent slots on cuda:0

### Arm status

| arm | rep | status | gen success% | train | eval (best) | eval (mean) |
|-----|-----|--------|-------------|-------|------------|------------|
| random_budget300 | 1 (null) | **done** | 25.6% (302/1180) | done | **0.490** | 0.470 |
| behavior_graph_budget300 | 1 (null) | **done** | 49.7% (313/630) | done | **0.560** | 0.534 |
| diversity_budget300 | 1 (null) | **done** | 40.0% (324/810) | done | **0.610** | 0.578 |
| random_budget300 | 2 (seed=1) | **done** | 24.7% (305/1233) | done | **0.480** | 0.468 |
| behavior_graph_budget300 | 2 (seed=1) | **done** | 38.3% (310/810) | done | **0.618** | 0.576 |
| diversity_budget300 | 2 (seed=1) | **done** | — | done | **0.594** | 0.573 |
| random_budget300 | 3 (seed=2) | **done** | 40.8% (331/810) | done | **0.530** | 0.514 |
| behavior_graph_budget300 | 3 (seed=2) | **done** | 58.3% (315/540) | done | **0.552** | 0.527 |
| diversity_budget300 | 3 (seed=2) | **done** | — | done | **0.624** | 0.606 |

### Results (best_success_rate, n=3)

| heuristic | mean ± std | min | max |
|-----------|-----------|-----|-----|
| random | 0.500 ± 0.026 | 0.480 | 0.530 |
| behavior_graph | 0.577 ± 0.036 | 0.552 | 0.618 |
| diversity | 0.609 ± 0.015 | 0.594 | 0.624 |

### Reference: D60 budget=100 nut-constrained (this session)

| heuristic | mean ± std (ddof=1) | best |
|-----------|-------------------|------|
| random | 0.350 ± 0.039 | 0.392 |
| behavior_graph | 0.422 ± 0.010 | 0.434 |
| diversity | 0.394 ± 0.003 | 0.396 |

---

## Per-seed analysis — D60 b=300, BG vs random (n=3 reps, 500 eval seeds)

Each eval seed deterministically fixes the initial object poses. Aggregating 3 independent
policy variants (each trained on a different MimicGen draw) across the same 500 seeds gives
a per-seed success count in {0, 1, 2, 3}.

**Note:** pooled mean = mean of means exactly when n per rep is balanced (500), so there is
no gain from pooling at the point-estimate level. The interest is in the shape of the
per-seed distribution.

### Per-seed success count distribution

| succeed / 3 reps | random | behavior_graph |
|-----------------|--------|----------------|
| 0/3 (all fail)  | 106 (21.2%) | 83 (16.6%) |
| 1/3             | 135 (27.0%) | 120 (24.0%) |
| 2/3             | 162 (32.4%) | 146 (29.2%) |
| 3/3 (all win)   |  97 (19.4%) | **151 (30.2%)** |

### Per-seed pairwise comparison

| | seeds |
|--|-------|
| BG strictly higher than random | 195 (39.0%) |
| random strictly higher than BG | 115 (23.0%) |
| tied | 190 (38.0%) |

### Significance tests (Welch's t-test across 3 rep means)

```
random:    [0.490, 0.480, 0.530]  mean=0.500 std=0.026
BG:        [0.560, 0.618, 0.552]  mean=0.577 std=0.036
diversity: [0.610, 0.594, 0.624]  mean=0.609 std=0.015

BG vs random:        t=2.97, p=0.046  ✓ significant
diversity vs random: t=6.23, p=0.007  ✓ significant
diversity vs BG:     t=1.45, p=0.253  ✗ not significant
```

### Interpretation

BG's +7.7 pp mean advantage is concentrated almost entirely in the **always-win bucket**:
BG has 151 seeds solved by all 3 policy variants vs random's 97 (+54 seeds, +10.8 pp).
The always-fail rate shrinks only modestly (83 vs 106, −4.6 pp).

This means BG is not winning by being marginally better everywhere — it is converting more
initial configurations into ones that its policies can handle robustly across independent
training runs. The nut constraint narrows the generation problem enough that BG's informed
seed selection (high-probability path through the behavior graph) translates into meaningfully
broader and more reliable coverage at eval time.

The difference is statistically significant at p≈0.04 (Welch's t-test, n=3 reps), which is
notable given the conservative df≈4 correction.

**Caveat:** with only 3 binary trials per seed, individual per-seed success counts are noisy
(a 50%-success seed fails all 3 with probability 0.125). The aggregate distributional shift
is real; per-seed labels are not.

---

## D300 Baseline — Budget 300 + 1000, Nut Constrained

**Launch:** 2026-05-08 18:35 (original); restarted after power outage (PID=4055)
**Log:** `logs/budget_rep_sweep_d300_nut_constrained.log`
**Run dir:** `mimicgen_square_apr26_seed1_d300_nut_constrained`
**Arms:** 3 heuristics × 2 budgets × 3 reps = 18 arms
**Devices:** 4 concurrent slots (cuda:0 ×2, cuda:1 ×2)

### Arm status

| arm | rep | status | gen success% | train | eval (best) | eval (mean) |
|-----|-----|--------|-------------|-------|------------|------------|
| random_budget300 | 1 (null) | **done** | 24.3% (306/1260) | done | **0.450** | 0.310 |
| behavior_graph_budget300 | 1 (null) | **done** | 55.7% (351/630) | done | **0.584** | 0.527 |
| diversity_budget300 | 1 (null) | **done** | 37.8% (306/810) | done | **0.576** | 0.558 |
| random_budget1000 | 1 (null) | **done** | 21.6% (1001/4629) | done | **0.656** | 0.644 |
| behavior_graph_budget1000 | 1 (null) | **done** | 57.0% (1197/2100) | done | **0.622** | 0.615 |
| diversity_budget1000 | 1 (null) | **done** | 39.6% (1068/2700) | done | **0.712** | 0.681 |
| random_budget300 | 2 (seed=1) | **done** | 37.4% (303/810) | done | **0.568** | 0.555 |
| behavior_graph_budget300 | 2 (seed=1) | **done** | — | done | **0.596** | 0.578 |
| diversity_budget300 | 2 (seed=1) | **done** | — | done | **0.480** | 0.452 |
| random_budget1000 | 2 (seed=1) | **done** | 37.6% (1015/2700) | done | **0.714** | 0.699 |
| behavior_graph_budget1000 | 2 (seed=1) | training (ep 1500/1751) | — | — | — | — |
| diversity_budget1000 | 2 (seed=1) | generating | — | — | — | — |
| random_budget300 | 3 (seed=2) | **done** | 25.3% (300/1188) | done | **0.524** | 0.512 |
| behavior_graph_budget300 | 3 (seed=2) | **done** | — | done | **0.566** | 0.542 |
| diversity_budget300 | 3 (seed=2) | **done** (manual eval) | — | done | **0.578** | 0.558 |
| random_budget1000 | 3 (seed=2) | **done** | 25.2% (1015/4035) | done | **0.686** | 0.669 |
| behavior_graph_budget1000 | 3 (seed=2) | training (ep 1451/1751) | — | — | — | — |
| diversity_budget1000 | 3 (seed=2) | generating | — | — | — | — |

### Phase A results (rep-1 / null seed only)

| heuristic | budget | best | mean |
|-----------|--------|------|------|
| random | 300 | 0.450 | 0.310 |
| behavior_graph | 300 | 0.584 | 0.527 |
| diversity | 300 | 0.576 | 0.558 |
| random | 1000 | 0.656 | 0.644 |
| behavior_graph | 1000 | 0.622 | 0.615 |
| **diversity** | **1000** | **0.712** | **0.681** |

Budget effect (b300→b1000): random +0.206, diversity +0.136, BG +0.038.
Ordering flips: BG leads at b300; diversity leads at b1000 by a wide margin.
diversity-b1000 (0.712) is the highest single-arm result across all experiments in this session.

### Results (best_success_rate, n=3 — pending Phase B)

| heuristic | budget | mean ± std | min | max |
|-----------|--------|-----------|-----|-----|
| random | 300 | 0.514 ± 0.060 | 0.450 | 0.568 |
| behavior_graph | 300 | 0.582 ± 0.015 | 0.566 | 0.596 |
| diversity | 300 | 0.545 ± 0.056 | 0.480 | 0.578 |
| random | 1000 | 0.685 ± 0.029 | 0.656 | 0.714 |
| behavior_graph | 1000 | 0.630 ± 0.019 | 0.616 | 0.652 |
| diversity | 1000 | 0.691 ± 0.050 | 0.634 | 0.726 |

---

## Notes / Incidents

- 2026-05-07: D60 b=100 nut-constrained run launched. Watching for failures.
- 2026-05-07 18:23: Phase A complete. diversity rep-1 best=0.396/mean=0.374. Phase B auto-started (random rep-2/3 generating).
- 2026-05-07 20:43: random rep-3 (seed=2) DONE best=0.392/mean=0.374. random rep-2 (seed=1) eval running. BG rep-2 (seed=1) generating.
- 2026-05-07 20:59: random all 3 reps done. rep-2 best=0.316/mean=0.296. BG rep-2/3 both training (ep 0, 150).
- 2026-05-07 23:06: BG rep-3 (seed=2) DONE best=0.416/mean=0.386. BG rep-2 eval running (5/5 ckpts). diversity rep-2 generating.
- 2026-05-07 23:11: BG all 3 reps done. rep-2 best=0.416, rep-3 best=0.434. BG mean=0.422±0.010. Both diversity rep-2/3 generating.
- 2026-05-08 ~01:31: D60 b=100 nut-constrained **COMPLETE**. diversity rep-2 best=0.396/mean=0.377, rep-3 best=0.390/mean=0.382. diversity mean=0.394±0.003. Final summary: random=0.350±0.039, BG=0.422±0.010, diversity=0.394±0.003. Budget=300 run launched.
- 2026-05-08 07:52: D60 b=300 **Phase A complete**. random=0.490/0.470, BG=0.560/0.534, diversity=0.610/0.578. Diversity leads at budget=300 — reversal from b=100 where diversity was lowest. Phase B auto-started (random rep-2/3 generating).
- 2026-05-08 12:27: random Phase B both reps done. rep-2 best=0.480, rep-3 best=0.530. random mean=0.500±0.026. BG rep-2/3 both training (ep 600, 250). diversity waiting.
- 2026-05-08 14:02: BG rep-2 (seed=1) training DONE (ep 1751), eval started on cuda:0 (ckpt 1/5). BG rep-3 (seed=2) training at ep 1589/1751, ~10 min remaining. diversity reps waiting for BG slots.
- 2026-05-08 14:34: BG rep-2 eval 4/5 ckpts done (partial scores: ep650=0.530, ep700=0.552, ep1000=0.592, ep1100=0.586; running ep1450). BG rep-3 training DONE, eval started (ckpt 1/5: ep300=0.458, running ep700). Diversity still pending.
- 2026-05-08 15:01: BG rep-2 DONE best=0.618/mean=0.576. BG rep-3 eval 4/5 done (ep300=0.458, ep700=0.544, ep850=0.544, ep1400=0.552; running ep1600). Diversity rep-2 generating (7/10 seeds active). Diversity rep-3 still pending.
- 2026-05-08 15:43: BG rep-3 DONE best=0.552/mean=0.527. **BG complete: mean=0.577±0.036 (n=3: 0.560/0.618/0.552)**. diversity rep-2 training ep475 (in-train peak 0.480), rep-3 training ep203 (in-train peak 0.420). Est. Phase B done ~18:30.
- 2026-05-08 17:29: diversity rep-2 training DONE (ep1751, in-train peak=0.640), eval started (ckpt 2/5: ep950=0.570, running ep1300). diversity rep-3 training ep1614/1751 (in-train peak=0.640), ~10 min remaining.
- 2026-05-10 22:51: diversity-b300-rep2 ep238 (49K log lines, confirmed progressing). BG-b1000 last ckpts ep1100/ep1000 (log ep1250/1219, ~500 epochs left, ~3.5h at 3-way GPU share). diversity-b1000 rep1 gen 868/1000, rep2 gen 261/1000. Phase B est. done ~08:00 May 11.
- 2026-05-11 22:46: **D300 PHASE B COMPLETE.** div-b1000 rep2 DONE best=0.726/mean=0.713. **div-b1000 n=3: mean=0.691±0.050 (0.712/0.634/0.726)**. Full d300 results: random b300=0.514/b1000=0.685, BG b300=0.582/b1000=0.630, diversity b300=0.545/b1000=0.691. Budget=1000 strongly benefits all heuristics; diversity b1000 leads.
- 2026-05-11 22:26: rep1 DONE. rep2 4/5 done (ep0900=0.716, ep1250=0.716, ep1600=0.702, ep1700=0.706). ep1200 relaunched on cuda:0 (PID=1897155) — was dying on cuda:1. Final eval ~22:45.
- 2026-05-11 20:22: rep1+rep2 both training done. 10 eval ckpts running (5 per rep, 320 processes, no training competition). GPU 0: 2.5GiB / GPU 1: 3.2GiB. Eval est. done ~21:30.
- 2026-05-11 13:08: rep1 (cuda:0 solo) at ep1079 — torch.compile recompiling, only 21 ep/hr (steps incrementing ✓). rep2 (cuda:1 solo) at ep946, now running 94 ep/hr → done ~21:30. Rep1 should accelerate after compile; ~2h left if it reaches ~300 ep/hr. Bottleneck: rep2 ~21:30.
- 2026-05-11 11:08: div-b1000 rep1 moved to cuda:0 solo (PID=1791074, ep1037 resuming) to cut ~16h→~2.5h. rep2 stays on cuda:1 solo (ep852, ~10h). Phase B est done ~21:00 May 11.
- 2026-05-11 10:08: **BG-b1000 rep1 DONE** best=0.652. **BG-b1000 rep2 DONE** best=0.616. **BG-b1000 n=3 mean=0.630±0.019**. **div-b300 n=3 mean=0.545±0.056** (Phase A=0.576, rep1=0.480, rep2=0.578). div-b1000 ep992/809 (still training, rep2 in-train peak=0.740!).
- 2026-05-11 09:10: **div-b300-rep2 DONE** best=0.578/mean=0.558 (manual eval). BG-b1000 rep1 eval 4/5 (peaks 0.652, running ep1700). rep2 eval 3/4 (peak 0.608). div-b1000 ep800/626 (38 ep/hr).
- 2026-05-11 08:10: BG-b1000 rep1 eval 4/5 done. rep2 eval 3/4 done. div-b300-rep2 eval still running (all 5 procs alive, GPU 0 loaded 3477MiB but 0% util — 126 concurrent sim instances CPU-bound). div-b1000 ep784/602 (38 ep/hr in rollout window).
- 2026-05-11 06:08: div-b300-rep2 training DONE (ep1750, peak=0.640), 5 eval processes launched manually (PIDs 1705748-52). BG-b1000 rep1 eval 3/5 (0.652/0.632/0.582, running ep1650). rep2 eval started (ckpt 1/5). div-b1000 ep642/460 (70/69 ep/hr).
- 2026-05-11 05:43: BG-b1000 rep2 training DONE (ep1750). rep1 eval 2/5 (ep1250=0.652, ep1300=0.632, running ep1400). div-b300-rep2 ep1695 (~56 left, done ~06:10, manual eval needed). div-b1000 ep642/460 (70/69 ep/hr → done ~21:45/00:40). RAM 85/124GiB fine, GPU contention slowing rates but stable.
- 2026-05-11 03:40: BG-b1000 rep1 eval running (still ckpt 1/5 — sim contention from training rollout evals). rep2 ep1706 (~11 ep/hr in rollout window). div-b300-rep2 ep1466. div-b1000 ep545/364 (also slow, ~23 ep/hr). Rates oscillate between fast (187+/hr) and slow (11-37/hr) windows every ~50 epochs.
- 2026-05-11 03:03: **Rates recovered**: BG-b1000 rep1 training DONE (ep1750, peak=0.700), eval running (ckpt 1/5). rep2 ep1699 (~18 min left). div-b300-rep2 ep1443 (~36 min left, manual eval needed after). div-b1000 ep531/350 (250/243 ep/hr → done ~08:00/08:50). Phase B est. done ~09:15 May 11.
- 2026-05-11 01:57: BG-b1000 ep1544/1493 (27/24 ep/hr, ~8-11h left). div-b300-rep2 ep907 (54 ep/hr, ~15h but will accelerate after BG done). div-b1000 ep256/83 (~28-31 ep/hr; after BG free, rates should improve 2-3×). Phase B realistic est: div-b1000 done ~May 12 11:00.
- 2026-05-11 00:54: 5-way GPU contention (3 on cuda:0, 2 on cuda:1) causing rollout-eval bottleneck. BG-b1000 ep1516/1468 (~18 ep/hr in eval-heavy window). div-b300-rep2 ep850, div-b1000 ep227/50. Conservative est: BG-b1000 done ~05:00, div-b300-rep2 done ~08:00 (after BG frees cuda:0), div-b1000 done ~19:00-21:00. Phase B est. done ~21:00 May 11.
- 2026-05-11 00:01: **REVISED ETA (much better)**: diversity-b1000 rep1/2 gen done, both training (ep206/33). BG-b1000 reps at ep1500/1451 (208/188 ep/hr). diversity-b300-rep2 ep820 (488 ep/hr). All 4 GPU slots (cuda:0×2, cuda:1×2) now fully utilized. BG-b1000 done ~01:30, diversity-b300-rep2 ~01:55 (manual eval needed), diversity-b1000 reps ~09:00. Phase B est. done ~09:00 May 11.
- 2026-05-10 23:55: **REVISED ETA**: BG-b1000 reps at ep1278/1250 (~28 ep/hr due to 3-way cuda:0 sharing with diversity-b300-rep2). 473/501 epochs left = ~17h each → done ~17:00 May 11. diversity-b300-rep2 ep300 (62 ep/hr, smaller dataset), ~23h left. diversity-b1000 rep1 gen 883/1000, rep2 gen 487/1000. Phase B est. done ~22:00 May 11.
- 2026-05-10 21:59: diversity-b300-rep2 training on cuda:0 ep201 (working!). BG-b1000 rep1/2 ep1237/1208 (peaks 0.600/0.680). diversity-b1000 rep1/2 both generating. 12/12 Phase B arms active/done.
- 2026-05-10 21:10: diversity-b300-rep2 killed and manually relaunched on cuda:0 (PID=1517780) — was hanging on cuda:1 since 17:47. BG-b1000 rep1/2 training ep903/906. diversity-b1000 rep1 generating.
- 2026-05-10 20:46: diversity-b300 rep1 DONE best=0.480/mean=0.452. diversity-b1000 rep1 generating (140/1000). BG-b1000 rep1/2 training ep903/906. diversity-b300 rep2 STUCK (train process hung on cuda:1 since 17:47, 0 progress). diversity-b1000 rep2 pending.
- 2026-05-10 19:41: All 4 training simultaneously (2 per GPU slowing pace). BG-b1000 rep1/2 at ep~596/623 (logs buffered). diversity-b300 rep1 near end (~ep1669→1751), rep2 training (logs buffered at ep0 but process alive). diversity-b1000 pending. Phase B est. done ~03:00 May 11.
- 2026-05-10 18:39: BG-b1000 rep1/2 gen done, training ep249/303 (peak 0.460/0.620). diversity-b300 rep1 gen done training ep800 (peak 0.500), rep2 training ep0. diversity-b1000 pending BG-b1000 slots (~21:30). Phase B est. done ~02:00 May 11.
- 2026-05-10 17:37: random-b1000 DONE: rep1=0.714/0.699, rep2=0.686/0.669. **random-b1000 n=3 mean=0.685±0.029**. BG-b1000 rep2 training ep0, rep1 still generating. diversity-b300 rep1/2 generating. diversity-b1000 pending.
- 2026-05-10 16:51: BG-b300 rep1/2 DONE (0.596/0.566). **BG-b300 n=3 mean=0.582±0.015**. random-b1000 rep1/2 evals running (in-train peaks 0.800/0.720). BG-b1000 rep1/2 generating. diversity-b300/b1000 pending slots.
- 2026-05-10 15:49: BG-b300 rep1/2 training done, evals 4/5 and 3/5 (partial peaks 0.596/0.566). random-b1000 rep1/2 training ep1572/1625 (~10-12 min left). All 4 slots free ~16:15 → BG-b1000 + diversity-b300 all start.
- 2026-05-10 14:47: All 4 arms ~20-43 min from training done. BG-b300 ep1465/1388 (peaks 0.640/0.580). random-b1000 ep1318/1334 (peaks 0.760/0.720). All 4 slots free ~15:45 → BG-b1000 + diversity-b300 all start. Phase B est. done ~00:30 May 11.
- 2026-05-10 13:45: BG-b300 rep1/2 gen done, training ep710/673. random-b1000 rep1/2 training ep1033/1003 (peaks 0.760/0.720 — very strong). BG-b1000/diversity pending. Est. random-b1000 done ~15:10, BG-b1000 starts ~15:15, Phase B done ~01:00 May 11.
- 2026-05-10 12:43: random-b300 DONE: rep1=0.568/0.555, rep2=0.524/0.512. **random-b300 n=3 mean=0.514±0.060** (vs Phase A 0.450). BG-b300 rep1/2 generating. random-b1000 rep1/2 training ep750/683 (peaks 0.760/0.640). BG-b1000/diversity pending.
- 2026-05-10 11:41: random-b300 rep1/2 training DONE (in-train peaks 0.700/0.540), evals running (ckpt 1/5 each). random-b1000 rep1/2 training ep490/383. BG arms start ~12:06 when b300 eval slots free. Phase B est. done ~23:00.
- 2026-05-10 10:39: All 4 random Phase B arms training. b300 rep1/2 at ep1200 (peaks 0.700/0.540). b1000 rep1/2 ep217/73. BG arms start when b300 reps free slots ~11:45. Revised est. Phase B done ~22:00.
- 2026-05-10 09:37: Phase B: random-b300 rep1/2 training (ep484/400), random-b1000 rep1/2 generating (724/527 of 1000). BG/diversity pending slots. Est. BG arms start ~11:30, Phase B done ~next morning.
- 2026-05-10 08:35: **D300 Phase A COMPLETE.** diversity-b1000 best=0.712/mean=0.681 — highest single arm in session. Phase B started: random_b300/b1000 rep1/2 (4 arms) generating. Full Phase A: random b300=0.450/b1000=0.656, BG b300=0.584/b1000=0.622, diversity b300=0.576/b1000=0.712.
- 2026-05-10 07:05: diversity-b1000 training DONE (ckpts: ep250=0.740, ep350=0.800, ep1000=0.800, ep1300=0.780, ep1350=0.760). Eval launching. Phase A completion ~07:30, Phase B to start immediately after.
- 2026-05-10 06:03: random-b1000 DONE best=0.656/mean=0.644 — large jump from b300 (0.450). diversity-b1000 training ep1071 (peak=0.800 sustained). Phase A 5/6 done; est. complete ~07:30.
- 2026-05-10 05:01: BG-b1000 DONE best=0.622/mean=0.615. diversity-b300 DONE best=0.576/mean=0.558. random-b1000 eval running (ckpt 1/5). diversity-b1000 training ep733 (peak=0.800). Phase A 5/6 done; est. complete ~07:00.
- 2026-05-10 03:59: BG-b1000 and diversity-b300 training DONE, evals running (1/5 ckpts each). random-b1000 training ep1460 (peak=0.720). diversity-b1000 training ep445 (peak=0.800 — notable). Phase A est. done ~06:00.
- 2026-05-10 02:57: All Phase A gen done. diversity-b1000 gen: 1068/2700 (39.6%). All 4 remaining arms training: BG-b1000 ep1497 (peak=0.660), diversity-b300 ep1207 (peak=0.660), random-b1000 ep1152 (peak=0.720), diversity-b1000 ep150 (peak=0.620). Phase A est. done ~05:30.
- 2026-05-10 01:55: BG-b300 DONE best=0.584/mean=0.527. **Both b300 arms done**: random=0.450, BG=0.584. BG-b1000 training ep1177 (peak=0.660). random-b1000 training ep836 (peak=0.680). diversity-b300 training ep467 (peak=0.580). diversity-b1000 generating (10/10 seeds). Phase A est. done ~07:15.
- 2026-05-09 00:53: D300 Phase A progress: random-b300 DONE best=0.450/mean=0.310. BG-b300 eval 9/10 (peak=0.564 at ep600). BG-b1000 training ep854, peak=0.660. random-b1000 training ep510, peak=0.680. diversity-b300 generating. diversity-b1000 waiting for BG-b300 slot (~01:03).
- 2026-05-08 18:27: **D60 b=300 nut-constrained COMPLETE.** diversity rep-2 best=0.594/mean=0.573, rep-3 best=0.624/mean=0.606. Final: random=0.500±0.026, BG=0.577±0.036, diversity=0.609±0.015. diversity > BG > random; diversity vs BG not significant (p=0.253), both vs random significant (p=0.046, p=0.007).
