# E1 Results Summary — `transport_mh` r512 / seed0

This document captures findings, methodological caveats, and queued follow-ups for Experiment E1 on the transport_mh r512x512 policy/rollouts/clustering. It is **separate from `experiment_e1_cluster_coherence.md`**, which describes how to run the experiment; this doc describes what the runs produced.

Last updated: 2026-05-04 (F10, F11 added)

---

## Headline numbers

All runs use Qwen3-VL-8B-Instruct (local, GPU-1 or GPU-0, bf16, greedy decoding) on clusterings of `transport_mh_seed0_r512` (8086 slices, 5-frame sliding windows, stride 2, kmeans). Default representation is InfEmbed unless noted.

⚠️ **UMAP dimension inconsistency** — F1–F10 and the original representation comparisons (F10, F13, F14 headline numbers) use **100D UMAP**. The policy_emb sweep (F14 variants), TRAK sweep, and window sweep (`run_window_sweep.py`) all use **50D UMAP** (set to stay below the 59D state feature ceiling). The within-sweep comparisons are self-consistent; cross-group comparisons (e.g. policy_emb vs original InfEmbed baseline) have this confound. A fresh **InfEmbed 50D UMAP baseline** is being run to anchor the comparison — see F15.

The E1 protocol classifies held-out rollout slices into K opaque cluster labels using K · n_example example storyboards in one prompt.

| Run | Repr | K | n_example | n_query | n_reps | Examples | Global disjoint | Headline acc | Clean (tier1_global) acc | Clean p (vs 1/K) |
|---|---|---|---|---|---|---|---|---|---|---|
| K=20 v1 | InfEmbed | 20 | 3 | 3 | 1 | random | no | 33.3% (20/60) | 28.6% (12/42) | 6.6e-7 |
| K=20 v2 | InfEmbed | 20 | 3 | 3 | 3 | centroid-proximal | yes | 31.7% (19/60) | 24.1% (13/54) | 1.9e-6 |
| K=15 v2 | InfEmbed | 15 | 3 | 3 | 3 | centroid-proximal | yes | 37.8% (17/45) | 30.8% (12/39) | 5.5e-6 |
| **K=10 v2** | **InfEmbed** | **10** | 3 | 3 | 3 | centroid-proximal | yes | **53.3% (16/30)** | **48.1% (13/27)** | **5.2e-7** |
| K=10 + 768² | InfEmbed | 10 | 3 | 3 | 3 | centroid-proximal | yes | 56.7% (17/30) | 51.9% (14/27) | 5.6e-8 |
| state K=10 + 768² | state | 10 | 3 | 3 | 3 | centroid-proximal | yes | 46.7% (14/30) | 46.7% (14/30)† | 3.1e-7 |
| state_action K=10 + 768² | state_action | 10 | 3 | 3 | 3 | centroid-proximal | yes | 40.0% (12/30) | 33.3% (9/27) | 8.7e-4 |
| policy_emb avg-t K=10 | policy_emb | 10 | 3 | 3 | 3 | centroid-proximal | yes | 46.7% (14/30) | 46.7% (14/30)† | 3.1e-7 |
| policy_emb t0 K=10 | policy_emb | 10 | 3 | 3 | 3 | centroid-proximal | yes | 40.0% (12/30) | 40.0% (12/30)† | 1.5e-5 |
| **policy_emb plan@t0 K=10** | **policy_emb** | **10** | 3 | 3 | 3 | centroid-proximal | yes | **63.3% (19/30)** | **59.3% (16/27)** | **4.4e-10** |

† All queries are tier1_global (no cross-cluster episode contamination), headline = clean.

Note: policy_emb uses w=5,s=2 (matching InfEmbed baseline). Clustering uses 50D UMAP on per-timestep features (window_first architecture). The `plan@t0` variant feeds the actual rollout action plan into the UNet at denoising step t=0; other variants use random noise.

"Clean" = queries in the `tier1_global` bucket — episode appears in NO cluster's example pool. This is the proper test; episode-confounded queries are excluded.

---

## Findings

### F1 — K is the dominant factor; cheap fixes were a wash at K=20

Holding the UMAP embedding fixed and only varying K (kmeans refit on the same 100D space), clean accuracy nearly **doubles** going from K=20 → K=10 (24% → 48%). The fraction of well-resolved clusters (≥ 2/3 correct) doubles too: K=20 = 25%, K=10 = 60%.

The cheap fixes — centroid-proximal example selection, n_repetitions=3 with majority vote, and the global episode-disjoint planner — combined did **not** raise the clean-bucket number at fixed K=20 (28.6% v1 → 24.1% v2, within noise). The improvement at K=10 came from reducing K, not from the fixes.

Implication for the paper: K=20 over-clusters this rollout pool. K=10 is closer to the visually-recoverable granularity. Downstream methods that depend on per-cluster crispness should use K ≤ 10 or be robust to over-splitting.

### F2 — Episode-level visual cues are a major confound

A first-pass disjointness check that only excluded queries whose episode appeared in the *same cluster's* examples missed cross-cluster contamination. With the global check (`tier1_global`) on the K=20 v1 data:
- 5/5 queries from the within_only bucket (episode in own examples only): 100% correct
- 0/9 queries from the cross_only bucket (episode in other clusters' examples): 0% correct
- 12/42 queries from the clean bucket: 28.6% correct

Both extremes (100% on within_only, 0% on cross_only) confirm the VLM exploits episode-level visual cues whenever possible. The corrected planner (`global_episode_disjoint=True`) blocks this in subsequent runs.

### F3 — Over-clustering signature is concentrated in K=20

At K=20 v2, 7 clusters are "concentrated" (0% correct, ≥ 2/3 misses on a single dominant target):
- Cluster 2 → cluster 6 (3/3)
- Cluster 18 → cluster 17 (3/3)
- Plus 5 others with strong but not unanimous targets

Plus 3 reciprocal pairs (3↔6, 4↔8, 10↔13). These are the over-splits that K=10 collapses.

### F4 — Same-episode "perfect" clusters are inflated

Across both runs, every "perfect" cluster (3/3 correct) was either same_episode-confounded or, at K=10 v2, the single-episode trapped cluster 7. The single clean perfect cluster across all v2 runs is K=20 v2 cluster 5. 5-frame visual ambiguity makes "perfect" outcomes rare on episode-disjoint data.

### F5 — VLM scale (8B → 32B-NF4) doesn't move the needle

Re-ran the K=10 v2 evaluation with **Qwen3-VL-32B at 4-bit NF4** (device_map=auto across both GPUs). Same sample plan, same prompt structure, only the model differs.

| Run | Headline | Clean (tier1_global) |
|---|---|---|
| Qwen3-VL-8B K=10 v2 | 0.533 | **0.481** |
| Qwen3-VL-32B-NF4 K=10 v2 | 0.533 | **0.481** |

Identical aggregate accuracy. Per-query, the two models disagree on **15 of 30** classifications — they're producing structurally different decisions but happen to both score 16/30 correct. Both runs have 1 perfect cluster (the same-episode cluster 7), 5 mostly_correct, 3 mixed, 1 diffuse — though *which* cluster is diffuse differs (8B: cluster 6; 32B: cluster 2).

**Implication:** the gap from 48% clean → ~100% is **not** primarily a VLM-capacity bottleneck. A 4× parameter scale-up at NF4 doesn't help. Pursuing larger frontier VLMs (Gemini 3, Claude Opus) is unlikely to produce dramatic gains either, unless they exceed Qwen-32B by a much larger margin.

The bottleneck must therefore be elsewhere — see F6.

### F6 — Visual context: image *resolution* per frame matters more than frame *count* or temporal *span*

Two-axis sweep on the K=10 v2 baseline (n_query=3, n_reps=3, global-disjoint, n=27 clean queries) varying `view_window_extension` (how far outside the cluster window to sample frames) and `max_frames_per_storyboard` (4 frames in a 2×2 grid vs 9 frames in a 3×3 grid, with the composite output fixed at 512×512).

| | mf=4 ext=0 | mf=4 ext=5 | mf=4 ext=10 | mf=4 ext=15 | mf=9 ext=0 | mf=9 ext=5 | mf=9 ext=10 | mf=9 ext=15 |
|---|---|---|---|---|---|---|---|---|
| Headline | 0.533 | 0.500 | 0.267 | 0.433 | 0.400 | 0.467 | (rerun pending) | 0.400 |
| Clean | **0.481** | 0.444 | 0.185 | 0.370 | 0.333 | 0.407 | — | 0.333 |
| Clean p (vs 0.10) | 5.2e-7 | 4.1e-6 | 0.13 | 1.7e-4 | — | — | — | — |

The baseline (mf=4, ext=0) is the winner. Two findings:

**F6a — Frame *count* helps only when frame *density* helps.** With max_frames_per_storyboard fixed and `view_window_extension > 0`, the 4 sampled frames are spaced uniformly across the wider window. With the 5-timestep cluster window centered, this means:
- ext=0: 4 frames at fractions [0, 1/3, 2/3, 1] of a 5-step span → all 4 inside the cluster.
- ext=5: 4 frames at [0, 5, 10, 15] of a 15-step span → only 1 frame inside.
- ext=10: 4 frames at [0, 8.3, 16.7, 25] of a 25-step span → **none** inside (collapse to 0.185, p=0.13 — not significant).
- ext=15: 4 frames at [0, 11.7, 23.3, 35] of a 35-step span → 1 happens to land in the cluster window again, partial recovery.

The dip at ext=10 followed by recovery at ext=15 is a sampling artifact, not a real "longer context is bad" finding.

**F6b — More frames at lower per-cell resolution hurts.** mf=9 (3×3 grid → ~170×170 px per cell at the fixed 512×512 composite) is uniformly worse than mf=4 (2×2 → ~256×256 per cell) at every comparable extension. The VLM gains visible motion-sequence information from more frames but loses the per-frame resolution needed to identify what's happening.

**F6c — But density-matched extension does help.** mf=9 ext=5 (0.407) beats mf=9 ext=0 (0.333). When frame density across the visual window is matched to the baseline (~0.6 frames/timestep), longer context recovers some accuracy — just not enough to overcome the resolution loss from a 3×3 grid.

**The right experimental knob is composite resolution, not frame count.** Bumping `make_storyboard` `target_size` from 512² to e.g. 1536² lets a 3×3 grid keep ~512×512 per cell. That separates "more frames" from "smaller frames" — the right test for whether visual context length is the actual bottleneck. **Not yet run** (needs the storyboard module to accept a target_size override; current default is hard-coded).

### F7 — Headline conclusion: the bottleneck is local-frame information density, not interpreter capacity

Combining F5 and F6:
- VLM scale doesn't help (F5).
- Visual extension at fixed frame count doesn't help, and *can hurt* due to sampling-position artifacts (F6a).
- More frames doesn't help when it costs per-frame resolution (F6b).
- Density-matched extension helps slightly when frame count is held (F6c).

The current configuration (mf=4 ext=0 at composite 512², image_max_pixels=1024²) appears close to a local optimum given the storyboard format. To break past 48% clean, the next moves are: (a) raise composite resolution so 3×3 / 4×4 grids stay readable, (b) feed numeric per-timestep state/action alongside images (implemented; see commit `6391f0c`, untested), and/or (c) revisit the clustering itself (representation, window width — see Q4/Q5 below).

### F8 — Visual context format sweep at K=10: bigger cells help, asymmetric formatting hurts; K=5 confirms format isn't the bottleneck

Tested three storyboard formats against the K=10 v2 baseline (composite mf=4 target=512², cells=256²), each holding sample plan, model, and seed fixed.

| Config | Headline | Clean (n=27) | p (vs 1/10) | Notes |
|---|---|---|---|---|
| baseline (composite 512², cells 256²) | 0.533 | 0.481 | 5.2e-7 | reference |
| **(C) larger composite (768², cells 384²)** | **0.567** | **0.519** | **5.6e-8** | +1 correct vs baseline; same image count |
| (A) hybrid (composite ex + frames query, mf=4) | 0.200 | 0.222 | 4.7e-2 | drops by half — VLM treats query frames as more examples |
| (B) pure frames (mf=4, image_max_pixels=147k) | OOM | — | — | 124 images per call exceeds 24GB activation memory |

**F8a — Larger composites give a small but consistent improvement; 768² is the VRAM ceiling.** Config (C) is the new best at K=10 v2, +0.04 clean accuracy over baseline at zero token cost. Confirms F6b (per-cell resolution matters) but the magnitude of the gain is modest — the gap to ceiling is mostly elsewhere. A follow-up attempt at 1024² (`composite_target_size=1024`, `image_max_pixels=1048576`) OOMed during the first vision-encoder forward pass on the 24GB GPU — the larger patch grid fills all 24GB at K=10/n_example=3. 768² is therefore not an arbitrary choice but the empirical VRAM ceiling for this config.

**F8b — Asymmetric multimodal formatting causes major regression.** Config (A) drops clean accuracy to 0.222 (almost halving from 0.481). Same image content as (B) at the visual level but split across formats: 30 example *composites* (one per example slice) followed by 4 query *frames* (one per timestep). The "Query:" text label between them isn't a strong enough boundary — the VLM apparently reads the 4 query frames as 4 more example images, fitting into whichever group's visual style they best match. Even the same-episode bucket goes 0/3 (vs baseline's 3/3 — episode cues stop helping when the prompt structure is broken).

Methodological consequence: when introducing per-slice multi-image variants in the future, **all slices** should be in the same format. Mixed formats break VLM grouping inference.

**F8c — Pure frames mode doesn't fit at K=10 on a single 24GB GPU.** At K=10/n_example=3/mf=4, frames mode produces 124 images per call. Even with `image_max_pixels=147456` (≈384²/frame), the vision encoder's per-image position-embedding allocations cumulatively overflow 24GB. To test pure frames cleanly we have to drop to K=5 (max ~75 frames per call) — see Q1'' / F9.

### F9 — K=5 follow-up: pure frames format is *not* the bottleneck

Built a fresh K=5 clustering (`/tmp/transport_mh_seed0_r512_clustering_k5`) and ran two configs at K=5/n_example=3/n_query=3/n_reps=3, both fitting comfortably in context.

| Config (K=5) | Headline | Clean (n=15) | p (vs 0.20) | Ratio above chance |
|---|---|---|---|---|
| baseline (composite mf=4 at 512²) | 0.467 | **7/15 = 0.467** | 0.018 | 2.3× |
| pure frames mf=4 (native 512² each) | 0.400 | 6/15 = 0.400 | 0.061 (NS) | 2.0× |

**Pure frames does not beat composite, even at full per-frame native resolution and full context fit.** Composite wins by 1 correct out of 15 (within sampling noise, but pointing the same direction).

**The K curve, summarized:**

| K | Chance | Clean acc | Ratio above chance |
|---|---|---|---|
| K=20 v2 | 0.05 | 0.241 | 4.8× |
| K=15 v2 | 0.067 | 0.308 | 4.6× |
| **K=10 v2** | 0.10 | 0.481 | **4.8×** |
| K=10 + 768² composite | 0.10 | 0.519 | 5.2× |
| K=5 | 0.20 | 0.467 | 2.3× |
| K=5 frames | 0.20 | 0.400 | 2.0× |

The signal-to-chance ratio is **stable at 4.6–5.2× from K=15 down to K=10**, then *drops* to ~2× at K=5. K=10 looks like the sweet spot for visual recoverability — going coarser (K=5) merges genuinely distinct behaviors that the VLM can no longer disambiguate.

**Synthesis (F8 + F9):** the storyboard composite is *not* the bottleneck. Format alone (composite vs frames vs hybrid) doesn't move the needle once we control for image count and per-cell resolution. The remaining gap from ~50% clean → ceiling is in the clustering itself or the slice content (5-frame windows + cluster-prototype ambiguity), not the prompt format. Larger composite cells (F8a) give a small honest gain (+0.04 clean); everything else moves it sideways or down.

### F11 — Resolution / n_example tradeoff: 768² remains the practical ceiling; trading examples for resolution is a net loss

Full sweep across K ∈ {5, 10, 15, 20}, n_example ∈ {2, 3}, composite_target_size ∈ {512², 768², 1024², 1536²}. All InfEmbed, GPU 0, seed 42, n_query=3, n_reps=3, global_episode_disjoint.

| K | n_ex | 512² clean | 768² clean | 1024² clean | 1536² clean |
|---|---|---|---|---|---|
| 5 | 3 | 0.467 (7/15) [ref] | 0.533 (8/15) | 0.400 (6/15) | OOM |
| 10 | 3 | 0.481 (13/27) [ref] | **0.519 (14/27)** [ref] | OOM | OOM |
| 10 | 2 | 0.407 (11/27) | 0.444 (12/27) | OOM | OOM |
| 15 | 3 | 0.308 (12/39) [ref] | OOM | OOM | OOM |
| 15 | 2 | 0.333 (13/39) | 0.385 (15/39) | OOM | OOM |
| 20 | 3 | 0.241 (13/54) [ref] | OOM | OOM | OOM |
| 20 | 2 | 0.241 (13/54) | OOM | OOM | OOM |

**F11a — Trading n_example=3 → n_example=2 to fit higher resolution is a net loss at K=10.** The resolution gain from 512² → 768² is +4pt clean (0.481 → 0.519 at n_ex=3). Dropping to n_ex=2 costs ~7pt at each resolution (512²: 0.481→0.407; 768²: 0.519→0.444). You cannot recover the example-count penalty with resolution, at least in the n_ex ∈ {2, 3} range.

**F11b — At K=15 and K=20, n_example=2 ≈ n_example=3.** K=15: n_ex=3 512²=0.308, n_ex=2 512²=0.333 (+1 query). K=20: both 0.241 (identical 13/54). At high K the limiting factor is cluster separability, not how many example storyboards the VLM sees. This has a useful implication: for K≥15, using n_ex=2 recovers VRAM headroom (31→21 images/call at K=15) at essentially no accuracy cost, enabling 768² at K=15.

**F11c — 768² at K=15 n_ex=2 = 0.385 clean** — the best K=15 result we have, +8pt over the n_ex=3 512² baseline (0.308). The gain comes partly from resolution and partly from the 30→39 clean-query difference between n_ex=2 (no episode contamination) and n_ex=3 (some contamination at K=15), so it isn't purely a resolution effect.

**F11d — VRAM ceiling map (empirical, 24GB GPU):**

| Composite | Max images/call | Fits if… |
|---|---|---|
| 512² | unlimited | always |
| 768² | ~30 | K·n_ex ≤ 29 (e.g. K=10/n_ex=3 ✓, K=15/n_ex=2 ✓, K=20/n_ex=2 ✗) |
| 1024² | ~16 | K·n_ex ≤ 15 (e.g. K=5/n_ex=3 ✓, K=10/n_ex=2 ✗) — LLM KV cache |
| 1536² | 0 | never — single-image vision encoder OOM |

OOM at 1024²+ (LLM decoder) has 1.4–1.6 GB reserved-but-unallocated; `expandable_segments=True` might squeeze through a few more images, but this is not worth pursuing — the gain over 768² at K=5 is already negative (0.533→0.400).

**F11e — Resolution helps monotonically at K=5, but needs n≥100 to see it.** The n_query=3 (n=15 clean) runs showed a non-monotonic 512²=0.467 → 768²=0.533 → 1024²=0.400, which appeared to disfavor 1024². Re-running at n_query=20 (n=100 clean) reveals the opposite: a clean monotonic trend.

| Comp | n_query=3 (n=15) | n_query=20 (n=100) | 95% Wilson CI |
|---|---|---|---|
| 512² | 0.467 | 0.480 | [0.385, 0.577] |
| 768² | 0.533 | 0.500 | [0.404, 0.596] |
| **1024²** | 0.400 | **0.530** | **[0.433, 0.625]** |

Each resolution step gains ~2-5pt clean accuracy at K=5. Pairwise differences are not individually significant at n=100 (CIs overlap ±9%), but the monotonic trend across three points is consistent. The n=15 reversal was entirely noise. **1024² (0.530 clean) is the best K=5 result**, and would likely improve further at 1536² if VRAM allowed.

The implication is important: **resolution consistently helps, but at K=10 the ceiling is VRAM (16 images/call fits at 1024², 21 does not), not the model's perceptual ability.** If the prompt could be split or GPU memory increased, higher resolution would continue to gain.

**Synthesis (F8, F9, F11):** Resolution helps monotonically wherever it fits in VRAM. The current ceiling is 768² at K=10/n_ex=3 (best reachable: 0.519 clean) and 1024² at K=5/n_ex=3 (0.530 clean). For K≥15, n_ex=2+768² is the best option and costs nothing accuracy-wise vs n_ex=3. The gap between observed performance and ceiling is mostly clustering quality, not image resolution — but resolution is a real lever wherever the budget allows.

### F10 — Representation comparison: InfEmbed > state ≈ state_action at K=10

All three representations evaluated at K=10 with the same best-known config (composite 768², mf=4, n_example=3, n_query=3, n_reps=3, global_episode_disjoint, seed=42).

| Representation | Headline | Clean | Ratio (acc / chance) | p |
|---|---|---|---|---|
| InfEmbed K=10 + 768² | 0.567 (17/30) | **0.519 (14/27)** | **5.2×** | 5.6e-8 |
| InfEmbed K=10 baseline (512²) | 0.533 (16/30) | 0.481 (13/27) | 4.8× | 5.2e-7 |
| **state K=10 + 768²** | 0.467 (14/30) | **0.467 (14/30)** | **4.7×** | 3.1e-7 |
| state_action K=10 + 768² | 0.400 (12/30) | 0.333 (9/27) | 3.3× | 8.7e-4 |

**F10a — InfEmbed clusters are more visually coherent than state/state_action at matched K.** InfEmbed's 768² clean acc (0.519) is ~56% higher than state_action's (0.333) and ~11% higher than state's (0.467). Same VLM, same prompt, same K — only the clustering changes.

**F10b — State is notably better than state_action (0.467 vs 0.333 clean).** Adding raw actions to the observation features *hurts* visual coherence. State-only clustering produces fully clean sample plans (all 30 queries are tier1_global — no cross-cluster episode contamination), while state_action loses 3 queries to contamination. The action concatenation may inflate feature dimensionality (59D → 79D) without adding cluster-separating signal, causing UMAP/kmeans to split on action-space noise rather than behaviorally meaningful structure.

**F10c — State clustering still well above chance.** State K=10 passes H1 comfortably (p=3.1e-7, ratio 4.7×), comparable to InfEmbed in the ratio metric — but the absolute clean acc is 5 points lower. The VLM can recover some structure from state-clustered slices; influence just produces tighter clusters.

**Implication:** influence captures behavioral structure that raw observations alone do not. The gap is real and meaningful (0.519 vs 0.467 at matched K, same evaluation). However, state clustering is not noise — it's a viable weaker baseline, not a flat-chance result. The influence signal's advantage likely reflects its sensitivity to *how the policy processes* the state, not just what the state is.

---

## Best-known config (post-F8/F9)

Defaults in `policy_doctor/configs/experiment/e1_cluster_coherence_vlm.yaml`, the Hydra step, and the runner script (`scripts/run_e1_transport_r512_qwen.py`) have been updated to:

| Knob | Default | Rationale |
|---|---|---|
| K | 10 (downstream choice) | Sweet spot in the K curve; K=15/20 over-cluster, K=5 over-merges |
| `n_repetitions` | 3 | majority vote; standard since v2 |
| `max_frames_per_storyboard` | 4 | mf=9 underperforms (cells too small) |
| `global_episode_disjoint` | **true** | blocks ~10pt of episode-cue inflation |
| `storyboard_mode` | composite | frames doesn't help, hybrid hurts |
| `composite_target_size` | **768** | 384² cells beat 256² by +0.04 clean at no token cost |
| `view_window_extension` | 0 | non-zero values hurt at fixed mf |
| `query_storyboard_mode` | unset (= storyboard_mode) | hybrid hurts the prompt structure |
| `random_seed` | 42 | as before |

Sweep eval driver (`run_e1_sweep_eval.py`) accepts the new knobs as CLI flags so a sweep can be evaluated under any of them.

Earlier sample plans and metrics (K=20 v1, K=10 v2 at 512², etc.) are bit-frozen on disk and remain comparable to one another. New runs from the current defaults will not be bit-identical to those, by design.

---

## Open methodological questions (informing the next sweep batch)

Of the original six confounds, two are now substantially settled:

| # | Confound | Status |
|---|---|---|
| 1 | Slice length / temporal context | **Partially answered (F6a)** — naive extension at fixed frame count *worsens* accuracy due to sampling-position artifacts. Density-matched extension helps slightly (F6c) but is bottlenecked by per-cell resolution (F6b). The clean test (raise composite resolution) hasn't been run yet. |
| 2 | VLM capacity | **Answered (F5)** — Qwen3-VL-32B-NF4 is identical to 8B. Not the bottleneck. |
| 3 | Image budget per storyboard | **Answered (F8a, F11)** — 768² is the VRAM ceiling at K=10/n_ex=3 (1024² OOMs). Trading n_ex=3→2 to reach 1024² costs −7pt clean at K=10 (net loss). At K≥15, n_ex=2+768² is the best reachable config and costs nothing accuracy-wise. 1536² is unreachable (single-image vision encoder OOM). |
| 4 | Clustering hyperparameters | Not yet swept on real data. Sweep harness built (`scripts/run_clustering_sweep.py`); spec at `sweep_specs/transport_r512_alt_clustering.yaml` (108 combos). |
| 5 | Representation choice | **Answered (F10)** — InfEmbed (0.519 clean) > state (0.467) > state_action (0.333) at K=10. InfEmbed captures behavioral structure beyond raw obs; state is a viable but weaker baseline; adding raw actions to state hurts. |
| 6 | Stronger frontier VLM | Pending. F5 makes this a lower-priority lever. |

---

## Queued / planned

The next implementation phase (in flight, see "Architecture" below) targets questions 1, 4, 5 — and lays the foundation for 2 + 6 by keeping the E1 evaluation interface stable.

### Q1 — Extended visual context (DONE — see F6)

`view_window_extension` is implemented and swept at K=10. Result: not the right knob in isolation. Composite-resolution scaling has been wired (`--composite_target_size`) and tested at K=10 mf=4 768² (config (C) in F8) — modest gain, not a breakthrough.

### Q1' — Storyboard-mode sweep at K=10 (DONE — see F8)

`--storyboard_mode {composite,frames}` and `--query_storyboard_mode` flags are implemented (commit pending). Results show larger composites help slightly; hybrid hurts; pure frames doesn't fit at K=10. Pure-frames testing has to drop to K=5.

### Q1'' — Pure frames at K=5 (PENDING — running now)

K=5 clustering built at `/tmp/transport_mh_seed0_r512_clustering_k5`. Two configs running on GPU 0:
- K=5 baseline (composite mf=4 target=512²) — to establish a fresh K=5 reference point
- K=5 pure frames (mf=4, frames at native 512² each — 64 images per call ≈ 86k tokens, fits)

Together these tell us whether the pure-frames format itself is informative once it actually fits in context. Note this *also* changes K, so direct comparison to K=10 results requires care — clean accuracy may rise simply from K=10 → K=5 (chance goes 0.10 → 0.20), as F1 already showed for K=20 → K=10.

### Q4 — Clustering hyperparameter sweeps

Beyond K, sweep:
- `window_width ∈ {3, 5, 10, 15}`
- `stride ∈ {1, 2, 5}`
- `aggregation ∈ {sum, mean, concat}`
- `prescale ∈ {none, standard, l2}`
- `umap_n_components ∈ {25, 50, 100, 200}`

This is a lot of clusterings (each a few minutes of CPU). The sweep harness lets us run them in batch and evaluate any subset through E1.

### Q5 — Baseline representations (DONE — see F10)

All three representations evaluated at K=10 with the best-known config (768² composite). Results: InfEmbed (0.519 clean) > state (0.467) > state_action (0.333). See F10 for full breakdown.

- `policy_doctor/data/slice_representations.py` — abstraction + 3 concretes (`infembed`, `state`, `state_action`) tested and committed (commit `c9d69ff`).
- `scripts/build_alt_clustering.py` — single-config CLI tested and committed.
- K=10 clusterings: `/tmp/transport_r512_state_k10/`, `/tmp/transport_r512_state_action_k10/`.
- E1 results: `experiments/e1_transport_r512_seed0_qwen3vl8b_state_k10/`, `experiments/e1_transport_r512_seed0_qwen3vl8b_state_action_k10/`.

---

## Architecture (implemented)

The new `SliceRepresentation` abstraction decouples *what* feature each slice is from *how* it's clustered:

```
policy_doctor/data/slice_representations.py
├── SliceWindowParams                    (dataclass: window_width, stride, aggregation)
├── SliceRepresentation (ABC)
│   ├── name: str
│   ├── extract(eval_dir, params, **kwargs) -> (features: ndarray, metadata: List[dict])
│   └── describe(params, **kwargs)       (JSON-able fingerprint for manifests)
├── InfEmbedRepresentation               (wraps extract_infembed_slice_windows)
├── StateRepresentation                  (proprioceptive obs from episode pickles)
│   └── kwargs: obs_strategy ∈ {current, full_history}
├── StateActionRepresentation            (obs + action concat per timestep)
│   └── kwargs: obs_strategy, action_strategy ∈ {executed, full_plan}
└── registry: get_slice_representation(name)
```

All three representations share the same window-aggregation engine (`build_windows_from_rollout_timestep_embeddings`). The output `(features, metadata)` contract is identical, so the downstream `normalize → prescale → reduce → cluster → save` pipeline is unchanged.

The output directory layout (`cluster_labels.npy`, `metadata.json`, `manifest.yaml`, `embeddings_reduced.npy`, `clustering_models.pkl`) matches what `RunClusteringStep` produces, so the existing E1 evaluation pipeline (runner script, Hydra step, analysis script) transparently consumes alternative clusterings without modification.

### Scripts

- **`scripts/build_alt_clustering.py`** — single-config CLI. Pick `--representation`, supply hyperparams (`--window_width`, `--stride`, `--aggregation`, `--prescale`, `--umap_n_components`, `--n_clusters`, etc.), get one clustering dir.
- **`scripts/run_clustering_sweep.py`** — cartesian product over a YAML/JSON spec. Filters rep-specific kwargs (e.g. `obs_strategy` only varied for state/state_action), de-duplicates, runs each combo as a subprocess. Skips combos whose `manifest.yaml` already exists unless `--force`. Writes a per-combo `build.log`.
- **`sweep_specs/transport_r512_alt_clustering.yaml`** — example spec: 108 unique combos across {infembed, state, state_action} × {window_width 3,5,10} × {aggregation sum,mean} × {umap_n_components 50,100} × {K 10,15,20}.

### Pipeline integration (unchanged for legacy users)

- `RunClusteringStep` (the Hydra step) is **not** modified yet — alternative clusterings are produced via the standalone runner. The sample plan / E1 step `validate_cluster_coherence_vlm` reads the new `view_window_extension` knob.
- The existing `vlm_cluster_classification.global_episode_disjoint` and the new `vlm_cluster_classification.view_window_extension` are both configurable via `policy_doctor/configs/experiment/e1_cluster_coherence_vlm.yaml`.

### Tests

- `tests/data/test_slice_representations.py` (13 tests): registry, helper flatteners (current/full_history, executed/full_plan), shape/metadata contracts for each concrete rep.
- `tests/data/test_build_alt_clustering.py` (2 tests): end-to-end smoke on a synthetic eval dir confirming `build_alt_clustering.py` produces a dir the existing E1 planner can consume in `global_episode_disjoint=True` mode.

---

## How-to (no runs yet — GPUs reserved for E2)

The pipeline has two stages: **build** clustering dirs (CPU only, no GPU) and **evaluate** them via E1 (GPU). They run independently — you can pre-build everything overnight on CPU, then evaluate any subset on GPU when time allows.

### Stage 1 — Build clusterings (CPU only)

#### One-off: a single configuration

```bash
python scripts/build_alt_clustering.py \
  --representation state_action \
  --eval_dir /mnt/ssdB/erik/cupid_data/outputs/eval_save_episodes/mar27/mar27_train_diffusion_unet_lowdim_transport_mh_0_r512x512/latest \
  --window_width 5 --stride 2 --aggregation mean \
  --prescale standard --umap_n_components 100 --n_clusters 10 \
  --obs_strategy current --action_strategy executed \
  --seed 42 \
  --out_dir /tmp/clustering_sweeps/sa_k10_mean
```

The output dir contains `manifest.yaml`, `cluster_labels.npy`, `metadata.json`, `embeddings_reduced.npy`, `clustering_models.pkl` — the same layout as the existing `RunClusteringStep` pipeline output.

#### Sweep: a grid of configurations

Edit (or duplicate) `sweep_specs/transport_r512_alt_clustering.yaml` to set the grid. Each list-valued entry under `grid:` is one axis of the cartesian product. Knobs that do not apply to a given representation are filtered out automatically (e.g. `obs_strategy` only varied for state/state_action).

```bash
# Dry-run first to see how many combos and what slugs they get
python scripts/run_clustering_sweep.py \
  --spec sweep_specs/transport_r512_alt_clustering.yaml \
  --dry_run

# Run for real (CPU only)
python scripts/run_clustering_sweep.py \
  --spec sweep_specs/transport_r512_alt_clustering.yaml
```

Output layout under `sweep_root` (defined in the spec):

```
/tmp/clustering_sweeps/transport_r512_seed0_alt/
├── infembed__w=3__agg=sum__pre=standard__d=50__K=10/
│   ├── manifest.yaml
│   ├── cluster_labels.npy
│   ├── embeddings_reduced.npy
│   ├── metadata.json
│   ├── clustering_models.pkl
│   └── build.log
├── infembed__w=3__agg=sum__pre=standard__d=50__K=15/
│   └── ...
├── state__w=5__agg=mean__pre=standard__d=100__K=10__obs=current/
│   └── ...
├── state_action__w=5__agg=mean__pre=standard__d=100__K=10__obs=current__act=executed/
│   └── ...
└── ...   (108 dirs for the default spec)
```

The pre-built spec produces 108 unique combos — roughly 60–90 min of CPU. Re-running the sweep skips combos whose `manifest.yaml` already exists; pass `--force` to redo.

### Stage 2 — Evaluate one or many clusterings via E1 (GPU)

#### Evaluate a single clustering dir

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_e1_transport_r512_qwen.py \
  --clustering_dir /tmp/clustering_sweeps/sa_k10_mean \
  --max_clusters 10 --n_example 3 --n_query 3 --n_repetitions 3 \
  --global_episode_disjoint \
  --view_window_extension 10 \
  --out_dir experiments/e1_state_action_k10_view10
```

#### Evaluate every clustering dir under a sweep root

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_e1_sweep_eval.py \
  --sweep_root /tmp/clustering_sweeps/transport_r512_seed0_alt \
  --results_root experiments/e1_sweep_transport_r512_seed0 \
  --eval_dir /mnt/ssdB/erik/cupid_data/outputs/eval_save_episodes/mar27/mar27_train_diffusion_unet_lowdim_transport_mh_0_r512x512/latest \
  --n_example 3 --n_query 3 --n_repetitions 3 \
  --global_episode_disjoint
```

Reads `n_clusters` from each `manifest.yaml`, runs E1 with the right `--max_clusters` per combo. Writes per-combo logs and an aggregated `sweep_eval_summary.jsonl` (one line per combo with `top1_accuracy`, `binomial_test_pvalue`, `elapsed_s`, etc.) into `--results_root`.

Useful filters:
- `--include_pattern infembed` — only evaluate InfEmbed combos
- `--include_pattern K=10` — only K=10 combos across all reps
- `--max_clusters_override 10` — force max_clusters=10 regardless of manifest (useful when capping a higher-K clustering)
- `--view_window_extension 0|5|10|15` — runs the same evaluation knob across the sweep
- `--force` — re-run combos whose `metrics.json` already exists

Single GPU → sequential evaluation. To split across GPUs, kick off two `run_e1_sweep_eval.py` processes pinned to different GPUs with **different `--include_pattern`** filters so they don't write the same out_dirs.

#### Estimating GPU time

Per-combo wall time scales with the prompt size:
- K=10 with n_reps=3: ~16 min (≈30 calls × 30 s/call)
- K=15 with n_reps=3: ~30 min
- K=20 with n_reps=3: ~50 min

For the default 108-combo spec at K∈{10,15,20} (36 combos × each K bucket), expect roughly: 36×16 + 36×30 + 36×50 ≈ 58 GPU-hours total on a single 8B model. Filter aggressively with `--include_pattern` for the most informative slices first.

### Stage 3 — Compare across the sweep

After eval, the per-combo `sweep_eval_summary.jsonl` is the easiest input to a comparison script. Each line has the clustering slug + headline accuracy + p-value. Use `scripts/analyze_e1_confusion_structure.py --metrics <results_dir>/metrics.json` for the per-cluster + per-query-origin breakdown of any individual combo.

A script for cross-combo aggregation (e.g. accuracy vs window_width holding K and rep fixed) is **not** in this commit — the `sweep_eval_summary.jsonl` is structured enough to drive ad-hoc pandas analysis once the data lands.

### Suggested initial evaluation matrix (when GPUs free up)

Most informative single-axis slices to evaluate first:

1. **Representation comparison at K=10** — filter `--include_pattern K=10` plus the rep prefix. Three combos per rep when fixing aggregation/window_width/umap_dim. Tests whether influence captures behavioral structure beyond raw obs.
2. **Extended visual context at K=10** — same K=10 InfEmbed clustering, four runs at `--view_window_extension ∈ {0, 5, 10, 15}`. Isolates the slice-length confound (changes only what the VLM sees, not the clustering).
3. **Window-width sweep** — InfEmbed combos at `window_width ∈ {3, 5, 10}`, K=10. Different question than (2): clustering window and visual window co-vary here.

---

## F12 — Window-parameter sweep: (w=3,s=2) is the best window for InfEmbed; architecture order matters

Full sweep: InfEmbed × (w,s) ∈ {(1,1),(2,2),(3,2),(5,5)} × K ∈ {5,10,15,20} × 3 seeds, 512² composite.
Run under two pipeline architectures:
- **window_first** (matches F1–F10): window → UMAP(100D) → kmeans. Results in `e1_window_sweep_wf_results.md`.
- **timestep_first** (new): UMAP(50D) on per-timestep features → window → kmeans. Results in `e1_window_sweep_results.md`.

### Unified comparison (InfEmbed, mean clean acc ± std across 3 seeds)

| K | (w,s) | window_first | ratio | timestep_first | ratio |
|---|---|---|---|---|---|
| 5 | (1,1) | 0.422±0.083 | 2.1× | 0.600±0.054 | 3.0× |
| 5 | (2,2) | 0.422±0.031 | 2.1× | 0.444±0.083 | 2.2× |
| **5** | **(3,2)** | **0.556±0.031** | **2.8×** | 0.444±0.083 | 2.2× |
| 5 | (5,5) | 0.511±0.063 | 2.6× | **0.644±0.031** | **3.2×** |
| 10 | (1,1) | 0.244±0.016 | 2.4× | 0.300±0.054 | 3.0× |
| 10 | (2,2) | 0.370±0.052 | 3.7× | 0.300±0.054 | 3.0× |
| **10** | **(3,2)** | **0.469±0.046** | **4.7×** | 0.311±0.057 | 3.1× |
| 10 | (5,5) | 0.411±0.016 | 4.1× | 0.367±0.072 | 3.7× |
| 15 | (1,1) | 0.310±0.019 | 4.6× | 0.310±0.051 | 4.6× |
| 15 | (2,2) | 0.310±0.085 | 4.6× | 0.286±0.019 | 4.3× |
| 15 | (3,2) | 0.294±0.022 | 4.4× | 0.206±0.030 | 3.1× |
| **15** | **(5,5)** | **0.356±0.018** | **5.3×** | 0.302±0.056 | 4.5× |
| 20 | (1,1) | 0.222±0.022 | 4.4× | 0.246±0.014 | 4.9× |
| 20 | (2,2) | 0.211±0.025 | 4.2× | 0.263±0.029 | 5.3× |
| **20** | **(3,2)** | **0.281±0.014** | **5.6×** | 0.216±0.022 | 4.3× |
| 20 | (5,5) | 0.250±0.014 | 5.0× | **0.298±0.089** | **6.0×** |

**F12a — window_first: (w=3,s=2) is the best or tied-best window at K=5, K=10, and K=20.** At K=15 `(5,5)` wins. Single-timestep `(1,1)` windows are worst at K=10 (0.244 vs 0.469). The original default `(w=5,s=2)` — not tested here — gave 0.481 at K=10; `(3,2)` at 0.469 is the closest match in this sweep.

**F12b — timestep_first inverts the pattern at K=5 and K=20.** At K=5, `(5,5)` wins (0.644 vs 0.556 for window_first's best). At K=10, all configs are weaker than window_first. At K=20 `(5,5)` again leads (0.298). The architecture change (50D UMAP on raw per-timestep features vs 100D UMAP on windowed features) shifts the optimal window, especially at the extremes of K.

**F12c — window_first consistently outperforms timestep_first at K=10** (all configs: 0.244–0.469 vs 0.300–0.367). Window aggregation *before* UMAP gives UMAP better-structured input for the K=10 sweet spot. The timestep_first approach loses this advantage because UMAP sees noisier individual timestep features.

**F12d — At K=15/20, the two architectures converge.** Both give 0.28–0.36 at K=20 for their respective best configs. The architecture choice matters less at high K where cluster separability is the binding constraint.

### Policy embedding naming convention

Layer names follow the pattern `{hook}_{action}_t{T}`:

**Hook** — which module in the diffusion UNet is tapped:

| Hook name | Module | UNet position |
|---|---|---|
| `encoder` | `down_modules[-1][1]` | End of the downsampling path |
| `bottleneck` | `mid_modules[-1]` | Bridge between down/up (most compressed) |
| `decoder` | `up_modules[-1][1]` | End of the upsampling path |

The naming follows the diffusion policy architecture: `encoder` = downsampling path, `bottleneck` = compressed middle, `decoder` = upsampling path. Somewhat counterintuitively, the `encoder` hook (early in the network) turns out to be the most discriminative for behavioral clustering.

**Action input** — what action is fed into the UNet:
- `plan` — the full 16-step rollout action plan (what was actually executed)
- `exec` — `action[0]` tiled across the full horizon (single executed step)
- `plan8` — first 8 executed steps, zero-padded to full horizon
- *(omit)* — random noise (action washes out when averaged over t)

**Diffusion timestep `t`** — the noise level at which the UNet is evaluated. The diffusion schedule runs from `t=0` (final denoising step, near-clean, σ≈0.01) up to `t=T-1` (pure noise). So `t=0` = the policy evaluating a nearly-committed action plan. Omitting `_t{N}` means the activation is averaged over 100 uniformly-spaced noise levels.

---

### F13 — Policy bottleneck embedding ≈ state: obs-dominated when averaged over noise levels

New representation `policy_emb` (layer=bottleneck): per-timestep UNet mid-module output
(512D, global-avg-pooled) averaged over all 100 denoising timesteps, using random noise
for the action input. Computed in ~9s for 16,956 timesteps on a 4090 (batch_size=128,
128×100=12,800 samples per forward pass). Saved to `eval_dir/policy_embeddings/bottleneck.npz`.

| Repr | Clean acc (K=10) | Ratio | p |
|---|---|---|---|
| InfEmbed | 0.519 (14/27) | 5.2× | 5.6e-8 |
| state | 0.467 (14/30) | 4.7× | 3.1e-7 |
| **policy_emb** | **0.467 (14/30)** | **4.7×** | **3.1e-7** |
| state_action | 0.333 (9/27) | 3.3× | 8.7e-4 |

`policy_emb` ties `state` exactly (same absolute count, same ratio). The bottleneck is
obs-dominated when averaged over 100 noise levels: averaging washes out the action-specific
signal, leaving the obs conditioning as the effective input. The result is a learned nonlinear
transform of the obs, which UMAP then treats similarly to raw obs.

**Full sweep results (all K=10, w=5,s=2, 50D UMAP, 512² or 768² composite):**

| Variant | Action input | t | Hook | Clean | Ratio | p |
|---|---|---|---|---|---|---|
| **encoder_plan_t0** | **actual plan** | **0** | **encoder** | **0.625 (15/24)** | **6.2×** | **5.4e-10** |
| bottleneck_plan_t0 | actual plan | 0 | bottleneck | 0.593 (16/27) | 5.9× | 4.4e-10 |
| bottleneck_plan_t5 | actual plan | 5 | bottleneck | 0.533 (16/30) | 5.3× | 3.7e-9 |
| bottleneck_exec_t0 | action[0]×16 | 0 | bottleneck | 0.519 (14/27) | 5.2× | 5.6e-8 |
| bottleneck_plan_t25 | actual plan | 25 | bottleneck | 0.519 (14/27) | 5.2× | 5.6e-8 |
| InfEmbed (100D) [ref] | — | — | — | 0.519 (14/27) | 5.2× | 5.6e-8 |
| decoder_plan_t0 | actual plan | 0 | decoder | 0.500 (15/30) | 5.0× | 3.6e-8 |
| bottleneck_plan_t10 | actual plan | 10 | bottleneck | 0.500 (12/24) | 5.0× | 8.5e-7 |
| InfEmbed (50D) [ref] | — | — | — | 0.444 (12/27) | 4.4× | 4.1e-6 |
| bottleneck (avg t) | random noise | avg | bottleneck | 0.467 (14/30) | 4.7× | 3.1e-7 |
| bottleneck_t0 | random noise | 0 | bottleneck | 0.400 (12/30) | 4.0× | 1.5e-5 |
| bottleneck_plan8_t0 | plan[:8]+zeros | 0 | bottleneck | 0.300 (9/30) | 3.0× | 2.0e-3 |

**F14a — encoder hook (up_modules[-1][1]) is the best policy embedding.** 0.625 clean (6.2×), beating bottleneck (0.593) and the InfEmbed 100D baseline (0.519). The encoder hook captures the most discriminative layer for behavioral clustering — it's the decoder's final synthesis before projecting to action space.

**F14b — actual rollout action is critical; zero-padding destroys signal.** plan_t0 (0.593) > exec_t0 (0.519) >> plan8_t0 (0.300). The zero-padded 8-step variant is dramatically worse — the zeros actively mislead the network. Using only action[0] tiled (exec) is fine (ties InfEmbed).

**F14c — Timestep sensitivity is non-monotonic.** t=0 (0.593) > t=5 (0.533) > t=10 (0.500) ≈ t=25 (0.519). Small noise levels (t=5) are slightly better than t=0 — a small amount of noise may regularise the representation. The network processes the action most sharply near t=0 but is not uniquely sharp there.

**F14d — Window params have little effect on policy_emb.** (w=3,s=2) = (w=5,s=2) = 0.467; (w=1,s=1) = (w=2,s=2) = 0.433. Slightly weaker at single-timestep windows; no strong preference above w=3.

**F14e — K sweep (bottleneck_plan_t0).** K=20: 0.429 (8.6×!), K=10: 0.467 (4.7×), K=15: 0.333 (5.0×), K=5: 0.400 (NS). The very high ratio at K=20 is notable — policy_emb maintains above-chance clustering at fine granularity where InfEmbed weakens (InfEmbed K=20 was 0.241).

**F14f — K sweep (encoder_plan_t0).** encoder_plan_t0 sweeps K=5–20 (w=5,s=2, 512² at K≥15):

| K | encoder_plan_t0 | Ratio | p | bottleneck_plan_t0 (ref) |
|---|---|---|---|---|
| 5 | 0.533 (8/15) | 2.7× | 4.2e-3 | 0.400 (NS) |
| 10 | 0.625 (15/24) | 6.2× | 5.4e-10 | 0.593 (16/27) |
| 15 | 0.444 (16/36) | 6.7× | 3.1e-10 | 0.333 (15/45) |
| 20 | 0.359 (14/39) | 7.2× | 2.8e-9 | 0.429 (17/40) |

encoder_plan_t0 beats bottleneck at every K except K=20 where bottleneck is +7pt. All four encoder results are statistically significant (bottleneck K=5 was not). The ratio increases monotonically with K (2.7× → 7.2×), meaning encoder maintains a growing discrimination advantage over chance as the problem gets harder. Raw accuracy peaks at K=10 (0.625) — the best single E1 result across all representations tested.

**F14f interpretation:** encoder_plan_t0 at K=10 is the recommended operating point. The encoder hook appears more discriminative for behavioral clustering than the bottleneck at all but the finest granularity (K=20). Both representations degrade gracefully at high K (unlike InfEmbed which drops sharply at K=20).

### F14 — Policy action at t=0 is the best representation tested: 0.593 clean at K=10

Three `policy_emb` variants at K=10 (w=5,s=2, 768², n_example=3, n_query=3, seed=42):

| Repr | Action input | Noise levels | Clean | Ratio | p |
|---|---|---|---|---|---|
| InfEmbed | — | — | 0.519 (14/27) | 5.2× | 5.6e-8 |
| state | raw obs | — | 0.467 (14/30) | 4.7× | 3.1e-7 |
| policy_emb (avg t) | random noise | avg over 100 | 0.467 (14/30) | 4.7× | 3.1e-7 |
| bottleneck_t0 | random noise | t=0 only | 0.400 (12/30) | 4.0× | 1.5e-5 |
| **bottleneck_plan_t0** | **actual rollout plan** | **t=0** | **0.593 (16/27)** | **5.9×** | **4.4e-10** |

**F14a — `bottleneck_plan_t0` beats InfEmbed by +7pt clean (0.593 vs 0.519).** Using the actual
rollout action plan at the final denoising step (t=0, near-clean limit) gives the strongest
representation tested so far. The key ingredient is the **action**: without it (random noise,
avg-t) the bottleneck ≈ state; with it, the bottleneck captures how the policy processes a
specific (obs, action) pair.

**F14b — t=0-only with random noise (bottleneck_t0) is *worse* than avg-t (0.400 vs 0.467).**
At t=0 alone, the random noise action is nearly absent (sigma≈0.01), so the bottleneck mostly
sees the obs — but with less averaging-induced smoothing. The avg-t version is better because
averaging over 100 noise levels stabilises the representation.

**F14c — The action plan is the critical signal, not t=0 alone.** Comparing bottleneck_t0
(0.400, random noise at t=0) vs bottleneck_plan_t0 (0.593, actual plan at t=0): same noise
level, 19pt gap purely from using the actual action. The policy's processing of the actual
executed plan is highly discriminative for clustering.

**Implication:** The policy bottleneck with actual action encodes richer behavioral information
than either raw observations (state) or influence gradients (InfEmbed) at K=10. This is the
first representation tested that exceeds InfEmbed at the sweet-spot K. Whether this advantage
holds across K and window widths is the next question.

### F15 — InfEmbed 50D UMAP is weaker than 100D; TRAK clustering fails

**F15a — InfEmbed 50D UMAP baseline.** Running InfEmbed with 50D instead of 100D UMAP (matching the policy_emb sweep pipeline) gives 0.444 clean at K=10 vs 0.519 with 100D. The UMAP dimensionality matters for InfEmbed — 7.5pt drop. This means that policy_emb comparisons within the 50D sweep moderately understate InfEmbed's advantage. The InfEmbed 100D reference (0.519) remains the strongest InfEmbed result.

**F15b — TRAK clustering (full 186k → SVD(200D) → UMAP(50D)) is at chance.** Raw TRAK score profiles produce no useful cluster structure:

| K | TRAK clean | Ratio | p |
|---|---|---|---|
| 5 | 0.200 (3/15) | 1.0× | 0.60 (NS) |
| 10 | 0.133 (4/30) | 1.3× | 0.35 (NS) |
| 15 | 0.044 (2/45) | 0.7× | 0.81 (NS) |
| 20 | 0.083 (5/60) | 1.7× | 0.18 (NS) |

All results are at or below chance level. TRAK clustering is a complete negative result. Raw per-timestep TRAK score rows are too noisy for behavioral clustering: each timestep's 186k-dim attribution vector contains gradient similarity noise that doesn't align with behaviorally meaningful segments. SVD(200D) cannot recover useful structure from this. This contrasts sharply with InfEmbed which applies the Arnoldi method to find the principal gradient directions — a more principled compression that preserves the behaviorally-relevant variance that TRAK scores spread across 186k dimensions.

---

**F12e — No simple "higher K → smaller window" rule.** Best window per K: K=5→(3,2), K=10→(3,2), K=15→(5,5), K=20→(3,2). The K=15 exception breaks any monotonic story, and by K=15/20 the differences between window configs are within noise anyway (except `(1,1)` which remains slightly worse).

The practical rules that do hold:
1. **Avoid w=1 at K≤10.** Single-timestep windows are consistently the worst and cost ~2× at K=10 (0.244 vs 0.469).
2. **Prefer overlapping stride (s < w) at moderate K.** At K=10, `(3,2)` and `(2,2)` both beat non-overlapping `(5,5)`.
3. **`(w=3, s=2)` is a safe default across all K.** It is best or tied-best at K=5, 10, and 20, and only 7pt below the K=15 winner. The benefit of tuning window width further is small once you're past the single-timestep baseline.
4. **The original `(w=5, s=2)` default remains the best known at K=10** (0.481 from F1–F10, not tested in the sweep but consistent with the pattern of overlapping medium-width windows).

---

### F16 — Head-to-head comparison: K=10, n_query=5, n_reps=5

Controlled re-run of all major representations under identical protocol (same K, same n_example=3, n_query=5, n_reps=5, global_episode_disjoint, 768², seed=42). Larger query pool (~40–50 clean queries per run vs 24–30 in the sweep) gives tighter estimates and more reliable cross-representation comparison.

| Representation | Clean | n | Ratio | p |
|---|---|---|---|---|
| **bottleneck_plan_t5** | **0.540 (27/50)** | 50 | **5.4×** | **1.1e-14** |
| encoder_plan_t0 | 0.525 (21/40) | 40 | 5.2× | 2.0e-11 |
| bottleneck_plan_t0 | 0.480 (24/50) | 50 | 4.8× | 8.9e-12 |
| state_action_full | 0.480 (24/50) | 50 | 4.8× | 8.9e-12 |
| infembed_100d | 0.378 (17/45) | 45 | 3.8× | 6.9e-7 |
| state_full | 0.378 (17/45) | 45 | 3.8× | 6.9e-7 |
| trak | 0.200 (10/50) | 50 | 2.0× | 2.5e-2 |

State representations: `state_full` = full 2-step obs history (118D → UMAP 100D); `state_action_full` = full history + full 16-step predicted action plan (438D → UMAP 100D).

**F16a — bottleneck_plan_t5 is the most reliable best representation (0.540).** The sweep result for encoder_plan_t0 (0.625, 15/24) was inflated by a small clean bucket. With 40 clean queries, encoder settles at 0.525 — consistent with but not significantly above bottleneck_plan_t5 (0.540). The top-3 (bottleneck_t5, encoder_t0, bottleneck_t0) are within 6pt and all share the same order of magnitude p-value. bottleneck_plan_t5 wins on point estimate.

**F16b — encoder advantage over bottleneck does not replicate cleanly.** In the sweep (n=24), encoder_plan_t0 appeared 16pt ahead of bottleneck_plan_t0. Here (n=40 vs n=50) the gap is 4.5pt. The earlier result was likely a sampling artifact from a small clean bucket. The representations are close; bottleneck_plan_t5 is arguably the more reproducible best.

**F16c — state_action_full (full history + full plan) matches bottleneck_plan_t0.** 0.480 vs 0.480, identical p-value. Adding the full 16-step predicted plan to raw state recovers as much clustering signal as the policy bottleneck with the same action — the plan itself is highly informative even without passing through the neural network. This is a notable finding: you don't need the UNet to cluster behaviourally if you have the raw action plan.

**F16d — InfEmbed 100D ties state_full (both 0.378, 3.8×).** Raw proprioceptive state (2-step history, 118D → UMAP) and the Arnoldi-projected gradient embeddings (100D) give identical results. InfEmbed's advantage over raw state likely came from the 50D UMAP inconsistency in earlier experiments (F15a). On equal footing they are equivalent.

**F16e — TRAK is now marginally significant (p=0.025, 2.0×) but remains the weakest.** With n=50 clean queries it clears the α=0.05 threshold, but the effect is small (2× vs chance). Not a complete null result, but far below every other representation.

**F16 headline:** `bottleneck_plan_t5 ≈ encoder_plan_t0 > bottleneck_plan_t0 ≈ state_action_full >> infembed_100d ≈ state_full >> trak`. The raw action plan is a surprisingly strong clustering signal on its own; the neural representation adds ~10pt over it.

## Findings to write up after the next batch

- **encoder_plan_t0 K sweep (F14f)**: K=5→0.533, K=10→0.625, K=15→0.444, K=20→0.359; all significant; encoder beats bottleneck at K=5/10/15

## Operational note

Going forward, only **GPU 0** is in scope for E1 work — GPU 1 is reserved for sander tonkens' streamlit. All sweep scripts pin via `CUDA_VISIBLE_DEVICES=0` accordingly.
