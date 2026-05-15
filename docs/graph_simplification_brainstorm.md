# Graph Simplification — Brainstorm

**Problem.** The behavior graph (see `policy_doctor/behaviors/behavior_graph.py`) has too many noisy
transitions between cluster nodes. Looking at the current `transport_mh_jan28` graph in the running
Policy Doctor app: 20 KMeans clusters with the layered BFS layout produce a hairball — many edges
criss-cross, no clear temporal "flow" from START to SUCCESS/FAILURE. We want a cleaner, more
interpretable graph.

This document brainstorms ~25 candidate methods, organized by where they live in the pipeline, and
ranks them by feasibility × effectiveness.

---

## 1. Diagnosis of the current pipeline

```
per-timestep influence row  ──►  sliding window (W=5, S=2, sum-agg)
                            ──►  StandardScaler
                            ──►  UMAP → 100D
                            ──►  KMeans (k=20)
                            ──►  run-length collapse per episode
                            ──►  Markov chain (counts + probs)  ← all observed edges kept
                            ──►  Plotly layered-BFS render      ← min_probability filter (display only)
```

Defaults from `policy_doctor/curation_pipeline/steps/run_clustering.py:76-87`:
window=5, stride=2, UMAP to 100D, KMeans k=20, normalize=none, prescale=standard.

The five root causes of the noise:

| # | Root cause | Where it shows up |
|---|---|---|
| R1 | No temporal coherence in clustering — adjacent windows can hop between clusters even when behavior hasn't changed | `behaviors/clustering.py:108` — `cluster_kmeans` is per-sample, no time prior |
| R2 | No edge pruning at construction — every observed transition becomes an edge regardless of count | `behaviors/behavior_graph.py:133-134` |
| R3 | k=20 is large for a flat graph; no auto-K | `run_clustering.py:79` (`clustering_n_clusters=20`) |
| R4 | Influence-vector feature space encodes "which training samples influence this prediction", not "what behavior is the policy doing" → cluster boundaries don't line up with causal phases | `computations/embeddings.py` |
| R5 | Layered BFS layout doesn't reflect time → unrelated nodes share columns, creating apparent edge clutter | `plotting/plotly/behavior_graph.py` |

The current pipeline only addresses (R5, weakly) via `min_probability` and (R3, weakly) via the K
knob. Everything else is open territory.

---

## 2. Brainstorm — methods by category

### A. Pre-clustering (smoothing, alternative features)

- **A1. Wider temporal window + Gaussian smoothing along time.** Bump `clustering_window_width` to
  ~15 (W=15, S=5) and apply Gaussian smoothing in UMAP space before clustering. Already
  parameterized; ~5-line addition for the smoother.
- **A2. Action-conditioned features.** Concatenate `action_{t-W..t}` (or even pre-trained image
  encoder features, à la ENAP's DINO) into the embedding. The InfEmbed influence vector is an
  *indirect* proxy for "what the policy is doing"; raw actions are direct and orders of magnitude
  cheaper to load.
- **A3. ENAP-style phase-aware RNN embedding.** Train an RNN on `(a_{0..t}, c_{0..t})` with a
  contrastive loss that pulls `h_t` close to `h_{t+1}` when `c_t = c_{t+1}` and pushes them apart
  when `c_t ≠ c_{t+1}`. This explicitly bakes "phase persistence" into the geometry of the
  embedding — clusters in the new space correspond to *task modes* rather than density modes in
  influence space. (See ENAP §IV-A, eq. 1.)

### B. Clustering algorithm

- **B1. Lower K.** Drop `clustering_n_clusters` to 6–10 and add silhouette/elbow auto-selection.
  Single-line config + ~20-line silhouette sweep.
- **B2. Hierarchical (Ward) clustering with a slider-controlled cut.** Build the dendrogram once;
  expose "level of detail" in the UI. Top level might be 4 macro-modes (Approach / Grasp / Transport
  / Release); zoom into each for finer sub-clusters. `sklearn.cluster.AgglomerativeClustering` makes
  this trivial.
- **B3. HDBSCAN with KNN noise propagation.** Switch back to HDBSCAN (the codebase already imports
  it) with a generous `min_cluster_size` (e.g. 50). Reassign noise points to the nearest
  centroid — `cluster_knn` in `clustering.py:73` already foreshadows this, just unused.
- **B4. Spectral clustering of the transition graph.** Two-pass: cluster fine (k=50), build the
  transition-count affinity between micro-clusters, then spectral-cluster that affinity to produce
  k=6 macro-clusters that are *transitionally cohesive by construction*. Directly attacks the noise.

### C. Sequence smoothing (post-cluster, pre-graph)

- **C1. Median / mode filter on the label sequence per episode.** Replace `label_t` with
  `mode(label_{t-w..t+w})` for w≈3. Removes the A→B→A→B flicker that survives run-length collapse
  (which only de-duplicates *consecutive* repeats, not oscillation).
- **C2. HMM Viterbi smoothing.** Fit `hmmlearn.GaussianHMM` on the UMAP embeddings; Viterbi-decode
  to get a smoothed state path. The transition matrix is learned from data, so smoothing strength
  is automatic.
- **C3. Sticky-decoder DP with explicit stickiness knob.** Solve
  `argmax_z Σ_t log p(z_t | x_t) − λ · 1[z_t ≠ z_{t-1}]` with a T × K DP table. One scalar knob (λ)
  that the user can crank up until the graph is clean. ~30 lines; gives the user the exact lever
  they're asking for.
- **C4. Sticky HDP-HMM (Fox et al.).** Bayesian nonparametric — learns both K *and* the persistence
  bias from data. Needs `dynamax` or `pyhsmm`. The gold standard for this kind of problem.
- **C5. Change-point detection (PELT / BOCPD).** Skip per-timestep clustering entirely: detect
  change-points in the embedding stream, then cluster the resulting *segments*. `ruptures` library
  makes this ~20 lines. Naturally suppresses high-frequency noise.

### D. Graph construction

- **D1. Edge pruning by count at construction time.** Drop edges with `count < N_min` or
  `prob < p_min`, then re-normalize. One numpy block in `from_cluster_assignments`. Different from
  the current display-time filter — these edges actually disappear from the graph, so values and
  paths reflect the cleaner topology.
- **D2. MDL / BIC edge pruning.** Keep only edges whose removal would significantly worsen the
  Markov model's log-likelihood. More principled than a fixed threshold; extends the existing
  `test_markov_property` in `behavior_graph.py:830`.
- **D3. K-shortest-paths skeleton.** Compute top-K paths from START to terminals; the union of
  these paths *is* the graph that gets drawn. Everything else fades into the background.
- **D4. Merge similar-centroid nodes.** Compute pairwise cosine similarity between cluster
  centroids in embedding space; merge pairs above a threshold (or also use KL-divergence between
  their outgoing-transition distributions). Iterate to fixed point. ~30 lines, very effective for
  collapsing redundant clusters.
- **D5. Mealy-machine split — condition transitions on next-input.** Currently a node's outgoing
  distribution is `P(next | current)`. A Mealy machine uses `P(next | current, observation)`. If a
  cluster is "ambiguous" (high-entropy outgoing), split it by next-observation context. Reduces
  hub-node clutter. (ENAP §III-B.)
- **D6. Build the graph from segments, not transitions.** After change-point segmentation (C5), one
  segment per node visit. Self-loops disappear by construction. Each segment carries its mean
  embedding → segment-to-segment edges are inherently cleaner.
- **D7. L\*-style state aggregation (ENAP §IV-B).** Treat `(cluster, k-step suffix)` as the state
  identifier. Merge two states if their distributions over next-cluster are statistically
  indistinguishable. Collapses behaviorally-redundant states using the chi-squared machinery
  already in `behavior_graph.py:830`. Theoretically grounded (Myhill–Nerode).
- **D8. Stable-phase pruning (ENAP §IV-B Phase 3).** Merge `q'` into `q` when the transition
  `q→q'` is triggered by some observation `c` *and* `q` already has a self-loop on `c`. Cheap, and
  directly removes the most common kind of noise: a stable phase getting split into two
  near-identical sub-states.

### E. Visualization / layout

- **E1. X-coordinate = mean timestep per node.** Replace the layered BFS x-coordinate with the
  median timestep that hits each node. This single change makes "temporal flow left-to-right" *real*
  rather than approximate. ~20 lines in `plotting/plotly/behavior_graph.py`.
- **E2. Sankey over time bins.** Discretize timesteps into 3–5 bins; render a Sankey of cluster
  occupation per bin with transitions as flows. Plotly has a `go.Sankey`. Different visual idiom,
  but very natural for "what happens over time".
- **E3. Edge bundling.** Reduces visual clutter without information loss. Existing libraries
  (`hammer` bundling in datashader, or simple force-directed bundling) — medium implementation cost.

---

## 3. Ranked recommendations

Ranking criterion: **(visual cleanup × interpretability gain) ÷ implementation cost**.

### Tier 1 — Quick wins (do these first; combined cost: 1–2 days)

| Rank | Method | Effort | Why now |
|---|---|---|---|
| 1 | **C3 — Sticky-decoder DP with λ knob** | ~30 lines | Gives the user the exact knob they're asking for ("crank up λ → fewer transitions"). Plays well with everything downstream. |
| 2 | **D1 — Edge pruning by count at construction** | ~10 lines | Most noisy edges are count-1 or count-2; removing them is essentially free and directly addresses the complaint. |
| 3 | **E1 — Mean-timestep layout** | ~20 lines | Makes "temporal flow" real. Doesn't change topology; pure win. |
| 4 | **B1 — Auto-pick K (silhouette sweep) or drop to k=8** | config + ~20 lines | k=20 is far too many for a readable graph; lowering it is the cheapest structural fix. |
| 5 | **C1 — Median filter on label sequence** | ~10 lines | Cheapest temporal smoothing; complements C3 or stands alone. |

**Expected outcome of Tier 1:** the same graph topology but with 30–50% fewer nodes (after B1+D1
merging), no flickering edges (C1+C3), and visible left-to-right time flow (E1). Should be enough
that the user actually wants to *use* the graph.

### Tier 2 — Moderate effort, structural improvements (cost: 3–5 days)

| Rank | Method | Effort | Why |
|---|---|---|---|
| 6 | **D4 — Merge similar-centroid nodes** | ~50 lines | Often there are 2–3 clusters that are obviously the same behavior; one merge pass collapses them. |
| 7 | **B2 — Hierarchical clustering with UI zoom** | ~100 lines | Lets the user navigate granularity. Better than a single K. |
| 8 | **C2 — HMM Viterbi smoothing** | ~50 lines (`hmmlearn`) | More principled than C1/C3; learns persistence from data. |
| 9 | **D8 — ENAP stable-phase pruning** | ~30 lines | Directly removes the "split-stable-phase" failure mode. Cheap, theoretically motivated. |
| 10 | **A1 — Wider window + smoothing** | ~10 lines | Trivial; gets you a chunk of the way to temporal coherence at the feature level. |

### Tier 3 — Bigger structural changes (cost: 1–2 weeks)

| Rank | Method | Effort | Why |
|---|---|---|---|
| 11 | **D6 — Change-point segments + segment-graph** (with **C5**) | ~150 lines | Drops the per-timestep paradigm. Result: one segment per phase visit, no oscillation possible. |
| 12 | **D7 — L\*-style state aggregation** | ~200 lines (chi-sq infra already exists) | Principled, formally justified; reuses the existing Markov-property test. Likely the highest-effectiveness pure-graph method. |
| 13 | **B4 — Spectral clustering of transition graph** | ~100 lines | Macro-clusters that are transitionally cohesive *by construction* — exactly what we want. Two-stage: micro→macro. |
| 14 | **A2 — Add action features to embedding** | ~100 lines (need action loading) | The influence vector is the wrong space. Actions directly encode behavior. |

### Tier 4 — Research project (cost: weeks)

| Rank | Method | Effort | Why |
|---|---|---|---|
| 15 | **A3 + D5 — Full ENAP recipe (RNN phase embedding + Mealy automaton)** | 1–3 weeks | The full paper. Most powerful, most expensive. Worth it for a final polish or paper figure. |
| 16 | **C4 — Sticky HDP-HMM** | 1–2 weeks (`dynamax`) | Bayesian gold standard. Learns K and persistence jointly. Strong theoretical pedigree. |

### Tier 5 — Specialized / situational

| | Method | Why situational |
|---|---|---|
| 17 | **D3 — K-shortest-paths skeleton** | Great for storytelling, hides structure that may matter. Use as an *overlay*, not the primary view. |
| 18 | **E2 — Sankey diagram** | Different idiom, doesn't replace the graph but complements it. |
| 19 | **B3 — HDBSCAN + KNN noise propagation** | Auto-K is nice but Tier 1 already covers most of the win. |
| 20 | **D2 — MDL/BIC edge pruning** | More principled than D1 but ~5× the code for marginally better result. |
| 21 | **D5 alone — Mealy split without RNN** | Effective in isolation, but the gain compounds with A2/A3. |
| 22 | **E3 — Edge bundling** | Cosmetic; Tier 1 makes it unnecessary. |

---

## 4. Suggested implementation order

**Sprint 1 (this week):** Tier 1 items 1–5, in order. Land them as independent commits behind config
flags (`clustering_smooth_method`, `graph_min_edge_count`, `graph_layout_mode`, `clustering_auto_k`),
all default-off so existing graphs don't change. Quick A/B in the Policy Doctor Streamlit tab.

**Sprint 2:** Items 6, 9, 10 — they share infrastructure with Tier 1. Items 7 and 8 if Tier 1 wasn't
enough.

**Sprint 3 (only if needed):** Pick one of {D6+C5, D7, B4} — these are alternative structural
overhauls and trying all three is wasted work. My recommendation is **D7 (L\*-style aggregation)**
because (i) it reuses the existing chi-squared test, (ii) it's theoretically grounded, and (iii) it
operates on the graph the rest of the pipeline already produces, so it composes cleanly.

**Sprint 4+ (research):** ENAP. Open question: is there a clear paper story / experiment that
justifies the cost? Otherwise stop at Sprint 3.

---

## 5. Implementation log (worktree `feat/graph-simplification`)

### What I built

Module **`policy_doctor/behaviors/graph_simplification.py`** — pure-function transformations:

| Function | Method | Status |
|---|---|---|
| `median_filter_labels` | C1 — per-episode mode filter | ✅ |
| `sticky_decoder` | C3 — sticky DP with explicit λ knob | ✅ |
| `hmm_smooth` | C2 — Gaussian-HMM Viterbi | ✅ (uses `hmmlearn`) |
| `prune_edges_by_count` | D1 — count-threshold edge prune | ✅ |
| `prune_edges_by_prob` | D1 — prob-threshold edge prune | ✅ |
| `merge_similar_centroids` | D4 — cosine-similar node merge | ✅ |
| `stable_phase_prune` | D8 — ENAP stable-phase merge | ✅ |
| `auto_k_kmeans` | B1 — silhouette-based auto-K | ✅ |
| `spectral_transition_clustering` | B4 — spectral on transition graph | ✅ |
| `change_point_segmentation` | C5 + D6 — PELT change-points + KMeans | ✅ (uses `ruptures`) |
| `temporal_layout` | E1 — rank-based mean-timestep x-coord | ✅ |

Also extended `plotting/plotly/behavior_graph.py:create_behavior_graph_plot` to accept an
optional `pos` argument (caller-supplied layout).

Streamlit app **`policy_doctor/streamlit_app/graph_simplification_app.py`** — 7 tabs:

1. Temporal smoothing (median / sticky / HMM)
2. Edge pruning (count / probability)
3. Node merging (cosine-centroid / ENAP stable-phase)
4. Re-clustering (auto-K / spectral / change-point)
5. Layout only (BFS-layered vs temporal)
6. Combined pipeline (stack methods)
7. Compare representations (different feature spaces, same simplification)

Run with: `streamlit run policy_doctor/streamlit_app/graph_simplification_app.py --server.port 8530`

### Findings on `transport_mh_jan28` (InfEmbed, k=20, baseline 23 nodes / 140 edges)

| Method | nodes | edges | Δedges | Notes |
|---|---|---|---|---|
| Baseline | 23 | 140 | — | k=20 KMeans + run-length collapse |
| Median filter w=5 | 23 | 120 | −14% | Cheap, modest impact |
| Sticky DP λ=2 | 23 | 87 | −38% | Big improvement, smooth knob |
| Sticky DP λ=10 | 17 | 33 | −76% | Some clusters become unused at high λ |
| Prune count ≥ 5 | 23 | 59 | −58% | One-line fix, strong baseline |
| Prune count ≥ 10 | 23 | 33 | −76% | |
| Prune prob ≥ 0.1 | 23 | 49 | −65% | |
| Cosine centroid merge ≥ 0.9 | 9 | 28 | −80% | Aggressive consolidation |
| Stable-phase prune | 17 | 85 | — | ENAP-style, conservative |
| HMM (n_states=8) | 11 | 39 | −72% | Re-discovers structure |
| Change-point (k=8, pen=10) | 11 | 46 | −67% | |
| Auto-K (silhouette → k=4) | 7 | 18 | **−87%** | **Cleanest result by far** |
| Spectral (k_macro=8) | 11 | 37 | −74% | |

**Headline result.** Auto-K selected **k=4** as optimal by silhouette score (0.55 at k=4
vs 0.46–0.50 at k=10+). Combined with the temporal layout, this produces an extremely
readable graph with clear START → Approach → Manipulate → Final → SUCCESS/FAILURE flow.

This is strong evidence that **k=20 is too granular for this task** — the data only
supports ~4 well-separated behavioral modes.

### Decisions

1. **Default re-running clustering with silhouette-based K** is the single highest-impact
   intervention. Lower K dramatically reduces both nodes and edges.
2. **Temporal layout (E1) is essential** — even when graph structure is identical, the
   BFS layered layout is misleading. Replacing it with rank-of-mean-timestep gives the
   user a real "time arrow" left-to-right.
3. **Sticky DP (C3)** is the best smoothing method for the user's intent: a single λ
   knob the user can dial. Defaults to λ=5 for combined pipeline.
4. **Edge pruning by count (D1)** at construction time is more useful than the existing
   display-only `min_probability` because it persists into downstream computations.
5. **Centroid merging (D4)** is dangerous at low thresholds (≤0.85) — it can collapse
   semantically distinct clusters. Recommend 0.9–0.95 as user-facing range.
6. **Stable-phase prune (D8)** is currently conservative — the implementation requires
   both dominant-incoming AND share-of-source > 30%. It's a useful complement to D1 but
   not a primary tool on its own.

### Representation comparison

The repo has clusterings on a different transport task (`mar27`, in `e1_experiments/`)
across many feature spaces: `policy_emb`, `bottleneck_plan_t0`, `encoder_plan_t0`,
`decoder_plan_t0`, `bottleneck_exec_t0`, `state_full`, `state_action_full`.

These were trained on a different policy/checkpoint, so apples-to-apples comparison
against `transport_mh_jan28` requires care. Tab 7 of the app surfaces them under their
own task family, so the user can A/B different representations on the same task.

On `transport_mh_jan28` (the actual task with the noisy graph the user complained about),
only two representations are available: `infembed` and `trak`. After sticky+prune
simplification:

| Representation | Raw edges | Smoothed edges | Clean? |
|---|---|---|---|
| InfEmbed k=20 | 65 | 16 | ✓ |
| TRAK k=20 (5 seeds) | 80–83 | 29–33 | partial |

InfEmbed produces a cleaner graph **after simplification**. TRAK clusterings lack
`embeddings_reduced.npy` so sticky-decoder cannot apply — they only get edge pruning.

**Open question for future work:** does re-running the clustering on this task with
`policy_emb` (the policy's own bottleneck features, ENAP-style) give a cleaner *raw*
graph? The infrastructure to do this exists in
`policy_doctor/curation_pipeline/steps/run_clustering.py` via the
`clustering_influence_source` config knob, but requires the policy checkpoint to be on
disk.

### Not yet implemented (deferred)

- **D7 — L\*-style state aggregation.** The chi-squared infrastructure exists
  (`behavior_graph.py:test_markov_property`) but the merge logic to actually collapse
  redundant states wasn't written. Tier 3 in the original brainstorm.
- **A3 — ENAP RNN phase-aware embedding.** Tier 4. Requires a real training run; skipped.
- **C4 — Sticky HDP-HMM.** Tier 4. The Gaussian-HMM (C2) covers most of the practical
  value with much less complexity.

## 6. Cross-cutting design notes

- **Keep everything reversible.** Wrap each new step in a config flag. The Streamlit app should let
  the user toggle smoothing / pruning / merging / layout and see the result live.
- **Save intermediate artifacts.** The smoothed-label sequence, the merged graph, etc. should be
  cached per clustering run (alongside the existing `clustering_models.pkl`), so the user-study app
  can show "graph at smoothing strength λ" without re-running clustering.
- **Re-run downstream value computation after any merge/prune.** `compute_values()` is fast but the
  cached node-V values must be invalidated.
- **Watch for breaking the slice search.** `get_rollout_slices_for_paths` depends on the
  cluster-label sequence; smoothing changes it. Test that "click node → see rollout slices" still
  produces semantically coherent segments.
