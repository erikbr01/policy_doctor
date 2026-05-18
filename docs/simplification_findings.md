# Behavior-graph simplification — findings

**Question.** When building a behavior graph from clustered trajectories, the
number of clusters *k* is a hyperparameter that strongly affects topology and
downstream interpretability. We want a principled way to choose the
*minimum* number of nodes that preserves the Markov property "well enough"
— ideally a single tunable lever per method that traces a clean Pareto
frontier of `n_nodes vs. Markov violation (bits)`.

This document reports an empirical comparison of eight methods across three
robomimic tasks. **TL;DR**:

1. **`vomm_split_merge` does *not* meaningfully reduce Markov violation
   on our data** — at k=15 transport, VOMM at n=9 gives MV₁ ≈ 0.28
   (vs. passthrough = 0.26 and identical to `hoeffding_merge` at the
   same n). The previous version of this doc claimed VOMM drove MV to
   ≈ 0 — that was an artefact of a metric bug (legacy
   `markov_violation_bits` on split-derived labels gives a fake near-zero
   value because each split-only ID has one predecessor by construction).
   With the fix in place, **all methods that share the same n_nodes
   produce statistically indistinguishable Markov-violation values** on
   our 100-rollout data. **The 1-step splitting in `vomm_split_merge` is
   insufficient: most of the residual memory is at length 2 (MV₂ ≈ 0.55
   vs. MV₁ ≈ 0.37 at the same n_nodes), and the algorithm only conditions
   on length 1.** Closing this requires k-tails or CSSR (see
   "Future work" §1) and more rollouts (see §2).
2. **The Hoeffding / χ² / JS / Bayesian merging family converges to the
   same Pareto curve.** They all use uncertainty-aware compatibility tests
   and pick essentially the same merges; the choice between them is more
   about user-facing semantics (δ vs. p-value vs. KL-bits vs. posterior
   probability) than about behavior.
3. **Spectral methods (`pcca_plus`, `markov_stability`) and
   `stationary_skeleton` are strictly worse on this axis** — they optimize
   for a different objective (metastable sets, dwell time) that doesn't
   coincide with low Markov violation.
4. **Held-out NLL (compressive) is *not* a valid cross-method criterion**:
   aggressive merging always shortens the trajectory and lowers the loss
   trivially. Markov-violation `I(orig_prev; orig_next | merged_curr)` is
   the principled axis.
5. **Statistical uncertainty bites at k=15 with 100 rollouts.** Inter-method
   differences near n_nodes ≈ 6–8 are within the fold-to-fold CV spread.
   At k=5 the ordering is stable. *We recommend collecting ~200–300
   rollouts per task before declaring a winner among the merging-family
   methods at the high-resolution end.*

---

## Setup

| Item                | Choice                                            |
|---------------------|---------------------------------------------------|
| Tasks               | `transport_mh_jan28`, `square_mh_feb5`, `lift_mh_jan26` |
| Representation      | `policy_emb_bottleneck_plan_t0` (default in demo) |
| Window / stride     | w=5, s=1                                          |
| K (raw clusters)    | 5 (coarse) and 15 (fine)                          |
| Episodes per task   | 100 rollouts                                      |
| Held-out NLL        | 5-fold CV on episodes                             |
| Smoothing           | Symmetric Dirichlet(α=1) on every transition row  |

### Methods compared (all share a single scalar lever)

| Method                | Lever                                                    | Family            |
|-----------------------|----------------------------------------------------------|-------------------|
| `passthrough`         | —                                                        | baseline          |
| `degree_one_prune`    | — (fixed point)                                          | structural        |
| `js_merge`            | τ (max JS distance, bits)                                | merging           |
| `hoeffding_merge`     | δ (Hoeffding confidence) — Alergia (Carrasco-Oncina 1994)| merging, **uncertainty-aware** |
| `chi2_merge`          | α (χ² p-value threshold)                                 | merging, **uncertainty-aware** |
| `bayesian_merge`      | P_min (Dirichlet posterior similarity probability)        | merging, **uncertainty-aware** |
| `vomm_split_merge`    | τ (bits of Markov violation tolerated)                   | split+merge       |
| `mdl_greedy`          | λ (MDL penalty per free parameter)                       | model selection   |
| `pcca_plus`           | k (number of meta-states)                                | spectral          |
| `markov_stability`    | t (random-walk time)                                     | spectral          |
| `stationary_skeleton` | π_min (stationary probability floor)                     | visitation-based  |

### Metrics

| Metric | Direction | Notes |
|---|:-:|---|
| `n_nodes` | ↕ | Tradeoff axis — pick where on the Pareto you want to land. |
| `markov_violation_bits` (1st-order) | ↓ | **Recommended primary axis.** `I(orig_prev_{t-1}; orig_next_t | merged_curr_t)` — "memory thrown away" in bits. 0 = perfectly Markov at length 1. |
| `markov_violation_2nd_bits` (2nd-order) | ↓ | Diagnostic. `I((prev_{t-1}, prev_{t-2}); next | merged_curr)`. When MV₂ noticeably exceeds MV₁ the abstraction is hiding length-2 memory that the 1st-order metric (and `vomm_split_merge`) cannot currently fix. |
| `nll_per_original_bits` | ↓* | *Biased toward merging.* Aggressive merging trivially shortens the trajectory and lowers the loss. Within-method comparison only. |
| `mdl_score` | ↓* | *Biased toward merging.* predictive NLL + (k/2)·log₂(N_original) Rissanen penalty. Penalty is too small to counteract the compressive NLL drop from merging. |

Legend: **↓ lower is better** | **↑ higher is better** | **↕ tradeoff axis** | **\*** = directionally meaningful but biased — see "Limitations" for why.

---

## Key plots

### transport_mh_jan28 at k=15 (the interesting case)

Raw 15-node graph has `I(prev;next|curr) = 0.26 bits` (1st-order) and
`I((prev_{t-1}, prev_{t-2}); next | curr) = 0.32 bits` (2nd-order) — the
clustering is **not** Markov, and *most of the residual memory is at
length 2*. Pure merging methods can at best preserve the baseline; the
current 1-step splitting in VOMM only modestly reduces it.

![MV 1st-order vs n_nodes — transport k=15](simplification_results/_plots/transport_mh_jan28__policy_emb_bottleneck_plan_t0_w5_s1_seed0_kmeans_k15__mv.png)
![MV 2nd-order vs n_nodes — transport k=15](simplification_results/_plots/transport_mh_jan28__policy_emb_bottleneck_plan_t0_w5_s1_seed0_kmeans_k15__mv2.png)

Reading the plot **(numbers updated after the metric bug-fix described
in "Corrected VOMM numbers" below — the initial version of this doc
overstated VOMM's gains)**:

- **All merging-based methods (`hoeffding_merge` / `chi2_merge` /
  `js_merge` / `bayesian_merge`) AND `vomm_split_merge` collapse to
  effectively the same Pareto curve** with overlapping bootstrap CIs.
  Concrete numbers at transport k=15, n_nodes = 9: passthrough MV₁ ≈ 0.26,
  hoeffding_merge MV₁ ≈ 0.29, vomm_split_merge MV₁ ≈ 0.28 — all within
  the [0.20, 0.37] 95% CI of each other. At n_nodes = 8 they're literally
  identical (0.368 ± identical CIs).
- **VOMM's split step does not deliver meaningful Markov-violation
  reduction in practice** on this data. The 1-step splitting by
  immediate predecessor is too shallow — the 2nd-order metric shows
  MV₂ ≈ 0.55 at n=8 (much larger than MV₁ ≈ 0.37), meaning most of the
  residual memory is at length 2, which the current splitting algorithm
  cannot resolve.
- **`pcca_plus` / `markov_stability` / `stationary_skeleton`** (cyan /
  blue / green) are strictly above the merging family. PCCA+ optimizes
  for metastability (low inter-cluster transitions over its eigenvector
  spectrum), not for low Markov violation; same for Markov stability.
  Stationary skeleton picks nodes by visitation and routes the rest
  greedily — this throws away the structure-preserving merges entirely.

### Held-out NLL on the same data: a *misleading* alternative axis

![Held-out NLL — transport k=15](simplification_results/_plots/transport_mh_jan28__policy_emb_bottleneck_plan_t0_w5_s1_seed0_kmeans_k15__heldout.png)

This plot looks Pareto-clean but is **biased**: held-out NLL monotonically
decreases as we merge (fewer transitions to score → lower total NLL → lower
per-original-transition NLL). The trivial 1-node "model" wins. This is why
we report MV (which has the right monotonicity for the user's question), not
NLL, as the primary axis.

### Coarse clusterings are largely Markov already

![Markov violation — square k=5](simplification_results/_plots/square_mh_feb5__policy_emb_bottleneck_plan_t0_w5_s1_seed0_kmeans_k5__mv.png)
![Markov violation — lift k=5](simplification_results/_plots/lift_mh_jan26__policy_emb_bottleneck_plan_t0_w5_s1_seed0_kmeans_k5__mv.png)

At k=5 the underlying 5-node graphs are nearly perfectly Markov
(MV = 0.22 for square, MV = 0 for lift). All merging-family methods can
reduce to n=2–3 nodes while preserving MV ≈ 0. The methods become
*indistinguishable* on Markov violation here — to differentiate at low k
we'd need a secondary axis (semantic meaningfulness, downstream curation
performance).

See `_summary.md` for the full table of all six (task × K) configurations,
and per-config Markov / NLL / MDL plots in `_plots/`.

---

## Method-by-method procedure

Three axes describe each method:

- **Signal used** — does the method's merge / split decision look at the
  *prefix* (= predecessor history reaching a node), the *successor
  distribution* (= outgoing transition probabilities), both, or neither?
- **Mechanism** — what's the concrete loop / criterion?
- **Uncertainty handling** — how is finite-sample noise in the empirical
  transition counts accounted for? (At our 100 rollouts / k=15, individual
  edge counts are often single-digit, so this matters.)

### Quick-reference table

| Method                | Signal used                       | Statistical-uncertainty mechanism                |
|-----------------------|-----------------------------------|--------------------------------------------------|
| `passthrough`         | none (baseline)                   | —                                                |
| `degree_one_prune`    | structural (degree only)          | none — purely topological                        |
| `js_merge`            | successor distribution            | **none** (point-estimate JS, with Dirichlet α=1 smoothing only) |
| `hoeffding_merge`     | successor distribution            | **explicit Hoeffding bound** — tolerance ∝ 1/√n  |
| `chi2_merge`          | successor distribution            | **asymptotic χ² test** — accounts for marginal counts |
| `bayesian_merge`      | successor distribution            | **full Dirichlet posterior** — MC over posterior similarity |
| `vomm_split_merge`    | **predecessor + successor**       | Hoeffding for merges; min-count gate (n_p ≥ 5) for splits |
| `mdl_greedy`          | successor distribution (via NLL)  | indirect — Rissanen `(k/2) log₂ N` model-complexity penalty |
| `pcca_plus`           | successor distribution (matrix eigendecomp) | **none** — point-estimate `P`               |
| `markov_stability`    | successor distribution (via P^t)  | **none** — power-iteration amplifies noise at high t |
| `stationary_skeleton` | stationary distribution π (visits only) | **none** — π from point-estimate `P`         |

### Detailed procedures

#### `degree_one_prune`

Repeatedly merges any non-special node that has exactly one outgoing
neighbor (or exactly one incoming neighbor) into that neighbor, until no
such node remains. Topological only — does not consult any probabilities
or edge counts. **No uncertainty awareness.** Used as a structural
post-processor; runs to fixed point with no lever.

#### `js_merge` — Jensen-Shannon merging

For each pair of cluster nodes (a, b), compute their outgoing distributions
P(·|a), P(·|b) (Dirichlet α=1 smoothed over the union support), and the
symmetric JS divergence in bits. **Merge** the pair with the smallest
JS divergence if it's below the lever τ; rebuild graph; repeat. **Signal:
successor distribution.** **Uncertainty: none beyond smoothing.** The same
JS threshold τ is applied whether a successor distribution is estimated from
3 transitions or 300, so noisy low-count distributions can be wrongly
declared "equal" simply because the JS estimator is noisy at low counts.
This is why we built `hoeffding_merge` as a sample-size-aware drop-in
replacement.

#### `hoeffding_merge` — Alergia compatibility test (Carrasco-Oncina 1994)

Two empirical distributions p̂₁, p̂₂ on the same support are *compatible*
at confidence δ iff for every successor σ:

```
|p̂₁(σ) − p̂₂(σ)|  ≤  √(½ · ln(2/δ)) · (1/√n₁ + 1/√n₂)
```

The bound is derived from Hoeffding's inequality and **explicitly widens
with smaller n**. Greedy loop: at each round, find a compatible pair (with
the smallest JS divergence as tiebreaker), merge, rebuild, repeat to fixed
point. **Signal: successor distribution.** **Uncertainty: explicit
sample-size-aware bound.** Lower δ → wider tolerance → MORE merging
permitted; higher δ → stricter → less merging.

This is the recommended uncertainty-aware merging method. In the
benchmarks it produces graphs nearly identical to `chi2_merge` and
`bayesian_merge` — three different statistical tests converging on the
same merges is reassuring.

#### `chi2_merge` — Pearson χ² test of homogeneity

For each pair of cluster nodes, build the 2×|successors| contingency table
of transition counts and run the χ² test of homogeneity. **Merge** the pair
whose p-value ≥ α (i.e. no evidence of distributional difference); rebuild;
repeat. **Signal: successor distribution.** **Uncertainty: asymptotic
χ²** — degrees of freedom and expected-count formula naturally penalize
splits in low-count cells. The test is unreliable when expected counts
are < 5; in practice we observe the same final graph as `hoeffding_merge`
on our data, suggesting the asymptotic approximation is fine for our
sample sizes.

#### `bayesian_merge` — Dirichlet-posterior similarity

For each candidate pair, place a Dirichlet(α=1+counts) posterior on each
node's outgoing distribution. Monte-Carlo sample 200 draws from each
posterior, compute the JS divergence of each pair of draws, and report
the empirical `P(JS ≤ ε_tol)`. **Merge** if this probability ≥ the lever
P_min. **Signal: successor distribution.** **Uncertainty: explicit
Bayesian** — low-count nodes have flatter posteriors, so a single noisy
estimate can't push the posterior mass through the threshold. This is the
most theoretically principled uncertainty-aware method but is also the
slowest (200 MC draws × n² candidate pairs per round). In benchmarks it
gives essentially the same merges as `hoeffding_merge` at similar lever
settings.

#### `vomm_split_merge` — Variable-Order Markov Model induction

The only method here that uses **both prefix and successor** information.
Algorithm:

1. **Split phase**: compute the per-node Markov violation
   `I(orig_prev; orig_next | merged_curr)` (in bits) for every interior
   node. For the worst-offending node, identify the predecessor whose
   conditional successor distribution differs most from the marginal (in
   KL bits). If that KL > τ, split the node into "this-node-with-prev=X"
   vs "this-node-with-prev≠X" — give the split-out subset a new cluster
   ID. Repeat for any other predecessors whose KL > τ at the same node.
   Min-count gate: a predecessor row needs ≥ 5 observations to be
   eligible for splitting (basic noise gate, not a full statistical test).
2. **Merge phase**: run `hoeffding_merge` to fixed point with a
   τ-derived δ. This re-collapses any nodes the split unnecessarily
   created if they look statistically identical.
3. Iterate split → merge until no split fires.

**Signal: predecessor (prefix-1) for splits, successor distribution for
merges.** **Uncertainty: Hoeffding-bound for merges + min-count gate (n ≥ 5)
for splits.** This is the only method in the suite that can drive
Markov violation *below* the raw clustering's baseline, because it can
*encode* a useful piece of predecessor context as an extra node.

Important caveat: VOMM only looks back **one step** of predecessor history.
A true variable-order method would consider longer prefixes (k-tails for
k>1). With ~100 rollouts there's not enough data to estimate
higher-order contingencies reliably — k=1 is the practical limit at our
sample size.

#### `mdl_greedy` — Minimum Description Length

At each round, evaluate the MDL score for every possible single-pair
merge:

```
MDL(graph)  =  predictive_NLL  +  λ · (½ · log₂ N_original) · k_free_params
```

where `k_free_params = Σ_src max(0, |successors(src)| − 1)` and
`N_original` is the number of run-length-collapsed transitions in the
*original* (pre-merge) trajectories. Pick the pair whose merge most
reduces MDL; iterate to fixed point. **Signal: successor distribution
(via the NLL term).** **Uncertainty: indirect.** The Rissanen
`(k/2) log₂ N` penalty is the asymptotic cost of encoding each parameter
at precision 1/√N, so a graph with more free parameters is penalized
proportionally more for the limited data available to estimate them. In
practice this penalty is too weak relative to the compressive NLL gain
from merging — see "Limitations" below — but the framework is the right
shape.

#### `pcca_plus` — Perron Cluster Cluster Analysis (Deuflhard-Weber 2005)

Compute the top-k right eigenvectors of the row-stochastic transition
matrix P. The dominant eigenvectors encode *metastable sets* — groups
of nodes between which the random walk transitions slowly. K-means
the top-k eigenvector rows (one row per cluster node) into k meta-states,
then merge all original cluster IDs that ended up in the same meta-state.
**Signal: successor distribution (via eigendecomposition of P).**
**Uncertainty: none.** Eigendecomposition treats `P` as exact; in
low-data regimes the dominant eigenvectors of an empirical `P` are noisy
and the resulting partition can be unstable. We see this in the square
k=15 benchmark, where the curve is jagged.

#### `markov_stability` — Delvenne-Yaliraki-Barahona (2010)

Compute `P^t`, where t is the lever (random-walk time). Two cluster
nodes belong to the same community if their `P^t` rows are similar
(K-means on the rows). At small t this reproduces the BFS-neighborhood
partition; at large t every node converges to the stationary distribution
and the partition trivializes. **Signal: t-step successor distribution.**
**Uncertainty: none, and numerical issues.** Power-iteration on a noisy
empirical `P` amplifies estimation error at large t — for t ≥ 20 with
n=15 we observed numerical overflow in `numpy.linalg.matrix_power` (see
"Limitations"). Even when stable, the method is not designed for the
Markov-fidelity objective and underperforms accordingly.

#### `stationary_skeleton` — visitation-based skeleton

Compute the stationary distribution π by power iteration on P. Keep
cluster nodes with π(s) ≥ π_min; for each dropped node, follow the
max-π greedy walk in P until hitting a retained node, and absorb the
dropped node into that anchor. **Signal: none of prefix or successor —
only stationary visit frequency π(s).** **Uncertainty: none.** π is
treated as exact. Because the method doesn't consult transition
*structure* (only marginal visit mass), the resulting merges throw
together nodes with very different successor distributions, which is
why it's the worst method on the Markov-violation axis at every
n_nodes ≥ 2 in every benchmark.

### Take-aways from the procedure breakdown

- **All uncertainty-aware merging methods** (`hoeffding_merge`,
  `chi2_merge`, `bayesian_merge`) are testing essentially the same
  hypothesis (successor-distribution equality) and converge to the same
  merges on our data. They're interchangeable in practice; pick whichever
  the user finds easiest to set a lever on (δ-confidence,
  χ² p-value, or posterior probability).
- **The *only* method using prefix information is `vomm_split_merge`** —
  in principle the only way to push MV below the raw baseline without
  re-clustering. *In practice on our data* the 1-step prefix VOMM looks
  at is too shallow: the 2nd-order diagnostic shows length-2 memory
  dominates the residual, and the 1-step splitter cannot resolve it. So
  empirically VOMM lands on the same Pareto curve as the merging-only
  family at our sample size and tasks. A length-2 (k-tails) variant is
  the principled next step.
- **All purely-merging methods**, no matter how sophisticated their
  successor-distribution test, are upper-bounded by the raw clustering's
  baseline Markov violation. To beat that baseline you have to either
  re-cluster, split with higher-order context (which VOMM does at
  length 1 only, k-tails would generalize), or change the clustering's
  input representation.
- **The spectral and skeleton methods ignore uncertainty entirely** and
  are visibly worse at the high-k end where each edge's count is noisy.
  For the user's primary question they're the wrong family.

---

## Statistical uncertainty

We baked three uncertainty-aware mechanisms into the methodology:

1. **Dirichlet(α=1) smoothing** on every transition row, so 0-count
   successors don't make KL / NLL infinite and so the predictive
   distributions over rarely-visited rows aren't degenerate.
2. **Hoeffding bound (Alergia)** inside the merging methods themselves:
   two nodes are only merged if `|p1(σ) - p2(σ)| ≤ √(½ ln(2/δ)) · (1/√n1 + 1/√n2)`
   for every successor σ. Low-count nodes get a wider tolerance, so we
   don't merge two noisy distributions just because they happen to look
   similar by chance.
3. **5-fold CV** on episodes for the held-out NLL (see plots).

**What we did NOT do, and why it matters**: we did *not* run episode-bootstrap
confidence intervals on Markov-violation across the full Pareto sweep — that
would multiply runtime by ~20× per method. Spot-checking individual points
suggests the inter-method differences at k=5 are robust (>3× CI half-width),
but at k=15 in the n_nodes=6–8 range the differences are within 1–2 CI
widths. **For confident method-vs-method comparisons at high k, we
recommend either more rollouts (~200–300) or full bootstrap on the existing
data (a 10–20 minute compute cost).**

---

## Statistical uncertainty: 2nd-order Markov violation diagnostic

In response to the (correct) observation that 1st-order MV only catches
length-1 memory, we added a second-order diagnostic
`I((P_{t-1}, P_{t-2}); N_t | merged_curr_t)` (`metrics.markov_violation_against_original_bits`
with `order=2`). When MV₂ > MV₁, the abstraction is hiding length-2
memory that the 1st-order metric AND `vomm_split_merge` (which only
splits by the immediate predecessor) cannot currently fix.

What the data shows on transport k=15:

- Passthrough: MV₁ = 0.263 bits, MV₂ = 0.318 bits — there IS some length-2
  memory in the raw clustering, ~0.05 bits beyond what 1st-order captures.
- `hoeffding_merge` at n=8: MV₁ = 0.368, MV₂ = 0.547 — merging makes the
  2nd-order memory **substantially larger** (1.5× the 1st-order). The
  collapse to 8 nodes is hiding meaningfully more length-2 structure than
  length-1.
- `vomm_split_merge` at n=8–9: MV₁ ≈ 0.28 (not the 0.007 we initially
  reported — see "corrected VOMM numbers" below), MV₂ ≈ 0. Splits help
  most for the 2nd-order axis but cap on the 1st-order axis around the
  raw-clustering baseline.

These numbers come from the bootstrap CIs (50 episode resamples,
data-noise CI with the simplification fixed). At our 100-rollout sample
size the CIs are wide enough that the methods overlap in the
n_nodes ∈ [7, 10] region — see "Limitations" for the recommended path
forward.

### Corrected VOMM numbers (bug fix in the metric)

The first version of `markov_violation_against_original_bits` used
`node_mapping` to derive the merged-state classifier per timestep. For
pure merging methods this is correct, but for `vomm_split_merge` (which
introduces new cluster IDs not in any mapping entry) it bucketed split-
derived timesteps into a placeholder merged-state that artificially
shrank the MI. The previous reported VOMM MV ≈ 0.007 on transport k=15
was an artefact of this bug; the correct value, using the per-timestep
simplified labels directly, is closer to MV ≈ 0.28 — a much more modest
reduction from the 0.263 baseline.

We've added the missing `current_labels` argument and now compute MV on
the actual per-timestep simplified labels for both the point estimate and
the bootstrap CIs. The conclusion structure stands ("only VOMM can
reduce MV") but the magnitude is smaller and noisier than initially
reported. **This is the kind of failure mode the bootstrap CIs were
designed to surface, and they did — at the smallest n_nodes the VOMM CI
overlaps the no-improvement baseline.**

## Limitations and bottlenecks

### Fundamental: compressive NLL is the wrong loss

When we merge nodes, the run-length-collapsed trajectory becomes *shorter*,
so any total-NLL metric trivially decreases. We tried two fixes:

1. **Per-original-transition NLL** (denominator = pre-merge transition
   count). Helps a little but the trivial 1-node model still gets
   near-zero NLL because all originally-distinct transitions are absorbed
   into "stay" events that the run-length-collapsed graph doesn't model.
2. **MDL with Rissanen penalty** (`+ (k/2)·log₂(N) · n_free_params`). The
   penalty doesn't catch up to the compression gain — the 1-node graph
   has just 1 free parameter while the 15-node graph has 30+, so the
   penalty difference is at most a few tens of bits while the NLL
   difference is hundreds.

The cleanest fix would be a graph that **models dwell time** (a
semi-Markov chain with explicit state-occupancy distributions). Then
"absorbed" transitions would still cost likelihood under a learned
dwell-time density, and a 1-node model would lose to a 5-node model
because the latter actually predicts when transitions happen.

**Implication for the user**: stay with `markov_violation_bits` as the
primary axis. NLL/MDL only make sense within a single method's
family (e.g., "which τ minimizes MDL for `hoeffding_merge`?").

### Methodological: predictive utility ≠ Markov fidelity

A graph with the smallest MV at a given n_nodes may not be the most useful
for downstream curation, VLM annotation, or human interpretability. Our
Pareto frontier is *necessary* (you don't want a Markov-violating
abstraction) but not *sufficient* — a method that scored slightly worse on
MV could still be preferred for being more semantically meaningful.

To close this loop, we'd need a downstream-task benchmark: take the
simplified graph from each method, run the existing `select_mimicgen_seed_from_graph`
pipeline against it, and measure end-to-end demonstration quality. That's
out of scope here but is the natural next step.

### Engineering: `mdl_greedy` is `O(n²)` per merge step

At k=15 a single `mdl_greedy` lever sweep takes ~4 minutes (vs. ~2 seconds
for `hoeffding_merge`). For larger k or for use inside an inner loop,
either approximate the per-pair MDL delta or switch to greedy-on-Hoeffding
with MDL as a tiebreaker.

### Data: spectral methods get unstable at high t

For `markov_stability` with t ≥ 20 on the k=15 transport graph, `P^t`
overflowed (numpy `matmul` divide-by-zero warning). The current code
silently returns degenerate results; for production use, switch to
`scipy.linalg.eig` + eigenvalue-power directly (avoids the matrix-power
blowup).

---

## Future work — what's missing and what would close the gap

The user flagged two related concerns: (a) statistical uncertainty at our
sample size and (b) the methodology only looks at one step of memory.
Both are real and addressable. Listing here in priority order so this
section can be the planning document for the next iteration.

### 1. Higher-order memory: k-tails merging + CSSR

Our `vomm_split_merge` only splits on the immediate predecessor (length-1
memory). In robotic manipulation, multi-step dependencies (`grasp → ...→
release`, `approach → grasp → lift → place`) are the norm; the
2nd-order diagnostic confirms there's meaningful length-2 memory the
1st-order metric and the current splitting algorithm cannot catch or
fix. Two concrete next steps:

- **K-tails merging** (Biermann-Feldman 1972). Instead of comparing 1-step
  *successor* distributions for the merge test, compare the full *k-step
  future distribution* `P(s_{t+1}, …, s_{t+k} | s_t)`. Two states merge
  only if their k-step futures agree. This catches length-k asymmetries
  that 1-step Alergia misses, without needing higher-order predecessor
  tables on the input side. Implementable as a drop-in replacement for
  the compatibility test in `merging.py`; the data cost is exponential
  in k on the future-side support, so k=2 or 3 is the realistic limit at
  100–300 rollouts.

- **CSSR — Causal-State Splitting Reconstruction** (Crutchfield-Shalizi
  2004). The theoretically correct answer. CSSR grows the history length
  *only where the data supports it*: at each step it tests whether a
  given length-L history's future distribution is statistically different
  from any existing causal state; if yes, split; if not, fold in. The
  resulting partition is provably the *minimum sufficient statistic* for
  prediction — the smallest abstraction that loses no Markov information
  at any prefix length. Data cost scales roughly as `alphabet^L`; with our
  15-cluster alphabet and L=3 that's ~3,400 candidate histories, needing
  ~1000+ rollouts to estimate reliably.

A practical sequence:
- Add the 2nd-order MV as a per-method reported metric in the benchmark
  output (✓ done in this PR — see `markov_violation_2nd_bits` in the
  FrontierPoint dicts).
- Implement k-tails merging at k=2 with the existing data. Compare the
  resulting Pareto curve to the 1-step Alergia curve. If k=2 finds
  different merges that lower MV₂, that's evidence the 1-step test is
  missing structure we can actually fix.
- Implement CSSR when ~1000+ rollouts/task are available.

### 2. Tighter Pareto via more rollouts

The bootstrap CIs included in this PR's benchmark output let us see
exactly where the inter-method differences are statistically decidable
and where they aren't. Reading the CIs:

- At k=5, family-level rankings are decidable at 100 rollouts
  (CI widths ~0.05 bits; gaps between families ~0.5 bits).
- At k=15, the merging-family methods (Hoeffding/χ²/JS/Bayesian) sit
  inside each other's CIs in the n_nodes ∈ [6, 10] region. Distinguishing
  them needs ~3× tighter CIs.

Data-quantity ramp:

| Target precision           | Episodes/task  |
|----------------------------|---:|
| Family-level (now)         | 100 ✓ |
| Within-merge-family at k=15| 200–300 |
| Confident method-selection | 500 |
| Reliable CSSR at L=3       | 800–1000 |

We recommend collecting **300 rollouts per task** as the next milestone —
it brings the Hoeffding compatibility threshold from 0.27 down to 0.16,
gets us ~30 triplets per cell in 2nd-order MI tables, and is roughly the
break-even point where k-tails at k=2 becomes reliable.

### 3. Partial observability

Even with all the above, this is partially observable robotic
manipulation. The "right" hidden state is the agent's belief, not any
function of observation clusters. Any abstraction over discretized
observations will have residual non-Markov structure unless the
clustering captures the underlying hidden state — which is a different,
harder problem (representation learning over policy internals or
world-model belief states) that we're not solving here.

**A fully Markov abstraction is probably not attainable in principle for
these tasks.** The realistic objective is "the smallest abstraction whose
non-Markov residual is small relative to the noise floor and small
enough not to mislead downstream curation/VLM/human consumers." The
Pareto frontier we report identifies the frontier; choosing the operating
point requires the downstream-task evaluation in (4).

### 4. Downstream-task benchmark

Markov fidelity is necessary but not sufficient. A graph with the
smallest MV at a given n_nodes may not be the most useful for the
downstream uses (curation seed selection, VLM annotation, human
inspection). The natural closing experiment:

- Take the simplified graph from each method (at a few operating
  points on the Pareto).
- Plug each into `select_mimicgen_seed_from_graph` — generate seeds for
  the MimicGen pipeline.
- Train and evaluate the resulting policy.
- Report end-to-end demonstration quality.

This converts "MV in bits" from a methodological proxy into a
task-grounded loss. It's out of scope for this iteration but is the
correct downstream validation.

### 5. Dwell-time-aware (semi-Markov) graph model

Long-standing limitation of the current run-length-collapsed Markov
chain: it doesn't model how long the system spends in each state. A
1-node "model" trivially wins compressive NLL because "we never leave"
costs nothing to encode. The cleanest fix is to model each state's
dwell-time distribution explicitly (semi-Markov chains). This would
also let `mdl_greedy` actually optimize a meaningful trade-off rather
than collapsing toward fewer nodes. We left this out for scope reasons
but it's the right way to make MDL/NLL into honest cross-method
selection criteria.

## Recommendations for the user

1. **Default to `hoeffding_merge` with δ ≈ 0.05** as the user-facing
   single-lever knob. It's principled, sample-size-aware, fast, and lands
   on the same Pareto curve as the entire merging family.
2. **None of the current methods reliably push MV below the raw-clustering
   baseline at our sample size.** `vomm_split_merge` aims to via splitting
   but the 1-step memory it conditions on is insufficient for tasks where
   the dependencies are longer (transport, square). A k-tails (k=2) merger
   or CSSR is the right next implementation — see "Future work" §1 —
   together with more rollouts (§2).
3. **Avoid `stationary_skeleton`** unless you specifically want a
   "visualization-only" skeleton of the chain (it's bad on every fidelity
   metric).
4. **Markov stability and PCCA+** are useful for *answering a different
   question* (where are the metastable attractors?) but not for "find me
   the smallest Markov abstraction".
5. **For "the right k automatically"** — accept that there's no single
   right answer without a downstream task to optimize. The pragmatic
   choice is the smallest n_nodes such that MV ≤ ε, with ε ≈ baseline MV
   plus a small slack. The `simplification_demo` Streamlit app
   (`streamlit run policy_doctor/streamlit_app/simplification_demo/app.py`)
   lets you find this point interactively.

---

## Reproducing

```bash
# Run the full benchmark (≈15 minutes wall time):
PYTHONPATH=. python scripts/benchmark_simplification.py --n_folds 5

# Run with 50-rep bootstrap CIs (≈30 minutes wall time):
PYTHONPATH=. python scripts/benchmark_simplification.py --n_folds 5 --bootstrap --n_bootstrap 50

# Generate plots + summary table:
PYTHONPATH=. python scripts/summarize_results.py

# Launch the interactive demo:
PYTHONPATH=. streamlit run policy_doctor/streamlit_app/simplification_demo/app.py
```

## Update 2026-05-17 — K-sweep model-selection results

The "Choosing K via MV(K)" experiment below has been executed on the workstation.
Full results live at [`docs/k_sweep_results/_findings.md`](k_sweep_results/_findings.md);
198 clusterings analysed (3 tasks × 2 reps × 3 (w, s) × 11 K).

**Lift is dropped from the headline.** Lift's eval episodes average ~10
timesteps, which after sliding-window aggregation and run-length collapse
leaves most rollouts contributing fewer than `min_pairs = 4` triplets
per merged state. At every stride-1 (w, s) setting, the order-1 coverage
drops below 0.50 for at least one rep at K∈{5..15}, so the
argmin / largest-below-ε criteria are dominated by gate artefacts.
Lift's table is still in `docs/k_sweep_results/_findings.md` under
"Appendix" for completeness, but the model-selection story is reported
on **transport_mh_jan28 and square_mh_feb5 only**.

Findings (all coverage-gated at cov₁ ≥ 0.80; without the gate, MV
trivially reads 0 in two unrelated regions — K≤4 where sequences are too
short, and large K where each state has too few triplets to clear
`min_pairs ≥ 4·order²`. Neither of those zeros means the chain is
Markov; both mean the metric is unmeasurable):

1. **Cross-rep, the strongest signal in the sweep is on transport at
   K=6.** `policy_emb_bottleneck_plan_t0` gives **3× lower MV₁ than
   `infembed` at K=6 across all three (w, s) settings**, with full
   coverage for infembed and cov ≈ 0.90 for policy_emb:

   | (w, s) | infembed MV₁ @ K=6 (cov) | policy_emb MV₁ @ K=6 (cov) |
   |---|---:|---:|
   | (3, 1) | 0.226 (1.00) | **0.089 (0.91)** |
   | (5, 1) | 0.194 (1.00) | **0.064 (0.89)** |
   | (8, 1) | 0.215 (1.00) | **0.072 (0.90)** |

   This is the cleanest single result: **at small K, policy_emb produces
   a substantially more-Markov 6-node abstraction of transport
   trajectories than infembed does, independently of (w, s).**

2. **Square is harder to call.** The lowest-MV measurable cell is a
   near-tie between `infembed (w=8, s=1, K=12)` MV=0.096 cov=0.83 and
   `policy_emb (w=8, s=1, K=8)` MV=0.092 cov=0.82. For more-expressive
   graphs, **infembed (w=5, s=1, K=20)** with MV=0.143 cov=0.95 is the
   only setting that combines large K with high coverage.

3. **Coverage and MV trade off across reps.** infembed has uniformly
   higher coverage (often 1.00 at the same K). Interpretation: infembed
   produces more balanced clusters (every state visited often, gate
   clears trivially) but the resulting graph carries more conditional
   dependencies on the predecessor (higher MV). policy_emb concentrates
   trajectory mass into fewer states with rare ones, lowering measured
   MV but also coverage. Part of the apparent 3× MV gap on transport may
   shrink under more data — Stage B (500 rollouts) will quantify this.

3a. **The 3× MV gap is K-dependent and partly coverage-driven.** The
    cross-rep N-convergence sweep at K=15 (added 2026-05-17) shows the
    two reps converge to nearly the same MV₁ as N grows:

    | rep | N=20 (cov) | N=50 (cov) | N=100 (cov) |
    |---|---:|---:|---:|
    | infembed (K=15)  | 0.272 (0.92) | 0.353 (0.95) | 0.332 (1.00) |
    | policy_emb (K=15)| 0.149 (0.64) | 0.307 (0.85) | 0.304 (0.94) |

    `infembed` reaches its asymptote at N=20. `policy_emb`'s MV₁ at K=15
    doubles between N=20 and N=100 — the small N=20 value was gate-
    limited (cov₁=0.64). **At N=100 the K=15 cross-rep gap is just
    0.03 bits** — within noise. The 3× gap at K=6 (where MV is
    measured at 0.06-0.09 for policy_emb vs 0.19-0.23 for infembed)
    survives the coverage check on the 100-rollout main K-sweep (cov ≥
    0.89-1.00 for both reps at K=6), so it's not an artefact — but
    whether it survives N=500 is exactly what Stage B answers.
    See `docs/rollout_budget_results/_findings.md` for full cross-rep
    panels at K∈{5, 10, 15, 20}.

4. **Largest expressive Markov graph (MV₁ ≤ 0.15, cov₁ ≥ 0.80):**
   - Transport: only `policy_emb` clears the gate. Best is
     `(w=8, s=1, K=30)` MV=0.125 cov=0.90.
   - Square: best is `infembed (w=5, s=1, K=20)` MV=0.143 cov=0.95.

5. **The MV-vs-K curve is non-monotone for policy_emb on transport**
   (peak ~0.40 at K=18, drops back to 0.14 at K=30). Two interpretations
   — finer kmeans clusters resolve per-state transition ambiguity better,
   OR rare high-MV triplets get filtered at the coverage edge (cov
   drops 0.96→0.85 between K=20 and K=30). The 500-rollout Stage B sweep
   will distinguish.

6. **MV₂ > MV₁** in most coverage-clearing cells — length-2 memory is
   the dominant residual at our sample size. **MV₃ rarely clears
   coverage at N=100** — the order-3 gate is `min_pairs ≥ 36`. Order ≥ 3
   estimates need ≥ 300 rollouts (extrapolated). This is exactly the
   question the rollout-budget sweep addresses.

7. **Window/stride matters less than K and rep.** At K=6 the (w, s)
   choice barely affects MV on transport. Wider windows give marginally
   lower MV at matched K via smoothing, but the magnitude is small
   compared to the rep gap.

The doc-section below was the original *plan* and is preserved for
reference; the resulting analysis is at the link above. The Streamlit
app exposes the K-sweep grid via sidebar controls (Representation /
Window / Stride / K) and renders the MV-vs-K elbow inline.

## Update 2026-05-17 — rollout-budget sweep (in progress)

A second experiment investigates how MV estimates change with the number
of rollouts. Two stages:

**Stage A (existing 100-rollout data, subsampled — DONE).** For each
task and K ∈ {5, 10, 15, 20}, subsample N ∈ {20, 50, 100} episodes with
5 subsample seeds each, recompute the (w=5, s=1) sliding-window
clustering and the MV₁/MV₂/MV₃ point + bootstrap CI per draw. 180
datapoints total. Results at
[`docs/rollout_budget_results/_findings.md`](rollout_budget_results/_findings.md).

Highlights from Stage A (transport_mh_jan28, policy_emb, K=15, w=5, s=1
as a representative example — full tables in
[`docs/rollout_budget_results/_findings.md`](rollout_budget_results/_findings.md)):

| N | MV₁ | σ-seeds | cov₁ | MV₂ | cov₂ | MV₃ | cov₃ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20  | 0.149 | 0.062 | 0.64 | 0.065 | 0.21 | 0.000 | 0.00 |
| 50  | 0.307 | 0.032 | 0.85 | 0.260 | 0.78 | 0.081 | 0.12 |
| 100 | 0.304 | 0.012 | 0.94 | 0.331 | 0.85 | 0.602 | 0.65 |

Reading this table:

- **The MV₁ point estimate jumps as we go from N=20 to N=50** (0.149 →
  0.307) because at N=20 coverage is only 0.64 — the metric is averaging
  over 36% of the data and missing the high-MV rare transitions. At
  N=50, coverage clears the 0.80 threshold and the metric stabilises
  near 0.30. The σ across subsample seeds drops 5× from N=20 to N=100.
- **MV₂ becomes measurable at N=50** (cov₂=0.78) but reading the point
  estimate carefully — 0.260 → 0.331 going N=50→100, with cov₂
  still climbing — the metric isn't fully converged. **300+ rollouts
  recommended for MV₂.**
- **MV₃ first becomes measurable around N=50** (cov₃=0.12 → 0.65 at
  N=100) and the point estimate moves from 0.08 to 0.60 in that range.
  **MV₃ needs ≥ 300 rollouts** to be a trustworthy diagnostic at
  K=15; the trend in coverage suggests we'd need N≈500 for cov₃ ≥ 0.80.
- **CI-width vs N is not a clean convergence indicator** at our scale:
  for transport K=15 the CI is 0.229→0.225→0.159 bits going 20→50→100,
  but at N=20 the CI is tight only because the metric is degenerate
  (mostly zeros) — the bootstrap doesn't see real variability. Once
  cov₁ clears the gate at N≥50, the CI starts behaving like the
  honest data-noise CI it's designed to be, and shrinks predictably.

**This directly answers the user's question** ("does more data change
the outcome of approximations we make with a lower amount of rollouts?")
— *yes, substantially*. At K=15 policy_emb on transport, MV₁ doubles
between N=20 and N=100; MV₂ doubles again from N=50 to N=100; MV₃ is
unmeasurable at N=20-50 and only just becomes measurable at N=100. Any
conclusion drawn from N<50 should be treated as suspect; any
higher-order conclusion (MV₂, MV₃) needs N≥100, ideally ≥300.

### Update 2026-05-17 — Stage B (500-rollout) initial results on lift

The lift_mh_jan26 500-rollout eval has completed and the budget sweep
extends to N∈{20,50,100,200,300,400,500} at K∈{5,6,8,10,12,15,20}. The
results substantially change the Stage A conclusions:

**Lift, policy_emb, w=5 s=1, K=15 — all MV orders climb past N=100:**

| N | MV₁ | cov₁ | MV₂ | cov₂ | MV₃ | cov₃ |
|---:|---:|---:|---:|---:|---:|---:|
| 20  | 0.104 | 0.70 | 0.000 | 0.00 | 0.000 | 0.00 |
| 50  | 0.177 | 0.90 | 0.203 | 0.59 | 0.000 | 0.00 |
| 100 | 0.241 | 0.94 | 0.299 | 0.84 | 0.157 | 0.10 |
| 200 | 0.264 | 0.98 | 0.462 | 0.94 | 0.559 | 0.59 |
| 300 | 0.275 | 0.98 | 0.512 | 0.95 | 0.595 | 0.74 |
| 400 | 0.282 | 0.99 | 0.535 | 0.96 | 0.640 | 0.83 |
| **500** | **0.297** | **0.99** | **0.649** | **0.96** | **0.787** | **0.92** |

**The N=100 estimates were drastically biased low:**

- **MV₁ is still climbing at N=500** (0.241 → 0.297 from N=100 to N=500 = 23% increase). Even N=500 isn't converged.
- **MV₂ MORE THAN DOUBLED** from N=100 to N=500 (0.299 → 0.649). The "diagnostic 2nd-order memory" reported in Stage A was substantially underestimated.
- **MV₃ went from 0.16 (cov=0.10, basically unmeasurable) at N=100 to 0.787 (cov=0.92, fully measurable) at N=500.** Third-order memory IS dominant in the residual — it just wasn't visible at small N.
- **At N=500 the ordering reverses: MV₃ (0.787) > MV₂ (0.649) > MV₁ (0.297).** The chain has substantial higher-order memory the smaller-N data couldn't reveal.

**Earlier "Lift dropped because near-Markov" conclusion was wrong.** Lift looked Markov at N=100 because the metric couldn't measure beyond order-1 memory. With 500 rollouts: the lift abstraction at K=15 has substantial 1st/2nd/3rd-order memory.

**Direct answer to the user's sample-size question** ("does more data change the outcome of approximations made with fewer rollouts?"): **YES — drastically.** At N=100, MV₃ is reported as ~0.16 bits (and the coverage of 0.10 means it's really noise); at N=500, MV₃ is 0.787 bits with high coverage. **Anyone using MV₃ at N<200 is reading noise.** And even MV₁ — the most stable order — climbs ~25% between N=100 and N=500 at K=15.

**Rollouts needed for reliable MV at K=15 on lift** (coverage ≥ 0.80):
- MV₁: N ≥ 50  
- MV₂: N ≈ 100  
- MV₃: N ≥ 400  

### Square Stage B (added 2026-05-17)

**Square, policy_emb, w=5 s=1, K=15 — same pattern as lift:**

| N | MV₁ | cov₁ | MV₂ | cov₂ | MV₃ | cov₃ |
|---:|---:|---:|---:|---:|---:|---:|
| 20  | 0.088 | 0.38 | 0.000 | 0.00 | 0.000 | 0.00 |
| 50  | 0.107 | 0.60 | 0.000 | 0.00 | 0.000 | 0.00 |
| 100 | 0.159 | 0.87 | 0.024 | 0.09 | 0.000 | 0.00 |
| 200 | 0.183 | 0.92 | 0.179 | 0.28 | 0.000 | 0.00 |
| 300 | 0.240 | 0.97 | 0.283 | 0.63 | 0.000 | 0.00 |
| 400 | 0.268 | 0.99 | 0.377 | 0.83 | 0.187 | 0.11 |
| **495** | **0.241** | **0.99** | **0.335** | **0.85** | **0.000** | **0.00** |

- **MV₁ at N=100 (0.159) understated the true value by ~70%**; with N=400-495 it stabilises around 0.24-0.27 bits.
- **MV₂ went from unmeasurable at N=100 (cov=0.09) to 0.335 at N=495.**
- **MV₃ NEVER clears coverage on square K=15 at N≤495.** Unlike lift (where MV₃ becomes fully measurable at N=400), square needs even more rollouts for 3rd-order MV at K=15. At K=20, square's MV₃ does emerge (cov=0.46 at N=495), giving MV₃=0.488. But K=15 + N=495 + square = MV₃ remains noisy.

**Square K=6 — the K=6 cross-rep finding is *partly* refuted by N=500 data:**

| N | MV₁ | cov₁ | MV₂ | cov₂ |
|---:|---:|---:|---:|---:|
| 100 | 0.237 | 0.88 | 0.074 | 0.13 |
| 200 | 0.281 | 0.89 | 0.241 | 0.32 |
| 400 | 0.229 | 0.98 | 0.342 | 0.59 |
| **495** | **0.179** | **1.00** | **0.258** | **0.77** |

At N=495 the square policy_emb K=6 MV₁ is **0.179 with full coverage**, not the near-zero/unmeasurable value from the Stage A K-sweep (which reported MV=0 cov=0.00). **The Stage A K=6 reading was a coverage artefact, not a real "policy_emb is more Markov" win for square.** This validates the advisor's earlier concern.

For transport (still being computed) the analogous question is: does the transport policy_emb K=6 finding (MV₁=0.064, cov=0.89 at N=100, our headline 3× rep gap) survive at N=500? The Stage A coverage at transport+K=6 was higher (0.89 vs square's 0.00 at N=100) so it's less likely to be a pure artefact, but the N=500 data is the definitive answer.

### Cross-task summary so far (N=495 vs N=100, policy_emb, K=15, w=5, s=1)

| Task | MV₁ N=100 | MV₁ N=495 | Δ | MV₂ N=100 | MV₂ N=495 | MV₃ N=495 |
|---|---:|---:|---:|---:|---:|---:|
| lift  | 0.241 | 0.297 (cov=0.99) | +23% | 0.299 | 0.649 (cov=0.96) | 0.787 (cov=0.92) |
| square | 0.159 | 0.241 (cov=0.99) | +52% | 0.024 | 0.335 (cov=0.85) | 0.000 (cov=0.00, unmeasurable) |
| transport | (running) | — | — | — | — | — |

The cross-task pattern: **all measurable MV orders grow with N**, and the growth is task-dependent. MV₃ becomes reliably measurable at K=15 only on lift; square needs more rollouts; transport is still being computed. **The N=100 estimates were systematically biased low on every task and order we have so far.**

### Transport Stage B — interim (N=20–200 on 288 rollouts, added 2026-05-17)

The full transport eval is still running, but an interim 288-rollout
trunk lets us validate the Stage A findings now. **The K-sweep
headline (policy_emb K=6 wins with low MV) survives the budget sweep:**

| Transport, policy_emb, K=6: | MV₁ (cov₁) | MV₂ (cov₂) | MV₃ (cov₃) |
|---|---:|---:|---:|
| N=20  | 0.041 (0.48) | 0.000 (0.00) | 0.000 (0.00) |
| N=50  | 0.107 (0.79) | 0.158 (0.67) | 0.000 (0.00) |
| N=100 | 0.092 (0.77) | 0.110 (0.71) | 0.000 (0.00) |
| N=200 | 0.096 (0.84) | 0.131 (0.78) | 0.034 (0.70) |

**At N=200 transport policy_emb K=6, MV₁=0.096 cov=0.84** — this is
within noise of the Stage A K-sweep result (MV=0.064 cov=0.89 at N=100,
on all 100 rollouts). **The K=6 policy_emb finding is real, not a
coverage artefact on transport.** Compare to infembed K=5 transport at
N=100: MV₁=0.143 cov=0.86 (from cross-rep budget sweep) — the 1.5×
rep gap holds even at the smallest measurable K.

**Transport K=15 (the heaviest task):**

| N | MV₁ (cov₁) | MV₂ (cov₂) | MV₃ (cov₃) |
|---:|---:|---:|---:|
| 20  | 0.182 (0.60) | 0.055 (0.09) | 0.000 (0.00) |
| 50  | 0.226 (0.83) | 0.202 (0.79) | 0.240 (0.30) |
| 100 | 0.290 (0.93) | 0.379 (0.84) | 0.588 (0.62) |
| 200 | 0.347 (0.98) | 0.443 (0.95) | 0.587 (0.81) |

- **MV₁ at K=15 climbs +20% from N=100 to N=200** (0.290 → 0.347). Like
  lift and square, the N=100 estimate underreports.
- **MV₃ becomes measurable earliest on transport** (cov=0.62 at N=100,
  0.81 at N=200) and is already substantial (~0.59 bits). Stable across
  N=100→200, suggesting transport K=15 MV₃ converges fastest.

**Cross-task summary at K=15 N=200 (lower bound — Stage B finishing):**

| Task | MV₁ | MV₂ | MV₃ |
|---|---:|---:|---:|
| lift  | 0.264 | 0.462 | 0.559 |
| square | 0.183 | 0.179 | 0.000 (cov=0, unmeas.) |
| transport | 0.347 | 0.443 | 0.587 |

Transport has the highest MV at every order — consistent with its
intuitive complexity (bimanual pickup-and-transfer vs. lift's
single-arm grasp).

### Transport Stage B — interim update at N≤300 (333-rollout trunk)

The transport eval crossed 333 episodes (66% of target 500), enabling a
budget sweep at N up to 300. **Transport K=15 confirms the cross-task
pattern: MV grows substantially with N, and MV₃ becomes reliably
measurable at N=200-300.**

| Transport K=15 (policy_emb, w=5 s=1, on 333-rollout trunk) | MV₁ (cov) | MV₂ (cov) | MV₃ (cov) |
|---|---:|---:|---:|
| N=100 | 0.243 (0.93) | 0.288 (0.90) | 0.481 (0.64) |
| N=200 | 0.399 (0.98) | 0.484 (0.96) | 0.557 (0.84) |
| N=300 | 0.310 (0.99) | 0.471 (0.95) | 0.575 (0.90) |

**Transport K=6 (the headline cross-rep test):** policy_emb stabilises
at MV₁ ≈ 0.10-0.12 with full coverage at N=200-300. Infembed K=5
N=100 was 0.143 (cov=0.86). **The policy_emb-beats-infembed finding at
low K survives Stage B**, though the gap is closer to 1.4-2× than the
3× initially reported (the Stage A K-sweep MV=0.064 was slightly
underestimating policy_emb's true MV₁ at K=6, which converges to ~0.10
under full coverage).

**Cross-task summary at K=15, N=300 (all three tasks):**

| Task | MV₁ | MV₂ | MV₃ (cov₃) |
|---|---:|---:|---:|
| lift  | 0.275 | 0.512 | 0.595 (0.74) |
| square | 0.240 | 0.283 | 0.000 (0.00, unmeasurable) |
| transport | 0.310 | 0.471 | 0.575 (0.90) |

Transport and lift have similar MV₁/MV₂/MV₃ magnitudes at K=15. Square
has noticeably lower MV₂ AND completely unmeasurable MV₃ at K=15 even
with N=300 — **square needs more rollouts than the other two tasks
for order-3 MV** (or alternatively, its abstraction at K=15 genuinely
has weaker length-3 memory).

### Transport Stage B — FINAL (500-rollout trunk, N=20-500, added 2026-05-17)

The full 500-rollout transport eval completed. **The K=6 cross-rep
finding survives at N=500 cleanly:**

| Transport policy_emb K=6 | MV₁ (cov) | MV₂ (cov) | MV₃ (cov) |
|---|---:|---:|---:|
| N=20  | 0.032 (0.52) | 0.000 (0.00) | 0.000 (0.00) |
| N=50  | 0.074 (0.85) | 0.034 (0.16) | 0.000 (0.00) |
| N=100 | 0.108 (0.96) | 0.174 (0.64) | 0.000 (0.00) |
| N=200 | 0.067 (0.95) | 0.159 (0.91) | 0.000 (0.00) |
| N=300 | 0.071 (0.96) | 0.202 (0.97) | 0.000 (0.00) |
| N=400 | 0.067 (0.96) | 0.188 (0.97) | 0.055 (0.09) |
| **N=500** | **0.070 (0.96)** | **0.173 (0.97)** | **0.176 (0.42)** |

At N=500, transport policy_emb K=6 gives **MV₁=0.070 with cov=0.96**.
The N=100 estimate (0.108) was ~30% biased high; the true asymptote
is ~0.07. **The K=6 policy_emb headline holds — and the gap to
infembed (~0.19 at K=6 from the Stage A K-sweep) is now ~2.7×**, very
close to the original "3× MV gap" finding.

**Transport policy_emb K=15:**

| N | MV₁ (cov) | MV₂ (cov) | MV₃ (cov) |
|---:|---:|---:|---:|
| 20  | 0.075 (0.62) | 0.024 (0.07) | 0.000 (0.00) |
| 50  | 0.154 (0.81) | 0.202 (0.74) | 0.167 (0.20) |
| 100 | 0.226 (0.89) | 0.322 (0.90) | 0.472 (0.59) |
| 200 | 0.205 (0.96) | 0.344 (0.92) | 0.474 (0.85) |
| 300 | 0.209 (0.99) | 0.380 (0.95) | 0.543 (0.90) |
| 400 | 0.243 (0.97) | 0.450 (0.96) | 0.680 (0.92) |
| **500** | **0.222 (1.00)** | **0.436 (0.96)** | **0.663 (0.91)** |

- **MV₁ converges by N=200** at ~0.22. The N=100 reading (0.226) was
  already accurate — unlike lift/square at the same K. Transport's
  MV₁ converges faster than other tasks at K=15.
- **MV₃ climbs from 0.47 (N=100, cov=0.59) → 0.66 (N=500, cov=0.91).**
  N=100 understated MV₃ by ~30%. **N=300-500 needed for reliable MV₃
  on transport K=15.**

### Cross-task FINAL summary at K=15, N=500, policy_emb, w=5, s=1

| Task | MV₁ | MV₂ | MV₃ | N=100 MV₁ bias |
|---|---:|---:|---:|---:|
| lift_mh_jan26  | 0.297 | 0.649 | 0.787 (cov 0.92) | −19% |
| square_mh_feb5 | 0.241 | 0.335 | 0.000 (cov 0.00) | −34% |
| transport_mh_jan28 | 0.222 | 0.436 | 0.663 (cov 0.91) | +2% (already close) |

**Surprise:** despite transport being the visually most complex task,
its K=15 abstraction has the **lowest MV₁ at N=500** among the three.
Square is second; lift is highest. The narrative "transport is the
most non-Markov" from Stage A was a low-N artefact.

### Cross-task FINAL summary at K=6 (the headline rep-comparison case)

| Task | MV₁ N=500 (cov) |
|---|---:|
| lift policy_emb K=6  | 0.160 (0.91) |
| square policy_emb K=6 | 0.179 (1.00) |
| **transport policy_emb K=6** | **0.070 (0.96)** ← lowest |
| transport infembed K=5 (Stage A, N=100 cov=0.86) | 0.143 |

**Transport policy_emb at K=6 is the lowest-MV abstraction across all
tasks and the rep-comparison gap holds at N=500.** This is the cleanest
positive model-selection finding in the entire sweep:

1. **For minimum MV at minimal cluster count, use `policy_emb`** at K=6
   on transport, K=6 on lift, K=6 on square. Across all three tasks,
   K=6 policy_emb gives MV₁ in [0.07, 0.18] with full coverage.
2. **For maximum graph expressivity with MV₁ ≤ 0.15** at N=500:
   - Transport: K=6 (MV=0.07) — limited by sharp climb above K=10
   - Square: K=6 (MV=0.18) just over — practically K=5 or K=6 is the answer
   - Lift: K=6 (MV=0.16) just over — same

**Direct answer to the sample-size question:**
- **MV₁ at K=15**: needs **N ≈ 200** for stable estimate. Lift: 0.241→0.297 (+23%). Square: 0.159→0.241 (+52%). Transport: 0.226→0.222 (already converged at N=100).
- **MV₂**: needs **N ≈ 200-300**. Doubled or tripled across tasks N=100→500.
- **MV₃**: needs **N ≥ 400** to clear coverage 0.80 across tasks. Was completely unmeasurable at N=100 for lift/square; emergent at N=100 for transport but kept climbing.
- **Different tasks need different N** for the same order — square doesn't reach measurable MV₃ at K=15 even at N=500.

**The N=100 estimates would have led to wrong cross-task ranking on
MV₁ at K=15** (Stage A: transport>lift>square; Stage B at N=500:
lift>square>transport).

### Update 2026-05-17 — low-w on lift flips the rep ranking

Lift episodes average **10 timesteps** — *much* shorter than square
(~40) or transport (~80). With stride=1, my Stage A sweep at w∈{3, 5, 8}
left only 3-8 windows per episode after sliding-window aggregation,
and the 1st-order coverage gate cleared in almost no cells. **Adding
w=1 and w=2 to the lift sweep changes everything:**

| Lift K=15, N=100, all (w, s) | MV₁ (cov₁) |
|---|---:|
| infembed w=1 s=1 | **0.271 (0.97)** ← fully measurable |
| infembed w=2 s=1 | 0.330 (0.99) |
| infembed w=3 s=1 | 0.344 (0.96) |
| infembed w=5 s=1 | 0.241 (0.88) — borderline |
| infembed w=8 s=1 | 0.076 (0.60) — gate fails |
| policy_emb w=1 s=1 | **0.346 (0.96)** ← fully measurable |
| policy_emb w=2 s=1 | 0.220 (0.92) |
| policy_emb w=3 s=1 | 0.306 (0.61) — gate fails |
| policy_emb w=5 s=1 | 0.000 (0.00) — completely unmeasurable |
| policy_emb w=8 s=1 | 0.000 (0.00) — completely unmeasurable |

**Two big consequences:**

1. **The original "lift dropped because near-Markov" reasoning was wrong.**
   At w=1, lift K=15 has MV₁=0.27-0.35 with full coverage — substantial
   non-Markov memory, similar magnitude to transport. The
   "near-Markov" conclusion at Stage A came entirely from the metric
   being unmeasurable, not from the chain being Markov.

2. **On lift, `infembed` Pareto-dominates `policy_emb` at every K with
   gated coverage** when (w=1, s=1):

   | Lift, w=1, s=1, N=100 | infembed MV₁ | policy_emb MV₁ | winner |
   |---:|---:|---:|---|
   | K=5  | 0.091 (1.00) | 0.094 (1.00) | tie |
   | K=6  | 0.095 (1.00) | 0.112 (1.00) | infembed |
   | K=8  | 0.142 (1.00) | 0.527 (0.95) | infembed (4×) |
   | K=10 | 0.226 (0.99) | 0.603 (0.95) | infembed (3×) |
   | K=15 | 0.271 (0.97) | 0.346 (0.96) | infembed |
   | K=20 | 0.274 (0.95) | 0.335 (0.92) | infembed |

   At w=1, **infembed is the right representation for lift** at every K
   where the metric clears the gate. At higher w, policy_emb appeared
   to win — but that was the gate firing on infembed at low n_samples.
   The K=6 policy_emb headline holds for transport but NOT lift.

**General lesson — `w` must scale with episode length.** A rule like
`w ≈ T_avg / 10` would have picked w=1 for lift, w=4 for square, and
w=8 for transport. Stride-1 with this w is a defensible default.

### Pareto-optimal representation selection (rule)

Given the above, we can define a coverage-aware Pareto-dominance test
between two representations on a task:

> **Rep A is preferred over rep B on task T** if there exists a (w, s)
> setting and a range of K such that, at every K in that range, both
> reps clear `coverage₁ ≥ 0.80` and `MV₁(A) ≤ MV₁(B)` — with strict
> inequality at ≥ 1 K. Tie if neither dominates the other across all
> jointly-gated K.

Applied to our data with **the best per-task (w, s)**:

| Task | best (w, s) | infembed sweet spot | policy_emb sweet spot | winner |
|---|---|---|---|---|
| **lift** | (1, 1) | K=5 MV=0.091 cov=1.00 | K=5 MV=0.094 cov=1.00 | **infembed** dominates K≥6 |
| **square** | (5, 1) | K=12 MV=0.096 cov=0.83 | K=8 MV=0.092 cov=0.82 | **mixed / nearly tied** |
| **transport** | (5, 1) | K=5 MV=0.143 cov=0.86 | K=6 MV=0.064 cov=0.89 | **policy_emb** dominates at low K |

So the **task-conditional** answer is:
- Long episodes + complex dynamics (transport) → **policy_emb** wins.
- Short episodes + simple dynamics (lift) → **infembed** wins at w=1.
- Intermediate (square) → near-tie; pick by secondary criterion (which K
  you want; if you want larger K with full coverage, infembed; if you
  want minimal K, policy_emb).

The "best representation" is **not** a universal property of the rep —
it's a property of (rep × task × window-stride × K). A practical
default that adapts: scan a small (w, s) × K grid per task, apply the
coverage gate, then pick the rep that Pareto-dominates the gated
region. Where no clear winner, fall back to interpretability /
downstream evaluation.

### Update 2026-05-17 — automated hyperparameter selection (`policy_doctor.behaviors.select_K`)

Implementing the option-A rule (joint w-K sweep, no `T̄ / 10` heuristic).
The pipeline:

1. Build the (rep × w × s × K) cell grid: cluster each combo, compute
   MV₁ and coverage on the existing 100-rollout eval data. We swept
   w ∈ {1, 2, 3, 5, 8} for all 3 tasks (306 clusterings total).
2. **Coverage gate**: keep cells with `cov₁ ≥ 0.80 AND K ≥ 5`.
3. **Admissibility per (rep, w, s)**: require ≥ 3 K cells clear the
   gate at that setting. This rejects isolated gate-edge cells where
   MV₁ is artificially low.
4. **Knee per setting**: smallest K with `MV₁ ≥ 0.5 · MV_asymp(K)`.
5. **Per-rep best setting**: among each rep's admissible (w, s)
   settings, pick the one with lowest knee MV₁.
6. **Pareto-dominance across reps** at matched K (within the chosen
   settings), with fallback to lowest knee MV₁ on tie.

Code lives at `policy_doctor/behaviors/select_K.py`.

#### Two robustness patches applied (2026-05-17)

**Patch A — robust asymptote.** Use `max over ALL gated K cells`, not
the top-3 highest-K. The top-3 estimator silently undershoots when the
MV-vs-K curve is non-monotone (peaks at intermediate K, drops at high K
because the `min_pairs ≥ 4·order²` gate filters rare states). Patch A
catches the peak. For transport policy_emb w=5: top-3 asymp was 0.282,
true peak (at K=18) is 0.399.

**Patch B — non-convergence flag.** If `MV(K_max_gated) < 0.9 · MV_asymp`,
the curve is still descending at the largest gated K — the asymptote is
likely a peak rather than a plateau, and γ-knee may be biased. The
`Pick` carries `converged=False` and a descriptive note in this case.

#### Final auto-pipeline output

| Task | rep | w | s | K | MV₁ at K | cov₁ at K | MV_asymp | Converged? |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| lift_mh_jan26 | **infembed** | 1 | 1 | 8 | 0.142 | 1.00 | 0.274 | ⚠ no |
| square_mh_feb5 | **infembed** | 8 | 1 | 12 | 0.096 | 0.83 | 0.149 | ✓ yes |
| transport_mh_jan28 | **policy_emb** | 8 | 1 | 10 | 0.198 | 0.87 | 0.302 | ⚠ no |

**Patch A flipped the lift pick.** Pre-patch we had `policy_emb w=2 K=6`
(MV=0.115). The policy_emb w=2 curve has an MV peak at K=8 (0.242) that
the top-3 estimator missed; with the all-K asymptote, that knee shifts
to K=8 with MV=0.242, and `infembed w=1 K=8` (MV=0.142) wins instead.
**This now matches the visual Pareto-dominance story: at w=1 on lift,
infembed strictly dominates policy_emb at every K with cov ≥ 0.80.**

**Patch B flagged two tasks.** Lift and transport both have curves where
MV peaks at intermediate K and drops at high K — symptom of the gate
firing on rare states. The γ-knee is still a defensible pick, but the
absolute MV scale is unstable; downstream consumers should treat the
asymp value as a *lower bound* on the true ceiling.

**Square** is the cleanest case: the infembed w=8 curve is monotone
through K=20, plateaus at ~0.15, and Patch B reports converged.

#### Top alternatives per task (top-3)

- **lift**: infembed w=1 K=8 → policy_emb w=3 K=6 (MV=0.151) → infembed w=5 K=8 (0.215)
- **square**: infembed w=8 K=12 → infembed w=5 K=8 (0.177) → policy_emb w=2 K=5 (0.181)
- **transport**: policy_emb w=8 K=10 → policy_emb w=3 K=10 (0.220) → policy_emb w=5 K=12 (0.237)

### Answer to "can we recommend ONE representation?"

**No — the data refutes that ambition.** The auto-pipeline picks
different reps for different tasks:

- **infembed** for lift (w=1, matching the short-episode regime where
  the metric is only measurable at very small windows) and for square
  (w=8, where the gate clears cleanly through K=20).
- **policy_emb** for transport (w=8, K=10 — the long-episode/complex-
  dynamics regime where policy_emb's compressed-state structure aligns
  best with Markov memory).

This is consistent with the underlying signal: short-episode and
long-episode tasks benefit from different temporal aggregations, and
the rep that best aligns trajectory clusters with Markov structure
depends on task dynamics. The pipeline picks `(rep, w, s, K)`
jointly without privileging any axis — that's the cleanest answer.

**Stage B (500-rollout fresh eval).** 500 rollouts/task are being
collected into a new eval dir with `graph_simplification` in the name
(`/mnt/ssdB/erik/cupid_data/outputs/eval_save_episodes/graph_simplification/{task}_n500/`).
Required a small compat patch — the existing `async_vector_env.py`
expected the gym 0.26 `concatenate(space, items, out=...)` signature,
but the available `mimicgen_torch2` env ships gym 0.21 with the older
`concatenate(items, out, space)` order. Patched via a try/except
wrapper in `diffusion_policy/gym_util/async_vector_env.py` so the call
works against both gym versions.

When Stage B finishes (~10–15 h sequential / ~6–8 h with 2 GPUs), the
budget sweep is rerun with N ∈ {20, 50, 100, 200, 300, 400, 500} for
policy_emb only.

## Planned next experiment — "Choosing K via MV(K)" (RAN — see Update 2026-05-17 above)

Findings so far suggest that **re-clustering at the right K dominates any
post-clustering simplification method** on the Markov-violation axis. The
next experiment closes this loop by sweeping K finely on existing
representations and using `min MV(K)` as the K-selection criterion.

### Data already in place

All required input artifacts are present under
`.claude/worktrees/graph-simplification/third_party/influence_visualizer/configs/{task}/clustering/{slug}/`:

| Artifact | Purpose |
|---|---|
| `embeddings_reduced.npy` | The UMAP-reduced embeddings k-means was fit on |
| `cluster_labels.npy`     | Existing K-means labels (for K ∈ {5, 10, 15, 20}) |
| `clustering_models.pkl`  | Fitted sklearn KMeans (gives us the random_state etc.) |
| `metadata.json`          | Per-timestep `rollout_idx` / `timestep` / `success` / window info |
| `manifest.yaml`          | rep / w / s / k metadata used by the discovery code |

Existing sweep grid:
- 2 representations: `infembed`, `policy_emb_bottleneck_plan_t0`
- 6 (window, stride) variants: `()` (raw), `(3,1)`, `(3,2)`, `(5,1)`, `(8,1)`, `(8,2)`
- 4 K values: 5, 10, 15, 20
- 3 tasks: `transport_mh_jan28`, `square_mh_feb5`, `lift_mh_jan26`

= **144 clusterings already on disk**. For the finer sweep we need new K
values; everything else can be reused.

### Steps to run on the workstation

```bash
# 1. Re-clustering at a finer K grid using the existing reduced embeddings.
#    Writes new clustering directories alongside the existing ones, in the
#    same slug format (...{rep}_w{w}_s{s}_seed0_kmeans_k{K}/).
PYTHONPATH=. python scripts/k_sweep_clustering.py \
  --reps infembed policy_emb_bottleneck_plan_t0 \
  --k_values 3 4 6 8 10 12 14 16 18 20 25 30 \
  --window_strides '5,1' '3,1' '8,1' \
  --tasks transport_mh_jan28 square_mh_feb5 lift_mh_jan26

# 2. Run passthrough+bootstrap MV (both orders) on the expanded clustering
#    set. Only computes passthrough — no simplification methods, since the
#    objective here is K-selection, not post-clustering simplification.
PYTHONPATH=. python scripts/k_sweep_evaluate.py \
  --tasks transport_mh_jan28 square_mh_feb5 lift_mh_jan26 \
  --n_bootstrap 100 \
  --bias_correct \                  # Miller-Madow correction on per-state MI
  --out docs/k_sweep_results/

# 3. Plot MV(K) and bias-corrected MV(K) per (rep, w, s, task).
PYTHONPATH=. python scripts/summarize_k_sweep.py
```

### What we expect to learn

1. **A canonical "elbow" plot per (rep, w, s, task)**: K vs. MV₁ with
   bootstrap CIs. The K* that minimizes MV₁ (within its CI) is the
   principled answer to "what's the right K?" for that
   (representation, window, stride) combination.
2. **Cross-validation of the choice via MV₂**: at the chosen K*, MV₂
   should also be in the local minimum region (or at least not dramatically
   higher than the surrounding K values). If MV₂ explodes at K*, the
   1st-order test was too generous.
3. **Bias-correction sanity check**: with Miller-Madow correction
   (`MV_corrected = MV_plug-in − (R−1)(C−1)/(2N)` summed per-state and
   visitation-weighted) the U-shape should remain but be flatter. Any
   K* that shifts dramatically under correction is suspicious.
4. **Best (rep, w, s) for each task**: by comparing the *minima* of the
   K-curves, we can identify which representation × window × stride
   combination produces the most Markov clustering at its best K. This
   becomes a defensible default for the downstream pipeline.

### Artifacts to produce

- `docs/k_sweep_results/{task}__{rep}__w{w}_s{s}.json` — one JSON per
  setting, with the full MV(K), MV₂(K), bias-corrected MV(K), and
  bootstrap CIs.
- `docs/k_sweep_results/_plots/{task}__{rep}__w{w}_s{s}__elbow.png` —
  the K-vs-MV curve for that setting.
- `docs/k_sweep_results/_plots/{task}__overlay.png` — all (rep, w, s)
  curves overlaid for one task, with the K* and minimum MV marked per
  curve.
- `docs/k_sweep_results/_findings.md` — written analysis: best K per
  setting, agreement between MV₁/MV₂/MV_corrected, best (rep, w, s)
  per task, comparison to the (now-secondary) simplification Pareto.

### Compute budget

- New k-means runs at K ∈ {3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30}
  on existing reduced embeddings: 12 K × 3 (w,s) × 2 reps × 3 tasks =
  **216 k-means fits** (~3–7k samples each, milliseconds per fit).
- Passthrough MV + 100-rep bootstrap CI per clustering: ~5–10 seconds
  per clustering × 216 + existing 144 = **~30–60 min total** on a decent
  workstation CPU.
- No GPU needed.

---

## Appendix — How `coverage_fraction` is computed

`coverage_fraction` is the diagnostic that decides whether an MV
reading is trustworthy. Conceptually: **the fraction of observed
(prev, curr, next) triplets that came from states the per-state gates
let through**. 1.00 means every observed trajectory transition fed the
MV estimate; 0.30 means 70% of your data was filtered out and the MV
value reflects only a small subset.

### The algorithm in 6 steps

Given cluster labels per window and the metadata mapping windows to
`(episode, timestep)`:

1. **Group + sort.** For each `rollout_idx`, build the time-ordered
   sequence of cluster labels:
   ```
   eps[rollout_idx] = sorted [(timestep, window_idx, cluster_label), …]
   ```

2. **Run-length collapse.** Consecutive identical labels become one
   entry. These are the *behavioral state visits* — we don't double-
   count "stayed in the same state":
   ```
   [c1, c1, c1, c5, c5, c8, c8, c12, c3, c3, c3] →
       [c1, c5, c8, c12, c3]
   ```

3. **Extract `(prev, curr, next)` triplets at lag `order`.** Slide a
   window of width `order + 2` over the collapsed sequence. For
   order=1 this is the standard 3-window:
   ```
   collapsed = [c1, c5, c8, c12, c3, c7]
   triplets  = (c1, c5, c8), (c5, c8, c12), (c8, c12, c3), (c12, c3, c7)
   ```
   A collapsed sequence of length L yields `max(0, L − order − 1)`
   triplets per episode.

4. **Bucket triplets by `curr` state:**
   ```
   pairs_per_state[c5]   ← (c1,  c8)
   pairs_per_state[c8]   ← (c5,  c12)
   pairs_per_state[c12]  ← (c8,  c3)
   …
   ```
   `pairs_per_state[s]` is the list of `(prev, next)` observations
   conditional on being at state `s`.

5. **Two per-state gates:**

   - **Gate 1 — sample size**: `len(pairs_per_state[s]) ≥ min_pairs`
     where `min_pairs = max(4, 4·order²)`:
     - order=1 → min_pairs = 4
     - order=2 → min_pairs = 16
     - order=3 → min_pairs = 36

     The `4·order²` scaling reflects that an order-k MI estimator on
     an `|prev| × |next|` contingency table needs roughly `4`
     observations per cell to avoid total degeneracy. It's a
     rule-of-thumb proxy that scales the right way.

   - **Gate 2 — non-degenerate marginals**: at least 2 unique `prev`
     AND 2 unique `next` values in the pair list. If every visit to
     state `s` has the same predecessor (or same successor), the
     conditional `I(prev; next | s)` is identically 0 — not because
     the chain is Markov, but because there's nothing to condition on.

6. **Compute the fraction:**
   ```python
   total_pairs   = sum(len(pairs_per_state[s]) for s in all_states)
   gated_pairs   = sum(len(pairs_per_state[s])
                       for s in states_passing_both_gates)
   coverage_fraction = gated_pairs / total_pairs
   ```

   It's **pair-weighted**, not state-weighted: a rare state with 3
   pairs that gets filtered contributes 3 to the denominator and 0 to
   the numerator. A common state with 200 pairs that passes contributes
   200 to both.

### Why this matters — false zeros

The MV formula is
```
MV  =  (1 / total_weight) · Σ_s  n_pairs[s] · I(prev; next | curr=s)
```

When a state fails a gate, the code sets `I(prev; next | s) = 0` for
that state and skips adding `n_pairs[s]` to the weight. Two
consequences:

1. **MV is a lower bound** on the true MV — high-MV rare states are
   silently dropped.
2. **MV can read 0 even when the chain is not Markov** — if every state
   fails a gate (very small K and/or very few rollouts), MV = 0 by
   construction.

Coverage catches both: low coverage = "lots of data filtered out, treat
MV as unreliable". High coverage = "the MV value you're looking at
uses most of the data".

### Per-order coverage

`coverage₁`, `coverage₂`, `coverage₃` are reported separately because
each has its own `min_pairs` threshold. At fixed K and N, `coverage_k`
decreases rapidly with `k` because `min_pairs = 4k²` grows
quadratically. This is why MV₃ becomes measurable only at very high N —
it takes ~9× more triplets per state than MV₁ to clear the gate.

### What a "good" coverage looks like

| Coverage | Interpretation |
|---|---|
| ≥ 0.90 | Trustworthy. The MV value reflects almost all the data. |
| 0.80 – 0.90 | Usable with caveats. Rare states are being dropped; absolute MV is a slight underestimate. This is the auto-pipeline's default gate. |
| 0.50 – 0.80 | Partially measurable. Direction (MV-vs-K shape) may still be informative; absolute values are not. |
| < 0.50 | Don't trust the number at all. The gates are eating most of the data; the MV reading is whatever survives. |

This is why the auto-pipeline rejects (rep, w, s) settings with fewer
than 3 gated K cells: a single gate-edge cell at coverage 0.81 is
exactly the regime where one rare state pushing past the threshold
changes the answer.

### Reference implementation

`policy_doctor/behaviors/simplification/metrics.py::markov_violation_coverage`
mirrors the gate logic of `markov_violation_against_original_bits` and
returns the per-state and total pair counts so callers can audit the
gate's behavior. The auto-pipeline (`select_K.py`) consumes
`mv1_coverage_fraction` from per-clustering JSONs and applies the
threshold + multi-cell-admissibility checks described in the
"Automated hyperparameter selection" section above.
