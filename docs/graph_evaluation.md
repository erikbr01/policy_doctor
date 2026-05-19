# Graph Evaluation: Cluster Quality and the Markov Property

This document describes the full methodology for evaluating the quality of behavior-graph
clusterings: geometric separability (silhouette), behavioral coverage, automatic model
selection, and the statistical test for the Markov property. All experiments are on
`transport_mh` task data unless noted.

---

## 1. Background and Motivation

The behavior graph represents policy rollouts as a probabilistic directed graph over
discrete behavioral states. Each node is a cluster of timestep embeddings; each directed
edge carries a transition probability estimated from the run-length-collapsed sequences
of cluster visits across episodes. Downstream uses — Bellman-equation value computation,
advantage-based data curation, path enumeration for seed selection — all rely on the
quality of the underlying clustering in two orthogonal senses:

1. **Geometric quality**: clusters should be compact and well-separated in the embedding
   space so that cluster membership is meaningful and stable.
2. **Behavioral quality**: the Markov property should hold — the probability of
   transitioning to state $s'$ given the current state $s$ should not depend on the
   predecessor state $s^{-1}$. If it does, first-order value iteration is biased.

Three complementary metrics operationalize these goals: the silhouette score for
geometric separability, coverage metrics for behavioral resolution, and the
second-order contingency test for the Markov property. Silhouette-based automatic
K selection connects geometric quality to hyperparameter selection.

---

## 2. Cluster Quality Metrics

### 2.1 Geometric Separability: Silhouette Score

The silhouette score measures how much better each sample fits its own cluster than the
nearest competing cluster.

**Setup.** Each window $i$ lives at a point $\mathbf{z}_i \in \mathbb{R}^D$ in the UMAP
embedding space ($D = 50$ for jan28, $D = 100$ for mar27). The distance between two windows
is the Euclidean distance $d(i, j) = \|\mathbf{z}_i - \mathbf{z}_j\|_2$. Each window is
assigned a cluster $c_i \in \{0, \ldots, K-1\}$; $C_c$ denotes the set of windows in cluster $c$.

**Per-sample score.** For sample $i$:

$$
a_i = \frac{1}{|C_{c_i}| - 1} \sum_{\substack{j \in C_{c_i} \\ j \neq i}} d(i, j)
$$

$a_i$ is the mean distance from $i$ to all other members of its own cluster — a measure of
how tightly $i$ is packed within its cluster. Small $a_i$ means $i$ is close to its cluster
centre.

$$
b_i = \min_{c \neq c_i} \;\frac{1}{|C_c|} \sum_{j \in C_c} d(i, j)
$$

$b_i$ is the mean distance from $i$ to every member of the nearest competing cluster,
minimised over all clusters $c \neq c_i$. It measures how far $i$ is from the cluster it
fits into second-best.

The per-sample silhouette is then:

$$
s_i = \frac{b_i - a_i}{\max(a_i,\; b_i)}
$$

The $\max$ in the denominator normalises $s_i$ to the interval $[-1, 1]$:
- When $b_i > a_i$ (i closer to own cluster): denominator $= b_i$, so $s_i = 1 - a_i / b_i \in (0, 1]$. Near $+1$ when $a_i \ll b_i$ (tight cluster, far from neighbours).
- When $a_i = b_i$: $s_i = 0$ (borderline case — equally close to own cluster and nearest other).
- When $a_i > b_i$ (i closer to a foreign cluster): denominator $= a_i$, so $s_i = b_i / a_i - 1 \in [-1, 0)$. Near $-1$ when $b_i \ll a_i$ (likely misassigned).
- Singleton clusters ($|C_{c_i}| = 1$): $a_i$ is undefined; $s_i = 0$ by convention.

**Dataset-level score:**

$$
\bar{s} = \frac{1}{N} \sum_{i=1}^{N} s_i \;\in\; [-1, 1]
$$

Values near $+1$ mean clusters are compact and well-separated; near $0$ indicates clusters
that overlap or are barely distinguishable; negative values indicate a systematically
mis-specified number of clusters.

**Computational note.** Computing $a_i$ and $b_i$ naively requires $O(N^2)$ distance
evaluations. In practice, the score is computed on a random subsample of 3,000 points
(drawn without replacement, seeded for reproducibility), keeping complexity tractable.
Variance across subsamples at this size is approximately $\pm 0.01$.

**Implementation.** `policy_doctor/behaviors/graph_simplification.py:auto_k_kmeans()` runs
K-Means for each $k$ in a sweep range, computes the silhouette on the 3,000-point subsample,
and returns the $(labels, k^*, \{k \to \bar{s}\})$ triple for the best-scoring $k$.

**Limitation.** Silhouette is a property of the embedding geometry, not of the clustering's
relationship to behavior. Large window strides mechanically push adjacent windows apart in
feature space — non-overlapping windows share no frames, so their embeddings decorrelate
regardless of behavioral similarity. This inflates silhouette for high-stride configurations
while behavioral swap rates simultaneously worsen (see §2.2). Silhouette should therefore be
used as one input to model selection, not as the sole criterion.

---

### 2.2 Behavioral Coverage Metrics

Coverage metrics answer whether the clustering captures the right behavioral structure
at the right resolution. They operate on the run-length-collapsed episode sequences
(§3.2), not on raw embedding coordinates.

#### 2.2.1 Mean Distinct Clusters per Episode

**Setup.** For episode $e$, let $(\hat{\ell}_1, \hat{\ell}_2, \ldots, \hat{\ell}_{N_e})$ be the
sequence of cluster labels for its $N_e$ windows in temporal order. Run-length encoding (RLE)
collapses consecutive identical labels into a single entry, giving the visit sequence
$\text{RLE}(e) = (c_1, c_2, \ldots, c_{T_e})$ where $c_k \neq c_{k+1}$ for all $k$ and
$T_e \leq N_e$. The number of distinct behavioral phases visited by episode $e$ is:

$$
D_e = \bigl|\{c : c \in \text{RLE}(e)\}\bigr|
$$

Note that $D_e$ counts unique labels, not transitions — if the episode revisits a cluster after
leaving it ($c_1 \to c_2 \to c_1$), that cluster is counted once.

**Metric:**

$$
\overline{D} = \frac{1}{|E|} \sum_{e \in E} D_e
$$

$\overline{D}$ measures behavioral resolution: how many distinct behavioral phases the
clustering assigns to a typical episode. For tasks with a fixed structure (e.g.,
reach → grasp → transport → place → outcome), $\overline{D}$ should be close to the
number of real task phases. If $\overline{D}$ is much lower, phases are being collapsed;
if much higher, the clustering is fragmenting single phases across multiple clusters.

For `transport_mh` (roughly 5 task phases), the target range is $\overline{D} \in [4, 6]$
at moderate K. Configuration W=5,S=1 at K=10 gives $\overline{D} = 2.99$ — episodes
visit on average fewer than three distinct clusters, indicating some phase collapse. The
W=5,S=5 configuration gives $\overline{D} = 5.30$ (closest to the true phase count) but at
the cost of high swap rate.

#### 2.2.2 Swap Rate (Per-Frame)

The per-frame swap rate measures temporal coherence: how often the clustering changes label
between adjacent windows, expressed relative to total observation frames rather than total
windows, making the metric independent of the stride parameter.

**Setup.** With window width $W$ and stride $S$, episode $e$ of length $F_e$ frames yields
$N_e = \lfloor(F_e - W)/S\rfloor + 1$ windows. Window $k$ spans frames $[kS,\, kS + W - 1]$
and carries label $\hat{\ell}_k$. Adjacent windows $k$ and $k+1$ overlap in $W - S$ frames
(or share no frames when $S = W$).

The number of label changes (swaps) within episode $e$ is:

$$
\text{swaps}(e) = \sum_{k=0}^{N_e - 2} \mathbf{1}\bigl[\hat{\ell}_{k+1} \neq \hat{\ell}_k\bigr]
$$

**Per-window swap rate** (not stride-fair):

$$
\text{swap}_{w} = \frac{\displaystyle\sum_e \text{swaps}(e)}{\displaystyle\sum_e (N_e - 1)}
$$

At stride $S$, the denominator $\sum_e (N_e - 1) \approx \sum_e F_e / S$, so
$\text{swap}_w$ scales as $S \times \text{swap}_f$. Larger stride mechanically deflates the
rate because there are fewer window pairs, even though each window jump crosses more frames.

**Per-frame swap rate** (stride-fair):

$$
\text{swap}_{f} = \frac{\displaystyle\sum_e \text{swaps}(e)}{\displaystyle\sum_e F_e}
$$

The denominator is the total number of observation frames across all episodes. Because
$\sum_e F_e$ does not depend on stride, this rate compares fairly across $(W, S)$
configurations. Equivalently, since $\sum_e N_e \approx \sum_e F_e / S$:

$$
\text{swap}_{f} \approx \text{swap}_{w} \cdot \frac{1}{S}
$$

so the per-frame rate undoes the stride deflation and reflects the true label-change density
per observation frame.

A low per-frame swap rate indicates stable cluster assignments — the policy stays in one
behavioral mode across many consecutive frames before transitioning. A high swap rate
indicates flicker: the clustering assigns consecutive frames to different clusters even
within what should be a single behavioral phase.

Reference values from the jan28 policy_emb sweep at K=10:

| W | S | sw/frm% | distinct/ep | MI(succ) |
|---|---|---------|-------------|----------|
| 1 | 1 | 4.79    | 4.59        | 0.096    |
| 3 | 1 | 3.28    | 3.71        | 0.133    |
| 5 | 1 | 2.37    | 2.99        | **0.188** |
| 5 | 5 | 5.47    | **5.30**    | 0.137    |
| 10| 2 | **2.07**| 2.94        | 0.150    |
| 10| 5 | 2.79    | 3.60        | 0.158    |

The W=10,S=2 configuration achieves the lowest swap rate (2.07%) but also the lowest
$\overline{D}$ (2.94), indicating phase collapse. The W=5,S=1 configuration achieves the
best outcome-predictiveness while maintaining reasonable swap rate.

#### 2.2.3 Mutual Information with Episode Outcome

**Setup.** Let $\mathcal{W} = \{w_1, \ldots, w_N\}$ be the set of all $N$ windows across all
rollout episodes. Each window $w_i$ carries two attributes:
- $\ell_i \in \{0, \ldots, K-1\}$: its cluster assignment.
- $o_i \in \{0, 1\}$: the binary success/failure outcome of the episode it belongs to ($1$ =
  success, $0$ = failure), broadcast uniformly to every window of the same episode.

**Empirical probability estimates.** Define counts:
- $n_{c,o} = |\{i : \ell_i = c,\, o_i = o\}|$ — windows in cluster $c$ from episodes with outcome $o$.
- $n_c = \sum_o n_{c,o} = |\{i : \ell_i = c\}|$ — all windows in cluster $c$.
- $n_o = \sum_c n_{c,o} = |\{i : o_i = o\}|$ — all windows from episodes with outcome $o$.

The empirical joint and marginal probabilities are:

$$
\hat{P}(\ell = c,\; \text{succ} = o) = \frac{n_{c,o}}{N}, \qquad
\hat{P}(\ell = c) = \frac{n_c}{N}, \qquad
\hat{P}(\text{succ} = o) = \frac{n_o}{N}
$$

**The MI formula.** Mutual information is the expectation (under the joint distribution) of
the log-ratio of the joint to the product of marginals:

$$
\text{MI}(\ell,\, \text{succ}) = \sum_{c=0}^{K-1} \sum_{o \in \{0,1\}} \hat{P}(\ell = c,\; \text{succ} = o)\;\log \frac{\hat{P}(\ell = c,\; \text{succ} = o)}{\hat{P}(\ell = c)\;\hat{P}(\text{succ} = o)}
$$

Substituting the counts:

$$
= \sum_{c,\,o} \frac{n_{c,o}}{N} \log \frac{n_{c,o} \cdot N}{n_c \cdot n_o}
$$

**Interpretation of the log-ratio.** The term $\log \dfrac{n_{c,o} \cdot N}{n_c \cdot n_o}$
is the log of the observed co-occurrence count $n_{c,o}$ relative to the count expected
under independence, $n_c \cdot n_o / N$:

- **Positive** ($n_{c,o} > n_c n_o / N$): cluster $c$ is enriched in outcome $o$ — windows in
  this cluster come disproportionately from episodes with that outcome. The clustering predicts
  this outcome.
- **Zero** ($n_{c,o} = n_c n_o / N$): cluster $c$ and outcome $o$ co-occur exactly as often as
  statistical independence predicts. This cluster carries no information about this outcome.
- **Negative** ($n_{c,o} < n_c n_o / N$): cluster $c$ is depleted in outcome $o$ — also
  predictive, but in the opposite direction (the cluster predicts the *other* outcome).

The outer weight $\hat{P}(\ell=c, \text{succ}=o)$ means the sum is dominated by the most
common (cluster, outcome) pairs; rare combinations contribute little regardless of their
log-ratio.

**MI as reduction in outcome uncertainty.** Equivalently:

$$
\text{MI}(\ell,\, \text{succ}) = H(\text{succ}) - H(\text{succ} \mid \ell)
$$

where the marginal entropy $H(\text{succ}) = -\sum_o \hat{P}(\text{succ}=o)\log\hat{P}(\text{succ}=o)$
is the a priori uncertainty about episode outcome, and the conditional entropy

$$
H(\text{succ} \mid \ell) = -\sum_{c} \hat{P}(\ell=c) \sum_{o} \hat{P}(\text{succ}=o \mid \ell=c) \log \hat{P}(\text{succ}=o \mid \ell=c)
$$

is the average remaining uncertainty about outcome after observing the cluster label. MI is
therefore the average reduction in outcome uncertainty provided by knowing the cluster.

**Non-negativity.** MI $\geq 0$ always, with equality iff the joint factorises:
$\hat{P}(\ell=c, \text{succ}=o) = \hat{P}(\ell=c)\hat{P}(\text{succ}=o)$ for all $(c, o)$.
This follows from the log-sum inequality (equivalently, from the KL divergence
$D_\text{KL}(\hat{P}(\ell, \text{succ}) \| \hat{P}(\ell)\hat{P}(\text{succ})) \geq 0$).

**Free-passenger clusters.** A cluster that appears equally in successful and failed episodes
has $n_{c,0}/n_c = n_0/N$ and $n_{c,1}/n_c = n_1/N$ — the within-cluster outcome mix equals
the dataset-level mix. Both log-ratios are then zero, so the cluster contributes zero to MI
regardless of how many windows it contains. This makes MI neutral to task-structure clusters
(e.g., the "reaching" phase) that must be traversed in every episode, rewarding only the
clusters that discriminate between outcomes.

**Key result.** On jan28 policy_emb at K=10, W=5,S=1 achieves MI=0.188 nats — nearly 2×
the W=10,S=2 value (0.150), despite having comparable swap rates. The large-window
configuration (W=10) blurs the discriminative late-manipulation phase into earlier
approach frames, destroying the MI signal.

#### 2.2.4 Testable State Fraction

A cluster state is *testable* for the Markov property if it satisfies three criteria
(detailed in §3.4): it must have at least 2 distinct predecessor states, at least 2
distinct successor states, and at least `min_total` total observed transitions. The
testable state fraction is:

$$
f_\text{testable} = \frac{|\{s : \text{testable}(s) = \text{True}\}|}{K}
$$

This metric reflects how well the rollout data covers the transition structure of the
graph. Low testable fraction means either (a) too many clusters for the rollout budget —
most states are too rarely visited to accumulate diverse predecessors/successors — or
(b) too much temporal concentration — some states dominate and others are near-empty.

Reference values:
- jan28 infembed, K=10, N=100 rollouts: 8/10 testable (80%)
- jan28 policy_emb, K=10, N=100 rollouts: 5/10 testable (50%)
- mar27 infembed, K=20, N=200 rollouts: 16/20 testable (80%)

---

### 2.3 Automatic K Selection via Silhouette Sweep

Selecting the number of clusters K involves a resolution–coherence trade-off. Too small a K
merges distinct behavioral phases; too large inflates the graph with noisy near-duplicate
nodes that violate the Markov property and degrade readability. The silhouette sweep provides
a data-driven default.

**Algorithm** (`auto_k_kmeans` in `graph_simplification.py`):

1. Fix a search range $[k_\text{min}, k_\text{max}]$ (default: $[4, 15]$).
2. For each $k$ in the range, run K-Means with `n_init=10` on the full embedding set.
3. Compute silhouette score $\bar{s}(k)$ on a 3,000-point random subsample.
4. Return $(labels, k^*, \{\bar{s}(k)\})$ where $k^* = \arg\max_k \bar{s}(k)$.

**Results on transport_mh_jan28 with InfEmbed:**
The sweep selected $k^* = 4$ with $\bar{s}(4) = 0.55$, dropping to $0.46$–$0.50$ at
$k \geq 10$. This reduced the graph from 23 nodes / 140 edges (K=20 baseline) to 7 nodes /
18 edges — an 87% edge reduction — producing a visually interpretable
START → Approach → Manipulate → Final → SUCCESS/FAILURE structure.

**Representation dependence.** The auto-K selection is sensitive to the feature space:

| Representation | auto-K | Silhouette (at auto-K) | Notes |
|---|---|---|---|
| InfEmbed | 4 | 0.55 | Collapses to 4 tight geometric modes |
| policy_emb/bottleneck | 14 | ~0.47 | Supports up to 14 separable clusters |
| state | 4 | — | Same collapse as InfEmbed |
| state_action | 4 | — | Same collapse |

The policy bottleneck features encode richer behavioral information — the policy's internal
plan representation at the final denoising step distinguishes finer-grained phases that
InfEmbed averages over. Consequently, silhouette-based auto-K selects lower K for
InfEmbed (fewer separable clusters) and higher K for policy_emb (more separable clusters),
even though the true number of task phases is the same. This means auto-K is useful as a
ceiling on the resolving power of the feature space, not as an estimate of behavioral structure.

**Practical recommendation.** For graph construction intended to be human-readable, use auto-K
on InfEmbed features with $[4, 12]$ range. For Markov property analysis requiring behavioral
granularity, use K=10 on policy_emb features: it provides enough resolution to detect
phase-level transitions while keeping the testable-state fraction high.

---

## 3. The Markov Property Test

### 3.1 Formal Statement

A discrete-time stochastic process $\{S_t\}$ satisfies the **first-order Markov property**
if, for all states $s, s', s^{-1}$:

$$
P(S_{t+1} = s' \mid S_t = s,\; S_{t-1} = s^{-1}) = P(S_{t+1} = s' \mid S_t = s)
$$

In words: the conditional distribution over next states depends only on the current state,
not on the history beyond the immediately preceding state. The behavior graph explicitly
assumes this property when estimating transition probabilities from single-step counts.

We test the Markov property per state $s$ by asking whether the conditional distribution
$P(S_{t+1} \mid S_t = s)$ is the same regardless of which state $S_{t-1}$ preceded $s$.
Equivalently, we test whether the pairs $(S_{t-1}, S_{t+1})$ are independent given $S_t = s$.

This is a second-order test: it detects exactly one degree of memory beyond the Markov
horizon. It is not a test of higher-order dependencies (e.g., $S_{t-2}$), and passing the
test does not rule them out.

---

### 3.2 Data Preprocessing: Sequence Extraction and Episode Augmentation

**Inputs.** The test takes two arrays as input:
- `cluster_labels: ndarray[int]` of shape $(N,)$, giving the cluster assignment
  $\hat{c}_i \in \{0, \ldots, K-1\}$ of each timestep window $i$. Windows assigned label
  $-1$ (noise label from HDBSCAN) are excluded.
- `metadata: List[Dict]`, length $N$, where each dict carries at least the episode index
  (key `"rollout_idx"` for `level="rollout"` or `"demo_idx"` for `level="demo"`) and the
  temporal sort key (`"timestep"` or `"window_start"`).

**Per-episode grouping.** Timestep windows are grouped by episode index. Within each
episode they are sorted by `window_start` (ascending). This establishes the temporal order
within each episode. The `success` field of each window's metadata, if present, is recorded
as the episode's outcome; the last observed value per episode is used.

**Run-length encoding (RLE).** Raw cluster sequences contain long runs of the same label —
consecutive timestep windows in the same behavioral phase will all be assigned the same
cluster. These runs are collapsed to avoid treating repeated within-phase transitions as
independent observations. The algorithm:

```
collapsed[ep_idx] = []
for (sort_key, label) in sorted(episodes[ep_idx]):
    if label != collapsed[ep_idx][-1]:  # label change
        collapsed[ep_idx].append(label)
```

The result is a sequence of distinct labels representing the succession of behavioral
phases — the "visit sequence" of the episode. Each element of the collapsed sequence
corresponds to one contiguous behavioral run, not one timestep.

**Episode augmentation.** To represent episode-level structure (start, outcome) within the
same framework as behavioral transitions, each collapsed sequence $[c_1, c_2, \ldots, c_T]$
is augmented to a full sequence:

$$
[\texttt{START}] \circ [c_1, c_2, \ldots, c_T] \circ [\texttt{TERMINAL}]
$$

where $\texttt{TERMINAL}$ is one of three special node IDs:
- $\texttt{SUCCESS}$ if `episode_outcome == True`
- $\texttt{FAILURE}$ if `episode_outcome == False`
- $\texttt{END}$ if outcome information is unavailable

The special node IDs are large negative integers outside the cluster-label space so they do
not collide with any cluster assignment.

**Effect of `exclude_terminals`.** The augmented sequence introduces a confound: states near
episode boundaries have position-dependent outcomes. A cluster state that appears
predominantly at the start of episodes will often be followed by SUCCESS (since the episode
hasn't had time to fail), regardless of any behavioral Markov violation. Because the
predecessor state carries information about temporal position, this creates an apparent
second-order dependency that is an artifact of the episode structure rather than genuine
history-dependence in the dynamics. Setting `exclude_terminals=True` filters out any
transition triple $(s^{-1}, s, s')$ where $s^{-1} \in \{\texttt{START}\}$ or
$s' \in \{\texttt{SUCCESS}, \texttt{FAILURE}, \texttt{END}\}$, isolating the interior
behavioral dynamics.

---

### 3.3 Second-Order Count Collection

The core data structure is a nested dict `counts[current][prev][next]` counting how many
times the triple $(S_{t-1} = \text{prev},\; S_t = \text{current},\; S_{t+1} = \text{next})$
was observed in the collapsed sequences. Collection iterates over all augmented sequences and
all interior positions $t \in \{1, \ldots, T-1\}$ (1-indexed, excluding first and last
elements which are boundary nodes):

```python
for ep_idx, seq in collapsed.items():
    full_seq = [START] + seq + [terminal]
    for i in range(1, len(full_seq) - 1):
        prev, current, next_ = full_seq[i-1], full_seq[i], full_seq[i+1]
        if prev in skip_nodes or next_ in skip_nodes:
            continue
        counts[current][prev][next_] += 1
```

When `exclude_terminals=True`, `skip_nodes = {START} ∪ {SUCCESS, FAILURE, END}`, so triples
where START is the predecessor or any terminal is the successor are skipped.

**Pooling.** Multiple independent rollout sets sharing the same clustering can be pooled by
summing their count dicts element-wise before running the tests. This is valid when the same
cluster IDs correspond to the same behavioral states across the datasets (e.g., evaluating
multiple eval batches under a fixed clustering from training). Pooling increases the total
observation count and improves statistical power. The API is `test_markov_property_pooled(
datasets=[(labels_1, meta_1), (labels_2, meta_2), ...])`.

---

### 3.4 Testability Criteria

For each cluster state $s$, a contingency table is constructed from `counts[s]` and then
tested. A state is **testable** only if all three of the following criteria are met:

1. **At least 2 distinct predecessor states.** The contingency table must have at least 2
   rows. A single predecessor means there is no variation in the conditioning variable —
   the test has no power to detect history-dependence.

2. **At least 2 distinct successor states.** The table must have at least 2 columns.
   A single possible next state makes independence trivially hold (there is no choice to
   be influenced by history).

3. **Sufficient total observations.** The table must have at least `min_total` observations
   in total. For the chi-squared test (`method="chi2"`), `min_total = 5`; for the exact
   and modal permutation tests, `min_total = 3`.

States failing any criterion are marked `testable=False` with a reason string:
`"no_interior_transitions"`, `"only_one_predecessor"`, `"only_one_successor"`, or
`"insufficient_data"`. These states are excluded from the summary violation count and the
overall `markov_holds` flag.

---

### 3.5 Statistical Tests

For each testable state $s$, let $r$ be the number of distinct predecessors and $c$ the
number of distinct successors. The contingency table $T_s$ is an $r \times c$ integer
matrix:

$$
T_s[i,j] = \text{counts}[s][\text{prev}_i][\text{next}_j]
$$

The null hypothesis is **independence of predecessor and successor given the current state**:
$P(S_{t+1} = \text{next}_j \mid S_t = s, S_{t-1} = \text{prev}_i) = P(S_{t+1} = \text{next}_j \mid S_t = s)$.

Three tests are implemented, selected via the `method` parameter.

#### 3.5.1 Asymptotic Chi-Squared Test (`method="chi2"`)

**Deriving the expected counts $E_{ij}$.** Under the null hypothesis $H_0$, predecessor and
successor are conditionally independent given the current state $s$:

$$
P(\text{prev}=p_i,\; \text{next}=n_j \mid \text{current}=s) = P(\text{prev}=p_i \mid s)\;\cdot\;P(\text{next}=n_j \mid s)
$$

The maximum-likelihood estimates of these conditional marginals from the table are:

$$
\hat{P}(\text{prev}=p_i \mid s) = \frac{R_i}{N_s}, \quad R_i = \sum_{j=1}^{c} T_s[i,j] \quad\text{(row sum)}
$$

$$
\hat{P}(\text{next}=n_j \mid s) = \frac{C_j}{N_s}, \quad C_j = \sum_{i=1}^{r} T_s[i,j] \quad\text{(column sum)}
$$

$$
N_s = \sum_{i=1}^{r}\sum_{j=1}^{c} T_s[i,j] \quad\text{(table total)}
$$

The expected count in cell $(i,j)$ under $H_0$ is therefore:

$$
E_{ij} = N_s \cdot \hat{P}(\text{prev}=p_i \mid s) \cdot \hat{P}(\text{next}=n_j \mid s) = N_s \cdot \frac{R_i}{N_s} \cdot \frac{C_j}{N_s} = \frac{R_i\, C_j}{N_s}
$$

**The Pearson statistic.** The chi-squared statistic measures the total squared deviation of
observed counts from expected counts, standardised by the expected count:

$$
\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{\bigl(T_s[i,j] - E_{ij}\bigr)^2}{E_{ij}}
$$

Each term $(T_s[i,j] - E_{ij})^2 / E_{ij}$ is a standardised squared residual for cell $(i,j)$:
large when the observed count deviates substantially from independence, weighted by $1/E_{ij}$
so that deviations in rare cells (small $E_{ij}$) contribute proportionally more.

Under $H_0$, the statistic follows asymptotically a chi-squared distribution with
$(r-1)(c-1)$ degrees of freedom. The df count comes from the number of free parameters in
the interaction: the full table has $rc - 1$ free cell probabilities; under independence,
these are parameterised by $(r-1) + (c-1)$ free marginals, leaving $(rc - 1) - (r - 1) -
(c - 1) = (r-1)(c-1)$ interaction degrees of freedom.

The p-value is the right-tail probability:

$$
p = P\!\left(\chi^2_{(r-1)(c-1)} \geq \chi^2_\text{obs}\right)
$$

Scipy's `chi2_contingency` skips cells with $E_{ij} = 0$ (structurally zero expected count
when a row sum or column sum is zero) rather than treating them as $0/0$.

**Validity condition.** The asymptotic approximation is reliable only when all expected
cell counts $E_{ij} \geq 5$. In practice, with 100 rollout episodes spread across
$K = 10$–$20$ clusters, many contingency tables have small expected counts, and the
asymptotic chi-squared distribution overestimates the right tail — inflating Type I
error (over-rejecting). For this regime, the permutation test is preferred.

#### 3.5.2 Exact Permutation Test (`method="exact"`)

The permutation test computes the same Pearson chi-squared statistic but derives the
p-value from an empirical null distribution rather than the asymptotic chi-squared family.

**Procedure:**

1. Expand the table $T_s$ into observation-level arrays: for each cell $(i, j)$ with count
   $T_s[i,j]$, add $T_s[i,j]$ copies of the pair $(\text{prev}_i, \text{next}_j)$. This gives two
   arrays `rows` and `cols` of length $N_s = \sum_{i,j} T_s[i,j]$.

2. Compute the observed statistic $\chi^2_\text{obs}$ from the original table.

3. For each permutation $b = 1, \ldots, B$:
   - Shuffle the `rows` array in place (permuting predecessor labels while keeping
     successor labels and column-marginals fixed).
   - Reconstruct a permuted table $T^{(b)}$ by counting $(rows_{\pi_b(k)}, cols_k)$ pairs.
   - Compute the permuted statistic $\chi^2_{(b)}$.

4. Compute the p-value with the +1 continuity correction:
$$
p = \frac{|\{b : \chi^2_{(b)} \geq \chi^2_\text{obs}\}| + 1}{B + 1}
$$

The +1 in numerator and denominator ensures the p-value is never exactly zero, and
corresponds to treating the observed data as one possible permutation of itself.

**Why permutation?** Permuting row labels while keeping column assignments fixed generates
data under the null (independence) while exactly preserving the column marginals and the
observed total. Under the null, every permutation of predecessor labels is equally likely;
the fraction of permutations that produce a statistic at least as extreme as observed is
the exact p-value.

**Default:** $B = 5{,}000$ permutations (used in the integration test; the code default is
$B = 10{,}000$). At $B = 5{,}000$ the minimum achievable p-value is $1/5001 \approx 0.0002$;
the expected standard error on a p-value of 0.05 is approximately $\sqrt{0.05 \times
0.95 / 5000} \approx 0.003$, sufficiently small for reliable threshold decisions.

#### 3.5.3 Modal Permutation Test (`method="modal"`)

The modal test asks a weaker question: does the most-likely next state change depending on
which state the agent came from? This tests for a qualitative violation of the Markov
property rather than a distributional one, and requires less data to answer reliably.

**Statistic.** For the contingency table $T_s$:
- The overall modal successor is $j^* = \arg\max_j \sum_i T_s[i,j]$ (most common next
  state across all predecessors).
- For each predecessor row $i$, the row-specific modal successor is $j^*_i = \arg\max_j T_s[i,j]$.
- The test statistic is the number of rows that disagree with the overall mode:
$$
M_\text{obs} = |\{i : j^*_i \neq j^*\}|
$$

The permutation procedure is identical to §3.5.2 — shuffle row labels, reconstruct table,
recompute $M$ — with the same +1-corrected p-value formula.

**Interpretation.** $M = 0$ means every predecessor sees the same most-likely next state;
the dominant transition is first-order. $M > 0$ means at least one predecessor's most-
likely successor differs from the population mode. Zero violations across all seeds and
states is strong evidence that the dominant graph structure is genuinely Markovian.

---

### 3.6 Multiple Testing

The test is applied independently to every testable state $s$ in the graph at a nominal
significance level $\alpha = 0.05$. With $m$ simultaneous tests, the expected number of
false rejections under the complete null hypothesis is $m \alpha$. At $m = 16$ testable
states (the mar27 baseline), the expected false rejection count is $0.8$ — less than one,
so the family-wise error rate concern is mild at this scale.

For settings with larger $m$ (e.g., $K = 30$ with high testable fraction), a Benjamini–
Hochberg false-discovery-rate correction should be applied: sort p-values $p_{(1)} \leq
\ldots \leq p_{(m)}$; reject $H_{(k)}$ for all $k \leq k^* = \max\{k : p_{(k)} \leq k\alpha/m\}$.
This controls the expected fraction of false positives among all rejections at level $\alpha$.

Currently no correction is applied: each per-state decision uses the raw p-value against
$\alpha = 0.05$. This is appropriate when the primary goal is to detect any violation rather
than to control the false-discovery rate across all states. For reporting violation counts,
focusing on states with $p \ll \alpha$ (e.g., $p < 0.001$) provides additional robustness.

---

### 3.7 Pooled Testing

`test_markov_property_pooled(datasets)` accepts a list of `(cluster_labels, metadata)` tuples
sharing a common set of cluster IDs, computes second-order counts independently for each
dataset, merges the count dicts by element-wise summation, and then runs the same per-state
tests on the pooled counts. This increases statistical power when the clustering is fixed and
multiple independent evaluation sets are available. It should not be used to pool data from
different clusterings (different cluster IDs), as cluster labels are not comparable across
clusterings.

---

## 4. Experimental Data

### 4.1 Mar27 Multi-Seed Experiment (All Three Tests)

Clustering of `transport_mh_seed0_r512` rollout episodes using InfEmbed features, 200
rollouts, K=20 K-Means. Chi-squared, exact permutation (B=5,000), and modal permutation
tests applied per-seed with `exclude_terminals=True` and `significance_level=0.05`.

Three independent policy seeds provide separate clusterings (cluster IDs are not comparable
across seeds). Each seed's clustering is run on 100 episodes drawn from that seed's policy.

| Parameter | Value |
|-----------|-------|
| Influence source | InfEmbed |
| Window width | 5; stride 2 |
| Reduction | UMAP 100D |
| Clustering | K-Means, $k = 20$ |
| Policy seeds | 0, 1, 2 (independent clusterings) |
| Episodes per seed | 100 |
| Rollout samples | ~3,755–3,879 per seed |
| Permutations $B$ | 5,000 |
| `exclude_terminals` | True |

### 4.2 Jan28 Single-Run Experiments (Chi-Squared Only)

Clustering of `transport_mh_jan28` rollout episodes using either InfEmbed or policy
bottleneck-plan-t0 features (50D UMAP), 100 rollouts, K=10 K-Means with sliding window
w=3, s=1. Chi-squared test only, `exclude_terminals=False` (default), `significance_level=0.05`.

| Parameter | Value |
|-----------|-------|
| Influence sources | `infembed`, `policy_emb` |
| Window width | 3; stride 1 |
| Reduction | UMAP 50D |
| Clustering | K-Means, $k = 10$ |
| Episodes | 100 |
| Rollout samples | 7,891 windows |
| Test method | chi-squared only |

Note: chi-squared may overreject when expected cell counts are small (see §3.5.1). These
results should be interpreted with that caveat; re-running with the exact permutation test
would give more reliable per-state p-values.

### 4.3 Mar27 Single-Run Experiment (Chi-Squared, K=20, EM Comparison)

InfEmbed clustering of 200 rollout episodes from a single policy (seed 0, `transport_mh_0_r512×512`),
K=20, 100D UMAP, w=5,s=2. Chi-squared test only, `exclude_terminals=False`. Baseline for the
RNN-EM representation learning experiment. Note: this is a single `test_markov_property` call,
not a pooled result — the 200 episodes come from one continuous eval run.

---

## 5. Results

### 5.1 Geometric Quality: Silhouette and Auto-K

On `transport_mh_jan28` with InfEmbed (K=20 baseline, 23 nodes / 140 edges), silhouette
sweep over $k \in [4, 15]$ selected $k^* = 4$ with $\bar{s} = 0.55$. The resulting graph
(7 nodes / 18 edges) has the canonical sequential structure of the task.

The same sweep on policy_emb/bottleneck features supports up to $k = 14$ well-separated
clusters ($\bar{s} \approx 0.47$). The policy bottleneck representation encodes richer
behavioral information — specific action-plan commitments at the final denoising step
distinguish finer-grained phases — so the data geometry supports higher K.

For InfEmbed, K=20 is therefore over-clustered: the geometry supports only 4 separable
modes. For policy_emb/bottleneck, K=10 is a conservative choice that leaves headroom.

### 5.2 Behavioral Coverage at Different (W, S)

Window/stride sweep on jan28 policy_emb (K=10, 100 episodes, ~52% failure rate):

| W | S | sw/frm% | distinct/ep | silhouette | MI(succ) | Assessment |
|---|---|---------|-------------|------------|----------|------------|
| 1 | 1 | 4.79 | 4.59 | 0.450 | 0.096 | High noise, single-frame windows |
| 3 | 1 | 3.28 | 3.71 | 0.425 | 0.133 | Moderate |
| **5** | **1** | 2.37 | 2.99 | 0.405 | **0.188** | Best MI — **recommended** |
| 5 | 5 | 5.47 | **5.30** | 0.490 | 0.137 | High silhouette is stride artifact |
| 10 | 2 | **2.07** | 2.94 | 0.423 | 0.150 | Lowest swap rate but phase-collapsed |
| 10 | 5 | 2.79 | 3.60 | **0.491** | 0.158 | Silhouette peak = stride artifact |

W=5,S=1 achieves the best MI (0.188 nats) and the second-lowest swap rate, at the cost of
moderate $\overline{D}$ (2.99 distinct clusters/episode). The large-window configurations
(W=10) achieve lower swap rates and higher silhouette but collapse distinct task phases,
destroying MI signal. This demonstrates that silhouette and swap rate alone are insufficient
metrics; MI with outcome captures the "useful" dimension.

### 5.3 Markov Violation Rates

#### 5.3.1 Mar27 Multi-Seed (Chi-Squared, Exact, Modal)

All tests run with `exclude_terminals=True`. The exact test uses $B = 5{,}000$ permutations.

| Seed | Method | Tested | Untestable | Violations | p-values of violations |
|------|--------|:------:|:----------:|:----------:|------------------------|
| 0 | chi2   | 7 | 13 | 3 | states 7, 17, 18 |
| 0 | exact  | 7 | 13 | **1** | state 18 (p≈0.02) |
| 0 | modal  | 7 | 13 | 0 | — |
| 1 | chi2   | 8 | 12 | 2 | states 1, 14 |
| 1 | exact  | 9 | 11 | **2** | states 1, 14 |
| 1 | modal  | 9 | 11 | 0 | — |
| 2 | chi2   | 7 | 13 | 1 | state 8 |
| 2 | exact  | 7 | 13 | **1** | state 5 |
| 2 | modal  | 7 | 13 | 0 | — |

**Interpretation.** Chi-squared finds 1–3 violations per seed; the exact test finds 1–2;
the modal test finds zero. The surviving exact-test violations have p-values of 0.01–0.04 —
significant at $\alpha = 0.05$ but not dramatically so. No state is flagged across all
three seeds, consistent with statistical noise from sparse data at K=20. The modal test's
clean result means the dominant transition from every testable state is the same regardless
of which state preceded it: the dominant graph structure is first-order Markovian.

The 55–65% untestable rate reflects the data regime: 100 episodes across 20 clusters yields
too few transitions per state to construct testable contingency tables for most states.

#### 5.3.2 Mar27 Full Dataset (Chi-Squared, K=20, N=200 Rollouts)

200 rollouts from a single policy (seed 0, `transport_mh_0_r512×512`). Chi-squared test only,
`exclude_terminals=False`.

| State | p-value | Notes |
|-------|---------|-------|
| 0  | 1.03e-22 | Extremely significant |
| 1  | 4.05e-05 | Significant |
| 3  | 7.82e-14 | |
| 4  | 8.49e-76 | Strongest violation |
| 8  | 1.33e-14 | |
| 9  | 5.74e-28 | |
| 10 | 6.02e-15 | |
| 11 | 3.99e-03 | Near threshold |
| 13 | 1.80e-04 | |
| 14 | 3.02e-03 | Near threshold |
| 16 | 5.42e-08 | |
| 17 | 2.80e-04 | |
| 19 | 2.27e-06 | |

Result: **13 violations / 16 testable states (81% violation rate)**. Untestable: 4/20.

This sharply contrasts with the 3-seed experiment (1–3 violations per seed with exact test),
which suggests that the 3-seed results at 100 rollouts understate true violation rates. With
more data and without the terminal-exclusion correction, many more states fail. The p-values
span a range of $10^{-76}$ to $10^{-2}$ — the strongest violations are not marginal.

**Scaling observation.** The chi-squared statistic grows as $O(n \cdot \phi)$ where $n$ is
the sample size and $\phi$ is the effect size (Cramér's V). Genuine Markov violations
(non-zero $\phi$) become more detectable — not less severe — as rollout count increases.
This means that the single-seed 100-rollout results systematically underestimate the
violation rate; the 200-rollout pooled result is more reliable.

#### 5.3.3 Jan28 (Chi-Squared, K=10, N=100 Rollouts)

| Representation | Tested | Untestable | Violations | Violation rate |
|---|:---:|:---:|:---:|:---:|
| InfEmbed | 8 | 2 | 7 | **88%** |
| policy_emb | 5 | 5 | 2 | **40%** |

InfEmbed at K=10 produces an 88% violation rate — nearly every testable state has
significant history-dependence. Policy bottleneck embeddings reduce this to 40%.

**Why policy_emb is more Markovian.** InfEmbed encodes "which training demonstrations
influence this prediction" — a gradient-based attribution signal that jitters across
consecutive windows as the policy's prediction changes. Consecutive windows within the same
behavioral phase still reflect different subsets of influential training examples, causing
the per-window embedding to vary non-smoothly. This variance in the conditioning signal
(cluster membership within a phase) introduces apparent history-dependence: the mix of
predecessors that lead to a cluster state will vary by where in the phase the agent is,
and this positional information propagates into the next-state distribution.

Policy bottleneck features encode "what the policy is planning to do" — a smooth,
behavior-aligned signal that changes primarily at genuine phase boundaries. Clusters defined
in this space correspond to genuine behavioral modes rather than attribution-space
neighborhoods, and the transition structure within each mode is more Markovian.

**Testable-state fraction.** InfEmbed at K=10, N=100: 8/10 testable (80%). Policy_emb at
K=10, N=100: only 5/10 testable (50%). Policy_emb clusters are larger and less visited in
diverse transition contexts — states that appear in long homogeneous runs have fewer distinct
predecessors and successors, reducing testability. This means the 40% violation rate is
computed over a smaller and potentially easier-to-pass subset of states; direct comparison
of rates across representations should account for this.

---

## 6. Discussion and Recommendations

### 6.1 Choosing the Test Method

| Context | Recommended method | Reason |
|---|---|---|
| N ≥ 300 episodes, K ≤ 10 | `method="chi2"` | Expected counts are large enough for reliable asymptotics |
| N < 200 episodes or K ≥ 15 | `method="exact"` | Asymptotic approximation fails for sparse tables |
| Quick sanity check | `method="modal"` | Tests the dominant transition only; robust to small N |
| Publication | `method="exact"` | Most reliable p-values; cite $B$ and the +1 correction |

The chi-squared test systematically over-rejects when expected cell counts are small —
that is, when $\sum_{i,j} E_{ij} < 5$ per table. With K=20 and N=100 episodes, roughly
half of all states fail this criterion entirely (they are marked untestable); the other
half often have sparse tables where the asymptotic approximation is marginal. The exact
permutation test is valid without a minimum expected-count requirement.

### 6.2 The Violation Rate Scales with Sample Size

The chi-squared statistic under a fixed non-null alternative grows approximately linearly
with $n$ (the total observation count in the table). A Markov violation with Cramér's V
effect size $V = \sqrt{\chi^2 / (n \cdot \min(r-1, c-1))} > 0$ will eventually be detected
with probability approaching 1 as more rollout data accumulates. This has an important
practical implication: **Markov violations are more detectable — not less severe — with
more data**. A result of "no violations at N=50 rollouts" does not mean the Markov property
holds; it may only mean the data is insufficient to detect the violation.

The apparent discrepancy between the 3-seed multi-seed result (1–2 exact-test violations per
seed) and the mar27 single-run result (81% violations at chi-squared) reflects three
simultaneous differences that cannot be separated from these data alone:

1. **Sample size.** 100 episodes per seed vs. 200 episodes — the larger dataset gives more
   observations per state, improving detection power for all non-zero effect sizes.
2. **Test method.** Exact permutation (B=5,000) vs. asymptotic chi-squared — the chi-squared
   test over-rejects when expected cell counts are small, potentially inflating the 81% figure.
3. **Terminal exclusion.** `exclude_terminals=True` in the 3-seed experiment vs. `False`
   in the single-run — including START as a valid predecessor introduces episode-position
   confounds that can inflate violation counts.

These three confounds act in opposite directions: the larger N and the inclusion of
terminals both push the violation count up in the single-run experiment; chi-squared's
over-rejection also inflates it. The exact-test, terminal-excluded, per-seed results are
therefore more conservative and probably understate the true violation rate; the single-run
chi-squared result likely overstates it. The true rate at N=200, method=exact,
exclude_terminals=True is unknown from current data.

### 6.3 Choosing K and the Feature Space

Based on the collected evidence:

1. **K=10 on policy_emb (bottleneck_plan_t0 or t5) features is the recommended default**
   for graph construction. In the E1 VLM cluster coherence experiment (K=10, `transport_mh_seed0_r512`,
   global-episode-disjoint evaluation), `policy_emb/plan@t0` achieves 63.3% clean accuracy
   (F1 sweep) vs. 53.3% for InfEmbed at the same K. In the larger head-to-head experiment
   (F16, n_query=5, n_reps=5), `bottleneck_plan_t5` achieves 54.0% clean vs. 37.8% for
   InfEmbed 100D — both differences are consistent. Combined with a 40% Markov violation
   rate (vs 88% for InfEmbed), policy_emb is the preferred representation for graph
   construction.

2. **Auto-K on InfEmbed** is useful as a graph simplification tool, selecting K=4 on
   transport_mh — a coarse but readable graph. This is appropriate when the goal is a
   high-level behavioral summary rather than fine-grained state analysis.

3. **Avoid K=20 on InfEmbed** unless N ≥ 300 rollouts. The 81% violation rate and the
   geometric auto-K selecting K=4 both indicate the feature space cannot support 20
   separable behavioral modes; the excess K splits geometrically adjacent embeddings that
   belong to the same behavioral phase, producing a graph that is both unreadable and
   non-Markovian.

### 6.4 Coverage and Power

**To achieve 80%+ testable-state fraction at K=10**, approximately 100 rollouts is
sufficient for InfEmbed (8/10 testable), but policy_emb requires more (5/10 at N=100).
Policy_emb clusters tend to have longer runs and fewer distinct transitions per episode
because the embedding is smoother — this reduces the diversity of predecessors/successors
per cluster state and reduces testability. To reach 80% testable states with policy_emb at
K=10, N ≈ 200–300 rollouts is recommended.

**To detect effect sizes of $V \approx 0.2$** (weak to moderate Markov violation, using Cramér's V) with
80% power using the chi-squared test on a $3 \times 3$ contingency table: $n \approx 60$
observations per state are required. Assuming roughly $N_\text{rollouts} / K$ observations
per state in the collapsed sequences, this translates to approximately 600 rollouts at K=10
to have adequate power for all states. With N=100, the test has limited power for small
effect sizes, and the "modal" test — which requires only qualitative dominance — is the
more useful low-N diagnostic.

---

## 7. Implementation Reference

### API

```python
from policy_doctor.behaviors.behavior_graph import (
    test_markov_property,
    test_markov_property_pooled,
    markov_test_result_to_jsonable,
)

# Single dataset
result = test_markov_property(
    cluster_labels,          # ndarray[int], shape (N,)
    metadata,                # List[Dict], length N
    level="rollout",         # "rollout" or "demo"
    significance_level=0.05,
    exclude_terminals=True,  # recommended: removes episode-boundary artifacts
    method="exact",          # "chi2", "exact", or "modal"
    n_permutations=10000,    # B; ignored for method="chi2"
    random_state=42,
)

# Pooled across multiple eval sets (same clustering)
result = test_markov_property_pooled(
    datasets=[(labels_1, meta_1), (labels_2, meta_2)],
    level="rollout",
    method="exact",
    n_permutations=10000,
    random_state=42,
)

# JSON serialisation
json_safe = markov_test_result_to_jsonable(result)
```

### Return Value

```python
{
    "markov_holds": bool | None,         # True iff all testable states pass
    "significance_level": float,
    "num_states_tested": int,
    "num_states_untestable": int,
    "per_state": {
        state_id: MarkovTestResult(
            state=int,
            testable=bool,
            chi2=float | None,           # observed statistic
            p_value=float | None,
            dof=int | None,              # degrees of freedom (chi2 only)
            markov_holds=bool | None,    # p_value >= significance_level
            contingency_table=ndarray,   # shape (r, c)
            previous_states=List[int],   # row labels
            next_states=List[int],       # column labels
            reason=str | None,           # why not testable, if untestable
        )
    }
}
```

### Automatic K Selection

```python
from policy_doctor.behaviors.graph_simplification import auto_k_kmeans

labels, best_k, scores_by_k = auto_k_kmeans(
    embeddings,          # ndarray, shape (N, D) — post-UMAP coordinates
    k_range=(4, 15),     # search range
    random_state=42,
)
# scores_by_k: Dict[int, float] — silhouette score per k
```

### Pooled Test (when to use)

Use `test_markov_property_pooled` when:
- The same clustering (same K, same cluster IDs) is applied to multiple independent eval
  rollout batches (e.g., three 100-episode eval runs under one training checkpoint).
- You want to increase N without re-running the clustering.

Do not use it to pool across different clusterings (different K, different seeds, or
different feature spaces). Cluster IDs are assigned by K-Means and are not semantically
comparable across fits.

---

## 8. Summary of Recommendations

| Decision | Recommendation | Evidence |
|---|---|---|
| Feature space for graph | `policy_emb/bottleneck_plan_t0` or `t5` | 40% violation rate vs 88% for InfEmbed; E1: 63% vs 53% (F1 sweep), 54% vs 38% (F16 head-to-head) |
| K value for policy_emb | 10–14 (auto-K ceiling is 14) | Auto-K analysis; E1 sweet spot at K=10 |
| K value for InfEmbed | 4–8 (auto-K selects 4) | Over-split above 8; 81% violation rate at K=20 |
| Window / stride | W=5, S=1 | Highest MI(succ)=0.188; second-lowest swap rate |
| Statistical test | `method="exact"`, B=10,000 | Chi-squared overrejects for sparse tables |
| Sample size for reliable testing | ≥200 rollouts at K=10 | 50% testable rate at N=100 for policy_emb |
| Multiple testing | Report raw p-values; flag states with p < 0.001 | No correction currently applied; FDR-BH recommended for K≥20 |
| exclude_terminals | True | Prevents episode-boundary confound from inflating violation count |
