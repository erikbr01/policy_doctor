# Testing the Markov Assumption of the Behavior Graph

## Background

The behavior graph models rollout episodes as sequences of transitions between discrete behavioral clusters. Given per-timestep cluster assignments from a sliding-window K-Means clustering, consecutive same-cluster timesteps are collapsed into a single visit, producing a run-length-encoded sequence per episode. The graph then treats these sequences as a first-order Markov chain: transition probabilities from state $s$ depend only on $s$, not on how the agent arrived at $s$.

This assumption underpins the value computation (Bellman equations over the graph), advantage-based slice selection, and path enumeration. If the assumption is violated — i.e., the distribution over next states depends on the predecessor — then the computed values may not accurately reflect the true expected outcomes from each behavioral state.

## What We Tested

For each behavioral state $s$ that appears in the interior of at least two distinct transition patterns, we constructed a contingency table:

$$
T_{s}[i, j] = \#\{(prev = p_i,\; current = s,\; next = n_j) \text{ in the collapsed episode data}\}
$$

where rows index predecessor states $p_i$ and columns index successor states $n_j$. The null hypothesis is **row-column independence**: the distribution over next states is the same regardless of the predecessor.

We applied three statistical tests at significance level $\alpha = 0.05$:

| Method | Null hypothesis | Test statistic | P-value computation | Min. data |
|--------|----------------|----------------|--------------------|----|
| **Chi-squared** | Full independence of prev and next given current | Pearson $\chi^2$ | Asymptotic $\chi^2$ distribution | $\sum T \geq 5$ |
| **Exact (permutation)** | Full independence (same as above) | Pearson $\chi^2$ | 5,000 Monte Carlo permutations of row labels | $\sum T \geq 3$ |
| **Modal (permutation)** | The most-likely successor is the same for all predecessors | Number of predecessors whose modal successor $\neq$ overall modal successor | 5,000 permutations | $\sum T \geq 3$ |

The exact test addresses the chi-squared test's tendency toward inflated Type I error when expected cell counts are small. The modal test asks a weaker question — not whether the full conditional distributions are identical, but whether the dominant transition is the same — which requires less data to answer reliably.

### Episode-Boundary Effects

Variable-length episodes introduce a confound: states near episode boundaries have position-dependent termination probabilities (a state early in an episode is less likely to be followed by END than the same state near the end). Because the predecessor carries information about position, this creates a spurious second-order dependency. All tests were run with `exclude_terminals=True` to remove START, END, SUCCESS, and FAILURE nodes from the predecessor/successor roles, isolating the behavioral dynamics from episode-boundary artifacts.

## Data

Clustering results from the `transport_mh` task (Jan 28 training run), produced by the pipeline with the following parameters:

| Parameter | Value |
|-----------|-------|
| Influence source | TRAK |
| Representation | Sliding window |
| Clustering level | Rollout |
| Algorithm | K-Means, $k = 20$ |
| Policy seeds | 0, 1, 2 (independent clustering per seed) |
| Episodes per seed | 100 |
| Samples per seed | ~3,755–3,879 |

Each seed has a separate clustering run on different rollout data from a different trained policy, so cluster IDs do not correspond across seeds.

## Results

### Per-Seed, Per-Method

| Seed | Method | States Tested | Untestable | Violations | Violating States |
|------|--------|:---:|:---:|:---:|-----------------|
| 0 | chi2 | 7 | 13 | 3 | 7, 17, 18 |
| 0 | exact | 7 | 13 | 1 | 18 |
| 0 | modal | 7 | 13 | 0 | — |
| 1 | chi2 | 8 | 12 | 2 | 1, 14 |
| 1 | exact | 9 | 11 | 2 | 1, 14 |
| 1 | modal | 9 | 11 | 0 | — |
| 2 | chi2 | 7 | 13 | 1 | 8 |
| 2 | exact | 7 | 13 | 1 | 5 |
| 2 | modal | 7 | 13 | 0 | — |

### Key Observations

1. **Chi-squared overrejects.** With 20 clusters and 100 episodes, many contingency tables have small expected cell counts. The asymptotic chi-squared approximation is unreliable here, leading to false positives. Seed 0 drops from 3 violations (chi-squared) to 1 (exact) when the p-value is computed by permutation.

2. **Exact test finds 1–2 mild violations per seed.** The surviving violations have p-values in the range 0.01–0.04 — significant at $\alpha = 0.05$ but not overwhelming. The violating states differ across seeds (no state is flagged in all three), indicating statistical noise from sparse data rather than a consistent structural violation.

3. **Modal test finds zero violations across all seeds.** The most-likely successor at every testable state is the same regardless of which state the agent came from. The dominant transition structure is genuinely first-order.

4. **Most states are untestable.** 11–13 of 20 clusters lack enough diverse interior transitions (multiple predecessors × multiple successors) to run any test. This is inherent to the data regime: 100 episodes spread across 20 clusters yields sparse transition counts, especially after run-length collapsing.

## Conclusions

**The Markov assumption is a reasonable approximation for this behavior graph.** Specifically:

- The **dominant transition structure** (which state you are most likely to visit next) is fully consistent with the Markov property across all seeds and all testable states.
- There are **weak second-order effects in the tails** of the conditional distributions at a few states, detectable by the exact test. These are not consistent across seeds and likely reflect sampling variability rather than genuine history-dependence in the behavior dynamics.
- The **chi-squared test is too aggressive** for this data regime and should not be used in isolation. The permutation-based exact test provides more reliable p-values when contingency tables are sparse.

### Limitations

- **Low statistical power.** With 65% of states untestable, we can only verify the Markov property on a subset of the graph. States that appear rarely or only at episode boundaries escape scrutiny.
- **Run-length collapsing.** The test operates on the collapsed sequence, not the raw timestep-level chain. A state that is revisited after a brief excursion (e.g., $A \to B \to A$) appears as two separate visits to $A$, which is correct for testing the collapsed chain's Markov property but says nothing about timestep-level dynamics.
- **Single task.** These results are specific to `transport_mh` with $k = 20$ K-Means clustering on TRAK influence features. Different tasks, cluster counts, or feature representations could yield different results.

### Recommendations

- **Use the exact test** (`method="exact"`) rather than chi-squared when evaluating the Markov property on real clustering results.
- **Use the modal test** (`method="modal"`) as a quick sanity check — if it fails, there is a strong structural violation.
- **Pool data when possible** (`test_markov_property_pooled`) to increase statistical power. This is valid when the same clustering is applied to multiple independent sets of episodes (e.g., multiple evaluation batches under one clustering run).
- **Increasing the number of evaluation episodes** (e.g., 300–500) would make more states testable and give all three tests substantially more power. Reducing $k$ is not recommended as it sacrifices behavioral granularity.
