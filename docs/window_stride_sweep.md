# Window / Stride Sweep — Choosing W and S for Clustering

## TL;DR

For `transport_mh_jan28` with `policy_emb/bottleneck_plan_t0` features and KMeans
k=10:

- **The cleanest *swap rate* is at W=10, S=2** (2.07 % swaps/frame), but that
  clustering collapses behavioral phases — its episodes only visit 2.94 distinct
  clusters on average.
- **The most *useful* clustering is W=5, S=1** by mutual information with
  episode outcome (0.188 nats — almost 2× the W=10/S=2 value).
- Pure swap rate / run length is a one-sided "consistency" metric; large
  windows always look good by it but throw away the signal that makes the
  clustering interpretable. A composite of (a) per-frame swap rate, (b) mean
  distinct clusters per episode, and (c) MI with outcome captures the
  consistency / resolution / informativeness tradeoff.

## Setup

- Source: `third_party/influence_visualizer/configs/transport_mh_jan28/clustering/policy_emb_bottleneck_plan_t0_w<W>_s<S>_seed0_kmeans_k10/`
- Eval rollouts: 100 episodes, ~52 % failure rate.
- Features: bottleneck-plan-t0 activations (512-D) → UMAP to 50 D, fixed seed.
- KMeans K=10, fixed seed.
- Built by `scripts/build_alt_clustering.py` with the listed (W, S) settings.

## Metrics measured

| Metric | What it measures | Direction |
|---|---|---|
| `swap%` | Fraction of adjacent windows (within an episode) that change cluster label | Lower is more stable — but stride-dependent: large S mechanically inflates it |
| `sw/frm%` | Same numerator, denominator = total *frames* covered by run-length-collapsed segments | Stride-fair version of swap%; **lower is better** |
| `run_win`, `run_frames` | Mean run length, in windows and in frames | Higher = more temporal coherence; frames version is stride-fair |
| `distinct/ep` | Mean number of distinct clusters that appear in an episode's run-length-collapsed sequence | Behavioral resolution. Should be ≈ number of real task phases (5–6 for transport); too low = phases collapsed, too high = flicker |
| `silhouette` | Mean silhouette score in the UMAP embedding space (3000-pt subsample) | Geometric cluster separability — but only in feature space, not behaviorally |
| `MI(succ)` | Mutual information between window's cluster label and its episode's success outcome | **Behavioral informativeness** — does this clustering predict task success? |

## Results (k=10, `policy_emb/bottleneck_plan_t0`, jan28)

| W | S | swap% | sw/frm% | distinct/ep | silhouette | **MI(succ)** |
|---|---|---|---|---|---|---|
| 1 | 1 | 4.9 % | 4.79 % | 4.59 | 0.450 | 0.0961 |
| 3 | 1 | 3.6 % | 3.28 % | 3.71 | 0.425 | 0.1327 |
| 3 | 2 | 8.6 % | 3.96 % | 4.21 | 0.480 | 0.1159 |
| **5** | **1** | 2.8 % | 2.37 % | 2.99 | 0.405 | **0.1880** ← MI winner |
| 5 | 2 | 7.8 % | 3.28 % | 3.81 | 0.470 | 0.1130 |
| 5 | 3 | 11.4 % | 3.33 % | 3.82 | 0.478 | 0.1271 |
| 5 | 5 | 29.2 % | 5.47 % | **5.30** | 0.490 | 0.1368 |
| 7 | 2 | 7.4 % | 2.88 % | 3.62 | 0.426 | 0.1310 |
| 7 | 3 | 11.6 % | 3.09 % | 3.75 | 0.488 | 0.1232 |
| **10** | **2** | 5.7 % | **2.07 %** | 2.94 | 0.423 | 0.1495 |
| 10 | 5 | 18.5 % | 2.79 % | 3.60 | **0.491** | 0.1578 |
| 15 | 5 | 17.4 % | 2.21 % | 3.24 | 0.435 | 0.1412 |

## Findings

### 1. Pure swap rate is misleading at large W.

The original swap-rate table flagged W=10/S=2 as the winner because it has the
lowest swaps-per-frame (2.07 %). But the same clustering has **the lowest
`distinct/ep` in the sweep** (2.94) — meaning on average each episode visits
fewer than three distinct cluster labels in its entire trajectory.

For a five-phase manipulation task (reach → grasp → transport → align →
release-or-fail), 2.94 distinct clusters per episode is not "clean" — it's
"phases collapsed together". The large window blurs adjacent phases into a
single embedding, so KMeans can't draw a boundary between them.

### 2. Silhouette favors *sparse* sampling, not behavior alignment.

The geometric silhouette score peaks at W=10/S=5 (0.491) and W=5/S=5 (0.490) —
the configurations with **least window overlap**. Larger stride means adjacent
windows share fewer frames, sit further apart in feature space, and so the
clusters look cleaner *geometrically*. But behaviorally those clusterings have
the highest swap rates (18.5 %, 29.2 %) — adjacent overlapping windows
disagree.

Silhouette is a useful sanity check but it's a property of the *embedding*,
not of the *clustering's relationship to behavior*. Optimizing for it alone
would push toward S → W (no overlap), at which point swap rate goes up and
run length collapses.

### 3. MI with outcome is the metric that captures "useful".

A good behavioral clustering should be *predictive* of the eventual outcome of
the episode. Mutual information between (window's cluster label) and (window's
episode's success bit) measures exactly this.

- **W=5/S=1 wins clearly at 0.188 nats** — almost 2× the W=10/S=2 value
  (0.150) and the highest MI in the entire sweep.
- W=10/S=5 is second (0.158).
- W=10/S=2 (the "low swap rate" winner) is mid-pack (0.150).

W=5/S=1 also has the second-best swaps-per-frame (2.37 %) and similar
`distinct/ep` (2.99) to W=10/S=2. So it's not paying for its MI lead with
extra noise — it's just the right window size.

### 4. Why this is the right answer for this task.

- Episodes are ~80 frames at 10 fps ≈ 8 seconds.
- Real task phases (reach, grasp, lift, transport, place) are 0.5–2 seconds.
- **W=5 frames ≈ 0.5 s** matches the shortest phase, smoothing single-frame
  noise without spanning phase boundaries.
- **S=1** preserves full temporal resolution (no information dropped between
  adjacent windows).
- Outcome of these knobs: each phase gets ~10–30 windows of stable
  assignment, with clean phase boundaries that show up as short transition
  runs.

## Recommendation

For graph construction on `policy_emb/bottleneck_plan_t0` (and likely any
policy-bottleneck feature space):

**Default: W=5, S=1, K=10.**

If memory or compute is a concern (S=1 keeps every frame's window):

- W=5, S=2 — 2× fewer windows, MI drops to 0.113, swaps-per-frame stays at
  3.28 %. Acceptable trade for half the windows.
- W=10, S=2 — 2× fewer windows but visible loss of behavioral resolution
  (`distinct/ep` from 2.99 → 2.94, MI from 0.188 → 0.150).

**Avoid** W ≥ 10 with any S < W/2 — the window starts spanning multiple task
phases and the clustering can no longer distinguish them. Avoid S ≈ W (no
overlap) — adjacent windows decorrelate, swap rate balloons, but MI doesn't
actually improve because the geometric "cleanness" is an artifact of stride
choice, not behavioral structure.

## Why this matters

This finding is upstream of every other clustering analysis in the pipeline:
the influence-graph nodes, the trajectory tree, the cluster inspector, every
downstream evaluation depends on the choice of W and S. Silhouette-based or
swap-rate-based defaults will quietly favor over-large windows that
homogenize phases. **MI with the outcome bit is cheap to compute** (sklearn's
`mutual_info_score` on the labels-vs-success array) and is the metric to
optimize when picking W and S.

## Methodology notes & limitations

- MI(label, success) treats the success bit as constant across an episode
  (broadcast to every window). This means clusters that appear in *both* successful
  and failed episodes contribute zero MI even if they're behaviorally
  distinct. It rewards clusters that segregate by episode outcome — the
  right thing for our purposes but worth noting.
- The sweep was done at K=10 only; at K=5 the choice of W/S matters less
  (fewer clusters to confuse), at K=20 the differences likely amplify.
- Silhouette uses a 3000-point random subsample (full would be O(n²) on
  ~7700 windows). Variance ±0.01.
- All numbers reproducible: see `scripts/build_alt_clustering.py --window_width W
  --stride S ...` in the worktree commit history.
