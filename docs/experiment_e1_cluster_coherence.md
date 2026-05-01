# Experiment E1: Validating Behavioral Coherence of Influence-Based Clusters via VLM Classification

## Purpose

The behavior graph methodology rests on the claim (Section 3.2) that influence embeddings, when clustered, recover behaviorally meaningful structure — slices in the same cluster correspond to the same behavioral mode; slices in different clusters correspond to distinguishable modes.

Experiment E1 tests this claim directly as a classification task: given a set of example slices from each cluster, can a vision-language model (VLM) correctly assign held-out slices to their cluster of origin? If clusters are behaviorally coherent and distinct, accuracy substantially exceeds chance (1/K). If not, accuracy is near chance.

## Hypotheses

- **H1 (Primary)**: VLM top-1 accuracy substantially and significantly exceeds chance (1/K).
- **H2 (Per-cluster)**: Accuracy varies across clusters, revealing which clusters are well-formed vs. over-merged/over-split.
- **H3 (Comparative, optional)**: Influence-based clustering produces more behaviorally coherent clusters than baseline methods (raw observations, action sequences, pretrained vision encoder features).

## Design Decisions

### Storyboard composites (token budget control)

Each slice is rendered as a single **storyboard composite image** (2×2 grid of up to 4 frames sampled uniformly from the window).  This keeps the full prompt for one query to K·n_example + 1 images — around 75 images for K=15, n_example=5 — rather than K·n_example·n_frames ≈ 300+ images.

### Pre-committed sample plan

`sample_plan.json` is written to disk **before any VLM call**.  It contains:
- `example_indices[cluster_id]` — n_example prototype slices per cluster
- `query_indices[cluster_id]` — n_query held-out slices per cluster
- `label_maps[cluster_id][query_idx][rep]` — per-rep opaque label permutation
- `random_seed` used

This makes sampling auditable and prevents cherry-picking after seeing results.

### Example slice selection (centroid proximity)

When `embeddings_reduced.npy` is present in the clustering result directory (saved by `run_clustering` since this experiment was introduced), the n_example prototype slices are those closest to the empirical cluster centroid in UMAP-reduced embedding space.

When absent (older clustering runs), the selection falls back to random sampling.  Run the `run_clustering` step again to generate `embeddings_reduced.npy`.

### Episode disjointness

Example and query slices for a given cluster are drawn from different rollout episodes wherever possible, preventing the VLM from classifying on episode-level visual cues (initial object placement, lighting) rather than behavioral content.  When disjointness is impossible (cluster spans only one episode), this is logged in `sample_plan.json` under `disjointness_status`.

### Opaque labels + randomisation

Cluster IDs are mapped to opaque labels ("Group A", "Group B", …) to prevent the VLM from inferring meaning from numerical ordering.  The mapping is re-shuffled independently for every (query, repetition) pair.

### Repetitions and majority vote

Each query is classified `n_repetitions` times (default 3) at temperature > 0.  The majority prediction is the reported result.  Per-query agreement rate (fraction of reps with the same prediction) is also recorded.

## Running the Experiment

### Prerequisites

1. A completed `run_clustering` step (produces `run_clustering/result.json` and clustering directories).
2. Eval episode pickles with `img` column (produced by `eval_save_episodes.py`).
3. A VLM backend configured (Gemini, Claude, or open-weights).

### Pipeline command

```bash
# Run just the E1 step on an existing run_dir
python -m policy_doctor.scripts.run_pipeline \
  data_source=mimicgen_square \
  experiment=mimicgen_square_pipeline_apr23 \
  steps=[validate_cluster_coherence_vlm] \
  vlm_cluster_classification.backend=gemini \
  "vlm_cluster_classification.backend_params={model_name: gemini-2.0-flash}"
```

Or using the Claude backend:

```bash
python -m policy_doctor.scripts.run_pipeline \
  steps=[validate_cluster_coherence_vlm] \
  vlm_cluster_classification.backend=claude \
  "vlm_cluster_classification.backend_params={model_name: claude-sonnet-4-6}"
```

### Experiment config YAML (recommended)

Add to your experiment YAML:

```yaml
vlm_cluster_classification:
  backend: gemini
  backend_params:
    model_name: gemini-2.0-flash
    api_key: ${oc.env:GOOGLE_API_KEY}
  n_example: 5          # prototype slices per cluster
  n_query: 5            # held-out slices per cluster
  n_repetitions: 3      # VLM calls per query (majority vote)
  max_frames_per_storyboard: 4   # frames in each composite storyboard image
  random_seed: 42       # fixed before any VLM call
  max_clusters: null    # null = all clusters; set to integer to cap
```

### API key setup

```bash
export GOOGLE_API_KEY=<your-key>    # for Gemini
export ANTHROPIC_API_KEY=<your-key> # for Claude
```

### Dry run

```bash
python -m policy_doctor.scripts.run_pipeline \
  steps=[validate_cluster_coherence_vlm] \
  vlm_cluster_classification.backend=mock \
  dry_run=true
```

This writes `sample_plan.json` and prints the call summary without making any VLM API calls.

## Output Files

Under `<run_dir>/validate_cluster_coherence_vlm/seed_<N>/`:

| File | Contents |
|------|----------|
| `sample_plan.json` | Pre-committed sampling plan (see above) |
| `predictions.jsonl` | One JSON record per query containing per-rep predictions, majority vote, `is_correct`, `agreement_rate` |
| `metrics.json` | Aggregate metrics (see below) |

### Metrics (`metrics.json`)

| Key | Description |
|-----|-------------|
| `top1_accuracy` | Fraction of valid (non-unclear) queries correctly classified |
| `top1_accuracy_ci_95` | 95% Wilson score confidence interval `[lo, hi]` |
| `chance_level` | `1/K` baseline |
| `binomial_test_pvalue` | One-sided binomial test p-value (H1: accuracy > chance) |
| `per_cluster_accuracy` | Per-cluster top-1 accuracy dict |
| `confusion_matrix` | K×K confusion counts (rows=true, cols=predicted) |
| `confusion_matrix_cluster_ids` | Sorted cluster IDs mapping rows/cols |
| `agreement_rate_mean` | Mean per-query agreement rate across repetitions |
| `unclear_rate` | Fraction of queries the VLM answered "unclear" |
| `n_total` | Total query count |
| `n_valid` | Queries with a definite prediction |
| `n_unclear` | Unclear responses (excluded from accuracy) |

## Statistical Analysis

### Power

With K=15 clusters and N_query=75 (5 per cluster), the experiment has high power to detect accuracy ≥ 0.3 (versus chance 1/15 ≈ 0.067).  Even N_query=30 is sufficient for a useful point estimate.

### Per-cluster corrections

Apply Bonferroni or Benjamini-Hochberg correction across K per-cluster binomial tests when reporting H2.

## Failure Modes

| Symptom | Likely cause | Diagnostic |
|---------|--------------|------------|
| Overall accuracy near chance | Clusters not behaviorally distinct, OR slice window too short, OR poor rendering | Inspect confused cluster pairs in confusion matrix |
| High accuracy on some clusters, near-chance on others | Over-merging in poor clusters | Per-cluster accuracy bar chart |
| Systematic confusion between specific pairs | Over-clustering (K too high) | Reduce K and re-cluster |
| High "unclear" rate | Storyboards ambiguous; short slices | Increase `max_frames_per_storyboard`; inspect "unclear" examples |
| Low run-to-run agreement | VLM uncertainty on boundary cases | Increase `n_example`; boundary slices near cluster edge |

## Scope and Limitations

- VLM classification is a **proxy** for human judgment.  A small follow-on human study is recommended.
- Results are task-specific.  Generalisation claims require running on additional tasks.
- The result depends on the VLM.  Running with at least two models (one API-based, one open-weights) guards against model artifacts.  Use `backend=gemini` and `backend=claude` for two runs; `backend=qwen2_vl` or `backend=molmo2` for an open-weights run.
- Behavioral coherence is not the same as behavioral correctness at the right granularity.  This experiment does not speak to whether the cluster count K is appropriate for downstream tasks.

## Two-VLM Protocol

The experiment spec recommends at least two VLMs.  Suggested runs:

```bash
# Run 1: Gemini Flash
vlm_cluster_classification.backend=gemini
vlm_cluster_classification.backend_params.model_name=gemini-2.0-flash

# Run 2: Claude Sonnet (Anthropic)
vlm_cluster_classification.backend=claude
vlm_cluster_classification.backend_params.model_name=claude-sonnet-4-6

# Run 3 (optional): open-weights
vlm_cluster_classification.backend=qwen2_vl
```

Use different `run_dir`s or different `seed_dir` suffixes so results don't overwrite each other.  Then compare `metrics.json` across runs — consistent accuracy across VLMs strengthens the claim.
