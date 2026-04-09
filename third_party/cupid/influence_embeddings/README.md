# Influence Embeddings Pipeline

Implementation of robot failure mode discovery via influence embeddings, as specified in `INFLUENCE_EMBEDDINGS_PLAN.md`.

## Overview

This pipeline automatically groups robotic failures into interpretable clusters (e.g., "Missed Grasp," "Wall Collision") and identifies which training demonstrations are responsible for those failures.

**Core Idea**: Combine CUPID (robotics influence functions) with InfEmbed (gradient-based clustering) using the TRAK random projection method for computational efficiency.

## Installation

The pipeline requires `scikit-learn` in addition to the existing CUPID dependencies:

```bash
pip install scikit-learn
```

## Quick Start

### Simple Example (Synthetic Data)

Validate the pipeline with synthetic data:

```bash
python -m influence_embeddings.example_simple
```

This creates a simple MLP, generates synthetic training/evaluation data with two failure modes, and runs the full pipeline.

### Full Pipeline (Real Data)

```bash
python -m influence_embeddings.pipeline \
    --train_dir /path/to/training_run \
    --eval_dir /path/to/eval_rollouts \
    --output_dir /path/to/output \
    --n_clusters 5 \
    --projection_dim 512
```

**Key Arguments:**
- `--train_dir`: Directory containing trained policy checkpoint
- `--eval_dir`: Directory with evaluation rollouts (must have `episodes/` subdirectory)
- `--train_ckpt`: Checkpoint to use (`best`, `latest`, or index)
- `--projection_dim`: Dimension of influence embeddings (default: 512)
- `--n_clusters`: Number of failure mode clusters (default: 5)
- `--top_k`: Number of training demos to attribute per cluster (default: 10)
- `--auto_k`: Automatically find optimal k using silhouette scores
- `--param_filter`: Comma-separated parameter filters (e.g., `model.up,model.final`)

## Architecture

```
influence_embeddings/
├── __init__.py              # Package exports
├── gradient_projector.py    # Phase 1: Gradient computation & projection
├── trajectory_aggregator.py # Phase 2: Temporal aggregation
├── embedding_computer.py    # Phase 3: Dataset processing
├── failure_clusterer.py     # Phase 4: K-Means clustering
├── training_attributor.py   # Phase 5: Training data attribution
├── pipeline.py              # Main orchestration script
└── example_simple.py        # Standalone test with synthetic data
```

### Component Details

#### 1. GradientProjector (`gradient_projector.py`)

Computes influence embeddings using random projections:

```
φ(z) = ∇L(z) @ P
```

Where `P` is a random Gaussian matrix (Johnson-Lindenstrauss lemma).

**Key Features:**
- Automatic parameter selection (defaults to last 2 layers)
- Configurable projection dimension
- Reproducible via seed parameter

#### 2. TrajectoryAggregator (`trajectory_aggregator.py`)

Aggregates per-timestep embeddings into trajectory-level embeddings:

```
μ_traj = Σ_t φ(s_t, a_t)
```

**Key Features:**
- Handles padded sequences via masking
- L2 normalization for clustering
- Reward weighting for test rollouts

#### 3. EmbeddingComputer (`embedding_computer.py`)

Processes full datasets to generate embeddings:

- **Training embeddings**: `φ_train` for all expert demonstrations
- **Evaluation embeddings**: `φ_test * reward` (flips direction for failures)

#### 4. FailureClusterer (`failure_clusterer.py`)

K-Means clustering of failure embeddings:

**Key Features:**
- Automatic filtering to failure-only embeddings
- Silhouette score computation
- Optional auto-k selection via `find_optimal_k()`

#### 5. TrainingAttributor (`training_attributor.py`)

Links failure clusters to training demonstrations via cosine similarity:

```
similarity(cluster_center, train_embedding) = cos(c_k, φ_train)
```

**Output:** Top-k training demos responsible for each failure mode.

## Mathematical Background

### Influence Approximation

The influence of training point on test point:

```
I(z_train, z_test) ≈ ∇L(z_test)ᵀ H⁻¹ ∇L(z_train)
```

### Random Projection Trick

Instead of computing H⁻¹, we use a random projection matrix P:

```
φ(z) = ∇L(z) @ P
I(z_train, z_test) ≈ φ(z_test) · φ(z_train)
```

### Reward Weighting (Critical)

For test rollouts, multiply embedding by reward:

```
φ_test_final = μ_traj * reward
```

- Success (reward=+1): Points toward helpful training data
- Failure (reward=-1): Flips direction to point toward harmful training data

## Output Files

The pipeline generates:

```
output_dir/
├── train_embeddings.pt      # Training embeddings [num_train, proj_dim]
├── eval_embeddings.pt       # Eval embeddings with rewards
├── gradient_projector.pt    # Saved projector for reproducibility
├── failure_clusterer.pt     # Fitted K-Means clusterer
├── attributions.pt          # Cluster-to-training attributions
├── attribution_report.txt   # Human-readable report
└── results.pt               # Complete results dictionary
```

## Programmatic Usage

```python
from influence_embeddings import (
    GradientProjector,
    TrajectoryAggregator,
    EmbeddingComputer,
    FailureClusterer,
    TrainingAttributor,
)

# Or run the full pipeline programmatically
from influence_embeddings.pipeline import run_pipeline

results = run_pipeline(
    train_dir="/path/to/train",
    eval_dir="/path/to/eval",
    output_dir="/path/to/output",
    n_clusters=5,
    projection_dim=512,
)
```

## Integration with Existing CUPID Infrastructure

This pipeline builds on the existing TRAK infrastructure in `diffusion_policy/data_attribution/`:

- `gradient_computers.py`: Per-sample gradient computation via `torch.func`
- `modelout_functions.py`: Loss function definitions for diffusion policies
- `trak_util.py`: Checkpoint loading and parameter filtering utilities

The new pipeline can work standalone or integrate with `train_trak_diffusion.py` for more advanced use cases.

---

## Implementation Notes

### Design Decisions Made

1. **Random Projections (TRAK method)**: Chosen over Arnoldi iteration for computational efficiency and memory friendliness. Mathematically proven to preserve distances.

2. **Parameter Filtering**: Defaults to last 2 layers (`model.up`, `model.final`) as these capture most semantic information while reducing computation by ~10x.

3. **L2 Normalization**: All embeddings are unit-normalized before clustering. This is crucial for K-Means to work well with cosine similarity semantics.

4. **Reward Weighting**: Applied after trajectory aggregation but before final normalization. This ensures failure embeddings point in the opposite direction.

5. **Diffusion Timestep Sampling**: Samples multiple diffusion timesteps per trajectory (default: 10) and averages gradients, following the existing TRAK infrastructure pattern.

6. **Modular Architecture**: Each component is independent and can be used/tested separately. This allows easy experimentation with different clustering methods, projection dimensions, etc.

### Insights from Implementation

1. **Existing Infrastructure**: The CUPID codebase already has excellent infrastructure for per-sample gradient computation via `torch.func.vmap` and `torch.func.grad`. The new pipeline leverages this.

2. **Memory Efficiency**: By projecting gradients immediately (before storage), we avoid memory issues with large models. A 100M parameter model's gradient (~400MB) becomes a 512-dim vector (~2KB).

3. **Numerical Stability**: Using float32 for clustering and proper normalization prevents numerical issues in K-Means.

4. **Episode vs Sample Granularity**: The current implementation works at the sample level (individual state-action pairs). For true trajectory-level analysis, the `TrajectoryAggregator` can aggregate across time.

### Open Questions for Future Work

1. **Optimal Number of Clusters (k)**
   - Currently defaults to 5 with optional auto-selection via silhouette scores
   - Should we use domain knowledge (e.g., known failure types) to set k?
   - Alternative: hierarchical clustering for discovering natural groupings

2. **Projection Dimension Trade-off**
   - Default 512 balances accuracy vs. computation
   - Johnson-Lindenstrauss suggests `d = O(log(n)/ε²)` for ε-approximate distances
   - Worth experimenting with 256, 1024, 2048

3. **Layer Selection Strategy**
   - Currently uses last 2 layers heuristically
   - Could experiment with: all layers, attention layers only, or learned selection
   - Different layers may capture different failure semantics

4. **Trajectory vs. Sample Attribution**
   - Current pipeline attributes at sample level
   - For robotics, might want trajectory-level attribution (which full demo caused failure?)
   - Would require episode-aware aggregation

5. **Temporal Weighting**
   - Currently sums embeddings equally across time
   - Could weight by: timestep importance, action magnitude, or learned weights
   - Failures often happen at specific moments (e.g., grasp attempt)

6. **Integration with Visualization**
   - UMAP/t-SNE visualization of embedding space
   - Video replay of attributed training demos
   - Interactive cluster exploration

7. **Validation Methodology**
   - How to validate that attributed training demos are actually responsible?
   - Could use leave-one-out retraining (expensive but ground truth)
   - Or synthetic experiments with known failure causes

8. **Scaling to Image Observations**
   - Current implementation supports `DiffusionUnetHybridImagePolicy`
   - Gradient computation for image encoders is more expensive
   - May need to project only policy head gradients, not encoder

9. **Online/Incremental Updates**
   - Current pipeline is batch (recompute everything)
   - For deployment, might want incremental updates as new failures occur
   - Would require approximate nearest neighbor structures

10. **Multi-Modal Failure Analysis**
    - Combine influence embeddings with visual embeddings (DinoV2)
    - Could reveal whether failures are "behaviorally similar" vs "visually similar"
    - The codebase already has DinoV2 integration in `eval_embeddings.py`
