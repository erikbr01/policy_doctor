# Performance Influence Visualization

This document explains how to use the updated `visualize_influence_matrix.py` script to visualize both action influence and performance influence matrices.

## Overview

The script now supports two types of influence visualization:

### 1. Action Influence Matrix (Default)
- **Level**: State-action pairs
- **Dimensions**: `(num_rollout_samples, num_demo_samples)`
- **Use case**: Understand influence at the granular state-action level
- **Description**: Shows how each state-action pair in rollout trajectories is influenced by each state-action pair in demonstration trajectories

### 2. Performance Influence Matrix (New Feature)
- **Level**: Trajectories
- **Dimensions**: `(num_rollout_trajectories, num_demo_trajectories)`
- **Use case**: CUPID's core metric for data curation
- **Description**: Shows how each rollout trajectory's performance is influenced by each demonstration trajectory
- **Computation**: Aggregates action influence using various aggregation functions

## How Performance Influence is Computed

The performance influence matrix is computed from the action influence matrix using the `pairwise_sample_to_trajectory_scores` function from `error_util.py`. This is the same computation used in CUPID (see `eval_demonstration_scores.py:online_trak_influence_routine`).

### Aggregation Functions

Different aggregation functions provide different perspectives on influence:

- **`mean_of_mean`**: Average influence across all state-action pairs
- **`mean_of_mean_success`**: Same as above, but only for successful rollouts (0 for failures)
- **`sum_of_sum`**: Total influence (sum across all state-action pairs)
- **`sum_of_sum_success`**: Same as above, but only for successful rollouts
- **`min_of_max`**: Conservative estimate (minimum of maximum influences)
- **`max_of_min`**: Optimistic estimate (maximum of minimum influences)

## Complete Workflow

Simply provide the training directory - dataset sizes are automatically determined from metadata:

```bash
python influence_embeddings/visualize_influence_matrix.py \
    --eval_dir data/outputs/eval_save_episodes/jan17/jan16_train_diffusion_unet_lowdim_lift_mh_0/latest \
    --exp_date default \
    --performance_influence \
    --train_dir data/outputs/train/jan16/jan16_train_diffusion_unet_lowdim_lift_mh_0 \
    --aggr_fn mean_of_mean \
    --output_path influence_heatmap.png \
    --no_debug
```

**Note:** `--train_set_size` and `--test_set_size` are optional when using `--train_dir`. They will be auto-determined from the dataset metadata.

This will:
1. Automatically load the training configuration from `train_dir/.hydra/config.yaml`
2. Load the training and holdout datasets
3. Extract episode metadata using `trak_util.get_dataset_metadata()`
4. Auto-determine `train_set_size` (train + holdout) and `test_set_size`
5. Load the action influence matrix and slice it to the training portion
6. Compute performance influence by aggregating with the specified function
7. Generate **both** visualizations:
   - `influence_heatmap_action.png` - Action influence (state-action level)
   - `influence_heatmap_performance.png` - Performance influence (trajectory level)

## Usage

### Action Influence (State-Action Level)

```bash
python influence_embeddings/visualize_influence_matrix.py \
    --eval_dir data/outputs/eval_save_episodes/jan17/train_name/latest \
    --exp_date 25.01.17 \
    --train_set_size 1000 \
    --test_set_size 500 \
    --output_path action_influence_heatmap.png
```

### Performance Influence (Trajectory Level)

```bash
python influence_embeddings/visualize_influence_matrix.py \
    --eval_dir data/outputs/eval_save_episodes/jan17/train_name/latest \
    --exp_date 25.01.17 \
    --train_set_size 1000 \
    --test_set_size 500 \
    --performance_influence \
    --train_ep_lens_file data/train_ep_lens.npy \
    --train_num_eps 50 \
    --aggr_fn mean_of_mean \
    --output_path performance_influence_heatmap.png
```

## Required Arguments for Performance Influence

When using `--performance_influence`, you must provide:

- **`--train_dir`**: Path to training directory containing `.hydra/config.yaml`
  - The script will automatically load the training config and dataset
  - Episode metadata is extracted using `trak_util.get_dataset_metadata()`
  - Dataset sizes (`train_set_size` and `test_set_size`) are auto-determined
  - No additional setup or file generation required

## Implementation Details

The new functionality adds:

1. **`compute_performance_influence()`**: Aggregates action influence to trajectory level
   - Located in: `visualize_influence_matrix.py:29-86`
   - Uses `error_util.pairwise_sample_to_trajectory_scores()`

2. **`load_episode_metadata()`**: Loads rollout episode metadata
   - Located in: `visualize_influence_matrix.py:89-124`
   - Reads from `eval_dir/episodes` using `BatchEpisodeDataset`

3. **CLI Arguments**:
   - `--performance_influence`: Enable performance influence mode
   - `--aggr_fn`: Choose aggregation function
   - `--train_ep_lens_file`: Path to demo episode lengths
   - `--train_num_eps`: Number of demo episodes

## Code References

The performance influence computation follows the same approach as used in:
- `eval_demonstration_scores.py:online_trak_influence_routine` (lines 617-667)
- `diffusion_policy/common/error_util.py:pairwise_sample_to_trajectory_scores` (lines 207-224)

This ensures consistency with CUPID's data curation pipeline.
