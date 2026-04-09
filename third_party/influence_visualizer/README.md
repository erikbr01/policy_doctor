# Influence Visualizer

An interactive web platform for fine-grained data curation through visualization of influence matrices. This tool enables model evaluators to understand how training demonstrations influence policy rollout behaviors at the state-action level.

## Overview

The influence visualizer provides:
- **Rollout Video Playback**: View policy evaluation episodes with frame-by-frame navigation
- **Influence Ranking**: For each rollout frame, see a ranked list of training demonstrations sorted by their influence on that specific state-action pair
- **Influence Heatmaps**: Aggregate visualizations showing which demonstrations most influence each rollout episode

## Current Implementation (v0.1)

### Features
- Web-based UI built with Streamlit
- Interactive rollout episode selection with success/failure indicators
- Frame-by-frame sample selection within episodes
- Top-K influential demonstration display with video previews
- Episode-aggregated influence heatmaps

### Architecture

```
influence_visualizer/
├── __init__.py              # Package exports
├── app.py                   # Streamlit web application
├── data_loader.py           # Data loading and sample-to-episode mapping
├── render_demo_videos.py    # Script to pre-render demonstration videos
├── README.md                # This documentation
└── tests/
    ├── __init__.py
    └── test_data_loader.py  # Unit tests for data loading
```

## Installation

The influence visualizer requires the following dependencies:

```bash
pip install streamlit plotly numpy pyyaml h5py
```

For running tests:
```bash
pip install pytest
```

## Usage

### Step 1: Pre-render Demonstration Videos

Before using the visualizer, you need to render videos for the training demonstrations:

```bash
python influence_visualizer/render_demo_videos.py \
    --dataset data/robomimic/datasets/lift/mh/low_dim.hdf5 \
    --output_dir data/outputs/demo_videos/lift_mh \
    --render_image_names agentview \
    --height 256 --width 256
```

This creates individual `.mp4` files for each demonstration episode.

### Step 2: Run the Visualizer

Launch the Streamlit app with your data paths:

```bash
streamlit run influence_visualizer/app.py -- \
    --eval_dir data/outputs/eval_save_episodes/jan17/jan16_train_diffusion_unet_lowdim_lift_mh_0/latest \
    --demo_video_dir data/outputs/demo_videos/lift_mh \
    --dataset_path data/robomimic/datasets/lift/mh/low_dim_abs.hdf5 \
    --train_set_size 27859 \
    --test_set_size 1242 \
    --exp_date default
```

For testing without real data, use mock mode:
```bash
streamlit run influence_visualizer/app.py -- --mock
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--eval_dir` | Path to evaluation output directory containing rollout episodes and TRAK scores |
| `--demo_video_dir` | Path to pre-rendered demonstration videos |
| `--dataset_path` | Path to HDF5 training dataset (for correct sample indexing) |
| `--train_set_size` | Number of training samples (from TRAK computation) |
| `--test_set_size` | Number of test/rollout samples |
| `--exp_date` | Experiment date prefix for TRAK scores (default: "default") |
| `--top_k` | Number of top influences to display (default: 10) |
| `--mock` | Use mock data for testing |

## Running Tests

Run the test suite to verify core functionality without a browser:

```bash
# Run all tests
pytest influence_visualizer/tests/ -v

# Run specific test file
pytest influence_visualizer/tests/test_data_loader.py -v

# Run with coverage
pytest influence_visualizer/tests/ -v --cov=influence_visualizer
```

## Data Structures

### Influence Matrix
The influence matrix has shape `(num_rollout_samples, num_demo_samples)` where:
- **Rows**: Rollout (test) samples, ordered sequentially by episode
- **Columns**: Demonstration (training) samples, ordered by the SequenceSampler used during training

### Sample-to-Episode Mapping
The data loader correctly maps between:
- Global sample indices (used in the influence matrix)
- Episode indices and timesteps (used for video display)

This mapping accounts for:
- Rollout episodes from `eval_save_episodes` pickle files
- Training samples from `SequenceSampler` with horizon, padding, and episode masking

## Future Development

### Planned Features (v0.2+)

1. **Slice-based Labeling**
   - Select temporal slices of rollouts (e.g., "grasping phase", "placement phase")
   - Label slices with behavior categories
   - Persist labels for later analysis

2. **Behavior Clustering**
   - Cluster rollout slices by influence pattern similarity
   - Discover recurring behavior modes automatically
   - Visualize cluster centers and boundaries

3. **Demonstration Slice Ranking**
   - Aggregate influence across labeled rollout slices
   - Rank demonstration slices by their contribution to specific behaviors
   - Support data curation decisions (keep/remove demonstrations)

4. **Export & Integration**
   - Export labeled data for downstream analysis
   - Integration with CUPID curation pipeline
   - Batch processing of multiple experiments

### Architecture Notes for Future Development

The current architecture is designed for extensibility:

- **`data_loader.py`**: The `InfluenceData` class can be extended to support slice-based queries. Consider adding:
  - `get_influences_for_slice(start_sample, end_sample)`: Aggregate influences over a sample range
  - `get_slice_similarity(slice1, slice2)`: Compare influence patterns between slices

- **`app.py`**: The Streamlit app uses session state for the current frame. For slicing, extend to store:
  - Selected slice boundaries
  - Slice labels/annotations
  - Clustering results

- **Video handling**: The current approach loads full videos. For slicing:
  - Consider extracting frame ranges on-demand
  - Or pre-compute frame thumbnails for timeline visualization

## Troubleshooting

### "Demo metadata not found"
Run `render_demo_videos.py` first to generate demonstration videos.

### Sample count mismatch warnings
Ensure `train_set_size` and `test_set_size` match the values used during TRAK score computation. These can be found in the training configuration or TRAK output logs.

### Video not available
- Check that rollout videos were generated with `save_episodes=True`
- Verify demo videos were rendered with `render_demo_videos.py`
- Ensure video paths are accessible from the machine running Streamlit

## Contributing

When adding new features:
1. Add tests in `tests/test_*.py` 
2. Update this README with usage instructions
3. Follow the existing code style (type hints, docstrings)

## References

- [CUPID: Improving Generalist Robots via Data Curation with Influence Functions](https://github.com/Stanford-ILIAD/cupid)
- [TRAK: Tracing Training Data Influence](https://github.com/MadryLab/trak)
- [Streamlit Documentation](https://docs.streamlit.io/)

## Prompt (Claude)

I have an idea for a project! I want to build a good interface for fine-grained data curation. The first step to that is annotation and enabling a human understanding of the influence matrix (first visualization script in [@visualize_influence_matrix.py](file:///home/erbauer/cupid/influence_embeddings/visualize_influence_matrix.py) ) that relates state-action pairs of rollouts to state-action pairs of demonstrations. What I am imagining is a labelling platform, where model evaluators can go through videos of policy rollouts, select slices of those rollouts and directly see videos of the slices of training demonstrations that influenced them. As a first step, I want to build a basic interpretability platform. There should be a web UI where a user can select a policy rollout, view a video of the policy evaluation and for every frame of the video, there should be a sorted list of demonstration videos, sorted by the influence on the frame of the rollout. This should be the first version of the visualizer - in future versions, I am thinking of adding slicing for labeling the rollouts and clustering the slices into behaviors, and then finding slices of demonstrations ranked by influence. Just keep that in mind for architectural purposes. 

This should be shown in a web interface, choose whichever framework is easy to work with to achieve this. 

Furthermore, write this code into a new folder called influence_visualizer. All the code there should be as self-contained as possible. Also write a test suite, where you can verify essential features without having to open the web browser. 

Write a markdown document in this new directory detailing these instructions that can serve as a documentation for this project, containing information on the current implementation and on future tasks that are open and any relevant notes for users or for coding agents such as yourself.
