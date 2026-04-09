# Using Interactive Scripts in Zed

This guide explains how to use the interactive Python scripts for exploring influence visualizations in Zed editor.

## Setup

### 1. Install ipykernel

The interactive scripts use Jupyter kernels for execution. Install ipykernel in your Python environment:

```bash
# If using conda 
conda activate your_environment
conda install ipykernel
python -m ipykernel install --user --name your_environment --display-name "Python (your_environment)"

# If using pip in a virtual environment
pip install ipykernel
python -m ipykernel install --user
```

**Note on macOS:** System Python won't work with Zed's REPL. Use conda or a virtual environment.

### 2. Configure Zed (Optional)

If you have multiple kernels installed, you can specify which one to use in Zed's `settings.json`:

```json
{
  "jupyter": {
    "kernel_selections": {
      "python": "your_environment"
    }
  }
}
```

### 3. Refresh Kernels in Zed

After installing ipykernel, refresh the available kernels in Zed:
- Open Command Palette: `Cmd+Shift+P` (macOS)
- Run: `repl: refresh kernelspecs`

## Using the Interactive Script

### Opening the Script

1. Open `influence_visualizer/interactive_aggregated_influence.py` in Zed
2. The script is organized into cells using `# %%` separators

### Executing Cells

There are two ways to run code:

1. **Run Selection/Cell**: 
   - Place cursor in a cell or select code
   - Press `Ctrl+Shift+Enter` (macOS default)
   - Results appear inline below your code

2. **View Available Kernels**:
   - Command Palette: `repl: sessions`

### Script Structure

The script is divided into logical cells:

1. **Imports** - Load necessary libraries
2. **Configuration** - Set the config name (e.g., `"pusht_jan26"`)
3. **Load Data** - Initialize the data loader and load influence data
4. **Settings** - Configure visualization parameters (demo split, top K, etc.)
5. **Helper Functions** - Plotting utilities
6. **Visualizations** - Individual visualization cells:
   - Trajectory-wise influence matrix heatmap
   - Performance influence per demonstration
   - Influence distribution by success/failure
7. **Exploration** - Raw data exploration
8. **Custom Analysis** - Your own analysis

### Workflow

1. **First Run** - Execute cells in order from top to bottom:
   ```python
   # Cell 1: Imports
   # Cell 2: Configuration (modify CONFIG_NAME if needed)
   # Cell 3: Load Data (this may take a moment)
   # Cell 4: Settings
   # Cell 5: Helper Functions
   ```

2. **Generate Visualizations** - Run individual visualization cells:
   ```python
   # Each visualization cell can be run independently
   # Plots appear inline in Zed
   ```

3. **Iterate** - Modify parameters and re-run cells:
   ```python
   # Change DEMO_SPLIT, TOP_K, or PERF_METRIC
   # Re-run the visualization cells to see updates
   ```

### Customizing the Analysis

#### Change Configuration

Edit the `CONFIG_NAME` variable in the Configuration cell:

```python
# %% Configuration
CONFIG_NAME = "pusht_jan26"  # Change to "lift_mh_jan26", etc.
```

Available configs (in `influence_visualizer/configs/`):
- `pusht_jan26`
- `lift_mh_jan26`
- `transport_mh_jan28`
- `mock` (for testing)

#### Change Visualization Settings

Edit parameters in the Settings cell:

```python
# %% Settings
DEMO_SPLIT = "train"  # Options: "train", "holdout", "both"
TOP_K = 20  # Number of top/bottom demonstrations
```

#### Modify Performance Metric

In the Performance Influence visualization cell:

```python
# %% Visualization 2
PERF_METRIC = "net"  # Options: "net", "succ", "fail"
```

- `"net"`: Success influence - failure influence
- `"succ"`: Only successful rollouts
- `"fail"`: Only failed rollouts

### Custom Analysis Cell

The last cell is reserved for your own exploratory analysis:

```python
# %% Custom Analysis
# Add your own analysis here!

# Example: Find most influential demo for a specific rollout
rollout_idx = 0
rollout_samples = data.get_samples_for_rollout_episode(rollout_idx)
influences = data.influence_matrix[rollout_samples, :].mean(axis=0)
top_demo = influences.argmax()
print(f"Most influential demo for rollout {rollout_idx}: Demo {top_demo}")
```

## Data Access

Once data is loaded, you have access to:

```python
# Main data object
data.influence_matrix  # Shape: (num_rollout_samples, num_demo_samples)

# Episode information
data.rollout_episodes  # List[EpisodeInfo]
data.demo_episodes     # List[EpisodeInfo] (train)
data.holdout_episodes  # List[EpisodeInfo] (holdout)

# Access specific data
data.get_rollout_frame(sample_idx)        # Get image frame
data.get_rollout_action(sample_idx)       # Get action
data.get_demo_frame(sample_idx)           # Get demo frame
data.get_demo_action_chunk(sample_idx)    # Get demo action

# Episode-level queries
data.get_samples_for_rollout_episode(ep_idx)
data.get_top_influences_for_sample(sample_idx, top_k=10)
```

## Tips

1. **Start Simple**: Run cells sequentially the first time to ensure everything loads correctly
2. **Data Persists**: Once loaded, the data stays in memory. You can re-run visualization cells without reloading
3. **Experiment**: The cell structure makes it easy to try different parameters and see results immediately
4. **Add Cells**: Add your own `# %%` cells anywhere for custom analysis
5. **Save Results**: Use `plt.savefig()` to save plots to disk

## Troubleshooting

### Kernel Not Found
- Run `conda install ipykernel` in your environment
- Refresh kernelspecs in Zed: `repl: refresh kernelspecs`

### Import Errors
- Ensure you're in the correct conda environment
- Check that the project root is in the Python path (the script handles this automatically)

### Data Loading Errors
- Verify the config file paths are correct
- Check that the data directories exist
- Try the `"mock"` config first to test the setup

### Plots Not Showing
- Zed's REPL should display matplotlib plots inline
- If not, try adding `%matplotlib inline` at the top of the script
- Or use `plt.savefig()` to save plots and view them externally

## Next Steps

After mastering the aggregated influence script, you can:

1. Request additional interactive scripts for other tabs (episode influence, clustering, etc.)
2. Combine insights from multiple visualizations
3. Export data for further analysis in notebooks or reports

## Example Session

```python
# 1. Load config and data
CONFIG_NAME = "pusht_jan26"
# ... (run cells)

# 2. Generate all visualizations with default settings
# ... (run visualization cells)

# 3. Focus on successful rollouts only
PERF_METRIC = "succ"
# ... (re-run performance influence cell)

# 4. Compare train vs holdout demos
DEMO_SPLIT = "holdout"
# ... (re-run all visualization cells)

# 5. Deep dive into specific demos
top_demos = np.argsort(perf_influence)[::-1][:5]
for demo_idx in top_demos:
    print(f"Demo {demo_idx}: {perf_influence[demo_idx]:.4f}")
```
