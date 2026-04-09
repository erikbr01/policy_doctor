# Interactive Influence Visualizer - Quick Start

## What Was Created

I've converted the **Aggregated Influence** tab from your Streamlit app into an interactive Python script that runs in Zed's REPL environment.

### Files Created

1. **`interactive_aggregated_influence.py`** - The main interactive script
2. **`INTERACTIVE_USAGE.md`** - Detailed usage guide
3. **`INTERACTIVE_QUICKSTART.md`** - This quick start guide

## Quick Start (3 Steps)

### 1. Install ipykernel

```bash
# Activate your conda environment first
conda activate your_env
conda install ipykernel
python -m ipykernel install --user --name your_env
```

### 2. Open in Zed

Open `influence_visualizer/interactive_aggregated_influence.py` in Zed editor.

### 3. Run Cells

- Place cursor in a cell (separated by `# %%`)
- Press `Ctrl+Shift+Enter` to execute
- Results appear inline below the code

## What's Included

The interactive script provides three main visualizations from the Streamlit app:

### 1. Trajectory-wise Influence Matrix
A heatmap showing how demonstration episodes influence rollout episodes (the original CUPID matrix).

### 2. Performance Influence per Demonstration  
Bar charts and histograms showing which demonstrations contribute most to successful rollouts.

### 3. Influence Distribution by Success/Failure
Overlapping histograms comparing influence patterns between successful and failed rollouts.

## Cell Structure

```python
# %% Configuration
CONFIG_NAME = "pusht_jan26"  # Change this to your config

# %% Load Data
# Loads influence data (run this once)

# %% Settings
DEMO_SPLIT = "train"  # Options: "train", "holdout", "both"
TOP_K = 20

# %% Visualization 1: Trajectory-wise Influence Matrix
# Run to generate heatmap

# %% Visualization 2: Performance Influence
PERF_METRIC = "net"  # Options: "net", "succ", "fail"
# Run to generate bar charts

# %% Visualization 3: Influence Distribution
# Run to compare success vs failure

# %% Custom Analysis
# Your own exploratory code here
```

## Advantages Over Streamlit

1. **Faster Iteration**: No page reloads - just re-run the cell
2. **Data Persistence**: Load data once, experiment with visualizations
3. **Inline Results**: Plots appear right below your code
4. **Customizable**: Easy to modify and extend for your analysis
5. **Version Control**: Scripts are easier to track than notebook state

## Key Configuration Options

Change these variables to customize your analysis:

```python
# Choose your config
CONFIG_NAME = "pusht_jan26"  # or "lift_mh_jan26", "transport_mh_jan28"

# Visualization settings
DEMO_SPLIT = "train"    # "train", "holdout", or "both"
TOP_K = 20              # Number of top/bottom demos to show
PERF_METRIC = "net"     # "net", "succ", or "fail"
```

## Example Workflow

1. **Initial Load**:
   - Run Configuration cell
   - Run Load Data cell (this takes a moment)
   - Run Settings cell

2. **Generate All Visualizations**:
   - Run Visualization 1 cell → See trajectory heatmap
   - Run Visualization 2 cell → See performance influence
   - Run Visualization 3 cell → See distribution comparison

3. **Iterate**:
   - Change `PERF_METRIC` to `"succ"` → Re-run Visualization 2
   - Change `DEMO_SPLIT` to `"both"` → Re-run all visualizations
   - Change `TOP_K` to `50` → Re-run Visualization 2

4. **Custom Analysis**:
   - Use the Custom Analysis cell to explore specific questions
   - Access `data.influence_matrix` directly
   - Query episode information with `data.rollout_episodes`

## Available Data After Loading

```python
# Influence scores
data.influence_matrix  # (num_rollout_samples, num_demo_samples)

# Episode metadata
data.rollout_episodes   # Rollout episode information
data.demo_episodes      # Training demo episodes
data.holdout_episodes   # Holdout demo episodes

# Configuration
data.horizon           # Action prediction horizon
data.n_obs_steps      # Number of observation steps
data.pad_before       # Padding before sequence
data.pad_after        # Padding after sequence

# Methods
data.get_rollout_frame(sample_idx)
data.get_rollout_action(sample_idx)
data.get_demo_frame(sample_idx)
data.get_top_influences_for_sample(sample_idx, top_k=10)
```

## Tips

- **Run cells sequentially** the first time through
- **Data stays loaded** - no need to reload when changing visualization params
- **Add your own cells** using `# %%` separator
- **Save plots** with `plt.savefig("my_plot.png")`
- **Print variables** to inspect data structures

## Next Steps

Once you're comfortable with this script:

1. Let me know if you want similar scripts for:
   - Episode-level influence (frame-by-frame analysis)
   - Clustering analysis
   - Behavior annotation
   
2. Customize the plotting functions in the script

3. Add your own analysis cells for specific research questions

## Troubleshooting

**Kernel not found?**
- Run `conda install ipykernel` in your environment
- Refresh in Zed: Command Palette → `repl: refresh kernelspecs`

**Import errors?**
- Make sure you're in the correct conda environment
- The script adds project root to path automatically

**Plots not showing?**
- Zed should display plots inline
- If not, save with `plt.savefig()` instead

**Data loading fails?**
- Check that paths in config file are correct
- Try `CONFIG_NAME = "mock"` first to test setup

## Feedback

Test the script and let me know:
- Does it work in Zed?
- Do the plots display correctly?
- What other visualizations or analyses would be helpful?

Once this works well, we can create similar interactive scripts for the other tabs!
