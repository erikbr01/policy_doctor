# Plotly Plots (Streamlit-Free)

The interactive notebook now uses **pure Plotly functions** with no Streamlit dependencies.

## New Module: `plot_heatmaps.py`

Located at: `influence_visualizer/plot_heatmaps.py`

This module provides Streamlit-free versions of the visualization functions that:
- **Return Plotly figures** instead of rendering to Streamlit
- Work in **any Python environment** (notebooks, scripts, REPL)
- Are **fully interactive** with hover, zoom, pan capabilities

## Available Functions

### 1. `plot_trajectory_influence_heatmap()`
```python
from influence_visualizer.plot_heatmaps import plot_trajectory_influence_heatmap

fig = plot_trajectory_influence_heatmap(
    data,
    split="train",  # "train", "holdout", or "both"
    annotation_file=None  # Optional annotation file path
)
fig.show()
```

**Features:**
- Interactive heatmap showing trajectory-level influences
- Hover to see demo/rollout details
- Color-coded by influence strength (red=negative, green=positive)
- Includes behavior labels if annotation file provided

### 2. `plot_performance_influence()`
```python
from influence_visualizer.plot_heatmaps import plot_performance_influence

fig = plot_performance_influence(
    data,
    split="train",
    metric="net",  # "net", "succ", or "fail"
    top_k=20
)
fig.show()
```

**Features:**
- Two-panel visualization: distribution histogram + top/bottom bar chart
- Shows which demos contribute most to performance
- Color-coded bars (green=positive, red=negative)
- Interactive tooltips with exact values

### 3. `plot_influence_distribution_by_success()`
```python
from influence_visualizer.plot_heatmaps import plot_influence_distribution_by_success

fig = plot_influence_distribution_by_success(
    data,
    split="train"
)
fig.show()
```

**Features:**
- Overlapping histograms comparing successful vs failed rollouts
- Density normalization for fair comparison
- Toggle success/failure in legend
- Hover to see density values

### 4. `plot_transition_statistics_density()` *(Placeholder)*
```python
from influence_visualizer.plot_heatmaps import plot_transition_statistics_density

fig = plot_transition_statistics_density(
    data,
    split="train"
)
fig.show()
```

**Note:** This is currently a placeholder. The full implementation requires `compute_transition_level_statistics` which is computationally expensive. Use the Streamlit app for this visualization.

## Advantages Over Streamlit Versions

### ✅ No Dependencies
- Pure Plotly - works anywhere Python runs
- No Streamlit required
- Easier to integrate into custom workflows

### ✅ Return Values
- Functions **return figures** instead of side effects
- Can save figures: `fig.write_html("plot.html")`
- Can customize further: `fig.update_layout(...)`

### ✅ Better for Notebooks
- Works in Jupyter, Zed REPL, IPython, etc.
- Figures display inline or in browser
- Easy to export and share

### ✅ Composable
- Combine multiple figures
- Create custom layouts
- Save to various formats (HTML, PNG, SVG, PDF)

## Example Workflow

```python
# Load data
from influence_visualizer.config import load_config
from influence_visualizer.data_loader import InfluenceDataLoader

config = load_config("pusht_jan26")
loader = InfluenceDataLoader(config)
loader.load()
data = loader.data

# Generate plots
from influence_visualizer.plot_heatmaps import (
    plot_trajectory_influence_heatmap,
    plot_performance_influence,
    plot_influence_distribution_by_success
)

# Plot 1: Trajectory heatmap
fig1 = plot_trajectory_influence_heatmap(data, split="train")
fig1.show()
# Or save: fig1.write_html("trajectory_heatmap.html")

# Plot 2: Performance influence
fig2 = plot_performance_influence(data, split="train", metric="net", top_k=20)
fig2.show()

# Plot 3: Distribution comparison
fig3 = plot_influence_distribution_by_success(data, split="train")
fig3.show()

# Customize a figure
fig1.update_layout(
    title="Custom Title",
    width=1200,
    height=800
)
fig1.show()
```

## Saving Figures

```python
# HTML (interactive)
fig.write_html("my_plot.html")

# Static image (requires kaleido)
fig.write_image("my_plot.png")
fig.write_image("my_plot.pdf")

# Show in browser
fig.show()
```

## Integration with Streamlit App

The Streamlit app (`influence_visualizer/app.py`) still uses the original `render_heatmaps.py` functions which call `st.plotly_chart()`.

The new `plot_heatmaps.py` module is specifically for **non-Streamlit contexts** like:
- Interactive notebooks (Zed, Jupyter)
- Standalone scripts
- Custom dashboards
- Report generation

## Future Enhancements

Potential additions to `plot_heatmaps.py`:
- `plot_episode_influence_heatmap()` - Frame-by-frame analysis
- `plot_clustering_dendrogram()` - Hierarchical clustering
- `plot_behavior_timeline()` - Temporal behavior patterns
- Full implementation of transition statistics (when optimized)

---

**See also:**
- `interactive_aggregated_influence.py` - Example usage in Zed REPL
- `render_heatmaps.py` - Original Streamlit versions
