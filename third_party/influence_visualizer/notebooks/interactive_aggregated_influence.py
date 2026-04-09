# %% [markdown]
# # Interactive Aggregated Influence Visualizer
#
# This script converts the "Aggregated Influence" tab from the Streamlit app
# into an interactive notebook-style script for Zed's REPL.
#
# ## Quick Start:
# 1. Run the "Imports & Setup" cell below (Ctrl+Shift+Enter)
# 2. Run the "Configuration" cell to load your config
# 3. Run the "Load Data" cell to load influence data
# 4. Run any visualization cells to generate plots
#
# **Note:** Run cells in order from top to bottom the first time!

# %% Imports & Setup
# ============================================================================
# IMPORTANT: Run this cell FIRST before running any other cells!
# ============================================================================
import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
# In REPL, __file__ is not defined, so we use getcwd() and navigate to project root
try:
    # Try __file__ first (works when run as script)
    _PROJECT_ROOT = Path(__file__).parent.parent.parent
except NameError:
    # Fallback for REPL environment
    _current = Path.cwd()
    if _current.name == "notebooks":
        _PROJECT_ROOT = _current.parent.parent
    elif _current.name == "influence_visualizer":
        _PROJECT_ROOT = _current.parent
    elif (_current / "influence_visualizer").exists():
        _PROJECT_ROOT = _current
    else:
        # Last resort: go up until we find influence_visualizer
        _PROJECT_ROOT = _current
        while (
            not (_PROJECT_ROOT / "influence_visualizer").exists()
            and _PROJECT_ROOT != _PROJECT_ROOT.parent
        ):
            _PROJECT_ROOT = _PROJECT_ROOT.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

print(f"Project root: {_PROJECT_ROOT}")

from influence_visualizer.config import load_config
from influence_visualizer.data_loader import InfluenceDataLoader

print("✓ Imports successful!")

# %% Configuration
# Change this to the config you want to use
CONFIG_NAME = (
    "pusht_jan26"  # Options: "pusht_jan26", "lift_mh_jan26", "transport_mh_jan28", etc.
)

print(f"Loading configuration: {CONFIG_NAME}")
config = load_config(CONFIG_NAME)
print(f"Task: {config.name}")
print(f"Eval dir: {config.eval_dir}")
print(f"Train dir: {config.train_dir}")

# %% Load Data
print("Loading influence data...")
loader = InfluenceDataLoader(config)
loader.load()
data = loader.data

print(f"\n✓ Data loaded successfully!")
print(f"\nDataset Info:")
print(f"  Rollout Episodes: {loader.num_rollout_episodes}")
print(f"  Train Demo Episodes: {len(loader.demo_episodes)}")
print(f"  Holdout Demo Episodes: {len(loader.holdout_episodes)}")
print(f"  Rollout Samples: {loader.num_rollout_samples}")
print(f"  Train Demo Samples: {len(loader.data.demo_sample_infos)}")
print(f"  Holdout Demo Samples: {len(loader.data.holdout_sample_infos)}")

print(f"\nConfig (from checkpoint):")
print(f"  Horizon: {loader.horizon}")
print(f"  Pad Before: {loader.data.pad_before}")
print(f"  Pad After: {loader.data.pad_after}")
print(f"  N Obs Steps: {loader.data.n_obs_steps}")
print(f"\nInfluence Matrix Shape: {data.influence_matrix.shape}")

print("  (num_rollout_samples, num_demo_samples)")

# %% Settings
# Configure visualization parameters
DEMO_SPLIT = "train"  # Options: "train", "holdout", "both"
ANNOTATION_FILE = None  # Set to path if you want to use annotations

print(f"\nVisualization Settings:")
print(f"  Demo Split: {DEMO_SPLIT}")
print(f"  Annotation File: {ANNOTATION_FILE or 'None'}")

# %% Visualization 1: Trajectory-wise Influence Matrix
print("\n" + "=" * 70)
print("VISUALIZATION 1: Trajectory-wise Influence Matrix")
print("=" * 70)
print("\nThis shows the original CUPID performance influence matrix where")
print("influences are aggregated from action-level to trajectory-level.")
print()

from influence_visualizer.plot_heatmaps import plot_trajectory_influence_heatmap

# This function returns a Plotly figure (no Streamlit dependency)
fig = plot_trajectory_influence_heatmap(
    data, split=DEMO_SPLIT, annotation_file=ANNOTATION_FILE
)

fig.show()

print("\n✓ Interactive plot displayed!")
print("  - Hover over cells to see values")
print("  - Click and drag to zoom")
print("  - Double-click to reset view")

# %% Visualization 2: Performance Influence per Demonstration
print("\n" + "=" * 70)
print("VISUALIZATION 2: Performance Influence per Demonstration")
print("=" * 70)
print("\nPerformance influence measures how each demonstration contributes")
print("to policy performance using sum_of_sum_influence aggregation.")
print()

# Settings for this visualization
PERF_METRIC = "net"  # Options: "net", "succ", "fail"
PERF_TOP_K = 20  # Number of top/bottom demos to show

from influence_visualizer.plot_heatmaps import plot_performance_influence

fig = plot_performance_influence(
    data,
    split=DEMO_SPLIT,
    metric=PERF_METRIC,
    top_k=PERF_TOP_K,
)

# Display the interactive plot
fig.show()

print(f"\n✓ Interactive plot displayed for metric='{PERF_METRIC}'!")
print("  - Hover over bars to see exact values")
print("  - Use legend to toggle views")

# %% Visualization 3: Influence Distribution by Success/Failure
print("\n" + "=" * 70)
print("VISUALIZATION 3: Influence Distribution - Success vs. Failure")
print("=" * 70)
print("\nDistribution comparison shows how individual influence values")
print("differ between successful and failed rollouts.")
print()

from influence_visualizer.plot_heatmaps import plot_influence_distribution_by_success

fig = plot_influence_distribution_by_success(
    data,
    split=DEMO_SPLIT,
)

# Display the interactive plot
fig.show()

print("\n✓ Interactive plot displayed!")
print("  - Hover over curves to see density values")

# %% Visualization 4: Transition Statistics Density (Optional)
print("\n" + "=" * 70)
print("VISUALIZATION 4: Transition Statistics Density (Optional)")
print("=" * 70)
print("\nThis shows overlapping density histograms for transition-level")
print("influence statistics, comparing successful and failed rollouts.")
print()

# This visualization can be heavy for large datasets
SHOW_TRANSITION_STATS = False  # Set to True to enable

if SHOW_TRANSITION_STATS:
    from influence_visualizer.plot_heatmaps import plot_transition_statistics_density

    fig = plot_transition_statistics_density(
        data,
        split=DEMO_SPLIT,
    )

    fig.show()

    print("\n✓ Interactive plot displayed!")
else:
    print("Skipped (set SHOW_TRANSITION_STATS=True to enable)")

# %% Custom Analysis: Explore Raw Data
print("\n" + "=" * 70)
print("CUSTOM ANALYSIS: Explore Raw Data")
print("=" * 70)

# Example: Look at influence distribution for a specific rollout sample
sample_idx = 0  # Change this to explore different samples
influences = data.influence_matrix[sample_idx, :]

print(f"\nInfluences for rollout sample {sample_idx}:")
print(f"  Min: {influences.min():.4f}")
print(f"  Max: {influences.max():.4f}")
print(f"  Mean: {influences.mean():.4f}")
print(f"  Std: {influences.std():.4f}")

# Create interactive histogram
fig = go.Figure()
fig.add_trace(
    go.Histogram(
        x=influences,
        nbinsx=50,
        name=f"Sample {sample_idx}",
        marker_color="steelblue",
        opacity=0.7,
    )
)

fig.update_layout(
    title=f"Influence Distribution for Rollout Sample {sample_idx}",
    xaxis_title="Influence Score",
    yaxis_title="Frequency",
    showlegend=True,
    hovermode="x unified",
)

fig.show()

# %% Custom Analysis: Success Rate Analysis
print("\n" + "=" * 70)
print("CUSTOM ANALYSIS: Success Rate")
print("=" * 70)

success_mask = np.array([ep.success for ep in data.rollout_episodes], dtype=bool)
num_success = success_mask.sum()
num_failure = (~success_mask).sum()

print(f"\nRollout Success Rate:")
print(f"  Successful: {num_success} ({num_success / len(success_mask) * 100:.1f}%)")
print(f"  Failed: {num_failure} ({num_failure / len(success_mask) * 100:.1f}%)")

# Create pie chart
fig = go.Figure(
    data=[
        go.Pie(
            labels=["Success", "Failure"],
            values=[num_success, num_failure],
            marker_colors=["green", "red"],
            hole=0.3,
        )
    ]
)

fig.update_layout(title="Rollout Success Rate", showlegend=True)

fig.show()

# %% Custom Analysis: Episode Length Distribution
print("\n" + "=" * 70)
print("CUSTOM ANALYSIS: Episode Length Distribution")
print("=" * 70)

rollout_lengths = [ep.num_samples for ep in data.rollout_episodes]

print(f"\nRollout Episode Lengths:")
print(f"  Mean: {np.mean(rollout_lengths):.1f}")
print(f"  Median: {np.median(rollout_lengths):.1f}")
print(f"  Min: {np.min(rollout_lengths)}")
print(f"  Max: {np.max(rollout_lengths)}")

# Create box plot
success_lengths = [ep.num_samples for ep in data.rollout_episodes if ep.success]
failure_lengths = [ep.num_samples for ep in data.rollout_episodes if not ep.success]

fig = go.Figure()
fig.add_trace(go.Box(y=success_lengths, name="Success", marker_color="green"))
fig.add_trace(go.Box(y=failure_lengths, name="Failure", marker_color="red"))

fig.update_layout(
    title="Episode Length Distribution by Success/Failure",
    yaxis_title="Number of Samples",
    showlegend=True,
)

fig.show()

# %% Your Custom Analysis Here
print("\n" + "=" * 70)
print("YOUR CUSTOM ANALYSIS")
print("=" * 70)
print("Add your own analysis code in this cell!")
print()

# Example: Access data directly
print("Available data structures:")
print(f"  data.influence_matrix - shape: {data.influence_matrix.shape}")
print(f"  data.rollout_episodes - {len(data.rollout_episodes)} episodes")
print(f"  data.demo_episodes - {len(data.demo_episodes)} episodes")
print(f"  data.holdout_episodes - {len(data.holdout_episodes)} episodes")
print()
print("Example operations:")
print("  - data.get_rollout_frame(sample_idx)")
print("  - data.get_rollout_action(sample_idx)")
print("  - data.get_top_influences_for_sample(sample_idx, top_k=10)")
print("  - data.compute_performance_influence(metric='net')")

# %%
