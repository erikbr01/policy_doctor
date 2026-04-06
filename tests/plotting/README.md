# Plotting tests

Run from repo root with the project environment active (numpy, plotly, pillow required; pyvis optional for interactive graph tests):

```bash
python -m pytest policy_doctor/tests/plotting/ -v
# or
python -m unittest discover -s policy_doctor/tests/plotting -p "test_*.py" -v
```

- **test_common.py** — `plotting.common`: label colors, colorscale, action labels.
- **test_plotly_heatmaps.py** — `create_influence_heatmap` (Figure, heatmap trace, z range).
- **test_plotly_clusters.py** — `create_cluster_scatter_2d` (Figure, scatter traces, noise).
- **test_plotly_behavior_graph.py** — `create_behavior_graph_plot` (BehaviorGraph → Figure).
- **test_plotly_frames.py** — `create_annotated_frame` (PIL Image), `create_action_plot` (Figure).
- **test_pyvis.py** — `create_interactive_behavior_graph`, `create_value_colored_interactive_graph` (HTML string); skipped if `pyvis` is not installed.
