# Interactive Notebooks for Influence Visualizer

This folder contains interactive Python scripts designed to run in Zed's REPL (or other notebook-style environments with ipykernel support).

## Files

- **`interactive_aggregated_influence.py`** - Interactive script for the "Aggregated Influence" tab
  - Trajectory-wise influence matrix heatmap
  - Performance influence per demonstration
  - Influence distribution by success/failure
  - Raw data exploration

- **`INTERACTIVE_QUICKSTART.md`** - Quick start guide to get up and running
- **`INTERACTIVE_USAGE.md`** - Detailed usage instructions and tips

## Quick Start

1. **Install ipykernel** in your conda environment:
   ```bash
   conda activate cupid
   conda install ipykernel
   python -m ipykernel install --user --name cupid
   ```

2. **Open the script** in Zed:
   ```
   influence_visualizer/notebooks/interactive_aggregated_influence.py
   ```

3. **Run cells sequentially**:
   - Place cursor in a cell (marked with `# %%`)
   - Press `Ctrl+Shift+Enter` to execute
   - Results appear inline

## Benefits vs Streamlit App

- **Faster iteration** - No page reloads
- **Data persistence** - Load once, visualize many times
- **Inline results** - Plots appear right below code
- **Easy customization** - Modify parameters and re-run cells
- **Better for exploration** - Full Python REPL access to data

## Configuration

The scripts work with the same config files as the Streamlit app:
- `influence_visualizer/configs/pusht_jan26.yaml`
- `influence_visualizer/configs/lift_mh_jan26.yaml`
- `influence_visualizer/configs/transport_mh_jan28.yaml`

Just change the `CONFIG_NAME` variable in the Configuration cell.

## Future Scripts

More interactive scripts can be added for other tabs:
- Episode-level influence (frame-by-frame analysis)
- Clustering analysis
- Behavior annotation
- Advanced analysis

## See Also

- [Zed REPL Documentation](https://zed.dev/docs/repl)
- Main Streamlit app: `influence_visualizer/app.py`
