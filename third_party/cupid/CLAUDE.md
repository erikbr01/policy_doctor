# Notes for Claude (AI Assistant)

This file contains design patterns and architectural decisions to follow when working on this codebase.

## Influence Visualizer Architecture

### Separation of Concerns Pattern

The `influence_visualizer` package follows a strict separation between UI logic and visualization logic:

**DO:**
- ✅ Keep Streamlit UI code (buttons, sliders, checkboxes) in `render_*.py` modules
- ✅ Create pure plotting functions in `influence_visualizer/plotting/` that return Plotly Figure objects
- ✅ Preprocess data (compute statistics, apply transformations) in the Streamlit layer
- ✅ Pass preprocessed data to plotting functions
- ✅ Export plotting functions from `plotting/__init__.py`

**DON'T:**
- ❌ Import Streamlit in `plotting/` modules
- ❌ Mix Plotly figure creation with Streamlit UI elements
- ❌ Put data preprocessing logic inside plotting functions

### Adding New Visualizations

When adding new visualizations, follow this workflow:

1. **Create Pure Plotting Function** in `plotting/heatmaps.py` (or other appropriate module):
   ```python
   def create_my_visualization(
       data: np.ndarray,
       labels: List[str],
       title: str,
       ...
   ) -> go.Figure:
       """Pure plotting function - no Streamlit dependencies.
       
       Args:
           data: Preprocessed data ready for plotting
           labels: Labels for the plot
           title: Plot title
           ...
       
       Returns:
           Plotly Figure object
       """
       fig = go.Figure()
       # ... create visualization ...
       return fig
   ```

2. **Export Function** in `plotting/__init__.py`:
   ```python
   from influence_visualizer.plotting.heatmaps import (
       ...
       create_my_visualization,  # Add here
   )
   
   __all__ = [
       ...
       "create_my_visualization",  # Add here
   ]
   ```

3. **Use in Streamlit Module** (e.g., `render_local_behaviors.py`):
   ```python
   from influence_visualizer import plotting
   
   @st.fragment
   def render_my_section(data, ...):
       # Streamlit UI controls
       show_option = st.checkbox("Show my option", value=True)
       
       if st.button("Generate Visualization"):
           # Preprocess data
           processed_data = compute_statistics(data)
           
           # Call pure plotting function
           fig = plotting.create_my_visualization(
               data=processed_data,
               labels=labels,
               title="My Visualization",
           )
           
           # Display in Streamlit
           st.plotly_chart(fig, width='stretch'')
   ```

### Benefits of This Pattern

- **Reusability**: Plotting functions can be used in notebooks, scripts, or other frontends
- **Testability**: Pure functions are easier to unit test
- **Maintainability**: Clear separation makes code easier to understand and modify
- **Consistency**: All modules follow the same architecture

### Examples in Codebase

Good examples to follow:
- `plotting/heatmaps.py`: `create_influence_heatmap()`, `create_influence_grid_plot()`, `create_influence_density_heatmaps()`, `create_influence_distribution_lines()`
- `render_local_behaviors.py`: `render_demo_influence_distribution()` (uses plotting functions)
- `render_heatmaps.py`: Various render functions that call plotting module

## Recent Refactoring (2026-02-02)

The `render_local_behaviors.py` module was refactored to follow this pattern:
- **Before**: ~346 lines of mixed Plotly + Streamlit code for density heatmaps and line plots
- **After**: ~133 lines that call `create_influence_density_heatmaps()` and `create_influence_distribution_lines()`
- **Result**: Reduced complexity, improved maintainability, plotting functions now reusable

## General Guidelines

- Always check existing patterns before adding new code
- When in doubt, look at how similar features are implemented
- Maintain consistency with the established architecture
- Document design decisions for future reference
