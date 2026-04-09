#!/bin/bash

# Set Numba threading layer for Streamlit compatibility
# Per Streamlit docs: https://docs.streamlit.io/develop/concepts/design/multithreading
# Use 'omp' (OpenMP) or 'tbb' for thread-safe parallel execution
# 'workqueue' is NOT thread-safe and will crash with Streamlit
export NUMBA_THREADING_LAYER=omp

# Run Streamlit app
streamlit run influence_visualizer/app.py "$@"
