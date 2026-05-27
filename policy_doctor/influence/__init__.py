"""Influence data + clustering persistence absorbed from the legacy ``influence_visualizer`` package.

This package collects the small handful of functions that the rest of
``policy_doctor`` actually used from the now-removed ``third_party/influence_visualizer/``:

- :mod:`policy_doctor.influence.clustering_io` — save/load clustering result directories
- :mod:`policy_doctor.influence.loader` — load TRAK / InfEmbed influence data + seed-path helpers
- :mod:`policy_doctor.influence.lazy_hdf5` — lazy HDF5 image dataset and replay-buffer adapters
- :mod:`policy_doctor.influence.annotations` — JSON annotation read helpers (pure data, no Streamlit)
- :mod:`policy_doctor.influence.frames` — PIL frame-annotation helper (no Streamlit)

All modules are UI-free; any Streamlit-driven rendering belongs under
``policy_doctor/streamlit_app/`` per the project-wide UI separation rule.
"""
