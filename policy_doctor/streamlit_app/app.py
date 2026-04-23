"""Policy Doctor Streamlit app: sidebar (config, paths), tabs. Orchestration only.

Uses policy_doctor.config for task config; policy_doctor.data.influence_loader for data;
tabs call computations/, behaviors/, curation/, and plotting/ — no heavy logic here.

Usage:
    streamlit run policy_doctor.streamlit_app.app
    # or from repo root:
    streamlit run policy_doctor.streamlit_app.app
"""

import pathlib
from typing import Optional

import streamlit as st

from policy_doctor.config import VisualizerConfig, list_configs, load_config
from policy_doctor.paths import REPO_ROOT

_REPO_ROOT = REPO_ROOT

# Session keys to drop when the sidebar task config changes (results + widget state).
_PD_TASK_SWITCH_RESET_KEYS: tuple[str, ...] = (
    # Clustering
    "clustering_labels",
    "clustering_metadata",
    "clustering_manifest",
    "clustering_viz_2d",
    "clustering_task_config",
    "clustering_params",
    "clustering_influence_source",
    "clustering_imported_config",
    "clustering_mode",
    "clust_task_cfg",
    "clust_result_select",
    "clust_ld_emb_source",
    "clust_ld_trak_split",
    "clust_ld_trak_level",
    "clust_ld_ww",
    "clust_ld_ww_na",
    "clust_ld_stride",
    "clust_ld_stride_na",
    "clust_ld_agg",
    "clust_ld_agg_na",
    "clust_ld_norm",
    "clust_ld_umap_dim",
    "clust_ld_umap_dim_na",
    "clust_ld_k",
    "clust_ld_umap_rs",
    "clust_ld_umap_rs_na",
    "clust_ld_algo",
    "clust_ld_repr",
    "clust_ld_nsamples",
    "clust_ld_nsamples_na",
    "clust_emb_source",
    "clust_trak_split",
    "clust_trak_level",
    "clust_ww",
    "clust_stride",
    "clust_agg",
    "clust_norm",
    "clust_umap_dim",
    "clust_k",
    "clust_umap_rs",
    "clust_color_by",
    "clust_export",
    "clust_exp_name",
    "sidebar_import",
    # Behavior graph
    "bg_graph",
    "bg_slice_labels",
    "bg_values",
    "bg_transition_values",
    "bg_q_values",
    "bg_advantages",
    "bg_params",
    "bg_task_cfg",
    "bg_clust_select",
    "bg_graph_view",
    "bg_graph_view_structure",
    "bg_graph_view_mrp",
    "bg_min_prob",
    "bg_min_prob_mrp",
    "bg_min_prob_prune",
    "bg_graph_view_prune",
    "bg_build_structure",
    "bg_prune_run",
    "bg_prune_rounds",
    "bg_prune_merges",
    "bg_compute_mrp",
    "bg_ep_num",
    "bg_gamma",
    "bg_r_success",
    "bg_r_failure",
    "bg_r_end",
    # Curation
    "cur_resolved",
    "cur_raw_selection",
    "cur_all_candidates",
    "cur_adv_threshold",
    "cur_pct",
    "cur_mode",
    "cur_ww",
    "cur_agg",
    "cur_topk",
    "cur_asc",
    # Comparison
    "cmp_baseline",
    "cmp_curated",
    "cmp_seeds",
    "cmp_mode",
    "cmp_baseline_pattern",
    "cmp_curated_pattern",
    "cmp_upload",
    # Sidebar pipeline YAML
    "imported_pipeline_config",
    # VLM: cached runs and clustering picker (prompt text areas use their own keys)
    "vlm_last_records",
    "vlm_last_prompt_version",
    "vlm_behavior_summaries",
    "vlm_behavior_prompt_version",
    "vlm_clust_name",
    "vlm_clust_dir_manual",
    "vlm_prompts_file",
    "vlm_prompts_sync_task",
    "vlm_task_hint",
    "vlm_user_tmpl",
    "vlm_sys_tmpl",
    # Runtime monitor
    "rm_df",
    "rm_path",
    "rm_threshold",
    "rm_episode",
    "rm_episode_list_idx",
    "rm_mode",
    "rm_timestep",
    "rm_intv_source",
    "rm_ep_filter",
    "rm_top_k",
    "rm_influence_window",
    "rm_agg",
    "rm_node_color_map",
    "rm_upload",
)


def _apply_task_config_change(config_name: str) -> None:
    """Clear task-scoped session state when the selected task config changes."""
    prev = st.session_state.get("_pd_active_task_config")
    if prev is not None and prev != config_name:
        for k in _PD_TASK_SWITCH_RESET_KEYS:
            st.session_state.pop(k, None)
        for k in list(st.session_state.keys()):
            if k.startswith("pd_cl_browse"):
                st.session_state.pop(k, None)
    st.session_state["_pd_active_task_config"] = config_name


def _render_config_selector() -> tuple[Optional[str], Optional[VisualizerConfig]]:
    """Render config selector in sidebar. Returns (config_name, config) or (None, None)."""
    st.sidebar.header("Configuration")
    config_names = list_configs()
    if not config_names:
        st.sidebar.error("No task configs found in package configs/*.yaml")
        return None, None
    selected = st.sidebar.selectbox(
        "Task config",
        options=config_names,
        index=0,
        help="Select a task configuration (YAML in configs/).",
    )
    config = load_config(selected)
    st.sidebar.caption(f"Task: {config.name}")
    if config.eval_dir:
        st.sidebar.caption(f"Eval: {config.eval_dir}")
    if config.train_dir:
        st.sidebar.caption(f"Train: {config.train_dir}")
    if config.use_mock:
        st.sidebar.info("Using mock data")
        return selected, config
    if not config.eval_dir or not config.train_dir:
        st.sidebar.warning("Set eval_dir and train_dir in the task config YAML to load data.")
        return selected, config
    return selected, config


@st.cache_resource
def _get_cached_data(_config_key: str, config: VisualizerConfig):
    """Load influence data (via policy_doctor.data.influence_loader). Cached by config key."""
    # Streamlit can keep old module objects alive across reruns; force a reload so that
    # schema changes to InfluenceDataContainer (e.g. adding frame accessors) take effect.
    import importlib

    import policy_doctor.data.influence_loader as _il

    _il = importlib.reload(_il)
    load_influence_data = _il.load_influence_data

    eval_dir = config.eval_dir
    train_dir = config.train_dir
    if not eval_dir or not train_dir:
        return None
    eval_path = pathlib.Path(eval_dir)
    train_path = pathlib.Path(train_dir)
    if not eval_path.is_absolute():
        eval_path = _REPO_ROOT / eval_path
    if not train_path.is_absolute():
        train_path = _REPO_ROOT / train_path
    return load_influence_data(
        eval_dir=str(eval_path),
        train_dir=str(train_path),
        train_ckpt=config.train_ckpt,
        exp_date=config.exp_date,
        include_holdout=True,
        image_dataset_path=config.image_dataset_path,
        lazy_load_images=config.lazy_load_images,
    )


def _render_sidebar_dataset_info(data) -> None:
    """Render dataset metrics in sidebar when data is loaded."""
    if data is None:
        return
    st.sidebar.divider()
    st.sidebar.header("Dataset")
    if st.sidebar.button("Clear cached influence data", key="clear_influence_cache"):
        # Reload influence data on next run (helps after code changes to InfluenceDataContainer)
        st.cache_resource.clear()
        st.rerun()
    # Match influence_visualizer/app.py::render_dataset_info (episode vs sample counts).
    n_train_demo_samples = len(data.demo_sample_infos)
    n_holdout_demo_samples = len(data.holdout_sample_infos)
    n_matrix_cols = int(data.influence_matrix.shape[1])

    st.sidebar.metric("Rollout episodes", len(data.rollout_episodes))
    st.sidebar.metric("Train demo episodes", len(data.demo_episodes))
    st.sidebar.metric("Holdout demo episodes", len(data.holdout_episodes))
    st.sidebar.metric("Rollout samples", data.influence_matrix.shape[0])
    st.sidebar.metric("Train demo samples", n_train_demo_samples)
    st.sidebar.metric("Holdout demo samples", n_holdout_demo_samples)
    st.sidebar.caption(
        "Train + holdout **demo samples** are TRAK **sequence indices** (horizon / pad / "
        "`create_indices`), same as the influence matrix columns — not raw HDF5 timesteps per episode."
    )
    if n_train_demo_samples + n_holdout_demo_samples != n_matrix_cols:
        st.sidebar.warning(
            f"Demo sample list length ({n_train_demo_samples + n_holdout_demo_samples}) ≠ "
            f"matrix columns ({n_matrix_cols}). See influence_visualizer load_influence_data warnings."
        )

    iv = getattr(data, "_iv_source", None)
    if iv is not None:
        st.sidebar.divider()
        st.sidebar.header("Config (from checkpoint)")
        for label, getter in (
            ("Horizon", lambda: iv.horizon),
            ("Pad Before", lambda: iv.pad_before),
            ("Pad After", lambda: iv.pad_after),
            ("N Obs Steps", lambda: iv.n_obs_steps),
        ):
            try:
                st.sidebar.text(f"{label}: {getter()}")
            except Exception:
                st.sidebar.text(f"{label}: —")

    with st.sidebar.expander("Debug (data source)", expanded=False):
        try:
            import policy_doctor
            from policy_doctor.data import influence_loader

            st.caption(f"`policy_doctor` from `{policy_doctor.__file__}`")
            st.caption(f"`influence_loader` from `{influence_loader.__file__}`")
        except Exception as e:
            st.caption(f"Import debug failed: {e}")
        st.caption(f"`data` type = `{type(data).__name__}`")
        st.caption(
            "frame accessors: "
            f"rollout={hasattr(data, 'get_rollout_frame')} "
            f"demo={hasattr(data, 'get_demo_frame')} "
            f"_iv_source={getattr(data, '_iv_source', None) is not None}"
        )
        if st.button("Clear Streamlit caches + rerun", key="pd_clear_caches"):
            try:
                st.cache_data.clear()
                st.cache_resource.clear()
            except Exception:
                pass
            st.rerun()


def _render_sidebar_config_import() -> None:
    """Render config import in sidebar for loading pipeline configs."""
    from policy_doctor.streamlit_app.config_io import (
        render_comparison_json_import,
        render_config_import,
    )

    st.sidebar.divider()
    st.sidebar.header("Pipeline Config")
    imported = render_config_import(
        label="Load pipeline config (YAML)",
        key="sidebar_import",
    )
    if imported is not None:
        st.session_state["imported_pipeline_config"] = imported
        st.session_state["clustering_imported_config"] = imported
        st.sidebar.success("Config loaded — values available in tabs")
        # Auto-populate clustering params if present
        for key_map in [
            ("clustering_window_width", "clust_ww"),
            ("clustering_stride", "clust_stride"),
            ("clustering_n_clusters", "clust_k"),
            ("clustering_umap_n_components", "clust_umap_dim"),
            ("advantage_threshold", "cur_adv_threshold"),
            ("selection_percentile", "cur_pct"),
            ("advantage_gamma", "bg_gamma"),
        ]:
            cfg_key, widget_key = key_map
            if cfg_key in imported:
                st.session_state[widget_key] = imported[cfg_key]

    st.sidebar.divider()
    st.sidebar.header("Comparison data")
    st.sidebar.caption("JSON with `baseline` and `curated` (seed → rate). Used in the Comparison tab.")
    render_comparison_json_import(key="cmp_upload")


def main() -> None:
    st.set_page_config(page_title="Policy Doctor", layout="wide")
    st.title("Policy Doctor")
    st.caption("Influence-based policy analysis and curation.")

    config_name, config = _render_config_selector()
    if config is None:
        st.info("Select a task config to continue.")
        return

    _apply_task_config_change(config_name)

    data = None
    if not config.use_mock and config.eval_dir and config.train_dir:
        try:
            # Bump this when the InfluenceDataContainer schema changes (cache bust).
            data_cache_version = "v2"
            config_key = f"{data_cache_version}_{config_name}_{config.eval_dir}_{config.train_dir}"
            data = _get_cached_data(config_key, config)
        except Exception as e:
            st.error(f"Failed to load influence data: {e}")
            if "influence_visualizer" in str(e):
                st.caption("Install influence_visualizer for this backend, or use a native loader.")
        if data is not None:
            _render_sidebar_dataset_info(data)

    _render_sidebar_config_import()

    (
        tab_clustering,
        tab_behavior,
        tab_vlm,
        tab_curation,
        tab_comparison,
        tab_mimicgen_eef,
        tab_runtime_monitor,
    ) = st.tabs([
        "Clustering",
        "Behavior Graph",
        "VLM annotation",
        "Curation",
        "Comparison",
        "MimicGen EEF",
        "Runtime Monitor",
    ])

    with tab_clustering:
        from policy_doctor.streamlit_app.tabs import clustering
        clustering.render_tab(config=config, data=data, task_config_stem=config_name)

    with tab_behavior:
        from policy_doctor.streamlit_app.tabs import behavior_graph
        behavior_graph.render_tab(config=config, data=data, task_config_stem=config_name)

    with tab_vlm:
        from policy_doctor.streamlit_app.tabs import vlm_annotation
        vlm_annotation.render_tab(config=config, data=data)

    with tab_curation:
        from policy_doctor.streamlit_app.tabs import curation
        curation.render_tab(config=config, data=data, task_config_stem=config_name)

    with tab_comparison:
        from policy_doctor.streamlit_app.tabs import comparison
        comparison.render_tab(config=config, data=data)

    with tab_mimicgen_eef:
        from policy_doctor.streamlit_app.tabs import mimicgen_eef
        mimicgen_eef.render_tab(config=config, data=data, task_config_stem=config_name)

    with tab_runtime_monitor:
        from policy_doctor.streamlit_app.tabs import runtime_monitor
        runtime_monitor.render_tab(config=config, data=data, task_config_stem=config_name)


if __name__ == "__main__":
    main()
