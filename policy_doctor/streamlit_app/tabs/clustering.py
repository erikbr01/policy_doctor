"""Clustering tab: run or load clustering, visualize, export config.

Workflow:
  1. Choose InfEmbed (rollout timestep embeddings) or TRAK (influence matrix) as the slice source
  2. Set UMAP / clustering params in the sidebar
  3. Run clustering interactively or load saved results
  4. Visualize: scatter plot, cluster sizes, silhouette, timeline
  5. Export clustering config as YAML for the pipeline
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

from policy_doctor.config import VisualizerConfig
from policy_doctor.streamlit_app.config_io import (
    merge_clustering_params_for_display,
    resolve_task_config_stem,
    sync_clustering_session_from_manifest,
)
from policy_doctor.paths import REPO_ROOT

_REPO_ROOT = REPO_ROOT


def _resolve_eval_path(eval_dir: str) -> pathlib.Path:
    eval_path = pathlib.Path(eval_dir)
    if not eval_path.is_absolute():
        eval_path = _REPO_ROOT / eval_path
    return eval_path


_AGG_OPTIONS = ("sum", "mean", "max", "min", "std", "median")
_NORM_OPTIONS = ("none", "l2", "standard", "minmax", "robust")
_DEMO_SPLITS = ("train", "holdout", "both")
_LEVELS = ("rollout", "demo")


def _selectbox_index(options: tuple, value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return options.index(value)
    except ValueError:
        return default


def _render_saved_clustering_parameters(
    merged: Dict[str, Any],
    manifest: Optional[Dict[str, Any]],
    config: VisualizerConfig,
) -> None:
    """Same layout as interactive clustering, but widgets are disabled (loaded run)."""
    st.subheader("Clustering parameters")
    st.caption("Read-only — values from the loaded run (manifest + any session fields).")
    if manifest and manifest.get("created"):
        st.caption(f"Saved: `{manifest.get('created')}`")

    inf = str(merged.get("clustering_influence_source", "infembed")).lower()
    use_infembed = inf != "trak"

    st.radio(
        "Embedding source",
        [
            "InfEmbed (rollout timestep embeddings)",
            "TRAK (influence / explanations matrix)",
        ],
        index=0 if use_infembed else 1,
        key="clust_ld_emb_source",
        disabled=True,
        help="From saved manifest (`influence_source`).",
    )

    eval_dir = config.eval_dir
    if eval_dir:
        eval_path = _resolve_eval_path(eval_dir)
        if use_infembed:
            st.caption(f"Eval dir (task config): `{eval_path}`")
        else:
            st.caption(f"Eval dir (task config): `{eval_path}` — TRAK windows use loaded influence data.")

    if not use_infembed:
        _trak_split_labels = {
            "train": "Train demos only (train-split columns)",
            "holdout": "Holdout demos only",
            "both": "Train + holdout (all demo columns)",
        }
        col_ts, col_tl = st.columns(2)
        demo_split = merged.get("clustering_demo_split", "train")
        level = merged.get("clustering_level", "rollout")
        with col_ts:
            st.selectbox(
                "Demonstration columns (TRAK)",
                list(_DEMO_SPLITS),
                index=_selectbox_index(_DEMO_SPLITS, demo_split),
                format_func=lambda k: str(_trak_split_labels.get(k, k)),
                key="clust_ld_trak_split",
                disabled=True,
            )
        with col_tl:
            st.selectbox(
                "Window level (TRAK)",
                list(_LEVELS),
                index=_selectbox_index(_LEVELS, level),
                key="clust_ld_trak_level",
                disabled=True,
            )

    st.markdown("##### Windowing & clustering")
    col1, col2 = st.columns(2)
    with col1:
        ww = merged.get("window_width")
        if ww is not None:
            st.number_input("Window width", 1, 50, int(ww), key="clust_ld_ww", disabled=True)
        else:
            st.text_input(
                "Window width",
                value="(not stored in manifest)",
                key="clust_ld_ww_na",
                disabled=True,
            )
        stride_val = merged.get("stride")
        if stride_val is not None:
            st.number_input("Stride", 1, 20, int(stride_val), key="clust_ld_stride", disabled=True)
        else:
            st.text_input("Stride", value="(not stored in manifest)", key="clust_ld_stride_na", disabled=True)
        agg = merged.get("aggregation")
        if agg is not None:
            st.selectbox(
                "Aggregation",
                list(_AGG_OPTIONS),
                index=_selectbox_index(_AGG_OPTIONS, agg),
                key="clust_ld_agg",
                disabled=True,
            )
        else:
            st.text_input("Aggregation", value="(not stored in manifest)", key="clust_ld_agg_na", disabled=True)
    with col2:
        norm = merged.get("normalize", "none")
        st.selectbox(
            "Normalization",
            list(_NORM_OPTIONS),
            index=_selectbox_index(_NORM_OPTIONS, norm),
            key="clust_ld_norm",
            disabled=True,
        )
        umap_d = merged.get("umap_n_components")
        if umap_d is not None:
            st.number_input("UMAP target dim", 2, 500, int(umap_d), key="clust_ld_umap_dim", disabled=True)
        else:
            st.text_input(
                "UMAP target dim",
                value="(not stored in manifest)",
                key="clust_ld_umap_dim_na",
                disabled=True,
            )
        _nk = max(2, min(100, int(merged.get("n_clusters", 20))))
        st.number_input(
            "K-Means clusters",
            2,
            100,
            _nk,
            key="clust_ld_k",
            disabled=True,
        )

    with st.expander("Advanced", expanded=False):
        rs = merged.get("umap_random_state")
        if rs is not None:
            st.number_input("UMAP random state", 0, 9999, int(rs), key="clust_ld_umap_rs", disabled=True)
        else:
            st.text_input(
                "UMAP random state",
                value="(not stored in manifest)",
                key="clust_ld_umap_rs_na",
                disabled=True,
            )
        if manifest:
            st.text_input(
                "Algorithm (manifest)",
                value=str(manifest.get("algorithm", "—")),
                key="clust_ld_algo",
                disabled=True,
            )
            st.text_input(
                "Representation (manifest)",
                value=str(manifest.get("representation", "—")),
                key="clust_ld_repr",
                disabled=True,
            )
            ns = manifest.get("n_samples")
            if ns is not None:
                st.number_input(
                    "Samples (manifest)",
                    min_value=0,
                    value=int(ns),
                    key="clust_ld_nsamples",
                    disabled=True,
                )
            else:
                st.text_input("Samples (manifest)", value="—", key="clust_ld_nsamples_na", disabled=True)


@st.cache_data
def _infembed_slice_windows_cached(
    eval_dir_resolved: str,
    window_width: int,
    stride: int,
    aggregation: str,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    from policy_doctor.data.clustering_embeddings import extract_infembed_slice_windows

    return extract_infembed_slice_windows(
        pathlib.Path(eval_dir_resolved),
        window_width,
        stride,
        aggregation,
    )


@st.cache_data
def _run_clustering_cached(
    cache_key: str,
    window_embeddings: np.ndarray,
    normalize: str,
    umap_n_components: int,
    n_clusters: int,
    umap_random_state: int,
) -> tuple:
    """UMAP + K-Means on pre-windowed slice embeddings (cached by params)."""
    from policy_doctor.behaviors.clustering import (
        cluster_kmeans,
        normalize_embeddings,
        reduce_dimensions,
    )

    windows = np.asarray(window_embeddings, dtype=np.float32)
    if normalize != "none":
        windows = normalize_embeddings(windows, method=normalize)
    reduced = reduce_dimensions(
        windows,
        method="umap",
        n_components=umap_n_components,
        random_state=umap_random_state,
    )
    labels = cluster_kmeans(reduced, n_clusters=n_clusters)

    from sklearn.manifold import TSNE

    if reduced.shape[1] > 2:
        viz_2d = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, len(reduced) - 1),
        ).fit_transform(reduced)
    else:
        viz_2d = reduced[:, :2]

    return labels, reduced, viz_2d, windows


def render_tab(
    config: VisualizerConfig,
    data: Any,
    task_config_stem: Optional[str] = None,
) -> None:
    """Render the Clustering tab."""
    st.header("Clustering")
    task_stem = resolve_task_config_stem(config, task_config_stem)

    mode = st.radio(
        "Mode",
        ["Run new clustering", "Load existing result"],
        horizontal=True,
        key="clustering_mode",
    )

    if mode == "Load existing result":
        _render_load_existing(config, data, task_stem)
    else:
        _render_run_new(config, data, task_stem)


def _render_load_existing(config: VisualizerConfig, data: Any, task_stem: str) -> None:
    """Load and visualize an existing clustering result."""
    from policy_doctor.streamlit_app.config_io import (
        clustering_roots_for_task,
        clustering_results_dir_for_task,
        discover_clustering_task_keys,
        list_clustering_results,
    )

    task_config = st.text_input(
        "Task config key",
        value=task_stem,
        key="clust_task_cfg",
        help="Same as the task YAML filename stem (e.g. transport_mh_jan28). Must match where clustering "
        "was saved (see path hint below if empty).",
    )
    try:
        available = list_clustering_results(task_config)
    except Exception as e:
        st.error(f"Could not list clustering results: {e}")
        available = []

    scan_roots = clustering_roots_for_task(task_config)
    if len(scan_roots) > 1:
        st.caption(
            f"Merged {len(available)} run(s) from {len(scan_roots)} clustering folders "
            "(installed `influence_visualizer` + monorepo `influence_visualizer/configs` when both exist)."
        )

    if available:
        selected = st.selectbox("Clustering result", available, key="clust_result_select")
        if st.button("Load", key="clust_load_btn"):
            from policy_doctor.streamlit_app.config_io import load_task_clustering_result

            labels, metadata, manifest = load_task_clustering_result(task_config, selected)
            st.session_state["clustering_labels"] = labels
            st.session_state["clustering_metadata"] = metadata
            st.session_state["clustering_manifest"] = manifest
            st.session_state["clustering_task_config"] = task_config
            sync_clustering_session_from_manifest(manifest)
            st.success(f"Loaded {selected}: {len(labels)} slices, {len(set(labels) - {-1})} clusters")
    else:
        expect = clustering_results_dir_for_task(task_config)
        discovered = discover_clustering_task_keys()
        st.info(
            f"No clustering results for key `{task_config}`. "
            f"Each run must be a subdirectory with `manifest.yaml` and `cluster_labels.npy`. "
            f"Primary location: `{expect}`."
        )
        if scan_roots:
            st.caption("Searched clustering roots: " + " · ".join(str(r) for r in scan_roots))
        else:
            st.caption(
                "No `clustering` folder found for this task (neither under the installed "
                "influence_visualizer package nor under `<repo>/influence_visualizer/configs/`)."
            )
        if discovered:
            st.caption("Task keys with at least one saved run: " + ", ".join(discovered))

    st.caption(
        "To import a pipeline/clustering YAML, use **Pipeline Config** → *Load pipeline config (YAML)* in the sidebar."
    )
    imported_cfg = st.session_state.get("imported_pipeline_config")
    if imported_cfg is not None:
        with st.expander("Imported YAML (from sidebar)", expanded=False):
            st.json(imported_cfg)

    if "clustering_labels" in st.session_state:
        ld_manifest = st.session_state.get("clustering_manifest")
        merged_ld = merge_clustering_params_for_display(ld_manifest, st.session_state.get("clustering_params"))
        _render_saved_clustering_parameters(merged_ld, ld_manifest, config)
        _render_visualizations(
            st.session_state["clustering_labels"],
            st.session_state["clustering_metadata"],
            viz_2d=None,
            data=data,
            config=config,
        )


def _render_run_new(config: VisualizerConfig, data: Any, task_stem: str) -> None:
    """Run clustering interactively with parameter controls."""
    if data is None and not config.eval_dir:
        st.info("Load a task config with eval_dir (and train_dir for TRAK) to run clustering.")
        return

    embedding_source = st.radio(
        "Embedding source",
        [
            "InfEmbed (rollout timestep embeddings)",
            "TRAK (influence / explanations matrix)",
        ],
        key="clust_emb_source",
        help=(
            "InfEmbed: sliding windows over vectors in ``default_trak_results-*/infembed_embeddings.npz`` "
            "(requires ``compute_infembed``). TRAK: sliding windows over the loaded influence matrix vs demos "
            "(same signal as pipeline ``clustering_influence_source=trak``)."
        ),
    )
    use_infembed = embedding_source.startswith("InfEmbed")

    eval_dir = config.eval_dir
    if not eval_dir:
        st.warning("eval_dir is required.")
        return
    eval_path = _resolve_eval_path(eval_dir)

    if use_infembed:
        st.caption(f"Eval dir: `{eval_path}` (latest ``default_trak_results-*`` + ``infembed_embeddings.npz``)")
    else:
        if data is None:
            st.warning(
                "TRAK mode needs influence data. Set **train_dir** and **eval_dir** in the task YAML so the "
                "app can load the matrix (same as other tabs)."
            )
            return
        st.caption(
            "Influence matrix shape is **(rollout timesteps × demo timesteps)**. "
            "Clustering still windows along **rollouts** (or along demos if you pick demo-level windows); "
            "the option below only changes which **demonstration** columns are kept."
        )

    demo_split = "train"
    level = "rollout"
    if not use_infembed:
        _trak_split_labels = {
            "train": "Train demos only (train-split columns)",
            "holdout": "Holdout demos only",
            "both": "Train + holdout (all demo columns)",
        }
        col_ts, col_tl = st.columns(2)
        with col_ts:
            demo_split = st.selectbox(
                "Demonstration columns (TRAK)",
                ["train", "holdout", "both"],
                index=0,
                format_func=lambda k: _trak_split_labels[k],
                key="clust_trak_split",
                help=(
                    "Splits refer to the **dataset split** used when building the replay buffer, "
                    "not the eval rollouts. **Train** = only columns for demonstrations in the training split "
                    "(influence of each rollout timestep on train demos — usual choice for curation). "
                    "**Holdout** = holdout demo columns only. **Train + holdout** = full matrix."
                ),
            )
        with col_tl:
            level = st.selectbox(
                "Window level (TRAK)",
                ["rollout", "demo"],
                key="clust_trak_level",
                help="Rollout: window along time per rollout episode. Demo: window along demos per timestep.",
            )

    st.subheader("Parameters")
    col1, col2 = st.columns(2)
    with col1:
        window_width = st.number_input("Window width", 1, 50, 5, key="clust_ww")
        stride = st.number_input("Stride", 1, 20, 2, key="clust_stride")
        aggregation = st.selectbox(
            "Aggregation",
            ["sum", "mean", "max", "min", "std", "median"],
            key="clust_agg",
        )
    with col2:
        normalize = st.selectbox(
            "Normalization",
            ["none", "l2", "standard", "minmax", "robust"],
            key="clust_norm",
        )
        umap_n_components = st.number_input("UMAP target dim", 2, 500, 100, key="clust_umap_dim")
        n_clusters = st.number_input("K-Means clusters", 2, 100, 20, key="clust_k")

    with st.expander("Advanced"):
        umap_random_state = st.number_input("UMAP random state", 0, 9999, 42, key="clust_umap_rs")

    influence_tag = "infembed" if use_infembed else "trak"

    if st.button("Run clustering", type="primary", key="clust_run"):
        try:
            with st.spinner("Building slice embeddings..."):
                if use_infembed:
                    window_emb, metadata = _infembed_slice_windows_cached(
                        str(eval_path.resolve()),
                        int(window_width),
                        int(stride),
                        aggregation,
                    )
                else:
                    from policy_doctor.data.clustering_embeddings import (
                        extract_trak_slice_windows_from_container,
                    )

                    window_emb, metadata = extract_trak_slice_windows_from_container(
                        data,
                        int(window_width),
                        int(stride),
                        aggregation,
                        demo_split=demo_split,
                        level=level,
                    )
        except FileNotFoundError as e:
            st.error(str(e))
            return
        except ValueError as e:
            st.error(str(e))
            return

        cache_key = (
            f"{influence_tag}_{eval_path}_{window_emb.shape}_{normalize}_"
            f"{umap_n_components}_{n_clusters}_{umap_random_state}"
        )
        with st.spinner("Running UMAP + K-Means..."):
            labels, reduced, viz_2d, _windows = _run_clustering_cached(
                cache_key,
                window_emb,
                normalize,
                umap_n_components,
                n_clusters,
                umap_random_state,
            )

        st.session_state["clustering_labels"] = labels
        st.session_state["clustering_metadata"] = metadata
        st.session_state["clustering_viz_2d"] = viz_2d
        st.session_state.pop("clustering_manifest", None)
        st.session_state["clustering_task_config"] = task_stem
        st.session_state["clustering_influence_source"] = influence_tag
        st.session_state["clustering_params"] = {
            "window_width": window_width,
            "stride": stride,
            "aggregation": aggregation,
            "normalize": normalize,
            "umap_n_components": umap_n_components,
            "n_clusters": n_clusters,
            "umap_random_state": umap_random_state,
            "clustering_influence_source": influence_tag,
            "clustering_demo_split": demo_split,
            "clustering_level": level,
        }
        n_actual = len(set(labels) - {-1})
        st.success(
            f"Done: {n_actual} clusters, {len(labels)} slices "
            f"(source={influence_tag}, slice matrix {window_emb.shape})"
        )

    if "clustering_labels" in st.session_state:
        _render_visualizations(
            st.session_state["clustering_labels"],
            st.session_state["clustering_metadata"],
            st.session_state.get("clustering_viz_2d"),
            data=data,
            config=config,
        )
        _render_export(config, task_stem)


def _render_visualizations(
    labels: np.ndarray,
    metadata: List[Dict],
    viz_2d: Optional[np.ndarray],
    data: Any = None,
    config: Optional[VisualizerConfig] = None,
) -> None:
    """Render cluster visualizations."""
    from policy_doctor import plotting

    st.subheader("Cluster Visualization")

    if viz_2d is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            color_by = st.selectbox(
                "Color by",
                ["cluster", "rollout_idx", "success"],
                key="clust_color_by",
            )
            fig = plotting.create_cluster_scatter_2d(
                viz_2d,
                labels,
                metadata,
                color_by=color_by,
                title="Cluster Scatter (t-SNE of UMAP embeddings)",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            cluster_ids = sorted(set(labels) - {-1})
            stats = [{"cluster_id": c, "size": int((labels == c).sum())} for c in cluster_ids]
            fig_sizes = plotting.create_cluster_size_chart(stats, title="Cluster Sizes")
            st.plotly_chart(fig_sizes, use_container_width=True)
    else:
        cluster_ids = sorted(set(labels) - {-1})
        stats = [{"cluster_id": c, "size": int((labels == c).sum())} for c in cluster_ids]
        fig_sizes = plotting.create_cluster_size_chart(stats, title="Cluster Sizes")
        st.plotly_chart(fig_sizes, use_container_width=True)

    with st.expander("Cluster quality metrics"):
        try:
            from sklearn.metrics import silhouette_samples

            if viz_2d is not None and len(set(labels)) > 1:
                valid = labels >= 0
                sil = silhouette_samples(viz_2d[valid], labels[valid])
                fig_sil = plotting.create_silhouette_plot(
                    sil,
                    labels[valid],
                    title="Silhouette Plot",
                )
                st.plotly_chart(fig_sil, use_container_width=True)
                st.metric("Mean silhouette", f"{sil.mean():.3f}")
            else:
                st.info("Need 2D embeddings and >1 cluster for silhouette plot.")
        except ImportError:
            st.warning("Install scikit-learn for silhouette metrics.")

    with st.expander("Cluster summary table"):
        import pandas as pd

        cluster_ids = sorted(set(labels) - {-1})
        rows = []
        for c in cluster_ids:
            mask = labels == c
            ep_indices = set()
            success_count = 0
            for i, m in enumerate(metadata):
                if mask[i]:
                    ep_indices.add(m.get("rollout_idx", m.get("demo_idx", -1)))
                    if m.get("success"):
                        success_count += 1
            rows.append(
                {
                    "Cluster": c,
                    "Slices": int(mask.sum()),
                    "Episodes": len(ep_indices),
                    "Success slices": success_count,
                    "Success %": f"{100 * success_count / max(mask.sum(), 1):.1f}%",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if data is not None:
        with st.expander("Episode video browser (cluster timeline + frames)", expanded=False):
            params = st.session_state.get("clustering_params") or {}
            level = str(params.get("clustering_level", "rollout"))
            demo_split = str(params.get("clustering_demo_split", "train"))
            ann_dict = None
            if config is not None and getattr(config, "annotation_file", None):
                try:
                    from pathlib import Path

                    from influence_visualizer.render_annotation import load_annotations

                    ap = Path(config.annotation_file)
                    if not ap.is_absolute():
                        ap = _REPO_ROOT / ap
                    task_key = st.session_state.get("clustering_task_config") or getattr(
                        config, "config_stem", None
                    ) or config.name
                    ann_dict = load_annotations(str(ap), task_config=str(task_key))
                except Exception:
                    ann_dict = None
            from policy_doctor.streamlit_app.clustering_browse import (
                render_cluster_episode_browser,
            )

            render_cluster_episode_browser(
                data,
                labels,
                metadata,
                level=level,
                demo_split=demo_split,
                key_prefix="pd_cl_vid",
                annotations=ann_dict,
            )


def _render_export(config: VisualizerConfig, task_stem: str) -> None:
    """Render config export controls."""
    from policy_doctor.streamlit_app.config_io import render_config_export, render_save_to_disk

    st.subheader("Export")
    params = st.session_state.get("clustering_params", {})
    task_config = st.session_state.get("clustering_task_config", task_stem)
    influence_src = st.session_state.get(
        "clustering_influence_source",
        params.get("clustering_influence_source", "infembed"),
    )

    pipeline_config = {
        "task_config": task_config,
        "clustering_influence_source": influence_src,
        "clustering_window_width": params.get("window_width", 5),
        "clustering_stride": params.get("stride", 2),
        "clustering_aggregation": params.get("aggregation", "sum"),
        "clustering_normalize": params.get("normalize", "none"),
        "clustering_umap_n_components": params.get("umap_n_components", 100),
        "clustering_n_clusters": params.get("n_clusters", 20),
        "clustering_demo_split": params.get("clustering_demo_split", "train"),
        "clustering_level": params.get("clustering_level", "rollout"),
    }

    col1, col2 = st.columns(2)
    with col1:
        render_config_export(
            pipeline_config,
            default_filename=f"clustering_{task_config}.yaml",
            label="Download clustering config",
            key="clust_export",
        )
    with col2:
        experiment_name = st.text_input("Experiment name", value="interactive", key="clust_exp_name")

    if st.button("Save clustering result to disk", key="clust_save_disk"):
        labels = st.session_state["clustering_labels"]
        metadata = st.session_state["clustering_metadata"]
        n_clusters = int(len(set(labels) - {-1}))
        name = f"{experiment_name}_kmeans_k{n_clusters}"
        try:
            from influence_visualizer.clustering_results import save_clustering_result

            result_dir = save_clustering_result(
                task_config=task_config,
                name=name,
                cluster_labels=labels,
                metadata=metadata,
                algorithm="kmeans",
                scaling=params.get("normalize", "none"),
                influence_source=influence_src,
                representation="sliding_window",
                level=params.get("clustering_level", "rollout"),
                n_clusters=n_clusters,
                n_samples=len(labels),
            )
            st.success(f"Saved to `{result_dir}`")
        except Exception as e:
            st.error(f"Failed to save: {e}")
