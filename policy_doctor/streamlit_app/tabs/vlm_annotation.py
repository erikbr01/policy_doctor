"""VLM slice annotation tab: configure backend/prompts, run subset, preview results."""

from __future__ import annotations

import datetime
import json
import pathlib
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from policy_doctor.config import VisualizerConfig
from policy_doctor.paths import REPO_ROOT

_REPO_ROOT = REPO_ROOT


def _abs_eval_dir(config: VisualizerConfig) -> Optional[pathlib.Path]:
    if not config.eval_dir:
        return None
    p = pathlib.Path(config.eval_dir)
    return p if p.is_absolute() else _REPO_ROOT / p


def _list_clustering_safe(task_config_name: str) -> List[str]:
    try:
        from policy_doctor.data.clustering_loader import list_clustering_results

        return list_clustering_results(task_config_name)
    except Exception:
        return []


def _clustering_path(task_config_name: str, name: str) -> pathlib.Path:
    from influence_visualizer.clustering_results import get_clustering_dir as iv_clustering_dir

    return iv_clustering_dir(task_config_name) / name


def render_tab(config: VisualizerConfig, data: Any = None) -> None:
    from policy_doctor import plotting
    from policy_doctor.vlm import (
        eval_dirs_equivalent,
        get_vlm_backend,
        list_vlm_backend_names,
        load_slice_annotations_jsonl,
        resolve_source_eval_dir_for_jsonl,
        run_behavior_summarization,
        run_slice_annotation_for_eval,
    )
    from policy_doctor.vlm.prompts import (
        DEFAULT_BEHAVIOR_USER_TEMPLATE,
        default_task_hint_for_vlm,
        merge_behavior_prompt_config,
        merge_prompt_config,
        resolve_vlm_prompts_file_for_task,
    )

    st.subheader("VLM slice annotation")
    st.caption(
        "Semantic labels for each clustering slice (sliding window) from rollout frames. "
        "Uses modular backends (mock / qwen2_vl / qwen3_vl / cosmos_reason2 / molmo / molmo2 / custom)."
    )

    with st.expander("Visualize pipeline output (load JSONL)", expanded=False):
        st.caption(
            "Load ``annotate_slices_vlm/annotations_seed*.jsonl`` from a pipeline run. "
            "Your task config **eval_dir** must match the rollouts used for that run so frame previews align."
        )
        default_suggest = (
            "data/pipeline_runs/mar27_pd_vlm_annotation_test/annotate_slices_vlm/annotations_seed0.jsonl"
        )
        jsonl_path = st.text_input(
            "Path to annotations JSONL (repo-relative or absolute)",
            placeholder=default_suggest,
            key="vlm_jsonl_path",
        )
        if st.button("Load into session", key="vlm_load_jsonl_btn"):
            raw = (jsonl_path or "").strip()
            if not raw:
                st.warning("Enter a path to the JSONL file.")
            else:
                p = pathlib.Path(raw)
                if not p.is_absolute():
                    p = _REPO_ROOT / p
                if not p.is_file():
                    st.error(f"File not found: {p}")
                else:
                    try:
                        loaded = load_slice_annotations_jsonl(p)
                    except Exception as e:
                        st.exception(e)
                    else:
                        st.session_state["vlm_last_records"] = loaded
                        pv = loaded[0].get("prompt_version", "") if loaded else ""
                        st.session_state["vlm_last_prompt_version"] = pv
                        st.session_state["vlm_loaded_source_eval_dir"] = (
                            resolve_source_eval_dir_for_jsonl(p, loaded)
                        )
                        st.success(f"Loaded {len(loaded)} rows from {p.name}")
                        st.rerun()

    eval_abs = _abs_eval_dir(config)
    if eval_abs is None or not eval_abs.is_dir():
        st.warning("Set a valid `eval_dir` in the task config (with episodes/ep*.pkl containing `img`).")
        return

    task_cfg_name = config.name
    if st.session_state.get("vlm_prompts_sync_task") != task_cfg_name:
        st.session_state["vlm_prompts_sync_task"] = task_cfg_name
        auto_pf = resolve_vlm_prompts_file_for_task(None, task_cfg_name, repo_root=_REPO_ROOT) or ""
        st.session_state["vlm_prompts_file"] = auto_pf
        hint = default_task_hint_for_vlm(None, task_cfg_name, repo_root=_REPO_ROOT)
        st.session_state["vlm_task_hint"] = hint or (config.task or task_cfg_name)

    clustering_names = _list_clustering_safe(task_cfg_name)
    c1, c2 = st.columns(2)
    with c1:
        if clustering_names:
            sel = st.selectbox("Clustering result", options=clustering_names, key="vlm_clust_name")
            clustering_dir = _clustering_path(task_cfg_name, sel)
        else:
            raw = st.text_input(
                "Clustering directory (absolute or repo-relative)",
                value="",
                key="vlm_clust_dir_manual",
                help="IV task clustering folder containing manifest.yaml, cluster_labels.npy, metadata.json",
            )
            clustering_dir = pathlib.Path(raw) if raw else None
            if clustering_dir and not clustering_dir.is_absolute():
                clustering_dir = _REPO_ROOT / clustering_dir
    with c2:
        backends = list_vlm_backend_names()
        backend = st.selectbox("VLM backend", options=backends, index=backends.index("mock") if "mock" in backends else 0)

    max_slices = st.number_input("Max slices (subset)", min_value=1, value=5, step=1)
    max_frames = st.number_input(
        "Max frames per slice (0 = all frames in window)",
        min_value=0,
        value=0,
        step=1,
        help="Matches Hydra: max_frames_per_slice null or ≤0 uses every timestep in the slice window.",
    )
    reasoning_effort = st.selectbox(
        "Slice reasoning effort",
        options=["none", "medium", "high"],
        index=2,
        help="High: chain-of-thought instructions + stored label from a line starting with FINAL:",
    )
    seed = st.number_input("Random seed (subset + frame sampling)", value=42, step=1)
    st.text_input(
        "Task hint for the model",
        key="vlm_task_hint",
        help="Filled from task VLM bundle when available (configs/vlm/tasks/). Edit freely.",
    )
    task_hint = str(st.session_state.get("vlm_task_hint") or config.task or task_cfg_name)

    st.text_input(
        "Prompts YAML path (repo-relative ok); empty = task bundle from registry",
        key="vlm_prompts_file",
        help="See policy_doctor/configs/vlm/tasks/registry.yaml — longest substring match on task config name.",
    )
    pf = str(st.session_state.get("vlm_prompts_file") or "").strip() or None
    st.caption(
        "Task-specific bundles live under `policy_doctor/policy_doctor/configs/vlm/tasks/`. "
        "Set `vlm_annotation.prompts_file` in Hydra to override the pipeline."
    )

    sys_default, user_default = merge_prompt_config(
        prompts_file=pf,
        prompts_inline=None,
        repo_root=_REPO_ROOT,
    )
    if "vlm_user_tmpl" not in st.session_state:
        st.session_state["vlm_user_tmpl"] = user_default
    if "vlm_sys_tmpl" not in st.session_state:
        st.session_state["vlm_sys_tmpl"] = sys_default or ""

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Reset slice prompts to task/file defaults"):
            s_d, u_d = merge_prompt_config(prompts_file=pf, prompts_inline=None, repo_root=_REPO_ROOT)
            st.session_state["vlm_user_tmpl"] = u_d
            st.session_state["vlm_sys_tmpl"] = s_d or ""
            st.rerun()
    with b2:
        if st.button("Reload prompts from YAML path above"):
            s_d, u_d = merge_prompt_config(prompts_file=pf, prompts_inline=None, repo_root=_REPO_ROOT)
            st.session_state["vlm_user_tmpl"] = u_d
            st.session_state["vlm_sys_tmpl"] = s_d or ""
            st.rerun()

    sys_t = st.text_area("Slice system prompt (optional)", value=st.session_state["vlm_sys_tmpl"], height=68)
    user_t = st.text_area(
        "Slice user prompt template",
        value=st.session_state["vlm_user_tmpl"],
        height=220,
        help="Placeholders: {task_hint}, {rollout_idx}, {window_start}, {window_end}, {cluster_id}, {num_frames}",
    )
    st.session_state["vlm_user_tmpl"] = user_t
    st.session_state["vlm_sys_tmpl"] = sys_t

    backend_params: Dict[str, Any] = {}
    if backend in ("qwen2_vl", "qwen3_vl", "cosmos_reason2"):
        with st.expander("Qwen2 / Qwen2.5 / Qwen3-VL / Cosmos-Reason2 parameters"):
            if backend == "cosmos_reason2":
                default_mid = "nvidia/Cosmos-Reason2-2B"
            elif backend == "qwen3_vl":
                default_mid = "Qwen/Qwen3-VL-8B-Instruct"
            else:
                default_mid = "Qwen/Qwen2.5-VL-7B-Instruct"
            backend_params["model_id"] = st.text_input(
                "model_id",
                value=default_mid,
                help="Cosmos-Reason2 is gated on Hugging Face (accept NVIDIA license). "
                "Qwen ids select Qwen2 / Qwen2.5 / Qwen3-VL from the name. bfloat16 is usually better than float16.",
            )
            backend_params["device"] = st.text_input("device", value="cuda:0")
            backend_params["torch_dtype"] = st.selectbox(
                "torch_dtype", options=["bfloat16", "float16", "float32"], index=0
            )
            default_tok = 2048 if backend == "cosmos_reason2" else 384
            backend_params["max_new_tokens"] = int(
                st.number_input("max_new_tokens", value=default_tok, min_value=16)
            )
            default_attn = "sdpa" if backend == "cosmos_reason2" else ""
            attn = st.text_input(
                "attn_implementation (optional, e.g. sdpa)",
                value=default_attn,
                help="Leave empty to use HF default; sdpa can lower VRAM on recent transformers.",
            )
            if attn.strip():
                backend_params["attn_implementation"] = attn.strip()
            if backend in ("qwen3_vl", "cosmos_reason2"):
                fi_label = (
                    "Frame input (Qwen3 / Cosmos; video mode requires pip install qwen-vl-utils; errors propagate)"
                )
                fi_choice = st.selectbox(
                    fi_label,
                    options=["images", "video"],
                    index=0,
                    format_func=lambda x: {
                        "images": "Separate images (bag of frames, default)",
                        "video": "Video-style (frame list + timestamps)",
                    }[x],
                    key="vlm_qwen_frame_input",
                )
                backend_params["qwen_frame_input"] = fi_choice
                backend_params["high_detail_pixels"] = st.checkbox(
                    "High-detail pixels (less aggressive resize; higher VRAM)",
                    value=True,
                    key="vlm_qwen_high_detail",
                )
                backend_params["video_sample_fps"] = float(
                    st.number_input(
                        "video_sample_fps (for temporal tokens)",
                        value=1.0,
                        min_value=0.25,
                        max_value=30.0,
                        step=0.25,
                        key="vlm_qwen_vid_fps",
                        help="Matches Qwen3-VL cookbook: FPS implied for the frame sequence.",
                    )
                )

    elif backend in ("molmo", "molmo2"):
        with st.expander("Molmo2 (Ai2) parameters"):
            backend_params["model_id"] = st.text_input(
                "model_id",
                value="allenai/Molmo2-4B",
                key="vlm_molmo_model",
                help="Molmo2 checkpoints use transformers AutoModelForImageTextToText (trust_remote_code). "
                "Try Molmo2-4B on ~24GB GPUs; Molmo2-8B needs more VRAM.",
            )
            backend_params["device"] = st.text_input(
                "device", value="cuda:0", key="vlm_molmo_dev"
            )
            backend_params["torch_dtype"] = st.selectbox(
                "torch_dtype",
                options=["bfloat16", "float16", "float32"],
                index=0,
                key="vlm_molmo_dtype",
            )
            backend_params["max_new_tokens"] = int(
                st.number_input("max_new_tokens", value=256, min_value=16, key="vlm_molmo_tok")
            )
            backend_params["images_before_text"] = st.checkbox(
                "Images before text (recommended for captioning)",
                value=True,
                key="vlm_molmo_img_first",
            )
            backend_params["do_sample"] = st.checkbox(
                "Sampling (reduces repeated canned captions vs greedy)",
                value=True,
                key="vlm_molmo_sample",
            )
            backend_params["temperature"] = float(
                st.slider("temperature (if sampling)", 0.1, 1.2, 0.4, 0.05, key="vlm_molmo_temp")
            )
            backend_params["top_p"] = float(
                st.slider("top_p (if sampling)", 0.5, 1.0, 0.9, 0.05, key="vlm_molmo_topp")
            )

    save_vlm_debug_pngs = st.checkbox(
        "Save debug PNGs (exact frames + prompts + model output on disk)",
        value=False,
        key="vlm_save_debug_pngs",
        help=f"Writes under `{_REPO_ROOT}/data/vlm_annotation_debug/gui_runs/<timestamp>/`.",
    )

    if st.button("Run annotation (subset)", type="primary"):
        if clustering_dir is None or not clustering_dir.is_dir():
            st.error("Select or enter a valid clustering directory.")
            return
        try:
            be = get_vlm_backend(backend, backend_params)
        except Exception as e:
            st.error(f"Backend failed: {e}")
            return
        prompts_inline = {
            "prompts": {"slice_system": sys_t or None, "slice_user_template": user_t},
            "reasoning_effort": reasoning_effort,
        }
        debug_plots_dir: Optional[pathlib.Path] = None
        if save_vlm_debug_pngs:
            root = _REPO_ROOT / "data" / "vlm_annotation_debug" / "gui_runs"
            root.mkdir(parents=True, exist_ok=True)
            debug_plots_dir = root / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_plots_dir.mkdir(parents=True, exist_ok=True)

        with st.spinner("Annotating…"):
            try:
                records, pver = run_slice_annotation_for_eval(
                    eval_dir=eval_abs,
                    clustering_dir=clustering_dir,
                    backend=be,
                    task_hint=task_hint,
                    prompts_file=pf,
                    prompts_inline=prompts_inline,
                    repo_root=_REPO_ROOT,
                    max_slices=int(max_slices),
                    max_frames_per_slice=(None if int(max_frames) <= 0 else int(max_frames)),
                    random_seed=int(seed),
                    debug_plots_dir=debug_plots_dir,
                )
            except Exception as e:
                st.exception(e)
                return
        st.session_state["vlm_last_records"] = records
        st.session_state["vlm_last_prompt_version"] = pver
        st.session_state["vlm_loaded_source_eval_dir"] = str(eval_abs.resolve())
        st.success(f"Annotated {len(records)} slices (prompt_version={pver}).")
        if debug_plots_dir is not None:
            st.info(f"Debug plots: `{debug_plots_dir}`")

    records = st.session_state.get("vlm_last_records")
    if records:
        src_ed = st.session_state.get("vlm_loaded_source_eval_dir")
        if src_ed and eval_abs is not None:
            if not eval_dirs_equivalent(pathlib.Path(src_ed), eval_abs):
                st.warning(
                    "**eval_dir mismatch:** these annotations were produced from rollouts under "
                    f"`{src_ed}`, but the selected task config uses `{eval_abs.resolve()}`. "
                    "Frame previews may not match the captions."
                )
        elif not src_ed:
            st.info(
                "Could not verify which rollout directory these annotations used "
                "(missing `source_eval_dir` in JSONL and no usable `result.json` next to the file). "
                "Re-run **annotate_slices_vlm** with an updated Policy Doctor, or keep `result.json` beside the JSONL."
            )
        st.caption(f"prompt_version: {st.session_state.get('vlm_last_prompt_version', '')}")
        df = pd.DataFrame(records)
        st.dataframe(df, width="stretch")

        idx = st.number_input("Preview slice row", min_value=0, max_value=len(records) - 1, value=0)
        rec = records[int(idx)]
        try:
            from policy_doctor.vlm.frames import extract_window_frames

            rng = np.random.default_rng(int(seed) + 17)
            imgs = extract_window_frames(
                eval_abs,
                int(rec["rollout_idx"]),
                int(rec["window_start"]),
                int(rec["window_end"]),
                max_frames=None,
                rng=rng,
            )
            strip_html = plotting.create_scrollable_frame_strip_html(
                imgs, thumb_height_px=168, gap_px=8
            )
            st.markdown(strip_html, unsafe_allow_html=True)
            st.caption(
                f"Cluster {rec['cluster_id']} — {len(imgs)} frame(s) in window "
                f"[{rec['window_start']}, {rec['window_end']}) (scroll horizontally). "
                f"{rec.get('label', '')[:400]}"
            )

            dl = json.dumps(records, indent=2, default=str)
            st.download_button("Download JSON", dl, file_name="vlm_slice_annotations.json")
        except Exception as e:
            st.warning(f"Could not render preview: {e}")

    st.divider()
    st.subheader("Behavior-level summarization")
    st.caption(
        "Aggregate slice captions by clustering ``cluster_id`` into one summary per behavior, "
        "including discrepancies when slice labels disagree."
    )

    if "vlm_behav_user_tmpl" not in st.session_state:
        _, bu = merge_behavior_prompt_config(
            prompts_file=pf,
            prompts_inline=None,
            repo_root=_REPO_ROOT,
        )
        st.session_state["vlm_behav_user_tmpl"] = bu
    if "vlm_behav_sys_tmpl" not in st.session_state:
        st.session_state["vlm_behav_sys_tmpl"] = ""

    if st.button("Reset behavior prompts to defaults", key="vlm_reset_behav_prompts"):
        st.session_state["vlm_behav_user_tmpl"] = DEFAULT_BEHAVIOR_USER_TEMPLATE
        st.session_state["vlm_behav_sys_tmpl"] = ""
        st.rerun()

    beh_sys = st.text_area(
        "Behavior summary system prompt (optional)",
        value=st.session_state["vlm_behav_sys_tmpl"],
        height=60,
        key="vlm_behav_sys_area",
    )
    beh_user = st.text_area(
        "Behavior summary user template",
        value=st.session_state["vlm_behav_user_tmpl"],
        height=200,
        key="vlm_behav_user_area",
        help="Placeholders: {task_hint}, {cluster_id}, {num_slice_labels}, {slice_labels_bulleted}",
    )
    st.session_state["vlm_behav_sys_tmpl"] = beh_sys
    st.session_state["vlm_behav_user_tmpl"] = beh_user

    max_beh_clusters = st.number_input(
        "Max clusters to summarize (0 = all)",
        min_value=0,
        value=0,
        step=1,
        key="vlm_max_beh_clusters",
    )
    max_labels_pc = st.number_input(
        "Max slice labels per cluster (0 = all)",
        min_value=0,
        value=0,
        step=1,
        key="vlm_max_labels_pc",
    )
    beh_backend = st.selectbox(
        "Backend for behavior summarization",
        options=list_vlm_backend_names(),
        index=list_vlm_backend_names().index("mock") if "mock" in list_vlm_backend_names() else 0,
        key="vlm_beh_backend",
        help="Uses the model's text-only path (no images). Same registry as slice annotation.",
    )
    beh_backend_params: Dict[str, Any] = {}
    if beh_backend in ("qwen2_vl", "qwen3_vl", "cosmos_reason2"):
        with st.expander("Qwen2 / Qwen2.5 / Qwen3-VL / Cosmos-Reason2 (behavior) parameters"):
            if beh_backend == "cosmos_reason2":
                _def_beh = "nvidia/Cosmos-Reason2-2B"
            elif beh_backend == "qwen3_vl":
                _def_beh = "Qwen/Qwen3-VL-8B-Instruct"
            else:
                _def_beh = "Qwen/Qwen2.5-VL-7B-Instruct"
            beh_backend_params["model_id"] = st.text_input(
                "model_id",
                value=_def_beh,
                key="vlm_beh_q_model",
            )
            beh_backend_params["device"] = st.text_input("device", value="cuda:0", key="vlm_beh_q_dev")
            beh_backend_params["torch_dtype"] = st.selectbox(
                "torch_dtype", options=["bfloat16", "float16", "float32"], index=0, key="vlm_beh_q_dtype"
            )
            _beh_tok = 2048 if beh_backend == "cosmos_reason2" else 512
            beh_backend_params["max_new_tokens"] = int(
                st.number_input("max_new_tokens", value=_beh_tok, min_value=16, key="vlm_beh_q_tok")
            )
            _beh_attn = "sdpa" if beh_backend == "cosmos_reason2" else ""
            beh_attn = st.text_input(
                "attn_implementation (optional)", value=_beh_attn, key="vlm_beh_q_attn"
            )
            if beh_attn.strip():
                beh_backend_params["attn_implementation"] = beh_attn.strip()

    elif beh_backend in ("molmo", "molmo2"):
        with st.expander("Molmo2 (behavior) parameters"):
            beh_backend_params["model_id"] = st.text_input(
                "model_id",
                value="allenai/Molmo2-4B",
                key="vlm_beh_molmo_model",
            )
            beh_backend_params["device"] = st.text_input(
                "device", value="cuda:0", key="vlm_beh_molmo_dev"
            )
            beh_backend_params["torch_dtype"] = st.selectbox(
                "torch_dtype",
                options=["bfloat16", "float16", "float32"],
                index=0,
                key="vlm_beh_molmo_dtype",
            )
            beh_backend_params["max_new_tokens"] = int(
                st.number_input("max_new_tokens", value=384, min_value=16, key="vlm_beh_molmo_tok")
            )

    if st.button("Summarize behaviors from slice results above", key="vlm_run_beh_sum"):
        recs = st.session_state.get("vlm_last_records")
        if not recs:
            st.warning("Run slice annotation first (or load results into session).")
        else:
            try:
                bbe = get_vlm_backend(beh_backend, beh_backend_params)
            except Exception as e:
                st.error(f"Backend failed: {e}")
            else:
                prompts_beh = {
                    "prompts": {
                        "behavior_system": beh_sys or None,
                        "behavior_user_template": beh_user,
                    }
                }
                with st.spinner("Summarizing behaviors…"):
                    try:
                        sums, bpver = run_behavior_summarization(
                            recs,
                            backend=bbe,
                            task_hint=task_hint,
                            prompts_file=pf,
                            prompts_inline=prompts_beh,
                            repo_root=_REPO_ROOT,
                            max_slice_labels_per_cluster=(
                                None if int(max_labels_pc) == 0 else int(max_labels_pc)
                            ),
                            max_clusters=None if int(max_beh_clusters) == 0 else int(max_beh_clusters),
                        )
                    except Exception as e:
                        st.exception(e)
                    else:
                        st.session_state["vlm_behavior_summaries"] = sums
                        st.session_state["vlm_behavior_prompt_version"] = bpver
                        st.success(f"Summarized {len(sums)} clusters (prompt_version={bpver}).")

    beh_sums = st.session_state.get("vlm_behavior_summaries")
    if beh_sums:
        st.caption(f"behavior_prompt_version: {st.session_state.get('vlm_behavior_prompt_version', '')}")
        st.dataframe(pd.DataFrame(beh_sums), width="stretch")
        dl2 = json.dumps(beh_sums, indent=2, default=str)
        st.download_button(
            "Download behavior summaries JSON",
            dl2,
            file_name="vlm_behavior_summaries.json",
            key="vlm_dl_beh",
        )
