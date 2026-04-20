"""Generate curation configs — pipeline step class."""

from __future__ import annotations

import pathlib
from typing import List

import yaml
from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.paths import PACKAGE_ROOT, REPO_ROOT, iv_task_configs_base

_REPO_ROOT = REPO_ROOT


class RunCurationConfigStep(PipelineStep[List[str]]):
    """Build curation configs from clustering + influence data for each seed.

    Result: list of saved config paths (as strings).

    Dependency resolution: if ``cfg.clustering_dir`` is not set, the step
    looks up the saved result of :class:`RunClusteringStep` from the run
    folder automatically.
    """

    name = "run_curation_config"

    def save(self, result: List[str]) -> None:
        import json

        self.step_dir.mkdir(parents=True, exist_ok=True)
        with open(self.step_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)
        (self.step_dir / "done").touch()

    def load(self) -> List[str]:
        import json

        p = self.step_dir / "result.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return []

    def compute(self) -> List[str]:
        cfg = self.cfg

        pipeline_yaml = PACKAGE_ROOT / "configs" / "pipeline" / "config.yaml"
        if pipeline_yaml.exists():
            with open(pipeline_yaml) as f:
                base_cfg = yaml.safe_load(f) or {}
        else:
            base_cfg = {}

        flat = {**base_cfg, **{k: v for k, v in OmegaConf.to_container(cfg, resolve=True).items()}}

        if self.dry_run:
            print(f"[dry_run] RunCurationConfigStep task_config={flat.get('task_config')}")
            return []

        # Resolve clustering dirs: prefer build_behavior_graph if available, then
        # explicit cfg value, then run_clustering output.
        from policy_doctor.curation_pipeline.steps.build_behavior_graph import BuildBehaviorGraphStep

        bbg_prior = BuildBehaviorGraphStep(cfg, self.run_dir).load()
        bbg_step_dir = self.run_dir / "build_behavior_graph"

        clustering_dir = flat.get("clustering_dir")
        bbg_dirs_map: dict = {}  # {seed: {"node_assignments_path": ..., "metadata_path": ...}}
        clustering_dirs_map: dict = {}

        if bbg_prior and bbg_prior.get("seeds"):
            # build_behavior_graph has run — use its per-seed node_assignments
            for seed_key, seed_info in bbg_prior["seeds"].items():
                if isinstance(seed_info, dict) and seed_info.get("node_assignments_path"):
                    bbg_dirs_map[seed_key] = {
                        "node_assignments_path": str(bbg_step_dir / seed_info["node_assignments_path"]),
                        "metadata_path": str(bbg_step_dir / seed_info["metadata_path"]),
                        "level": seed_info.get("level", "rollout"),
                    }
        elif not clustering_dir:
            from policy_doctor.curation_pipeline.steps.run_clustering import RunClusteringStep

            prior = RunClusteringStep(cfg, self.run_dir).load()
            if prior:
                clustering_dirs_map = prior.get("clustering_dirs", {})
        if clustering_dir:
            clustering_dirs_map = {}

        policy_seeds = flat.get("policy_seeds", [0, 1, 2])
        if isinstance(policy_seeds, (int, float)):
            policy_seeds = [policy_seeds]
        seeds = [str(s) for s in policy_seeds]

        out_paths: List[str] = []
        if bbg_dirs_map:
            # ENAP or build_behavior_graph path: pass node_assignment paths directly
            for seed in seeds:
                seed_info = (
                    bbg_dirs_map.get(seed)
                    or bbg_dirs_map.get("_single")
                    or bbg_dirs_map.get("_enap")
                )
                if not seed_info:
                    continue
                seed_cfg = dict(flat)
                seed_cfg["policy_seeds"] = [int(seed) if seed.isdigit() else 0]
                seed_cfg["_bbg_node_assignments_path"] = seed_info["node_assignments_path"]
                seed_cfg["_bbg_metadata_path"] = seed_info["metadata_path"]
                seed_cfg["_bbg_level"] = seed_info.get("level", "rollout")
                seed_cfg["repo_root"] = str(self.repo_root)
                paths = run_pipeline_from_config(seed_cfg)
                out_paths.extend(str(p) for p in paths)
        elif clustering_dirs_map:
            for seed in seeds:
                seed_cfg = dict(flat)
                seed_cfg["policy_seeds"] = [int(seed)]
                cdir = clustering_dirs_map.get(seed)
                if cdir:
                    seed_cfg["clustering_dir"] = cdir
                paths = run_pipeline_from_config(seed_cfg)
                out_paths.extend(str(p) for p in paths)
        else:
            flat["repo_root"] = str(self.repo_root)
            paths = run_pipeline_from_config(flat)
            out_paths.extend(str(p) for p in paths)

        return out_paths


# ---------------------------------------------------------------------------
# Core implementation (unchanged logic, extracted from prior module)
# ---------------------------------------------------------------------------

def run_pipeline_from_config(cfg: dict) -> list:
    """Run the full curation-config pipeline for each policy seed.

    Returns list of paths to saved curation configs.
    """
    import numpy as np

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable=None, total=None, desc="", unit="step", **kwargs):
            if iterable is not None:
                return iterable

            class _DummyBar:
                def update(self, n=1): pass
                def set_description(self, s): pass
                def __enter__(self): return self
                def __exit__(self, *a): pass

            return _DummyBar()

    task_config = cfg["task_config"]
    config_root = cfg.get("config_root", "iv")
    clustering_name = cfg.get("clustering_name")
    clustering_dir = cfg.get("clustering_dir")
    if not clustering_dir and not clustering_name:
        raise ValueError(
            "clustering_name or clustering_dir is required. "
            "Pass clustering_dir=<path> or include run_clustering as a prior step."
        )
    repo_root = pathlib.Path(cfg.get("repo_root") or _REPO_ROOT).resolve()
    if clustering_dir:
        clustering_dir = pathlib.Path(clustering_dir)
        if not clustering_dir.is_absolute():
            clustering_dir = repo_root / clustering_dir

    if config_root == "iv":
        base = iv_task_configs_base(repo_root)
    else:
        base = PACKAGE_ROOT / "configs"
    task_yaml = base / f"{task_config}.yaml"
    if not task_yaml.exists():
        raise FileNotFoundError(f"Task config not found: {task_yaml}")

    with open(task_yaml) as f:
        task_cfg = yaml.safe_load(f)
    eval_dir = cfg.get("eval_dir") or task_cfg.get("eval_dir")
    train_dir = cfg.get("train_dir") or task_cfg.get("train_dir")
    if not eval_dir or not train_dir:
        raise ValueError("eval_dir and train_dir required (from task config or cfg override)")

    policy_seeds = cfg.get("policy_seeds", [0, 1, 2])
    if isinstance(policy_seeds, (int, float)):
        policy_seeds = [policy_seeds]
    seeds = [str(s) for s in policy_seeds]
    reference_seed = str(cfg.get("reference_seed", 0))

    from policy_doctor.data.path_utils import get_eval_dir_for_seed, get_train_dir_for_seed
    from policy_doctor.data.influence_loader import load_influence_data
    from policy_doctor.data.clustering_loader import (
        load_clustering_result_from_path,
        load_clustering_result,
    )

    # Load node assignments: prefer _bbg_* paths (from build_behavior_graph step),
    # then explicit clustering_dir, then clustering_name lookup.
    bbg_na_path = cfg.get("_bbg_node_assignments_path")
    bbg_meta_path = cfg.get("_bbg_metadata_path")
    bbg_level = cfg.get("_bbg_level")

    if bbg_na_path and bbg_meta_path:
        import json as _json
        cluster_labels = np.load(bbg_na_path)
        with open(bbg_meta_path) as _f:
            metadata = _json.load(_f)
        manifest = {"level": bbg_level or "rollout", "source": "build_behavior_graph"}
    elif clustering_dir:
        cluster_labels, metadata, manifest = load_clustering_result_from_path(clustering_dir)
    else:
        # Clustering artifacts are stored under influence_visualizer/configs/<task>/clustering/
        # (see save_clustering_result), regardless of config_root for the task YAML.
        cluster_labels, metadata, manifest = load_clustering_result(
            task_config, clustering_name
        )

    total_steps = 1 + len(seeds) * 5
    out_paths = []
    with tqdm(total=total_steps, desc="Pipeline", unit="step", dynamic_ncols=True) as pbar:
        pbar.set_description("Load clustering")
        pbar.update(1)

        for seed in seeds:
            pbar.set_description(f"Seed {seed}: load data")
            eval_dir_seed = get_eval_dir_for_seed(eval_dir, seed, reference_seed)
            train_dir_seed = get_train_dir_for_seed(train_dir, seed, reference_seed)
            data = load_influence_data(
                eval_dir=str(repo_root / eval_dir_seed),
                train_dir=str(repo_root / train_dir_seed),
                train_ckpt=task_cfg.get("train_ckpt", "latest"),
                exp_date=task_cfg.get("exp_date", "default"),
                include_holdout=True,
                image_dataset_path=None,
                lazy_load_images=True,
                quality_labels=task_cfg.get("quality_labels"),
            )
            pbar.update(1)

            pbar.set_description(f"Seed {seed}: build matrix")
            from policy_doctor.data.structures import EpisodeInfo, GlobalInfluenceMatrix

            def _to_pd_ep(ep):
                return EpisodeInfo(
                    index=ep.index,
                    num_samples=ep.num_samples,
                    sample_start_idx=ep.sample_start_idx,
                    sample_end_idx=ep.sample_end_idx,
                    success=getattr(ep, "success", None),
                    raw_length=getattr(ep, "raw_length", None) or ep.num_samples,
                )

            rollout_episodes_pd = [_to_pd_ep(ep) for ep in data.rollout_episodes]
            demo_episodes_pd = [_to_pd_ep(ep) for ep in data.demo_episodes]
            holdout_episodes_pd = [_to_pd_ep(ep) for ep in data.holdout_episodes]
            all_demo_episodes_pd = demo_episodes_pd + holdout_episodes_pd
            num_train_samples = len(data.demo_sample_infos)
            num_holdout_samples = len(data.holdout_sample_infos)

            global_matrix = GlobalInfluenceMatrix(
                data.influence_matrix,
                rollout_episodes_pd,
                all_demo_episodes_pd,
            )
            pbar.update(1)

            pbar.set_description(f"Seed {seed}: behavior graph & advantage")
            from policy_doctor.behaviors.behavior_values import (
                get_behavior_graph_and_slice_values,
                slice_indices_to_rollout_slices,
            )

            gamma = cfg.get("advantage_gamma", 0.99)
            reward_success = cfg.get("advantage_reward_success", 1.0)
            reward_failure = cfg.get("advantage_reward_failure", -1.0)
            reward_end = cfg.get("advantage_reward_end", 0.0)
            advantage_threshold = cfg.get("advantage_threshold")
            if advantage_threshold is None:
                raise ValueError("advantage_threshold is required (e.g. 0.1 for 'advantage_based_A0.1')")

            _, _, _, advantages = get_behavior_graph_and_slice_values(
                cluster_labels,
                metadata,
                gamma=gamma,
                reward_success=reward_success,
                reward_failure=reward_failure,
                reward_end=reward_end,
            )
            valid = np.isfinite(advantages) & (cluster_labels >= 0)
            if not np.any(valid):
                raise RuntimeError("No valid advantage values for selection")

            curation_mode = cfg.get("curation_mode", "selection")
            threshold = float(advantage_threshold)
            if curation_mode == "filter":
                selected_indices = np.where(valid & (advantages < threshold))[0]
            else:
                selected_indices = np.where(valid & (advantages >= threshold))[0]

            rollout_slices = slice_indices_to_rollout_slices(
                metadata,
                rollout_episodes_pd,
                cluster_labels,
                selected_indices,
            )
            if not rollout_slices:
                raise RuntimeError("No rollout slices selected from advantage threshold")
            pbar.update(1)

            pbar.set_description(f"Seed {seed}: slice search")
            window_width = cfg.get("window_width", 5)
            aggregation_method = cfg.get("aggregation_method", "mean")
            per_slice_top_k = cfg.get("per_slice_top_k", 100)
            use_all = cfg.get("use_all_demos_per_slice", True)
            ascending = cfg.get("ascending", False)

            from policy_doctor.curation.attribution import (
                run_slice_search,
                resolve_candidates_to_demo_slices,
                per_slice_n_sigma_selection,
                per_slice_percentile_selection,
            )

            if curation_mode == "filter":
                demo_start = 0
                demo_end = num_train_samples
            else:
                demo_start = num_train_samples
                demo_end = num_train_samples + num_holdout_samples

            selection_n_sigma = cfg.get("selection_n_sigma", None)
            selection_pct = cfg.get("selection_percentile", None)
            if selection_n_sigma is None and selection_pct is None:
                selection_pct = 99.0

            run_selection_pct = selection_pct if selection_n_sigma is None else None
            all_candidates, per_slice_candidates = run_slice_search(
                global_matrix,
                rollout_slices,
                all_demo_episodes_pd,
                window_width_demo=window_width,
                per_slice_top_k=per_slice_top_k,
                ascending=ascending,
                demo_start_idx=demo_start,
                demo_end_idx=demo_end,
                use_all_demos_per_slice=use_all,
                show_progress=True,
                aggregation_method=aggregation_method,
                selection_percentile=run_selection_pct,
            )
            pbar.update(1)

            pbar.set_description(f"Seed {seed}: resolve & save")
            if selection_n_sigma is not None:
                raw_selection = per_slice_n_sigma_selection(per_slice_candidates, float(selection_n_sigma))
            else:
                raw_selection = per_slice_percentile_selection(per_slice_candidates, float(selection_pct))
            if cfg.get("remove_negative_influence", False):
                raw_selection = [c for c in raw_selection if c.get("score", 0) >= 0]
            if not raw_selection:
                raise RuntimeError("No candidates after slice selection")

            if curation_mode == "filter":
                resolve_sample_infos = data.demo_sample_infos
                resolve_episodes = demo_episodes_pd
            else:
                resolve_sample_infos = data.holdout_sample_infos
                resolve_episodes = holdout_episodes_pd
            resolved = resolve_candidates_to_demo_slices(
                raw_selection,
                resolve_sample_infos,
                resolve_episodes,
                window_width=window_width,
            )
            if not resolved:
                raise RuntimeError("Could not resolve any candidates to demo slices")

            from policy_doctor.curation.config import (
                CurationConfig,
                CurationSlice,
                compute_dataset_fingerprint,
                merge_overlapping_slices,
                save_curation_config,
            )

            default_source = "behavior_search_train" if curation_mode == "filter" else "behavior_search_holdout"
            slice_label = cfg.get("slice_label", f"advantage_based_A{advantage_threshold}")
            slice_source = cfg.get("slice_source", default_source)
            raw_slices = [
                CurationSlice(episode_idx=ep, start=s, end=e, label=slice_label, source=slice_source)
                for (ep, s, e) in resolved
            ]
            slices = raw_slices if curation_mode == "filter" else merge_overlapping_slices(raw_slices)

            try:
                from policy_doctor.data.dataset_episode_ends import load_dataset_episode_ends
                train_dir_abs = repo_root / train_dir_seed
                episode_ends_cumulative_train = load_dataset_episode_ends(
                    train_dir_abs, task_cfg.get("train_ckpt", "latest"), repo_root
                )
            except Exception as e:
                if cfg.get("repo_root") is not None:
                    raise RuntimeError(
                        "load_dataset_episode_ends failed (repo_root was set; fingerprint must match)"
                    ) from e
                train_lengths = np.array(
                    [int(getattr(ep, "raw_length", None) or ep.num_samples) for ep in demo_episodes_pd],
                    dtype=np.int64,
                )
                episode_ends_cumulative_train = np.cumsum(train_lengths)

            fingerprint = compute_dataset_fingerprint(episode_ends_cumulative_train)
            total_raw = int(episode_ends_cumulative_train[-1]) if len(episode_ends_cumulative_train) > 0 else 0
            episode_lengths = {
                ep.index: int(getattr(ep, "raw_length", None) or ep.num_samples)
                for ep in all_demo_episodes_pd
            }

            metadata_dict = {
                "task_config": task_config,
                "split": "train" if curation_mode == "filter" else "holdout",
                "curation_mode": curation_mode,
                "num_slices": len(slices),
                "dataset_fingerprint": fingerprint,
                "total_raw_samples": total_raw,
                "policy_seed": int(seed) if seed.isdigit() else seed,
                "eval_dir": eval_dir,
                "train_dir": train_dir,
                "selection_method_metadata": {
                    "selection_method": "advantage_based",
                    "window_width": window_width,
                    "aggregation_method": aggregation_method,
                    "selection_normalization": "none",
                    "selection_mode": "per_slice_n_sigma" if selection_n_sigma is not None else "per_slice_percentile",
                    "n_sigma": float(selection_n_sigma) if selection_n_sigma is not None else None,
                    "global_top_k": per_slice_top_k,
                    "percentile": float(selection_pct) if selection_pct is not None else None,
                },
            }

            config_obj = CurationConfig(
                slices=slices,
                metadata=metadata_dict,
                episode_lengths=episode_lengths,
            )
            base_name = cfg.get("curation_output_name", "recreated_advantage_selection")
            output_name = f"{base_name}_seed{seed}"
            out_path = save_curation_config(
                task_config,
                output_name,
                config_obj,
                episode_ends=episode_ends_cumulative_train,
            )
            out_paths.append(out_path)

            covered_pairs: set = set()
            for sl in slices:
                ep_len = episode_lengths.get(sl.episode_idx, 0)
                for t in range(sl.start, min(sl.end + 1, ep_len)):
                    covered_pairs.add((sl.episode_idx, t))
            covered_samples = sum(
                1 for si in data.demo_sample_infos
                if (si.episode_idx, si.timestep) in covered_pairs
            )
            num_train = len(data.demo_sample_infos)
            pct_filtered = 100.0 * covered_samples / num_train if num_train else 0.0
            mode_str = f"n_sigma={selection_n_sigma}" if selection_n_sigma is not None else f"percentile={selection_pct}"
            print(
                f"[Seed {seed}] rollout_slices={len(rollout_slices):,}  "
                f"raw_candidates={len(raw_selection):,}  "
                f"unique_demo_windows={len(slices):,}  "
                f"covered_samples={covered_samples:,}/{num_train:,}  "
                f"filtered={pct_filtered:.1f}%  ({mode_str})"
            )
            pbar.update(1)

    for out_path in out_paths:
        print(f"Saved curation config: {out_path}")
    return out_paths


