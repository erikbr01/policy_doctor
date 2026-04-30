"""Run sliding-window clustering — pipeline step class."""

from __future__ import annotations

import os

if "NUMBA_THREADING_LAYER" not in os.environ:
    os.environ["NUMBA_THREADING_LAYER"] = "omp"

import pathlib
from typing import Dict

import numpy as np
from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.curation_pipeline.paths import get_eval_dir
from policy_doctor.data.clustering_embeddings import (
    extract_infembed_slice_windows,
    extract_trak_slice_windows,
)


class RunClusteringStep(PipelineStep[Dict[str, str]]):
    """Run sliding-window clustering for each seed.

    Result: ``{"clustering_dirs": {seed: path, ...}}``
    """

    name = "run_clustering"

    def compute(self) -> Dict[str, str]:
        from policy_doctor.behaviors.clustering import (
            fit_cluster_kmeans,
            fit_normalize_embeddings,
            fit_reduce_dimensions,
        )
        from policy_doctor.data.clustering_loader import save_clustering_models
        from influence_visualizer.clustering_results import save_clustering_result
        from influence_visualizer.data_loader import get_eval_dir_for_seed

        cfg = self.cfg

        # Resolve eval_dir_base: prefer explicit override, then evaluation config.
        clustering_eval_dir_override = OmegaConf.select(cfg, "clustering_eval_dir")
        if clustering_eval_dir_override:
            eval_dir_base = clustering_eval_dir_override
        else:
            evaluation = OmegaConf.select(cfg, "evaluation") or {}
            eval_date = (
                OmegaConf.select(evaluation, "train_date")
                or OmegaConf.select(cfg, "evaluation.eval_date")
                or OmegaConf.select(cfg, "train_date")
            )
            eval_task = OmegaConf.select(evaluation, "task")
            eval_policy = OmegaConf.select(evaluation, "policy")
            eval_output_dir = OmegaConf.select(evaluation, "eval_output_dir") or "data/outputs/eval_save_episodes"
            if eval_date and eval_task and eval_policy:
                eval_dir_base = get_eval_dir(eval_output_dir, eval_date, eval_task, eval_policy, 0)
            else:
                raise ValueError(
                    "Cannot resolve eval_dir for clustering: set evaluation.train_date, "
                    "evaluation.task, and evaluation.policy in the Hydra config. "
                    "The task_config YAML fallback has been removed."
                )
        print(f"  eval_dir_base: {eval_dir_base}")

        train_dir_base = OmegaConf.select(cfg, "clustering_train_dir")
        train_ckpt = OmegaConf.select(cfg, "evaluation.train_ckpt") or "latest"
        exp_date = OmegaConf.select(cfg, "evaluation.exp_date") or "default"

        reference_seed = str(OmegaConf.select(cfg, "reference_seed") or 0)
        seeds = OmegaConf.select(cfg, "seeds") or OmegaConf.select(cfg, "policy_seeds") or [0, 1, 2]
        seeds = [str(s) for s in seeds]

        window_width = OmegaConf.select(cfg, "clustering_window_width") or 5
        stride = OmegaConf.select(cfg, "clustering_stride") or 2
        umap_n_components = OmegaConf.select(cfg, "clustering_umap_n_components") or 100
        n_clusters = OmegaConf.select(cfg, "clustering_n_clusters") or 20
        normalize = OmegaConf.select(cfg, "clustering_normalize") or "none"
        aggregation = OmegaConf.select(cfg, "clustering_aggregation") or "sum"
        experiment_name = OmegaConf.select(cfg, "experiment_name") or OmegaConf.select(cfg, "train_date") or "default"
        influence_source = OmegaConf.select(cfg, "clustering_influence_source") or "infembed"
        demo_split = OmegaConf.select(cfg, "clustering_demo_split") or "both"
        level = OmegaConf.select(cfg, "clustering_level") or "rollout"
        umap_n_jobs = OmegaConf.select(cfg, "clustering_umap_n_jobs") or -1
        umap_prescale = OmegaConf.select(cfg, "clustering_umap_prescale") or "standard"

        if self.dry_run:
            for seed in seeds:
                print(
                    f"[dry_run] RunClusteringStep seed={seed}: "
                    f"source={influence_source}, level={level}, split={demo_split}, "
                    f"window={window_width}, stride={stride}, umap_dim={umap_n_components}, "
                    f"k={n_clusters}, normalize={normalize}, prescale={umap_prescale}, agg={aggregation}"
                )
            return {"clustering_dirs": {}}

        result_dirs: Dict[str, str] = {}
        for seed in seeds:
            eval_dir_seed = get_eval_dir_for_seed(eval_dir_base, seed, reference_seed)
            eval_dir_abs = self.repo_root / eval_dir_seed

            if influence_source == "trak":
                embeddings_arr, all_metadata = extract_trak_slice_windows(
                    eval_dir_abs, train_dir_base, self.repo_root,
                    seed, reference_seed,
                    window_width, stride, aggregation, demo_split, level,
                    train_ckpt=train_ckpt, exp_date=exp_date,
                )
            else:
                if level == "demo":
                    raise ValueError(
                        "level='demo' requires clustering_influence_source='trak' "
                        "(infembed embeddings are per-rollout-timestep only)"
                    )
                embeddings_arr, all_metadata = extract_infembed_slice_windows(
                    eval_dir_abs, window_width, stride, aggregation,
                )

            print(f"  Slice embeddings: {embeddings_arr.shape}")
            print(f"  Normalizing: {normalize}")
            embeddings_norm, normalizer_model = fit_normalize_embeddings(embeddings_arr, method=normalize)

            print(f"  Pre-UMAP scaling: {umap_prescale}")
            embeddings_scaled, prescaler_model = fit_normalize_embeddings(embeddings_norm, method=umap_prescale)

            print(f"  UMAP: {embeddings_scaled.shape[1]}d -> {umap_n_components}d (n_jobs={umap_n_jobs})")
            embeddings_reduced, umap_model = fit_reduce_dimensions(
                embeddings_scaled, method="umap", n_components=umap_n_components, n_jobs=umap_n_jobs
            )

            print(f"  K-Means: k={n_clusters}")
            labels, kmeans_model = fit_cluster_kmeans(embeddings_reduced, n_clusters=n_clusters)

            n_actual = len(set(labels) - {-1})
            print(f"  Clusters: {n_actual}, noise: {int((labels == -1).sum())}")

            clustering_name = f"{experiment_name}_seed{seed}_kmeans_k{n_clusters}"
            result_dir = save_clustering_result(
                name=clustering_name,
                cluster_labels=labels,
                metadata=all_metadata,
                algorithm="kmeans",
                scaling=normalize,
                influence_source=influence_source,
                representation="sliding_window",
                level=level,
                n_clusters=n_actual,
                n_samples=len(labels),
                output_dir=self.step_dir / "clustering",
            )
            models_path = save_clustering_models(
                result_dir=result_dir,
                normalizer=normalizer_model,
                normalizer_method=normalize,
                prescaler=prescaler_model,
                prescaler_method=umap_prescale,
                reducer=umap_model,
                reducer_method="umap",
                kmeans=kmeans_model,
            )
            print(f"  Saved: {result_dir}  (models: {models_path.name})")
            result_dirs[seed] = str(result_dir)

        return {"clustering_dirs": result_dirs}
