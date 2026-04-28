"""Flywheel pipeline sub-steps: per-iteration eval, infembed, clustering, and training."""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
from typing import Any

import yaml
from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.curation_pipeline.diffusion_overrides import baseline_diffusion_extra_overrides
from policy_doctor.curation_pipeline.paths import expand_seeds, get_train_name
from policy_doctor.paths import CUPID_ROOT


def _best_checkpoint_stem(ckpt_dir: pathlib.Path) -> str:
    """Return the stem of the checkpoint with the highest test_mean_score, or 'latest'."""
    if not ckpt_dir.exists():
        return "latest"
    ckpt_files = [
        p for p in ckpt_dir.iterdir()
        if p.suffix == ".ckpt" and p.stem != "latest"
    ]
    if not ckpt_files:
        return "latest"
    best_ckpt = None
    best_score = -1.0
    for p in ckpt_files:
        m = re.search(r"test_mean_score=([\d.]+)", p.stem)
        if m:
            score = float(m.group(1))
            if score > best_score:
                best_score = score
                best_ckpt = p
    if best_ckpt is None:
        best_ckpt = sorted(ckpt_files)[-1]
    return best_ckpt.stem


class EvalFlywheelPolicyStep(PipelineStep[dict]):
    """Full eval_save_episodes on the best checkpoint from TrainOnCombinedDataStep.

    Unlike EvalMimicgenCombinedStep (which uses --save_episodes=False), this step
    saves full rollout data so subsequent infembed and clustering can consume it.

    Config keys (under ``evaluation``):
        num_episodes       Episodes to collect (default 200).
        test_start_seed    RNG seed (default 100000).
        overwrite          Re-run if output already exists (default false).
        eval_output_dir    Base output dir (default data/outputs/eval_save_episodes).

    Result JSON:
        eval_dir             Absolute path to the eval output directory.
        rollouts_hdf5_path   Absolute path to episodes/rollouts.hdf5.
        best_checkpoint      Checkpoint stem used for evaluation.
        train_dir            Path to the training directory evaluated.
    """

    name = "eval_flywheel_policy"

    def compute(self) -> dict[str, Any]:
        from policy_doctor.curation_pipeline.steps.train_on_combined_data import (
            TrainOnCombinedDataStep,
        )

        cfg = self.cfg
        evaluation = OmegaConf.select(cfg, "evaluation") or {}

        num_episodes: int = int(OmegaConf.select(evaluation, "num_episodes") or 200)
        test_start_seed: int = int(OmegaConf.select(evaluation, "test_start_seed") or 100000)
        overwrite: bool = bool(OmegaConf.select(evaluation, "overwrite") or False)
        device: str = OmegaConf.select(cfg, "device") or "cuda:0"
        eval_output_dir: str = (
            OmegaConf.select(evaluation, "eval_output_dir")
            or "data/outputs/eval_save_episodes"
        )
        conda_env: str = (
            OmegaConf.select(cfg, "data_source.conda_env_train") or "mimicgen_torch2"
        )

        # Load train dirs from TrainOnCombinedDataStep (name="train_on_combined_data")
        train_step = TrainOnCombinedDataStep(cfg, self.run_dir)
        if not train_step.is_done():
            raise RuntimeError(
                "EvalFlywheelPolicyStep requires TrainOnCombinedDataStep to have run first."
            )
        train_result = train_step.load()
        train_dirs: list[str] = train_result.get("train_dirs", [])
        if not train_dirs:
            raise RuntimeError("TrainOnCombinedDataStep result has no train_dirs.")

        train_dir = pathlib.Path(train_dirs[0])
        if not train_dir.exists():
            raise FileNotFoundError(f"Train dir not found: {train_dir}")

        best_ckpt = _best_checkpoint_stem(train_dir / "checkpoints")
        output_dir = str(
            self.repo_root / eval_output_dir / f"{train_dir.name}_flywheel_eval"
        )

        if self.dry_run:
            print(
                f"[dry_run] EvalFlywheelPolicyStep  train={train_dir.name}  "
                f"ckpt={best_ckpt}  output={output_dir}"
            )
            rollouts_hdf5 = str(pathlib.Path(output_dir) / "episodes" / "rollouts.hdf5")
            return {
                "eval_dir": output_dir,
                "rollouts_hdf5_path": rollouts_hdf5,
                "best_checkpoint": best_ckpt,
                "train_dir": str(train_dir),
            }

        print(
            f"  [eval_flywheel_policy] train={train_dir.name}  "
            f"ckpt={best_ckpt}  episodes={num_episodes}  conda_env={conda_env}"
        )
        cmd = [
            "conda", "run", "-n", conda_env, "--no-capture-output",
            "python", str(CUPID_ROOT / "eval_save_episodes.py"),
            f"--output_dir={output_dir}",
            f"--train_dir={train_dir}",
            f"--train_ckpt={best_ckpt}",
            f"--num_episodes={num_episodes}",
            f"--test_start_seed={test_start_seed}",
            f"--overwrite={overwrite}",
            f"--device={device}",
            "--n_test_vis=0",
        ]
        result = subprocess.run(cmd, cwd=str(CUPID_ROOT))
        if result.returncode != 0:
            raise RuntimeError(
                f"[eval_flywheel_policy] subprocess failed with exit code {result.returncode}"
            )

        # Locate rollouts.hdf5 (standard layout: episodes/rollouts.hdf5)
        rollouts_hdf5 = pathlib.Path(output_dir) / "episodes" / "rollouts.hdf5"
        if not rollouts_hdf5.exists():
            alt = pathlib.Path(output_dir) / "rollouts.hdf5"
            if alt.exists():
                rollouts_hdf5 = alt

        return {
            "eval_dir": output_dir,
            "rollouts_hdf5_path": str(rollouts_hdf5),
            "best_checkpoint": best_ckpt,
            "train_dir": str(train_dir),
        }


class ComputeInfembedFlywheelStep(PipelineStep[None]):
    """Compute InfEmbed embeddings for a flywheel-trained policy.

    Reads train_dir and eval_dir from prior sub-steps' result.json files instead
    of resolving via config dates (as the base ComputeInfembedStep does).

    Reads:
        train_on_combined_data/result.json["train_dirs"][0]
        eval_flywheel_policy/result.json["eval_dir"]
        train_on_combined_data/result.json["combined_hdf5_path"]  (--dataset_path override)
    """

    name = "compute_infembed"

    def compute(self) -> None:
        from policy_doctor.curation_pipeline.steps.train_on_combined_data import (
            TrainOnCombinedDataStep,
        )
        from policy_doctor.curation_pipeline.steps.compute_infembed import (
            _call_compute_infembed_embeddings,
        )

        cfg = self.cfg
        attribution = OmegaConf.select(cfg, "attribution") or {}

        # Resolve train_dir and eval_dir from sibling step results
        train_step = TrainOnCombinedDataStep(cfg, self.run_dir)
        if not train_step.is_done():
            raise RuntimeError(
                "ComputeInfembedFlywheelStep requires TrainOnCombinedDataStep."
            )
        train_result = train_step.load()
        train_dirs = train_result.get("train_dirs", [])
        if not train_dirs:
            raise RuntimeError("No train_dirs in TrainOnCombinedDataStep result.")
        train_dir = str(train_dirs[0])
        combined_hdf5_path = train_result.get("combined_hdf5_path")

        eval_step = EvalFlywheelPolicyStep(cfg, self.run_dir)
        if not eval_step.is_done():
            raise RuntimeError(
                "ComputeInfembedFlywheelStep requires EvalFlywheelPolicyStep."
            )
        eval_result = eval_step.load()
        eval_dir = eval_result["eval_dir"]

        # InfEmbed config — mirrors ComputeInfembedStep
        modelout_fn = OmegaConf.select(attribution, "modelout_fn") or "DiffusionLowdimFunctionalModelOutput"
        loss_fn = OmegaConf.select(attribution, "loss_fn") or "square"
        num_timesteps = OmegaConf.select(attribution, "num_timesteps") or 64
        batch_size = (
            OmegaConf.select(attribution, "infembed_batch_size")
            or OmegaConf.select(attribution, "batch_size")
            or 32
        )
        device = OmegaConf.select(attribution, "device") or OmegaConf.select(cfg, "device") or "cuda:0"
        exp_seed = OmegaConf.select(attribution, "seed") or 0
        featurize_holdout = OmegaConf.select(attribution, "featurize_holdout")
        if featurize_holdout is None:
            featurize_holdout = True
        projection_dim = OmegaConf.select(attribution, "projection_dim") or 100
        arnoldi_dim = OmegaConf.select(attribution, "arnoldi_dim") or 200
        overwrite = bool(OmegaConf.select(attribution, "overwrite") or False)
        model_keys = OmegaConf.select(attribution, "model_keys") or "model."
        train_ckpt = OmegaConf.select(attribution, "train_ckpt") or "latest"

        cmd_args = [
            "--exp_name=auto",
            f"--eval_dir={eval_dir}",
            f"--train_dir={train_dir}",
            f"--train_ckpt={train_ckpt}",
            f"--modelout_fn={modelout_fn}",
            f"--loss_fn={loss_fn}",
            f"--num_timesteps={num_timesteps}",
            f"--batch_size={batch_size}",
            f"--device={device}",
            f"--seed={exp_seed}",
            f"--projection_dim={projection_dim}",
            f"--arnoldi_dim={arnoldi_dim}",
        ]
        if model_keys:
            cmd_args.append(f"--model_keys={model_keys}")
        if featurize_holdout:
            cmd_args.append("--featurize_holdout")
        if overwrite:
            cmd_args.append("--overwrite")
        # Override stale checkpoint dataset path with the combined HDF5
        if combined_hdf5_path:
            cmd_args.append(f"--dataset_path={combined_hdf5_path}")

        if self.dry_run:
            print(f"[dry_run] ComputeInfembedFlywheelStep")
            print(f"[dry_run]   {' '.join(cmd_args)}")
            return

        print(f"  [compute_infembed_flywheel] train={pathlib.Path(train_dir).name}")
        _call_compute_infembed_embeddings(cmd_args=cmd_args)


class RunClusteringFlywheelStep(PipelineStep[dict]):
    """Cluster a flywheel-trained policy's rollouts.

    Like RunClusteringStep but reads eval_dir directly from
    EvalFlywheelPolicyStep's result instead of from the task config YAML.

    Result JSON: same format as RunClusteringStep: ``{"clustering_dirs": {"0": path}}``
    """

    name = "run_clustering"

    def compute(self) -> dict[str, str]:
        if "NUMBA_THREADING_LAYER" not in os.environ:
            os.environ["NUMBA_THREADING_LAYER"] = "omp"

        from policy_doctor.behaviors.clustering import (
            fit_cluster_kmeans,
            fit_normalize_embeddings,
            fit_reduce_dimensions,
        )
        from policy_doctor.data.clustering_loader import save_clustering_models
        from policy_doctor.data.clustering_embeddings import extract_infembed_slice_windows
        from influence_visualizer.clustering_results import save_clustering_result

        cfg = self.cfg

        # Resolve eval_dir from EvalFlywheelPolicyStep result
        eval_step = EvalFlywheelPolicyStep(cfg, self.run_dir)
        if not eval_step.is_done():
            raise RuntimeError(
                "RunClusteringFlywheelStep requires EvalFlywheelPolicyStep."
            )
        eval_result = eval_step.load()
        eval_dir_abs = pathlib.Path(eval_result["eval_dir"])

        seed = str(OmegaConf.select(cfg, "mimicgen_datagen.policy_seed") or 0)

        window_width = OmegaConf.select(cfg, "clustering_window_width") or 5
        stride = OmegaConf.select(cfg, "clustering_stride") or 2
        umap_n_components = OmegaConf.select(cfg, "clustering_umap_n_components") or 100
        n_clusters = OmegaConf.select(cfg, "clustering_n_clusters") or 20
        normalize = OmegaConf.select(cfg, "clustering_normalize") or "none"
        aggregation = OmegaConf.select(cfg, "clustering_aggregation") or "sum"
        experiment_name = (
            OmegaConf.select(cfg, "experiment_name")
            or OmegaConf.select(cfg, "train_date")
            or "flywheel"
        )
        umap_n_jobs = OmegaConf.select(cfg, "clustering_umap_n_jobs") or -1
        umap_prescale = OmegaConf.select(cfg, "clustering_umap_prescale") or "standard"
        task_config = OmegaConf.select(cfg, "task_config") or "flywheel"

        if self.dry_run:
            print(
                f"[dry_run] RunClusteringFlywheelStep  eval_dir={eval_dir_abs}  "
                f"window={window_width}  stride={stride}  umap_dim={umap_n_components}  "
                f"k={n_clusters}  normalize={normalize}"
            )
            return {"clustering_dirs": {}}

        print(f"  [run_clustering_flywheel] eval_dir={eval_dir_abs.name}")
        embeddings_arr, all_metadata = extract_infembed_slice_windows(
            eval_dir_abs, window_width, stride, aggregation,
        )

        print(f"  [run_clustering_flywheel] embeddings: {embeddings_arr.shape}")
        embeddings_norm, normalizer_model = fit_normalize_embeddings(embeddings_arr, method=normalize)
        embeddings_scaled, prescaler_model = fit_normalize_embeddings(embeddings_norm, method=umap_prescale)

        print(f"  [run_clustering_flywheel] UMAP {embeddings_scaled.shape[1]}d → {umap_n_components}d  (n_jobs={umap_n_jobs})")
        embeddings_reduced, umap_model = fit_reduce_dimensions(
            embeddings_scaled, method="umap", n_components=umap_n_components, n_jobs=umap_n_jobs,
        )

        print(f"  [run_clustering_flywheel] K-Means k={n_clusters}")
        labels, kmeans_model = fit_cluster_kmeans(embeddings_reduced, n_clusters=n_clusters)

        n_actual = len(set(labels) - {-1})
        print(f"  [run_clustering_flywheel] clusters={n_actual}  noise={(labels == -1).sum()}")

        clustering_name = f"{experiment_name}_flywheel_seed{seed}_kmeans_k{n_clusters}"
        result_dir = save_clustering_result(
            task_config=task_config,
            name=clustering_name,
            cluster_labels=labels,
            metadata=all_metadata,
            algorithm="kmeans",
            scaling=normalize,
            influence_source="infembed",
            representation="sliding_window",
            level="rollout",
            n_clusters=n_actual,
            n_samples=len(labels),
        )
        save_clustering_models(
            result_dir=result_dir,
            normalizer=normalizer_model,
            normalizer_method=normalize,
            prescaler=prescaler_model,
            prescaler_method=umap_prescale,
            reducer=umap_model,
            reducer_method="umap",
            kmeans=kmeans_model,
        )
        print(f"  [run_clustering_flywheel] saved: {result_dir}")
        return {"clustering_dirs": {seed: str(result_dir)}}


class TrainFlywheelIterStep(PipelineStep[dict]):
    """Train on base dataset + all generated demos accumulated through iteration i.

    At iteration i trains on: original_dataset + gen_iter_0 + ... + gen_iter_i.
    The combined HDF5 is built by iteratively appending generated datasets.

    name = "train_on_combined_data" so that EvalMimicgenCombinedStep can locate
    this step's result via its usual sibling-step lookup.

    Args (constructor, in addition to the standard PipelineStep args):
        iteration_idx:       Current flywheel iteration index (0-based).
        all_prior_iter_dirs: Step dirs of all PREVIOUS iterations (iter_0 through iter_{i-1}).
                             The current iteration's generated data is read from self.run_dir.
    """

    name = "train_on_combined_data"

    def __init__(
        self,
        cfg,
        run_dir: pathlib.Path,
        parent_run_dir: pathlib.Path | None = None,
        *,
        iteration_idx: int = 0,
        all_prior_iter_dirs: list[pathlib.Path] | None = None,
    ) -> None:
        super().__init__(cfg, run_dir, parent_run_dir)
        self.iteration_idx = iteration_idx
        self.all_prior_iter_dirs = all_prior_iter_dirs or []

    def compute(self) -> dict[str, Any]:
        from policy_doctor.curation_pipeline.steps.generate_mimicgen_demos import (
            GenerateMimicgenDemosStep,
        )
        from policy_doctor.mimicgen.combine_datasets import combine_hdf5_datasets

        import h5py

        cfg = self.cfg
        cfg_mg = OmegaConf.select(cfg, "mimicgen_datagen") or {}
        baseline = OmegaConf.select(cfg, "baseline") or {}

        heuristic_name: str = OmegaConf.select(cfg_mg, "seed_selection_heuristic") or "behavior_graph"

        # Collect generated HDF5 paths: prior iters first, then current iter
        generated_hdf5_paths: list[pathlib.Path] = []
        for iter_dir in self.all_prior_iter_dirs:
            gen_result_path = iter_dir / "generate_mimicgen_demos" / "result.json"
            if gen_result_path.exists():
                with open(gen_result_path) as f:
                    gen_result = json.load(f)
                gen_path = pathlib.Path(gen_result["generated_hdf5_path"])
                if gen_path.exists():
                    generated_hdf5_paths.append(gen_path)

        current_gen_step = GenerateMimicgenDemosStep(cfg, self.run_dir)
        if not current_gen_step.is_done():
            raise RuntimeError("TrainFlywheelIterStep requires GenerateMimicgenDemosStep.")
        current_gen_result = current_gen_step.load()
        current_gen_path = pathlib.Path(current_gen_result["generated_hdf5_path"])
        if current_gen_path.exists():
            generated_hdf5_paths.append(current_gen_path)

        # Resolve original (base) dataset
        original_hdf5_path = self._resolve_original_dataset_path(baseline)

        with h5py.File(original_hdf5_path, "r") as f:
            num_base = sum(1 for k in f["data"].keys() if k.startswith("demo_"))
        total_generated = 0
        for gp in generated_hdf5_paths:
            with h5py.File(gp, "r") as f:
                total_generated += sum(1 for k in f["data"].keys() if k.startswith("demo_"))

        print(
            f"  [train_flywheel_iter] iter={self.iteration_idx}  heuristic={heuristic_name!r}  "
            f"base={num_base}  generated={total_generated} (across {len(generated_hdf5_paths)} sources)"
        )

        # Build combined HDF5: chain combine_hdf5_datasets over all generated sources
        self.step_dir.mkdir(parents=True, exist_ok=True)
        combined_hdf5_path = self.step_dir / "combined.hdf5"

        if not generated_hdf5_paths:
            shutil.copy2(original_hdf5_path, combined_hdf5_path)
        else:
            current_src = original_hdf5_path
            for i, gen_path in enumerate(generated_hdf5_paths):
                is_final = (i == len(generated_hdf5_paths) - 1)
                dst = combined_hdf5_path if is_final else self.step_dir / f"_tmp_{i}.hdf5"
                combine_hdf5_datasets(current_src, gen_path, dst)
                # Remove intermediate temp files
                if i > 0 and current_src != original_hdf5_path:
                    try:
                        current_src.unlink(missing_ok=True)
                    except Exception:
                        pass
                current_src = dst

        with h5py.File(combined_hdf5_path, "r") as f:
            num_combined = sum(1 for k in f["data"].keys() if k.startswith("demo_"))

        print(f"  [train_flywheel_iter] combined → {combined_hdf5_path}  demos={num_combined}")

        # Training config — mirrors TrainOnCombinedDataStep
        config_dir = (
            OmegaConf.select(baseline, "config_dir")
            or OmegaConf.select(cfg, "config_dir")
        )
        if not config_dir:
            raise ValueError("baseline.config_dir is required for training")
        config_name = OmegaConf.select(baseline, "config_name") or "config.yaml"
        task = OmegaConf.select(baseline, "task") or OmegaConf.select(cfg, "task")
        policy = OmegaConf.select(baseline, "policy") or "diffusion_unet_lowdim"
        seeds = expand_seeds(
            OmegaConf.select(cfg, "seeds")
            or OmegaConf.select(baseline, "seeds")
            or [0]
        )
        num_epochs = int(
            OmegaConf.select(baseline, "num_epochs")
            or OmegaConf.select(cfg, "num_epochs")
            or 1001
        )
        checkpoint_topk = int(OmegaConf.select(baseline, "checkpoint_topk") or 3)
        checkpoint_every = int(OmegaConf.select(baseline, "checkpoint_every") or 50)
        val_ratio = float(OmegaConf.select(baseline, "val_ratio") or 0.04)
        output_dir = OmegaConf.select(cfg, "output_dir") or "data/outputs/train"
        project = OmegaConf.select(cfg, "project") or "influence-clustering"
        wandb_tags = OmegaConf.select(cfg, "wandb_tags") or []
        if isinstance(wandb_tags, str):
            wandb_tags = [wandb_tags]
        train_date = (
            OmegaConf.select(cfg, "train_date")
            or OmegaConf.select(baseline, "train_date")
            or "default"
        )
        device = OmegaConf.select(cfg, "device") or "cuda:0"
        run_tag = OmegaConf.select(cfg, "run_tag")
        tf32 = bool(OmegaConf.select(baseline, "tf32") or False)
        compile_ = bool(OmegaConf.select(baseline, "compile") or False)
        num_gpus = int(
            OmegaConf.select(baseline, "num_gpus")
            or OmegaConf.select(cfg, "num_gpus")
            or 1
        )
        conda_env = (
            OmegaConf.select(baseline, "conda_env")
            or OmegaConf.select(cfg, "data_source.conda_env_train")
            or "mimicgen_torch2"
        )
        exp_name = f"train_{policy}"

        train_dirs: list[str] = []
        for seed in seeds:
            base_name = get_train_name(train_date, task, policy, seed)
            train_name = f"{base_name}-flywheel_iter{self.iteration_idx}-{heuristic_name}"
            if run_tag:
                train_name = f"{train_name}-{run_tag}"
            run_output_dir = str(self.repo_root / output_dir / train_date / train_name)
            train_dirs.append(run_output_dir)

            overrides = [
                f"name={exp_name}",
                f"training.device={device}",
                f"training.seed={seed}",
                f"training.num_epochs={num_epochs}",
                f"checkpoint.topk.k={checkpoint_topk}",
                f"training.checkpoint_every={checkpoint_every}",
                f"training.rollout_every={checkpoint_every}",
                f"task.dataset.seed={seed}",
                f"task.dataset.val_ratio={val_ratio}",
                f"logging.name={train_name}",
                f"logging.group={train_date}_{exp_name}_{task}",
                f"logging.project={project}",
                f"multi_run.wandb_name_base={train_name}",
                f"multi_run.run_dir={run_output_dir}",
                f"hydra.run.dir={run_output_dir}",
                f"++task.dataset.dataset_path={combined_hdf5_path}",
                "++task.env_runner.n_train_vis=0",
                "++task.env_runner.n_test_vis=0",
                f"+training.tf32={str(tf32).lower()}",
                f"+training.compile={str(compile_).lower()}",
            ]
            if wandb_tags:
                tags_str = "[" + ",".join(str(t) for t in wandb_tags) + "]"
                overrides.append(f"logging.tags={tags_str}")
            overrides.extend(baseline_diffusion_extra_overrides(baseline))

            if self.dry_run:
                print(
                    f"[dry_run] TrainFlywheelIterStep iter={self.iteration_idx}  "
                    f"seed={seed}  output={run_output_dir}"
                )
                continue

            pathlib.Path(run_output_dir).mkdir(parents=True, exist_ok=True)
            if num_gpus > 1:
                launcher = ["torchrun", f"--nproc_per_node={num_gpus}", str(CUPID_ROOT / "train.py")]
            else:
                launcher = ["python", str(CUPID_ROOT / "train.py")]
            cmd = [
                "conda", "run", "-n", conda_env, "--no-capture-output",
                *launcher,
                "--config-path", config_dir,
                "--config-name", config_name,
                *overrides,
            ]
            env_vars = {**os.environ, "WANDB_RESUME": "never"}
            print(
                f"  [train_flywheel_iter] iter={self.iteration_idx}  "
                f"seed={seed}  conda_env={conda_env}  output={run_output_dir}"
            )
            result = subprocess.run(cmd, cwd=str(CUPID_ROOT), env=env_vars)
            if result.returncode != 0:
                raise RuntimeError(
                    f"[train_flywheel_iter] iter={self.iteration_idx} seed={seed} "
                    f"failed with exit code {result.returncode}"
                )

        return {
            "combined_hdf5_path": str(combined_hdf5_path.resolve()),
            "original_hdf5_path": str(original_hdf5_path),
            "generated_hdf5_paths": [str(p) for p in generated_hdf5_paths],
            "num_combined_demos": num_combined,
            "num_generated_demos": total_generated,
            "heuristic": heuristic_name,
            "iteration_idx": self.iteration_idx,
            "train_dirs": train_dirs,
        }

    def _resolve_original_dataset_path(self, baseline: Any) -> pathlib.Path:
        from policy_doctor.paths import PROJECT_ROOT

        cfg = self.cfg
        dataset_path = OmegaConf.select(cfg, "mimicgen_datagen.original_dataset_path")
        if not dataset_path:
            dataset_path = (
                OmegaConf.select(cfg, "task.dataset.dataset_path")
                or OmegaConf.select(cfg, "dataset_path")
            )
        if dataset_path is None:
            config_dir = (
                OmegaConf.select(baseline, "config_dir")
                or OmegaConf.select(cfg, "config_dir")
            )
            config_name = OmegaConf.select(baseline, "config_name") or "config.yaml"
            if config_dir:
                hydra_config_path = self.repo_root / config_dir / config_name
                if hydra_config_path.exists():
                    with open(hydra_config_path) as f:
                        hydra_cfg = yaml.safe_load(f) or {}
                    dataset_path = (
                        hydra_cfg.get("task", {}).get("dataset", {}).get("dataset_path")
                    )
        if dataset_path is None:
            raise ValueError("Cannot determine original dataset path.")
        p = pathlib.Path(dataset_path)
        if not p.is_absolute():
            candidate = (self.repo_root / p).resolve()
            if not candidate.exists():
                candidate = (PROJECT_ROOT / p).resolve()
            p = candidate
        if not p.exists():
            raise FileNotFoundError(f"Original dataset not found: {p}")
        return p
