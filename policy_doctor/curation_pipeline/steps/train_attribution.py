"""Train TRAK attribution model — pipeline step."""

from __future__ import annotations

from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.curation_pipeline.paths import expand_seeds, get_eval_dir, get_train_dir


def _call_train_trak_diffusion(cmd_args: list) -> None:
    from train_trak_diffusion import main  # noqa: PLC0415

    main(cmd_args, standalone_mode=False)


class TrainAttributionStep(PipelineStep[None]):
    """Run TRAK training per seed to compute attribution scores."""

    name = "train_attribution"

    def compute(self) -> None:
        cfg = self.cfg
        attribution = OmegaConf.select(cfg, "attribution") or {}
        baseline = OmegaConf.select(cfg, "baseline") or {}

        train_date = OmegaConf.select(attribution, "train_date") or OmegaConf.select(cfg, "train_date")
        eval_date = OmegaConf.select(attribution, "eval_date") or OmegaConf.select(cfg, "eval_date") or train_date
        task = OmegaConf.select(attribution, "task") or OmegaConf.select(baseline, "task") or OmegaConf.select(cfg, "task")
        policy = OmegaConf.select(attribution, "policy") or OmegaConf.select(baseline, "policy")
        seeds = expand_seeds(
            OmegaConf.select(attribution, "seeds")
            or OmegaConf.select(cfg, "seeds")
            or OmegaConf.select(baseline, "seeds")
            or [0]
        )
        train_ckpt = OmegaConf.select(attribution, "train_ckpt") or "latest"
        eval_ckpt = OmegaConf.select(attribution, "eval_ckpt") or "latest"
        eval_as_train_seed = OmegaConf.select(attribution, "eval_as_train_seed")
        if eval_as_train_seed is None:
            eval_as_train_seed = True
        train_output_dir = OmegaConf.select(attribution, "train_output_dir") or "data/outputs/train"
        eval_output_dir = OmegaConf.select(attribution, "eval_output_dir") or "data/outputs/eval_save_episodes"

        num_ckpts = OmegaConf.select(attribution, "num_ckpts") or 1
        exp_seed = OmegaConf.select(attribution, "seed") or 0
        result_date = OmegaConf.select(attribution, "result_date") or "default"
        num_timesteps = OmegaConf.select(attribution, "num_timesteps") or 64
        loss_fn = OmegaConf.select(attribution, "loss_fn") or "square"
        featurize_holdout = OmegaConf.select(attribution, "featurize_holdout")
        if featurize_holdout is None:
            featurize_holdout = True
        finalize_scores = OmegaConf.select(attribution, "finalize_scores")
        if finalize_scores is None:
            finalize_scores = True
        finalize_on_train = OmegaConf.select(attribution, "finalize_on_train")
        if finalize_on_train is None:
            finalize_on_train = True

        proj_dim = OmegaConf.select(attribution, "proj_dim") or 4000
        lambda_reg = OmegaConf.select(attribution, "lambda_reg") or 0.0
        exp_name = (
            f"{result_date}_trak_results"
            f"-proj_dim={proj_dim}"
            f"-lambda_reg={lambda_reg}"
            f"-num_ckpts={num_ckpts}"
            f"-seed={exp_seed}"
            f"-loss_fn={loss_fn}"
            f"-num_timesteps={num_timesteps}"
        )

        for seed_idx, seed in enumerate(seeds):
            use_model_id = OmegaConf.select(attribution, "use_model_id")
            model_id = use_model_id if use_model_id is not None else seed_idx

            train_dir = str(self.repo_root / get_train_dir(train_output_dir, train_date, task, policy, seed))
            eval_dir = str(self.repo_root / get_eval_dir(
                eval_output_dir, eval_date, task, policy, seed,
                train_ckpt=eval_ckpt, eval_as_train_seed=eval_as_train_seed,
            ))

            cmd_args = [
                f"--exp_name={exp_name}",
                f"--eval_dir={eval_dir}",
                f"--train_dir={train_dir}",
                f"--train_ckpt={train_ckpt}",
                f"--model_keys={OmegaConf.select(attribution, 'model_keys') or 'model.'}",
                f"--model_id={model_id}",
                f"--modelout_fn={OmegaConf.select(attribution, 'modelout_fn') or 'DiffusionLowdimFunctionalModelOutput'}",
                f"--gradient_co={OmegaConf.select(attribution, 'gradient_co') or 'DiffusionLowdimFunctionalGradientComputer'}",
                f"--proj_dim={proj_dim}",
                f"--proj_max_batch_size={OmegaConf.select(attribution, 'proj_max_batch_size') or 32}",
                f"--lambda_reg={lambda_reg}",
                f"--use_half_precision={OmegaConf.select(attribution, 'use_half_precision') or False}",
                f"--batch_size={OmegaConf.select(attribution, 'batch_size') or 32}",
                f"--device={OmegaConf.select(attribution, 'device') or OmegaConf.select(cfg, 'device') or 'cuda:0'}",
                f"--seed={exp_seed}",
                f"--loss_fn={loss_fn}",
                f"--num_timesteps={num_timesteps}",
                f"--featurize_holdout={featurize_holdout}",
                f"--finalize_scores={finalize_scores and (num_ckpts == 1 or finalize_on_train)}",
            ]

            if self.dry_run:
                print(f"[dry_run] TrainAttributionStep seed={seed}  model_id={model_id}")
                print(f"[dry_run]   {' '.join(cmd_args)}")
                continue

            print(f"  [train_attribution] seed={seed}  model_id={model_id}")
            _call_train_trak_diffusion(cmd_args=cmd_args)
