"""Compute demonstration scores (TRAK influence) — pipeline step."""

from __future__ import annotations

from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.curation_pipeline.paths import expand_seeds, get_eval_dir, get_train_dir


def _call_eval_demonstration_scores(cmd_args: list) -> None:
    from eval_demonstration_scores import main  # noqa: PLC0415

    main(cmd_args, standalone_mode=False)


class ComputeDemonstrationScoresStep(PipelineStep[None]):
    """Evaluate demonstration scores for each seed."""

    name = "compute_demonstration_scores"

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
        eval_as_train_seed = OmegaConf.select(attribution, "eval_as_train_seed")
        if eval_as_train_seed is None:
            eval_as_train_seed = True
        train_output_dir = OmegaConf.select(attribution, "train_output_dir") or "data/outputs/train"
        eval_output_dir = OmegaConf.select(attribution, "eval_output_dir") or "data/outputs/eval_save_episodes"

        result_date = OmegaConf.select(attribution, "result_date") or "default"
        exp_seed = OmegaConf.select(attribution, "seed") or 0
        exp_name = f"{result_date}_demonstration_scores-seed={exp_seed}"

        for seed in seeds:
            train_dir = str(self.repo_root / get_train_dir(train_output_dir, train_date, task, policy, seed))
            eval_dir = str(self.repo_root / get_eval_dir(
                eval_output_dir, eval_date, task, policy, seed,
                train_ckpt=train_ckpt, eval_as_train_seed=eval_as_train_seed,
            ))

            compute_holdout = OmegaConf.select(attribution, "compute_holdout")
            eval_trak = OmegaConf.select(attribution, "eval_online_trak_influence")
            cmd_args = [
                f"--exp_name={exp_name}",
                f"--eval_dir={eval_dir}",
                f"--train_dir={train_dir}",
                f"--train_ckpt={train_ckpt}",
                f"--result_date={result_date}",
                f"--overwrite={OmegaConf.select(attribution, 'overwrite') or False}",
                f"--device={OmegaConf.select(attribution, 'device') or OmegaConf.select(cfg, 'device') or 'cpu'}",
                f"--seed={exp_seed}",
                f"--use_half_precision={OmegaConf.select(attribution, 'use_half_precision') or False}",
                f"--compute_holdout={compute_holdout if compute_holdout is not None else True}",
                f"--eval_offline_policy_loss={OmegaConf.select(attribution, 'eval_offline_policy_loss') or False}",
                f"--eval_offline_action_diversity={OmegaConf.select(attribution, 'eval_offline_action_diversity') or False}",
                f"--eval_offline_state_diversity={OmegaConf.select(attribution, 'eval_offline_state_diversity') or False}",
                f"--eval_online_state_similarity={OmegaConf.select(attribution, 'eval_online_state_similarity') or False}",
                f"--eval_online_demo_score={OmegaConf.select(attribution, 'eval_online_demo_score') or False}",
                f"--eval_online_trak_influence={eval_trak if eval_trak is not None else True}",
            ]

            if self.dry_run:
                print(f"[dry_run] ComputeDemonstrationScoresStep seed={seed}")
                print(f"[dry_run]   {' '.join(cmd_args)}")
                continue

            print(f"  [compute_demonstration_scores] seed={seed}")
            _call_eval_demonstration_scores(cmd_args=cmd_args)
