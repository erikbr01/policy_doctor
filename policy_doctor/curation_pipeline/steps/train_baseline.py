"""Train baseline policy — pipeline step."""

from __future__ import annotations

import pathlib
import subprocess

from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.curation_pipeline.diffusion_overrides import baseline_diffusion_extra_overrides
from policy_doctor.curation_pipeline.paths import expand_seeds, get_train_name
from policy_doctor.paths import CUPID_CONDA_ENV_NAME, CUPID_ROOT


def _train_baseline_worker(
    run_output_dir: str,
    config_dir_str: str,
    config_name: str,
    overrides: list,
) -> None:
    """Hydra-compose training in an isolated child process.

    Must run in a separate process because ``hydra.initialize_config_dir``
    manages a global singleton that cannot be re-initialized in-process.
    """
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=config_dir_str, version_base=None)
    try:
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
        OmegaConf.resolve(cfg)
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=run_output_dir)
        workspace.run()
    finally:
        GlobalHydra.instance().clear()


class TrainBaselineStep(PipelineStep[None]):
    """Train the baseline policy for each seed.

    For data sources using the ``cupid`` conda env (robomimic, MimicGen), training
    runs via ``hydra.initialize_config_dir`` in an isolated child process.

    For data sources that require a different conda env (e.g. ``robocasa`` for
    RoboCasa image training), training is dispatched via ``conda run`` in a
    subprocess so that the correct interpreter and packages are used.  Set
    ``baseline.conda_env: robocasa`` (or the appropriate env name) in the baseline
    config to activate this path.  The ``data_source.conda_env_train`` field is
    used as a fallback.
    """

    name = "train_baseline"

    def compute(self) -> None:
        cfg = self.cfg
        baseline = OmegaConf.select(cfg, "baseline") or {}

        config_dir = OmegaConf.select(baseline, "config_dir") or OmegaConf.select(cfg, "config_dir")
        if not config_dir:
            raise ValueError("baseline.config_dir is required")
        config_dir_abs = str((self.repo_root / config_dir).resolve())
        if not pathlib.Path(config_dir_abs).exists():
            raise FileNotFoundError(f"Config dir not found: {config_dir_abs}")

        config_name = OmegaConf.select(baseline, "config_name") or "config.yaml"
        task = OmegaConf.select(baseline, "task") or OmegaConf.select(cfg, "task")
        policy = OmegaConf.select(baseline, "policy")
        # Prefer top-level cfg.seeds (CLI / experiment) over baseline.seeds from task YAML.
        seeds_raw = OmegaConf.select(cfg, "seeds")
        if seeds_raw is None:
            seeds_raw = OmegaConf.select(baseline, "seeds")
        seeds = expand_seeds(seeds_raw or [0])
        num_epochs = OmegaConf.select(baseline, "num_epochs") or 1001
        checkpoint_topk = OmegaConf.select(baseline, "checkpoint_topk") or 3
        checkpoint_every = OmegaConf.select(baseline, "checkpoint_every") or 50
        train_ratio = OmegaConf.select(baseline, "train_ratio") or 0.64
        val_ratio = OmegaConf.select(baseline, "val_ratio") or 0.04
        max_train_episodes_raw = OmegaConf.select(baseline, "max_train_episodes")
        max_train_episodes: int | None = int(max_train_episodes_raw) if max_train_episodes_raw is not None else None
        uniform_quality = OmegaConf.select(baseline, "uniform_quality")
        if uniform_quality is None:
            uniform_quality = True
        output_dir = OmegaConf.select(baseline, "output_dir") or "data/outputs/train"
        # Prefer top-level cfg.project (experiment / CLI) over baseline.project from task YAML.
        project = OmegaConf.select(cfg, "project") or OmegaConf.select(baseline, "project") or "influence-clustering"
        train_date = OmegaConf.select(baseline, "train_date") or OmegaConf.select(cfg, "train_date") or "default"
        script_name = OmegaConf.select(baseline, "script_name") or "train"
        exp_name = OmegaConf.select(baseline, "exp_name") or f"{script_name}_{policy}"
        device = OmegaConf.select(cfg, "device") or "cuda:0"
        num_gpus = int(OmegaConf.select(baseline, "num_gpus") or OmegaConf.select(cfg, "num_gpus") or 1)
        tf32    = bool(OmegaConf.select(baseline, "tf32") or False)
        compile_ = bool(OmegaConf.select(baseline, "compile") or False)

        # Determine training dispatch mode.
        # Always use subprocess when a conda_env is specified — the pipeline orchestrator runs
        # in the policy_doctor env, which does not have cupid's training packages installed.
        conda_env = (
            OmegaConf.select(baseline, "conda_env")
            or OmegaConf.select(cfg, "data_source.conda_env_train")
        )
        use_subprocess = bool(conda_env)

        for seed in seeds:
            train_name = get_train_name(train_date, task, policy, seed)
            run_output_dir = str(self.repo_root / output_dir / train_date / train_name)

            if use_subprocess:
                # Subprocess mode: invoke `conda run -n <env> python train.py` so that
                # the correct interpreter (e.g. cupid_torch2 env) is used.
                overrides = self._subprocess_overrides(
                    exp_name=exp_name,
                    device=device,
                    seed=seed,
                    num_epochs=num_epochs,
                    checkpoint_topk=checkpoint_topk,
                    checkpoint_every=checkpoint_every,
                    val_ratio=val_ratio,
                    train_name=train_name,
                    train_date=train_date,
                    task=task,
                    policy=policy,
                    project=project,
                    run_output_dir=run_output_dir,
                    max_train_episodes=max_train_episodes,
                )
                overrides.extend(baseline_diffusion_extra_overrides(baseline))
                overrides.extend([
                    f"+training.tf32={str(tf32).lower()}",
                    f"+training.compile={str(compile_).lower()}",
                ])

                if self.dry_run:
                    print(f"[dry_run] TrainBaselineStep (subprocess) seed={seed}")
                    print(f"[dry_run]   conda_env={conda_env}  config_dir={config_dir}")
                    print(f"[dry_run]   output_dir={run_output_dir}")
                    print(f"[dry_run]   overrides={overrides}")
                    continue

                print(f"  [train_baseline/subprocess] conda_env={conda_env}  seed={seed}  output_dir={run_output_dir}")
                self._run_subprocess_train(
                    conda_env=conda_env,
                    config_dir=config_dir,
                    config_name=config_name,
                    overrides=overrides,
                    run_output_dir=run_output_dir,
                    num_gpus=num_gpus,
                )
            else:
                # In-process (hydra.initialize_config_dir) mode: standard cupid training.
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
                    *(
                        [f"task.dataset.max_train_episodes={max_train_episodes}"]
                        if max_train_episodes is not None
                        else [
                            f"+task.dataset.dataset_mask_kwargs.train_ratio={train_ratio}",
                            f"+task.dataset.dataset_mask_kwargs.uniform_quality={uniform_quality}",
                        ]
                    ),
                    f"logging.name={train_name}",
                    f"logging.group={train_date}_{exp_name}_{task}",
                    f"logging.project={project}",
                    f"multi_run.wandb_name_base={train_name}",
                    f"multi_run.run_dir={run_output_dir}",
                    f"+training.tf32={str(tf32).lower()}",
                    f"+training.compile={str(compile_).lower()}",
                ]
                overrides.extend(baseline_diffusion_extra_overrides(baseline))

                if self.dry_run:
                    print(f"[dry_run] TrainBaselineStep seed={seed}")
                    print(f"[dry_run]   config_dir={config_dir_abs}  config_name={config_name}")
                    print(f"[dry_run]   output_dir={run_output_dir}")
                    print(f"[dry_run]   overrides={overrides}")
                    continue

                worker_kwargs = {
                    "run_output_dir": run_output_dir,
                    "config_dir_str": config_dir_abs,
                    "config_name": config_name,
                    "overrides": overrides,
                }
                if num_gpus > 1:
                    from diffusion_policy.common.ddp_util import spawn_ddp
                    print(f"  [train_baseline/ddp] num_gpus={num_gpus}  seed={seed}  output_dir={run_output_dir}")
                    spawn_ddp(
                        worker_fn=_train_baseline_worker,
                        worker_kwargs=worker_kwargs,
                        num_gpus=num_gpus,
                    )
                else:
                    print(f"  [train_baseline] seed={seed}  output_dir={run_output_dir}")
                    self._run_in_process(_train_baseline_worker, worker_kwargs)

    # ------------------------------------------------------------------
    # Subprocess training helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _subprocess_overrides(
        exp_name: str,
        device: str,
        seed,
        num_epochs: int,
        checkpoint_topk: int,
        checkpoint_every: int,
        val_ratio: float,
        train_name: str,
        train_date: str,
        task: str,
        policy: str,
        project: str,
        run_output_dir: str,
        max_train_episodes: "int | None" = None,
    ) -> list:
        """Build the core Hydra override list for subprocess training.

        Omits ``dataset_mask_kwargs`` / ``train_ratio`` overrides that only apply
        to HDF5-backed cupid dataset classes.  Pass ``max_train_episodes`` to cap
        the training split (e.g. for baseline vs. combined-data conditions).
        """
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
            # Override Hydra's output dir so checkpoints land in run_output_dir
            # instead of the datetime-based default (outputs/YYYY-MM-DD/HH-MM-SS/).
            f"hydra.run.dir={run_output_dir}",
        ]
        if max_train_episodes is not None:
            overrides.append(f"task.dataset.max_train_episodes={max_train_episodes}")
        return overrides

    def _run_subprocess_train(
        self,
        conda_env: str,
        config_dir: str,
        config_name: str,
        overrides: list,
        run_output_dir: str,
        num_gpus: int = 1,
    ) -> None:
        """Run training in an isolated conda env via ``conda run``.

        ``config_dir`` must be relative to ``CUPID_ROOT`` (e.g.
        ``configs/image/robocasa_lerobot_atomic/diffusion_policy_transformer``).
        The subprocess is run with ``cwd=CUPID_ROOT`` so that Hydra's
        ``--config-path`` resolves correctly.

        When ``num_gpus > 1``, ``torchrun`` is used instead of plain
        ``python`` so that each GPU gets its own rank process.
        """
        pathlib.Path(run_output_dir).mkdir(parents=True, exist_ok=True)
        if num_gpus > 1:
            launcher = [
                "torchrun",
                f"--nproc_per_node={num_gpus}",
                str(CUPID_ROOT / "train.py"),
            ]
        else:
            launcher = ["python", str(CUPID_ROOT / "train.py")]
        cmd = [
            "conda", "run", "-n", conda_env, "--no-capture-output",
            *launcher,
            "--config-path", config_dir,
            "--config-name", config_name,
            *overrides,
        ]
        result = subprocess.run(cmd, cwd=str(CUPID_ROOT))
        if result.returncode != 0:
            raise RuntimeError(
                f"[train_baseline] subprocess (conda_env={conda_env}) failed with exit code {result.returncode}"
            )
