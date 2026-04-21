"""Lightweight wandb helpers for ENAP pipeline steps.

Pipeline steps that do in-process training (ENAP RNN, ResidualMLP) call
:func:`init_wandb_run` at the start of ``compute()`` and
:func:`finish_wandb_run` at the end.  If ``cfg.wandb.enabled`` is false
(the default) or wandb is not installed, these are silent no-ops.

Config expected under ``cfg.wandb``::

    wandb:
      enabled: false          # set true to activate
      project: policy_doctor_enap
      entity: null            # wandb entity / team (null = default)
      group: null             # run group for sweeps / experiments
      tags: []                # extra tags appended to step-specific ones
"""

from __future__ import annotations

from typing import Any, Optional

from omegaconf import OmegaConf


def init_wandb_run(
    cfg,
    step_name: str,
    extra_config: Optional[dict] = None,
) -> Any:
    """Initialise a wandb run for an in-process training step.

    Returns the ``wandb.Run`` object on success, or ``None`` if wandb is
    disabled / unavailable.

    Args:
        cfg: Pipeline OmegaConf config.
        step_name: Name of the pipeline step (used as the run name suffix
            and as a tag).
        extra_config: Additional config dict logged to wandb.
    """
    wandb_cfg = OmegaConf.select(cfg, "wandb") or {}
    if not wandb_cfg.get("enabled", False):
        return None

    try:
        import wandb
    except ImportError:
        return None

    run_name_base = OmegaConf.select(cfg, "run_name") or ""
    run_name = f"{step_name}_{run_name_base}" if run_name_base else step_name

    tags = list(wandb_cfg.get("tags", []))
    tags.append(step_name)

    log_cfg: dict = {}
    if extra_config:
        log_cfg.update(extra_config)
    # Always log the ENAP hyper-params so runs are reproducible
    enap_cfg = OmegaConf.select(cfg, "graph_building.enap")
    if enap_cfg is not None:
        log_cfg["enap"] = OmegaConf.to_container(enap_cfg, resolve=True)

    run = wandb.init(
        project=wandb_cfg.get("project", "policy_doctor_enap"),
        entity=wandb_cfg.get("entity") or None,
        group=wandb_cfg.get("group") or None,
        name=run_name,
        tags=tags,
        config=log_cfg or None,
        reinit=True,
    )
    return run


def finish_wandb_run(run: Any, summary: Optional[dict] = None) -> None:
    """Finish an active wandb run, optionally logging summary metrics.

    Safe to call with ``run=None`` (no-op).
    """
    if run is None:
        return
    if summary:
        try:
            import wandb
            if wandb.run is not None:
                for k, v in summary.items():
                    wandb.run.summary[k] = v
        except ImportError:
            pass
    try:
        run.finish()
    except Exception:
        pass
