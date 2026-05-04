#!/usr/bin/env python
"""E2 sim runner: pull DemonstrationRequests from a proposal server and execute them.

Run from third_party/cupid/ in the cupid env:

    python ../../scripts/run_e2_sim.py task=square_mh \\
        train_dir=/path/to/train_dir \\
        proposal_server=http://localhost:5003 \\
        viz_url=http://localhost:5002 \\
        output_dir=/tmp/e2_demos

CRITICAL: never log request.target_cluster or request.source_condition. The
``GET /requests/active`` endpoint already strips them from the operator-view
payload — but if you read raw requests elsewhere, treat those fields as opaque.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure policy_doctor + diffusion_policy are importable from any cwd
_PD_ROOT = Path(__file__).resolve().parent.parent
_CUPID_ROOT = _PD_ROOT / "third_party" / "cupid"
for _p in [str(_PD_ROOT), str(_CUPID_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hydra
from omegaconf import DictConfig, OmegaConf

from policy_doctor.paths import CONFIGS_DIR


@hydra.main(config_path=str(CONFIGS_DIR / "e2"), config_name="run_sim", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    from policy_doctor.envs.e2_runner import run_e2_session

    n = run_e2_session(
        proposal_server_url=cfg.proposal_server,
        output_dir=Path(cfg.output_dir),
        task=cfg.task,
        train_dir=cfg.get("train_dir"),
        train_ckpt=cfg.get("train_ckpt", "best"),
        dataset_path=cfg.get("dataset_path"),
        device=cfg.get("device", "auto"),
        viz_url=cfg.get("viz_url"),
        dagger_config=cfg.get("dagger_config", "keyboard_default"),
        max_demos=cfg.get("max_demos"),
        max_steps=int(cfg.get("max_steps", 500)),
        random_actions=bool(cfg.get("random_actions", False)),
        random_action_scale=float(cfg.get("random_action_scale", 1.0)),
        random_seed=cfg.get("random_seed"),
        poll_interval_s=float(cfg.get("poll_interval_s", 2.0)),
    )
    print(f"[run_e2_sim] completed {n} demonstration(s)")


if __name__ == "__main__":
    main()
