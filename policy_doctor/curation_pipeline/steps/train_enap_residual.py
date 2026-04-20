"""Train ENAP ResidualMLP — M-step pipeline step.

Implements the **M-step** of the ENAP EM algorithm:

    PMM (frozen) + visual features → ResidualMLP training → refined policy

Loads the PMM from ``extract_enap_graph``, the visual features and cluster
centres from ``train_enap_perception``, and the node assignments from
``extract_enap_graph``, then trains a :class:`~policy_doctor.enap.residual_policy.ResidualMLP`
to refine PMM action priors using the current visual context.

Saved outputs (in ``step_dir/``):
- ``residual_checkpoint.pt``  — best :class:`ResidualMLP` checkpoint
- ``result.json`` / ``done``  — standard PipelineStep outputs
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep


class TrainENAPResidualStep(PipelineStep[Dict[str, Any]]):
    """Train ResidualMLP conditioned on the learned PMM topology.

    Config keys consumed (all under ``graph_building.enap``):
    - ``residual_epochs``: training epochs (default 300)
    - ``residual_lr``: Adam learning rate (default 3e-4)
    - ``residual_batch_size``: mini-batch size (default 512)
    - ``residual_noise_std``: feature/action noise injection (default 0.01)
    - ``residual_hidden``: hidden layer width (default 512)
    - ``residual_val_fraction``: validation fraction (default 0.1)
    - ``device``: torch device string
    """

    name = "train_enap_residual"

    def save(self, result: Dict[str, Any]) -> None:
        self.step_dir.mkdir(parents=True, exist_ok=True)
        with open(self.step_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        (self.step_dir / "done").touch()

    def load(self) -> Optional[Dict[str, Any]]:
        p = self.step_dir / "result.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return None

    def compute(self) -> Dict[str, Any]:
        from policy_doctor.enap.pmm import PMM
        from policy_doctor.enap.residual_policy import ResidualMLP, train_residual_mlp
        from policy_doctor.curation_pipeline.steps.extract_enap_graph import (
            ExtractENAPGraphStep,
        )
        from policy_doctor.curation_pipeline.steps.train_enap_perception import (
            TrainENAPPerceptionStep,
        )

        cfg = self.cfg
        enap_cfg = OmegaConf.select(cfg, "graph_building.enap") or {}

        # --- Verify prior steps ---
        graph_prior = ExtractENAPGraphStep(cfg, self.run_dir).load()
        if not graph_prior:
            raise RuntimeError(
                "train_enap_residual: extract_enap_graph has not been completed yet."
            )
        perception_prior = TrainENAPPerceptionStep(cfg, self.run_dir).load()
        if not perception_prior:
            raise RuntimeError(
                "train_enap_residual: train_enap_perception has not been completed yet."
            )

        graph_dir = self.run_dir / "extract_enap_graph"
        perception_dir = self.run_dir / "train_enap_perception"

        if self.dry_run:
            print("  [dry_run] TrainENAPResidualStep: would train ResidualMLP")
            return {"dry_run": True}

        # --- Load data ---
        features = np.load(perception_dir / "feature_embeddings.npy")  # (N, feat_dim)
        symbols = np.load(perception_dir / "symbol_assignments.npy")    # (N,)
        actions = np.load(graph_dir / "actions.npy")                    # (N, a_dim)
        node_assignments = np.load(graph_dir / "node_assignments.npy")  # (N,)
        cluster_centers = np.load(graph_dir / "cluster_centers.npy")   # (K, feat_dim)

        feat_dim = int(features.shape[1])
        a_dim = int(actions.shape[1] if actions.ndim > 1 else 1)
        print(
            f"  Data: {len(features)} timesteps, "
            f"feat_dim={feat_dim}, a_dim={a_dim}"
        )

        # --- Load PMM ---
        pmm_pkl = graph_dir / "pmm.pkl"
        pmm_obj = PMM()
        pmm_obj.load_pmm(str(pmm_pkl))
        print(
            f"  PMM loaded: {len(pmm_obj.pmm['Q'])} nodes, "
            f"{sum(len(d) for d in pmm_obj.pmm['delta'].values())} edges"
        )

        # --- Config ---
        num_epochs = int(OmegaConf.select(enap_cfg, "residual_epochs") or 300)
        lr = float(OmegaConf.select(enap_cfg, "residual_lr") or 3e-4)
        batch_size = int(OmegaConf.select(enap_cfg, "residual_batch_size") or 512)
        noise_std = float(OmegaConf.select(enap_cfg, "residual_noise_std") or 0.01)
        hidden = int(OmegaConf.select(enap_cfg, "residual_hidden") or 512)
        val_fraction = float(OmegaConf.select(enap_cfg, "residual_val_fraction") or 0.1)

        device_str = str(
            OmegaConf.select(enap_cfg, "device")
            or OmegaConf.select(cfg, "device")
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        device = torch.device(device_str)

        # --- Build and train ResidualMLP ---
        model = ResidualMLP(feat_dim=feat_dim, a_dim=a_dim, hidden=hidden)
        print(
            f"  ResidualMLP: feat_dim={feat_dim}, a_dim={a_dim}, "
            f"hidden={hidden}, epochs={num_epochs}"
        )
        stats = train_residual_mlp(
            model=model,
            features=features,
            actions=actions,
            symbols=symbols,
            node_assignments=node_assignments,
            cluster_centers=cluster_centers,
            pmm=pmm_obj,
            num_epochs=num_epochs,
            lr=lr,
            batch_size=batch_size,
            noise_std=noise_std,
            val_fraction=val_fraction,
            device=device,
            verbose=True,
        )

        # --- Persist ---
        self.step_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = str(self.step_dir / "residual_checkpoint.pt")
        model.save_checkpoint(ckpt_path)

        return {
            "feat_dim": feat_dim,
            "a_dim": a_dim,
            "num_epochs": num_epochs,
            "best_epoch": int(stats["best_epoch"]),
            "best_val_loss": float(stats["best_val_loss"]),
            "checkpoint_path": ckpt_path,
        }
