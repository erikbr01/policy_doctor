"""Train GRU-based ENAP RNN encoder (custom/modified variant) — pipeline step.

This is the *modified* ENAP E-step variant that uses a GRU
(:class:`~policy_doctor.enap.rnn_encoder.ENAPRNNEncoder`) rather than the
vanilla RNN used by the faithful port (:class:`~policy_doctor.enap.rnn_encoder.PretrainRNN`).

It produces hidden states ``h_t`` that are loaded directly by
:class:`~policy_doctor.curation_pipeline.steps.extract_enap_graph_custom.ExtractENAPGraphCustomStep`
for the Extended L* algorithm.

Saved outputs (in ``step_dir/``):
- ``hidden_states.npy``   — ``(N, hidden_dim)`` per-timestep h_t
- ``rnn_checkpoint.pt``   — ``ENAPRNNEncoder`` state dict
- ``result.json`` / ``done``
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep


class TrainENAPRNNCustomStep(PipelineStep[Dict[str, Any]]):
    """Train GRU encoder (modified ENAP) on (action, symbol) sequences → h_t.

    Config keys consumed (all under ``graph_building.enap``):
    same set as :class:`TrainENAPRNNStep` (``rnn_hidden_dim``, ``rnn_epochs``,
    etc.) — both variants share the same hyper-parameter namespace.
    """

    name = "train_enap_rnn_custom"

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
        from policy_doctor.enap.rnn_encoder import (
            ENAPRNNEncoder,
            extract_hidden_states,
            train_rnn_encoder,
        )
        from policy_doctor.curation_pipeline.steps.train_enap_perception import (
            TrainENAPPerceptionStep,
        )

        cfg = self.cfg
        enap_cfg = OmegaConf.select(cfg, "graph_building.enap") or {}

        perception_prior = TrainENAPPerceptionStep(cfg, self.run_dir).load()
        if not perception_prior:
            raise RuntimeError(
                "train_enap_rnn_custom: train_enap_perception has not been completed yet."
            )
        perception_dir = self.run_dir / "train_enap_perception"

        if self.dry_run:
            print("  [dry_run] TrainENAPRNNCustomStep: would train GRU encoder")
            return {"dry_run": True}

        symbols_all = np.load(perception_dir / "symbol_assignments.npy")
        actions_all = np.load(perception_dir / "actions.npy")
        with open(perception_dir / "metadata.json") as f:
            metadata: List[Dict] = json.load(f)

        num_symbols = int(perception_prior["num_symbols"])
        action_dim = int(actions_all.shape[1] if actions_all.ndim > 1 else 1)

        # Build per-episode lists
        episodes_actions: List[np.ndarray] = []
        episodes_symbols: List[np.ndarray] = []
        current_ep: Optional[int] = None
        ep_acts: List[np.ndarray] = []
        ep_syms: List[int] = []

        for i, meta in enumerate(metadata):
            ep_idx = meta["rollout_idx"]
            if current_ep is None:
                current_ep = ep_idx
            if ep_idx != current_ep:
                if ep_acts:
                    episodes_actions.append(np.array(ep_acts, dtype=np.float32))
                    episodes_symbols.append(np.array(ep_syms, dtype=np.int64))
                ep_acts, ep_syms, current_ep = [], [], ep_idx
            act = actions_all[i]
            if act.ndim == 0:
                act = act.reshape(1)
            ep_acts.append(act)
            ep_syms.append(int(symbols_all[i]))

        if ep_acts:
            episodes_actions.append(np.array(ep_acts, dtype=np.float32))
            episodes_symbols.append(np.array(ep_syms, dtype=np.int64))

        print(f"  Episodes: {len(episodes_actions)}")

        hidden_dim = int(OmegaConf.select(enap_cfg, "rnn_hidden_dim") or 64)
        symbol_embed_dim = int(OmegaConf.select(enap_cfg, "rnn_symbol_embed_dim") or 16)
        num_layers = int(OmegaConf.select(enap_cfg, "rnn_num_layers") or 1)
        num_epochs = int(OmegaConf.select(enap_cfg, "rnn_epochs") or 50)
        batch_size = int(OmegaConf.select(enap_cfg, "rnn_batch_size") or 32)
        lr = float(OmegaConf.select(enap_cfg, "rnn_lr") or 1e-3)
        contrastive_margin = float(OmegaConf.select(enap_cfg, "rnn_contrastive_margin") or 0.5)

        loss_weights_cfg = OmegaConf.select(enap_cfg, "rnn_loss_weights")
        loss_weights = (
            {k: float(v) for k, v in loss_weights_cfg.items()}
            if loss_weights_cfg is not None
            else None
        )

        device_str = str(
            OmegaConf.select(enap_cfg, "device")
            or OmegaConf.select(cfg, "device")
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        device = torch.device(device_str)

        encoder = ENAPRNNEncoder(
            action_dim=action_dim,
            num_symbols=num_symbols,
            hidden_dim=hidden_dim,
            symbol_embed_dim=symbol_embed_dim,
            num_layers=num_layers,
        )
        print(
            f"  ENAPRNNEncoder (GRU): action_dim={action_dim}, num_symbols={num_symbols}, "
            f"hidden_dim={hidden_dim}, epochs={num_epochs}"
        )
        train_rnn_encoder(
            encoder=encoder,
            episodes_actions=episodes_actions,
            episodes_symbols=episodes_symbols,
            num_epochs=num_epochs,
            lr=lr,
            batch_size=batch_size,
            loss_weights=loss_weights,
            contrastive_margin=contrastive_margin,
            device=device,
            verbose=True,
        )

        h_all, _ = extract_hidden_states(
            encoder=encoder,
            episodes_actions=episodes_actions,
            episodes_symbols=episodes_symbols,
            device=device,
        )
        print(f"  Hidden states: {h_all.shape}")

        self.step_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.step_dir / "hidden_states.npy", h_all)
        torch.save(encoder.state_dict(), self.step_dir / "rnn_checkpoint.pt")

        return {
            "num_timesteps": int(h_all.shape[0]),
            "hidden_dim": hidden_dim,
            "action_dim": action_dim,
            "num_symbols": num_symbols,
            "num_episodes": len(episodes_actions),
        }
