"""Train ENAP PretrainRNN encoder — pipeline step.

Implements **Stage 2** of the ENAP E-step:

    (a_t, c_t) → PretrainRNN (vanilla RNN + PER) → checkpoint

Loads symbol assignments and actions from the completed
``train_enap_perception`` step, trains the vanilla RNN with prioritised
experience replay and the multi-objective phase-aware contrastive loss
(faithful to the ENAP repository's ``rnn_train.py``), then saves a
PMM-compatible checkpoint that :class:`~policy_doctor.enap.pmm.PMM` can load
directly.

Saved outputs (in ``step_dir/``):
- ``pretrain_checkpoint.pt``  — PMM-compatible checkpoint
  ``{'model_state': ..., 'dims': {'a','s','e','h'}}``
- ``result.json`` / ``done``  — standard PipelineStep outputs
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep


class TrainENAPRNNStep(PipelineStep[Dict[str, Any]]):
    """Train PretrainRNN on (action, symbol) sequences.

    Config keys consumed (all under ``graph_building.enap``):
    - ``rnn_hidden_dim``: RNN hidden state dimension (default 64)
    - ``rnn_symbol_embed_dim``: symbol embedding dim (default 16)
    - ``rnn_epochs``: training epochs (default 100)
    - ``rnn_batch_size``: episodes per mini-batch (default 32)
    - ``rnn_lr``: Adam learning rate (default 1e-3)
    - ``rnn_loss_weights``: ``{act, state, contrast}`` weight dict
    - ``rnn_contrastive_margin``: margin for phase-aware loss (default 0.5)
    - ``rnn_noise_std``: action noise injection std (default 0.01)
    - ``rnn_use_per``: use Prioritized Experience Replay (default True)
    - ``device``: torch device string
    """

    name = "train_enap_rnn"

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
        from policy_doctor.enap.rnn_encoder import PretrainRNN, train_pretrain_rnn
        from policy_doctor.curation_pipeline.steps.train_enap_perception import (
            TrainENAPPerceptionStep,
        )

        cfg = self.cfg
        enap_cfg = OmegaConf.select(cfg, "graph_building.enap") or {}

        # --- Load perception step outputs ---
        perception_prior = TrainENAPPerceptionStep(cfg, self.run_dir).load()
        if not perception_prior:
            raise RuntimeError(
                "train_enap_rnn: train_enap_perception has not been completed yet."
            )
        perception_dir = self.run_dir / "train_enap_perception"

        if self.dry_run:
            print("  [dry_run] TrainENAPRNNStep: would train PretrainRNN")
            return {"dry_run": True}

        symbols_all = np.load(perception_dir / "symbol_assignments.npy")  # (N,) int
        actions_all = np.load(perception_dir / "actions.npy")              # (N, a_dim)
        with open(perception_dir / "metadata.json") as f:
            metadata: List[Dict] = json.load(f)

        num_symbols = int(perception_prior["num_symbols"])
        action_dim = int(actions_all.shape[1] if actions_all.ndim > 1 else 1)

        # --- Split flat arrays into per-episode sequences ---
        # Build episodes in PMM format: {'S': (T, s_dim) one-hot, 'A': (T, a_dim)}
        episodes: List[Dict] = []
        current_ep: Optional[int] = None
        ep_acts: List[np.ndarray] = []
        ep_syms: List[int] = []

        def _flush_episode(ep_acts, ep_syms):
            if not ep_acts:
                return
            T = len(ep_acts)
            A = np.array(ep_acts, dtype=np.float32)
            S_idx = np.array(ep_syms, dtype=np.int64)
            S_onehot = np.zeros((T, num_symbols), dtype=np.float32)
            S_onehot[np.arange(T), S_idx] = 1.0
            episodes.append({"S": S_onehot, "A": A})

        for i, meta in enumerate(metadata):
            ep_idx = meta["rollout_idx"]
            if current_ep is None:
                current_ep = ep_idx
            if ep_idx != current_ep:
                _flush_episode(ep_acts, ep_syms)
                ep_acts = []
                ep_syms = []
                current_ep = ep_idx
            act = actions_all[i]
            if act.ndim == 0:
                act = act.reshape(1)
            ep_acts.append(act)
            ep_syms.append(int(symbols_all[i]))

        _flush_episode(ep_acts, ep_syms)

        print(f"  Episodes: {len(episodes)}, action_dim={action_dim}, num_symbols={num_symbols}")

        # --- Config ---
        hidden_dim = int(OmegaConf.select(enap_cfg, "rnn_hidden_dim") or 64)
        embed_dim = int(OmegaConf.select(enap_cfg, "rnn_symbol_embed_dim") or 16)
        num_epochs = int(OmegaConf.select(enap_cfg, "rnn_epochs") or 100)
        batch_size = int(OmegaConf.select(enap_cfg, "rnn_batch_size") or 32)
        lr = float(OmegaConf.select(enap_cfg, "rnn_lr") or 1e-3)
        contrastive_margin = float(
            OmegaConf.select(enap_cfg, "rnn_contrastive_margin") or 0.5
        )
        noise_std = float(OmegaConf.select(enap_cfg, "rnn_noise_std") or 0.01)
        use_per = bool(OmegaConf.select(enap_cfg, "rnn_use_per") if OmegaConf.select(enap_cfg, "rnn_use_per") is not None else True)

        loss_weights_cfg = OmegaConf.select(enap_cfg, "rnn_loss_weights")
        if loss_weights_cfg is not None:
            loss_weights = {
                "act": float(loss_weights_cfg.get("act", 1.0)),
                "state": float(loss_weights_cfg.get("state", 1.0)),
                "contrast": float(loss_weights_cfg.get("contrast", 0.5)),
            }
        else:
            loss_weights = None

        device_str = str(
            OmegaConf.select(enap_cfg, "device")
            or OmegaConf.select(cfg, "device")
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        device = torch.device(device_str)

        # --- Build and train PretrainRNN ---
        model = PretrainRNN(
            a_dim=action_dim,
            s_dim=num_symbols,
            e_dim=embed_dim,
            h_dim=hidden_dim,
        )
        print(
            f"  PretrainRNN: a_dim={action_dim}, s_dim={num_symbols}, "
            f"e_dim={embed_dim}, h_dim={hidden_dim}, epochs={num_epochs}"
        )

        from policy_doctor.curation_pipeline.wandb_utils import (
            init_wandb_run, finish_wandb_run,
        )
        wandb_run = init_wandb_run(cfg, step_name=self.name)

        train_pretrain_rnn(
            model=model,
            episodes=episodes,
            num_epochs=num_epochs,
            lr=lr,
            batch_size=batch_size,
            loss_weights=loss_weights,
            contrastive_margin=contrastive_margin,
            device=device,
            verbose=True,
            noise_std=noise_std,
            use_per=use_per,
            wandb_prefix="enap_rnn",
        )

        # --- Persist (PMM-compatible checkpoint) ---
        self.step_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = str(self.step_dir / "pretrain_checkpoint.pt")
        model.save_checkpoint(ckpt_path)

        result = {
            "num_episodes": len(episodes),
            "hidden_dim": hidden_dim,
            "embed_dim": embed_dim,
            "action_dim": action_dim,
            "num_symbols": num_symbols,
            "num_epochs": num_epochs,
            "checkpoint_path": ckpt_path,
        }
        finish_wandb_run(wandb_run, summary=result)
        return result
