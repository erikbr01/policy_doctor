"""Residual MLP and PMMAgent for the ENAP M-step.

Faithful port of the ENAP repository's ``agent/pmm_agent.py`` and
``scripts/train/residual_train.py``.

Classes:
- :class:`ResidualMLP` — refines a PMM action prior given current visual
  features and the nearest HDBSCAN cluster centre.
- :class:`PMMAgent` — runtime agent that combines :class:`~policy_doctor.enap.pmm.PMM`
  structure with a trained :class:`ResidualMLP` for closed-loop execution.

Training entry point: :func:`train_residual_mlp`.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Residual MLP
# ---------------------------------------------------------------------------

class ResidualMLP(nn.Module):
    """Residual action-refinement network for the ENAP M-step.

    Given the current visual-proprioceptive embedding ``z_t``, the PMM action
    prior ``a_base = mean(actions on edge (q,x))``, and the HDBSCAN cluster
    centre for symbol ``x``, outputs a refined action:

        delta  = z_t − cluster_center_x
        output = net([delta; a_base]) + a_base

    This is a faithful port of ``ResidualMLP`` in ``pmm_agent.py``.

    Args:
        feat_dim: Dimension of the visual embedding ``z_t``.
        a_dim: Action dimension.
        hidden: Hidden layer width (default 512).
    """

    def __init__(self, feat_dim: int, a_dim: int, hidden: int = 512) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.a_dim = a_dim
        self.net = nn.Sequential(
            nn.Linear(feat_dim + a_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, a_dim),
        )
        self.logstd = nn.Parameter(torch.ones(1, a_dim) * -0.5)

    def forward(
        self,
        action_base: torch.Tensor,      # (B, a_dim) — PMM mean-action prior
        cluster_center: torch.Tensor,   # (B, feat_dim)
        concat_feat: torch.Tensor,      # (B, feat_dim) — current z_t
    ) -> torch.Tensor:
        """Return refined action ``(B, a_dim)``."""
        delta = concat_feat - cluster_center
        x = torch.cat([delta, action_base], dim=-1)
        return self.net(x) + action_base

    def get_log_prob(
        self,
        action_base: torch.Tensor,
        cluster_center: torch.Tensor,
        concat_feat: torch.Tensor,
        target_action: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian NLL log-probability for *target_action*, shape ``(B,)``."""
        mu = self.forward(action_base, cluster_center, concat_feat)
        std = torch.exp(self.logstd).clamp(min=1e-6).expand_as(mu)
        return torch.distributions.Normal(mu, std).log_prob(target_action).sum(-1)

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "feat_dim": self.feat_dim,
                "a_dim": self.a_dim,
                "hidden": self.net[0].out_features,
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls, path: str, device: Optional[torch.device] = None
    ) -> "ResidualMLP":
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        model = cls(ckpt["feat_dim"], ckpt["a_dim"], hidden=ckpt.get("hidden", 512))
        model.load_state_dict(ckpt["state_dict"])
        if device is not None:
            model.to(device)
        return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_residual_mlp(
    model: ResidualMLP,
    features: np.ndarray,           # (N, feat_dim) — z_t for all timesteps
    actions: np.ndarray,            # (N, a_dim)
    symbols: np.ndarray,            # (N,) int — HDBSCAN symbol c_t
    node_assignments: np.ndarray,   # (N,) int — PMM state q_t
    cluster_centers: np.ndarray,    # (num_symbols, feat_dim)
    pmm: Any,                       # PMM object with .predict(q, x) method
    num_epochs: int = 300,
    lr: float = 3e-4,
    batch_size: int = 512,
    noise_std: float = 0.01,
    val_fraction: float = 0.1,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train :class:`ResidualMLP` with Gaussian NLL loss.

    Faithful to ``residual_train.py`` in the ENAP repository.  Constructs a
    dataset of ``(z_t, c_t, q_t, a_t)`` tuples, computes PMM action priors
    and cluster centres on-the-fly, and minimises ``−log p(a_t | ...)`` using
    cosine-annealing Adam.

    Args:
        model: :class:`ResidualMLP` to train in-place.
        features: Visual embeddings ``(N, feat_dim)``.
        actions: Ground-truth actions ``(N, a_dim)``.
        symbols: HDBSCAN symbols ``(N,)`` int.
        node_assignments: PMM states ``(N,)`` int.
        cluster_centers: HDBSCAN centroids ``(num_symbols, feat_dim)``.
        pmm: Trained :class:`~policy_doctor.enap.pmm.PMM` with
            ``.predict(q, x)`` method returning the mean action prior.
        num_epochs: Training epochs (default 300).
        lr: Adam learning rate (default 3e-4).
        batch_size: Samples per mini-batch (default 512).
        noise_std: Action noise injection std (default 0.01).
        val_fraction: Fraction held out for validation (default 0.1).
        device: Training device.
        verbose: Print per-10-epoch summaries.

    Returns:
        Dict with ``'train_losses'``, ``'val_losses'``, and
        ``'best_epoch'`` keys.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    N = len(features)
    a_dim = model.a_dim
    feat_dim = model.feat_dim

    # Build dataset: (feat, a_base, cc, action_target)
    feat_list: List[np.ndarray] = []
    a_base_list: List[np.ndarray] = []
    cc_list: List[np.ndarray] = []
    action_list: List[np.ndarray] = []

    for i in range(N):
        q = int(node_assignments[i])
        x = int(symbols[i])
        try:
            a_base = pmm.predict(q, x)
        except (KeyError, ValueError):
            a_base = np.zeros(a_dim, dtype=np.float32)

        cc = cluster_centers[x].astype(np.float32)
        feat_list.append(features[i].astype(np.float32))
        a_base_list.append(a_base.astype(np.float32))
        cc_list.append(cc)
        action_list.append(actions[i].astype(np.float32))

    feat_arr = np.stack(feat_list)       # (N, feat_dim)
    a_base_arr = np.stack(a_base_list)   # (N, a_dim)
    cc_arr = np.stack(cc_list)           # (N, feat_dim)
    action_arr = np.stack(action_list)   # (N, a_dim)

    # Train / val split
    indices = np.random.permutation(N)
    n_val = max(1, int(N * val_fraction))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    def _to_device(*arrays):
        return [torch.from_numpy(a).to(device) for a in arrays]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(num_epochs):
        model.train()
        batch_start = np.random.permutation(train_idx)
        total_train_loss = 0.0
        n_batches = 0

        for start in range(0, len(batch_start), batch_size):
            bidx = batch_start[start : start + batch_size]
            if len(bidx) == 0:
                continue

            f_t, ab_t, cc_t, a_t = _to_device(
                feat_arr[bidx], a_base_arr[bidx], cc_arr[bidx], action_arr[bidx]
            )

            # Action noise injection
            if noise_std > 0.0:
                f_t = f_t + torch.randn_like(f_t) * noise_std
                ab_t = ab_t + torch.randn_like(ab_t) * noise_std

            log_prob = model.get_log_prob(ab_t, cc_t, f_t, a_t)
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += float(loss.detach())
            n_batches += 1

        scheduler.step()
        avg_train = total_train_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        # Validation
        model.eval()
        with torch.no_grad():
            f_v, ab_v, cc_v, a_v = _to_device(
                feat_arr[val_idx], a_base_arr[val_idx],
                cc_arr[val_idx], action_arr[val_idx],
            )
            val_loss = float(-model.get_log_prob(ab_v, cc_v, f_v, a_v).mean())
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"  [Residual epoch {epoch+1}/{num_epochs}] "
                f"train={avg_train:.4f}  val={val_loss:.4f}"
            )

    # Restore best checkpoint
    model.load_state_dict(best_state)
    model.eval()

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


# ---------------------------------------------------------------------------
# Runtime agent
# ---------------------------------------------------------------------------

class PMMAgent:
    """Runtime agent combining :class:`~policy_doctor.enap.pmm.PMM` structure
    with a trained :class:`ResidualMLP`.

    Inference loop at each timestep:

    1. Identify symbol ``c_t`` (nearest HDBSCAN centroid in feature space).
    2. Look up PMM action prior: ``a_base = PMM.predict(q_t, c_t)``.
    3. Refine: ``a_t = ResidualMLP(a_base, cluster_center[c_t], z_t)``.
    4. Advance PMM state: ``q_{t+1} = PMM.step(q_t, c_t, z_t)``.

    Args:
        pmm: Trained :class:`~policy_doctor.enap.pmm.PMM` object.
        residual_mlp: Trained :class:`ResidualMLP`.
        cluster_centers: HDBSCAN centroids ``(num_symbols, feat_dim)``.
        device: Inference device.
    """

    def __init__(
        self,
        pmm: Any,
        residual_mlp: ResidualMLP,
        cluster_centers: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> None:
        self.pmm = pmm
        self.residual_mlp = residual_mlp
        self.cluster_centers = cluster_centers
        self.device = device or torch.device("cpu")
        self.residual_mlp.to(self.device).eval()
        self._q: int = 0  # current PMM state

    def reset(self) -> None:
        self._q = 0

    @torch.no_grad()
    def act(self, z_t: np.ndarray, c_t: int) -> np.ndarray:
        """Produce an action given current visual embedding and symbol.

        Args:
            z_t: Visual-proprioceptive embedding ``(feat_dim,)``.
            c_t: HDBSCAN symbol index for current observation.

        Returns:
            Refined action ``(a_dim,)``.
        """
        try:
            a_base = self.pmm.predict(self._q, c_t)
        except (KeyError, ValueError):
            a_base = np.zeros(self.pmm.a_dim, dtype=np.float32)

        cc = self.cluster_centers[c_t]

        f_t = torch.from_numpy(z_t.astype(np.float32)).unsqueeze(0).to(self.device)
        ab_t = torch.from_numpy(a_base.astype(np.float32)).unsqueeze(0).to(self.device)
        cc_t = torch.from_numpy(cc.astype(np.float32)).unsqueeze(0).to(self.device)

        a_out = self.residual_mlp(ab_t, cc_t, f_t)
        action = a_out.squeeze(0).cpu().numpy()

        self._q = self.pmm.step(self._q, c_t, z_t)
        return action

    def identify_symbol(self, z_t: np.ndarray) -> int:
        """Nearest-centroid symbol assignment for a feature vector."""
        cc = self.cluster_centers  # (K, feat_dim)
        diffs = cc - z_t[None, :]
        dists = np.linalg.norm(diffs, axis=1)
        return int(np.argmin(dists))
