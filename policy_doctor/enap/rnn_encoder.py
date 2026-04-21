"""GRU-based history encoder for ENAP with phase-aware contrastive loss.

Implements the Markovian History Encoding stage of ENAP:

    (a_{0:t}, c_{0:t}) → ENAPRNNEncoder → h_t   (64-dim Markovian state)

The encoder is trained with a multi-objective loss:

    L = w_act  · L_act     (MSE on next action prediction)
      + w_state · L_state   (CrossEntropy on next symbol prediction)
      + w_contrast · L_contrast (phase-aware margin contrastive loss)

The contrastive loss minimises cosine distance between consecutive h_t when
the symbol is unchanged (self-loop) and maximises it when the symbol changes
(phase transition).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Phase-aware contrastive loss
# ---------------------------------------------------------------------------

def compute_contrastive_loss(
    h_t: torch.Tensor,
    h_t1: torch.Tensor,
    c_t: torch.Tensor,
    c_t1: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """Phase-aware contrastive loss between consecutive hidden states.

    For consecutive timestep pairs ``(t, t+1)``:

    - If ``c_t == c_{t+1}`` (same phase / self-loop): minimise cosine distance
      → loss = ``(1 − cos_sim(h_t, h_{t+1})) / 2``
    - If ``c_t != c_{t+1}`` (phase transition): maximise cosine distance up to
      margin → loss = ``max(0, cos_sim(h_t, h_{t+1}) − (1 − margin))``

    Args:
        h_t: Hidden states at time t, shape ``(B, D)``.
        h_t1: Hidden states at time t+1, shape ``(B, D)``.
        c_t: Symbol at time t, shape ``(B,)`` int tensor.
        c_t1: Symbol at time t+1, shape ``(B,)`` int tensor.
        margin: Controls the inter-phase separation distance.
            A value of 0.5 means cos_sim should be ≤ 0.5 for transition pairs.

    Returns:
        Scalar loss tensor.
    """
    cos_sim = F.cosine_similarity(h_t, h_t1, dim=-1)  # (B,)
    same_phase = (c_t == c_t1).float()                  # 1 = same, 0 = transition

    # Same phase: want cos_sim → 1, loss = (1 - cos_sim) / 2 ∈ [0, 1]
    same_loss = (1.0 - cos_sim) / 2.0

    # Transition: want cos_sim → (1 - margin), penalise similarity above threshold
    transition_threshold = 1.0 - margin
    transition_loss = torch.clamp(cos_sim - transition_threshold, min=0.0)

    loss = same_phase * same_loss + (1.0 - same_phase) * transition_loss
    return loss.mean()


# ---------------------------------------------------------------------------
# ENAP RNN Encoder
# ---------------------------------------------------------------------------

class ENAPRNNEncoder(nn.Module):
    """GRU that compresses action-symbol history into Markovian embeddings h_t.

    Input at each timestep: ``[a_t; embed(c_t)]`` where ``a_t`` is the
    continuous action and ``embed(c_t)`` is a learnable embedding of the
    discrete symbol.

    Three prediction heads share the hidden state ``h_t``:

    - **Action head**: predicts ``a_{t+1}`` (MSE loss).
    - **Symbol head**: predicts ``c_{t+1}`` (CrossEntropy loss).
    - (Contrastive loss is computed externally via
      :func:`compute_contrastive_loss` on consecutive ``h_t`` pairs.)

    Args:
        action_dim: Dimensionality of continuous actions ``a_t``.
        num_symbols: Vocabulary size of the discrete symbol alphabet ``Σ``.
        hidden_dim: GRU hidden state dimension.
        symbol_embed_dim: Learnable symbol embedding dimension.
        num_layers: Number of stacked GRU layers.
    """

    def __init__(
        self,
        action_dim: int,
        num_symbols: int,
        hidden_dim: int = 64,
        symbol_embed_dim: int = 16,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.num_symbols = num_symbols
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.symbol_embed = nn.Embedding(num_symbols, symbol_embed_dim)
        gru_input_dim = action_dim + symbol_embed_dim
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.symbol_head = nn.Linear(hidden_dim, num_symbols)

    def forward(
        self,
        actions: torch.Tensor,
        symbols: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the GRU over a sequence.

        Args:
            actions: ``(B, T, action_dim)`` action sequence.
            symbols: ``(B, T)`` int symbol sequence.
            h0: Optional initial hidden state ``(num_layers, B, hidden_dim)``.

        Returns:
            Tuple of:
            - ``h_seq``: Hidden states ``(B, T, hidden_dim)``
            - ``a_pred``: Predicted next actions ``(B, T, action_dim)``
            - ``c_pred``: Predicted next symbol logits ``(B, T, num_symbols)``
        """
        sym_emb = self.symbol_embed(symbols)  # (B, T, symbol_embed_dim)
        gru_in = torch.cat([actions, sym_emb], dim=-1)  # (B, T, gru_input_dim)
        h_seq, _ = self.gru(gru_in, h0)  # (B, T, hidden_dim)
        a_pred = self.action_head(h_seq)   # (B, T, action_dim)
        c_pred = self.symbol_head(h_seq)   # (B, T, num_symbols)
        return h_seq, a_pred, c_pred

    @torch.no_grad()
    def encode_sequence(
        self,
        actions: torch.Tensor,
        symbols: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return hidden states h_t for a sequence (no grad)."""
        h_seq, _, _ = self.forward(actions, symbols, h0)
        return h_seq


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def compute_rnn_loss(
    a_pred: torch.Tensor,
    c_pred: torch.Tensor,
    h_seq: torch.Tensor,
    actions_target: torch.Tensor,
    symbols_target: torch.Tensor,
    loss_weights: Optional[Dict[str, float]] = None,
    contrastive_margin: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the combined multi-objective RNN training loss.

    Args:
        a_pred: Predicted actions ``(B, T, action_dim)`` — targets ``a_{t+1}``.
        c_pred: Predicted symbol logits ``(B, T, num_symbols)`` — targets ``c_{t+1}``.
        h_seq: Hidden states ``(B, T, hidden_dim)``.
        actions_target: Ground-truth actions shifted by 1 ``(B, T, action_dim)``.
        symbols_target: Ground-truth symbols shifted by 1 ``(B, T)`` int.
        loss_weights: Dict with keys ``"act"``, ``"state"``, ``"contrast"``
            (default: 1.0 each).
        contrastive_margin: Margin parameter for
            :func:`compute_contrastive_loss`.

    Returns:
        ``(total_loss, loss_components_dict)`` where the dict contains
        ``"act"``, ``"state"``, and ``"contrast"`` scalar values.
    """
    weights = loss_weights or {"act": 1.0, "state": 1.0, "contrast": 0.5}

    # Action prediction loss (MSE on next step)
    l_act = F.mse_loss(a_pred[:, :-1, :], actions_target[:, 1:, :])

    # Symbol prediction loss (CE on next step)
    B, T, V = c_pred.shape
    l_state = F.cross_entropy(
        c_pred[:, :-1, :].reshape(-1, V),
        symbols_target[:, 1:].reshape(-1),
    )

    # Contrastive loss on consecutive h_t pairs
    h_t = h_seq[:, :-1, :].reshape(-1, h_seq.shape[-1])   # (B*(T-1), D)
    h_t1 = h_seq[:, 1:, :].reshape(-1, h_seq.shape[-1])
    c_t = symbols_target[:, :-1].reshape(-1)
    c_t1 = symbols_target[:, 1:].reshape(-1)
    l_contrast = compute_contrastive_loss(h_t, h_t1, c_t, c_t1, margin=contrastive_margin)

    total = (
        weights.get("act", 1.0) * l_act
        + weights.get("state", 1.0) * l_state
        + weights.get("contrast", 0.5) * l_contrast
    )
    return total, {
        "act": float(l_act),
        "state": float(l_state),
        "contrast": float(l_contrast),
    }


def train_rnn_encoder(
    encoder: ENAPRNNEncoder,
    episodes_actions: List[np.ndarray],
    episodes_symbols: List[np.ndarray],
    num_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    loss_weights: Optional[Dict[str, float]] = None,
    contrastive_margin: float = 0.5,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> List[Dict[str, float]]:
    """Train the RNN encoder on per-episode action-symbol sequences.

    Sequences of different lengths are padded to the maximum length in each
    mini-batch (they are drawn from randomly shuffled episodes).

    Args:
        encoder: :class:`ENAPRNNEncoder` to train in-place.
        episodes_actions: List of per-episode action arrays, each of shape
            ``(T_i, action_dim)``.
        episodes_symbols: List of per-episode integer symbol arrays, each of
            shape ``(T_i,)``.
        num_epochs: Number of passes over all episodes.
        lr: Adam learning rate.
        batch_size: Number of episodes per mini-batch.
        loss_weights: ``{"act": float, "state": float, "contrast": float}``.
        contrastive_margin: Margin for the contrastive loss.
        device: Training device (defaults to CUDA if available).
        verbose: Print loss every 10 epochs.

    Returns:
        List of per-epoch loss dicts.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device).train()
    optimiser = torch.optim.Adam(encoder.parameters(), lr=lr)

    n_eps = len(episodes_actions)
    epoch_losses: List[Dict[str, float]] = []

    for epoch in range(num_epochs):
        indices = np.random.permutation(n_eps)
        total_loss = 0.0
        total_steps = 0
        comp_sum: Dict[str, float] = {"act": 0.0, "state": 0.0, "contrast": 0.0}

        for start in range(0, n_eps, batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_acts = [episodes_actions[i] for i in batch_idx]
            batch_syms = [episodes_symbols[i] for i in batch_idx]
            max_len = max(a.shape[0] for a in batch_acts)

            # Pad sequences
            action_dim = batch_acts[0].shape[1]
            pad_acts = np.zeros((len(batch_idx), max_len, action_dim), dtype=np.float32)
            pad_syms = np.zeros((len(batch_idx), max_len), dtype=np.int64)
            for j, (a, s) in enumerate(zip(batch_acts, batch_syms)):
                pad_acts[j, : len(a)] = a
                pad_syms[j, : len(s)] = s

            acts_t = torch.from_numpy(pad_acts).to(device)
            syms_t = torch.from_numpy(pad_syms).to(device)

            h_seq, a_pred, c_pred = encoder(acts_t, syms_t)
            loss, comps = compute_rnn_loss(
                a_pred, c_pred, h_seq,
                acts_t, syms_t,
                loss_weights=loss_weights,
                contrastive_margin=contrastive_margin,
            )
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimiser.step()

            total_loss += float(loss) * len(batch_idx)
            total_steps += len(batch_idx)
            for k in comp_sum:
                comp_sum[k] += comps[k] * len(batch_idx)

        avg = {k: v / total_steps for k, v in comp_sum.items()}
        avg["total"] = total_loss / total_steps
        epoch_losses.append(avg)
        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"  [RNN epoch {epoch+1}/{num_epochs}] "
                f"total={avg['total']:.4f} "
                f"act={avg['act']:.4f} "
                f"state={avg['state']:.4f} "
                f"contrast={avg['contrast']:.4f}"
            )

    encoder.eval()
    return epoch_losses


@torch.no_grad()
def extract_hidden_states(
    encoder: ENAPRNNEncoder,
    episodes_actions: List[np.ndarray],
    episodes_symbols: List[np.ndarray],
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Run encoder inference on all episodes and return concatenated h_t.

    Args:
        encoder: Trained :class:`ENAPRNNEncoder`.
        episodes_actions: Per-episode action arrays.
        episodes_symbols: Per-episode symbol arrays.
        device: Inference device.

    Returns:
        Tuple of:
        - ``h_all``: ``(N_total, hidden_dim)`` hidden state array.
        - ``episode_lengths``: List of per-episode timestep counts (for
          splitting ``h_all`` back into episodes).
    """
    if device is None:
        device = next(encoder.parameters()).device
    encoder.eval()
    all_h = []
    lengths = []
    for acts, syms in zip(episodes_actions, episodes_symbols):
        T = len(acts)
        acts_t = torch.from_numpy(acts.astype(np.float32)).unsqueeze(0).to(device)
        syms_t = torch.from_numpy(syms.astype(np.int64)).unsqueeze(0).to(device)
        h = encoder.encode_sequence(acts_t, syms_t)  # (1, T, D)
        all_h.append(h.squeeze(0).cpu().numpy())
        lengths.append(T)
    return np.concatenate(all_h, axis=0), lengths


# ---------------------------------------------------------------------------
# Faithful ENAP port: PretrainRNN + PrioritizedReplayBuffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """Episode-level prioritized experience replay.

    Samples episodes with probability proportional to ``priority ** alpha``.
    IS weights anneal from ``beta_start`` to ``1.0`` over ``beta_steps`` updates.

    Args:
        capacity: Maximum number of stored episodes.
        alpha: Priority exponent (default 0.6).
        beta_start: Initial IS weight exponent (default 0.4).
        beta_end: Target IS weight exponent (default 1.0).
        beta_steps: Update steps over which beta anneals
            (defaults to ``capacity``).
    """

    def __init__(
        self,
        capacity: int = 10_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: Optional[int] = None,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_end = beta_end
        _steps = beta_steps or capacity
        self.beta_step_size = (beta_end - beta_start) / max(_steps, 1)

        self._storage: List[Dict] = []
        self._priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self._pos: int = 0
        self._max_priority: float = 1.0

    def add(self, episode: Dict) -> None:
        """Add an episode with maximum current priority."""
        if len(self._storage) < self.capacity:
            self._storage.append(episode)
        else:
            self._storage[self._pos] = episode
        self._priorities[self._pos] = self._max_priority
        self._pos = (self._pos + 1) % self.capacity

    def fill(self, episodes: List[Dict]) -> None:
        """Bulk-add a list of episodes."""
        for ep in episodes:
            self.add(ep)

    def sample(
        self, batch_size: int
    ) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """Sample a batch weighted by priority.

        Returns:
            ``(episodes, is_weights, indices)``
        """
        n = len(self._storage)
        raw = self._priorities[:n] ** self.alpha
        probs = raw / raw.sum()
        indices = np.random.choice(n, size=min(batch_size, n), replace=False, p=probs)
        episodes = [self._storage[i] for i in indices]
        weights = (n * probs[indices]) ** (-self.beta)
        weights = (weights / weights.max()).astype(np.float32)
        return episodes, weights, indices

    def update_priorities(
        self, indices: np.ndarray, losses: np.ndarray
    ) -> None:
        """Update priorities from per-episode losses."""
        for idx, loss in zip(indices, losses):
            prio = float(abs(loss)) + 1e-6
            self._priorities[idx] = prio
            if prio > self._max_priority:
                self._max_priority = prio

    def anneal_beta(self) -> None:
        self.beta = min(self.beta_end, self.beta + self.beta_step_size)

    def __len__(self) -> int:
        return len(self._storage)


class PretrainRNN(nn.Module):
    """Vanilla RNN encoder matching the ENAP repository's ``Pretrain`` class.

    Architecture: ``state_embed`` + ``enc`` (``nn.RNN``) + ``act_head`` +
    ``cls_head``.  Saves checkpoints in the PMM-compatible format expected by
    :class:`~policy_doctor.enap.pmm.TrainableRNN`::

        {'model_state': state_dict, 'dims': {'a': int, 's': int, 'e': int, 'h': int}}

    Args:
        a_dim: Action dimension.
        s_dim: Number of HDBSCAN symbols.
        e_dim: Symbol embedding dimension (default 16).
        h_dim: RNN hidden state dimension (default 64).
    """

    def __init__(
        self,
        a_dim: int,
        s_dim: int,
        e_dim: int = 16,
        h_dim: int = 64,
    ) -> None:
        super().__init__()
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.e_dim = e_dim
        self.h_dim = h_dim

        self.state_embed = nn.Embedding(s_dim, e_dim)
        self.enc = nn.RNN(a_dim + e_dim, h_dim, batch_first=True)
        self.act_head = nn.Linear(h_dim, a_dim)
        self.cls_head = nn.Linear(h_dim, s_dim)

    def forward(
        self,
        A: torch.Tensor,            # (B, T, a_dim)
        S: torch.Tensor,            # (B, T) int — symbol indices
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run RNN over a sequence.

        Returns:
            ``(h_seq, a_pred, s_pred)`` each ``(B, T, *)``
        """
        s_emb = self.state_embed(S)                   # (B, T, e_dim)
        x = torch.cat([A, s_emb], dim=-1)             # (B, T, a_dim + e_dim)
        h_seq, _ = self.enc(x, h0)                    # (B, T, h_dim)
        a_pred = self.act_head(h_seq)                 # (B, T, a_dim)
        s_pred = self.cls_head(h_seq)                 # (B, T, s_dim)
        return h_seq, a_pred, s_pred

    @torch.no_grad()
    def extract_hidden_states_episode(
        self,
        A: np.ndarray,    # (T, a_dim)
        S: np.ndarray,    # (T,) int
    ) -> np.ndarray:
        """Return hidden states ``(T, h_dim)`` for a single episode."""
        A_t = torch.from_numpy(A.astype(np.float32)).unsqueeze(0)
        S_t = torch.from_numpy(S.astype(np.int64)).unsqueeze(0)
        h_seq, _, _ = self.forward(A_t, S_t)
        return h_seq.squeeze(0).cpu().numpy()

    def save_checkpoint(self, path: str) -> None:
        """Save in PMM-compatible checkpoint format."""
        torch.save(
            {
                "model_state": self.state_dict(),
                "dims": {
                    "a": self.a_dim,
                    "s": self.s_dim,
                    "e": self.e_dim,
                    "h": self.h_dim,
                },
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls, path: str, device: Optional[torch.device] = None
    ) -> "PretrainRNN":
        """Load from a PMM-compatible checkpoint."""
        ckpt = torch.load(path, map_location=device or "cpu")
        dims = ckpt["dims"]
        model = cls(dims["a"], dims["s"], dims["e"], dims["h"])
        model.load_state_dict(ckpt["model_state"])
        if device is not None:
            model.to(device)
        return model


def phase_aware_contrastive_loss_pretrain(
    h_t: torch.Tensor,
    h_t1: torch.Tensor,
    s_t: torch.Tensor,
    s_t1: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """Phase-aware contrastive loss matching the ENAP repository formulation.

    Uses cosine **distance** (``1 − cos_sim``) rather than similarity:

    - Same phase  → loss = distance   (minimise)
    - Transition  → loss = ``relu(margin − distance)``  (push apart)

    This matches ``rnn_train.py::phase_aware_contrastive_loss`` exactly.
    """
    cos_sim = F.cosine_similarity(h_t, h_t1, dim=-1)   # (B,)
    dist = 1.0 - cos_sim
    same = (s_t == s_t1).float()
    loss = same * dist + (1.0 - same) * torch.clamp(margin - dist, min=0.0)
    return loss.mean()


def train_pretrain_rnn(
    model: PretrainRNN,
    episodes: List[Dict],
    num_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    loss_weights: Optional[Dict[str, float]] = None,
    contrastive_margin: float = 0.5,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    noise_std: float = 0.01,
    use_per: bool = True,
    per_alpha: float = 0.6,
    per_beta_start: float = 0.4,
    wandb_prefix: str = "enap_rnn",
) -> List[Dict[str, float]]:
    """Train :class:`PretrainRNN` with PER and phase-aware contrastive loss.

    Faithful to the ENAP repository's ``rnn_train.py`` training procedure.

    Args:
        model: :class:`PretrainRNN` to train in-place.
        episodes: List of dicts ``{'S': (T, s_dim) one-hot float32,
            'A': (T, a_dim) float32}``.
        num_epochs: Training epochs.
        lr: Adam learning rate.
        batch_size: Episodes per mini-batch (sampled from PER buffer).
        loss_weights: ``{'act', 'state', 'contrast'}`` weights.
        contrastive_margin: Margin for phase-aware contrastive loss.
        device: Training device (defaults to CUDA if available).
        verbose: Print per-10-epoch loss summary.
        noise_std: Standard deviation of Gaussian noise added to actions.
        use_per: Use Prioritized Experience Replay.
        per_alpha: PER priority exponent.
        per_beta_start: Initial IS-weight annealing start.
        wandb_prefix: Metric prefix for wandb logging (e.g. ``"enap_rnn"``
            → ``enap_rnn/loss/total``).  Logging is skipped when no wandb
            run is active (``wandb.run is None``).

    Returns:
        List of per-epoch loss dicts
        ``{'total', 'act', 'state', 'contrast'}``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()

    weights = loss_weights or {"act": 1.0, "state": 1.0, "contrast": 0.5}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    updates_per_epoch = max(1, len(episodes) // batch_size)
    buffer = PrioritizedReplayBuffer(
        capacity=max(len(episodes), 1000),
        alpha=per_alpha,
        beta_start=per_beta_start,
        beta_steps=num_epochs * updates_per_epoch,
    )
    buffer.fill(episodes)

    s_dim = episodes[0]["S"].shape[1]
    a_dim = episodes[0]["A"].shape[1]

    epoch_losses: List[Dict[str, float]] = []

    for epoch in range(num_epochs):
        batch_eps, is_weights, sample_indices = buffer.sample(batch_size)
        is_weights_t = torch.from_numpy(is_weights).to(device)

        max_len = max(ep["S"].shape[0] for ep in batch_eps)
        B = len(batch_eps)

        A_pad = np.zeros((B, max_len, a_dim), dtype=np.float32)
        S_pad = np.zeros((B, max_len), dtype=np.int64)

        for j, ep in enumerate(batch_eps):
            T = ep["S"].shape[0]
            A_ep = ep["A"].astype(np.float32)
            if noise_std > 0.0:
                A_ep = A_ep + (np.random.randn(*A_ep.shape) * noise_std).astype(np.float32)
            A_pad[j, :T] = A_ep
            S_pad[j, :T] = np.argmax(ep["S"], axis=1)

        A_t = torch.from_numpy(A_pad).to(device)
        S_t = torch.from_numpy(S_pad).to(device)

        h_seq, a_pred, s_pred = model(A_t, S_t)

        # Action prediction loss (next-step MSE)
        l_act = F.mse_loss(a_pred[:, :-1, :], A_t[:, 1:, :])

        # Symbol prediction loss (next-step cross-entropy)
        l_state = F.cross_entropy(
            s_pred[:, :-1, :].reshape(-1, s_dim),
            S_t[:, 1:].reshape(-1),
        )

        # Phase-aware contrastive (repo formulation)
        h_flat_t = h_seq[:, :-1, :].reshape(-1, model.h_dim)
        h_flat_t1 = h_seq[:, 1:, :].reshape(-1, model.h_dim)
        s_flat_t = S_t[:, :-1].reshape(-1)
        s_flat_t1 = S_t[:, 1:].reshape(-1)
        l_contrast = phase_aware_contrastive_loss_pretrain(
            h_flat_t, h_flat_t1, s_flat_t, s_flat_t1, margin=contrastive_margin
        )

        total = (
            weights.get("act", 1.0) * l_act
            + weights.get("state", 1.0) * l_state
            + weights.get("contrast", 0.5) * l_contrast
        )

        # IS-weighted gradient
        if use_per:
            total = total * is_weights_t.mean()

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if use_per:
            per_losses = np.full(len(batch_eps), float(l_act.detach()), dtype=np.float32)
            buffer.update_priorities(sample_indices, per_losses)
            buffer.anneal_beta()

        entry: Dict[str, float] = {
            "total": float(total.detach()),
            "act": float(l_act.detach()),
            "state": float(l_state.detach()),
            "contrast": float(l_contrast.detach()),
        }
        epoch_losses.append(entry)

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"  [Pretrain epoch {epoch+1}/{num_epochs}] "
                f"total={entry['total']:.4f}  "
                f"act={entry['act']:.4f}  "
                f"state={entry['state']:.4f}  "
                f"contrast={entry['contrast']:.4f}"
            )

        try:
            import wandb as _wandb
            if _wandb.run is not None:
                _wandb.log(
                    {
                        f"{wandb_prefix}/loss/total": entry["total"],
                        f"{wandb_prefix}/loss/act": entry["act"],
                        f"{wandb_prefix}/loss/state": entry["state"],
                        f"{wandb_prefix}/loss/contrast": entry["contrast"],
                        f"{wandb_prefix}/epoch": epoch + 1,
                    },
                    step=epoch + 1,
                )
        except ImportError:
            pass

    model.eval()
    return epoch_losses
