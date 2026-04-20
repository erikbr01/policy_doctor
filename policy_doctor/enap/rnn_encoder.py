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
