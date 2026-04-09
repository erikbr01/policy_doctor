"""
Trajectory Aggregator: Aggregates per-timestep embeddings into trajectory-level embeddings.

For robotics data, trajectories consist of T timesteps. This module:
1. Computes embeddings for each (state, action) pair in the trajectory
2. Applies masking to handle padded trajectories
3. Sums embeddings over the time dimension
4. Normalizes to unit length (L2 norm) for clustering

For test rollouts, it also applies reward weighting:
- If Failure (reward=-1): embedding is flipped, aligning with training data that caused the failure
- If Success (reward=+1): embedding points toward helpful training data
"""

from typing import Optional

import torch
from torch import Tensor


class TrajectoryAggregator:
    """Aggregates per-timestep embeddings into trajectory embeddings.

    Handles:
    - Masking for padded sequences
    - Summation over time dimension
    - L2 normalization for clustering
    - Reward weighting for test rollouts
    """

    def __init__(
        self,
        normalize: bool = True,
        eps: float = 1e-8,
    ):
        """Initialize TrajectoryAggregator.

        Args:
            normalize: Whether to L2-normalize final embeddings.
            eps: Small constant for numerical stability in normalization.
        """
        self.normalize = normalize
        self.eps = eps

    def aggregate(
        self,
        timestep_embeddings: Tensor,
        mask: Optional[Tensor] = None,
        rewards: Optional[Tensor] = None,
    ) -> Tensor:
        """Aggregate timestep embeddings into trajectory embedding.

        Args:
            timestep_embeddings: Tensor of shape [batch, time, embedding_dim]
                containing per-timestep influence embeddings.
            mask: Optional boolean tensor of shape [batch, time] indicating
                valid (non-padded) timesteps. True = valid, False = padding.
            rewards: Optional tensor of shape [batch] containing trajectory
                rewards. For test rollouts, this flips failure embeddings.
                Expected values: +1 (success) or -1 (failure).

        Returns:
            Tensor of shape [batch, embedding_dim] with aggregated embeddings.
        """
        batch_size, time_steps, embedding_dim = timestep_embeddings.shape

        # Apply mask if provided.
        if mask is not None:
            # Expand mask to match embedding dimensions: [batch, time, 1]
            mask_expanded = mask.unsqueeze(-1).float()
            timestep_embeddings = timestep_embeddings * mask_expanded

        # Sum over time dimension.
        trajectory_embeddings = timestep_embeddings.sum(dim=1)  # [batch, embedding_dim]

        # Apply reward weighting for test rollouts.
        if rewards is not None:
            # rewards: [batch] -> [batch, 1]
            rewards_expanded = rewards.unsqueeze(-1)
            trajectory_embeddings = trajectory_embeddings * rewards_expanded

        # L2 normalize if requested.
        if self.normalize:
            norms = trajectory_embeddings.norm(dim=1, keepdim=True).clamp(min=self.eps)
            trajectory_embeddings = trajectory_embeddings / norms

        return trajectory_embeddings

    def aggregate_batch_sequential(
        self,
        get_embedding_fn,
        obs_sequence: Tensor,
        action_sequence: Tensor,
        mask: Optional[Tensor] = None,
        rewards: Optional[Tensor] = None,
        timesteps: Optional[Tensor] = None,
    ) -> Tensor:
        """Aggregate embeddings by computing them sequentially over time.

        This method is memory-efficient as it computes and projects embeddings
        one timestep at a time, never storing full gradient tensors.

        Args:
            get_embedding_fn: Function(obs, action, timesteps) -> [batch, embed_dim]
                that computes projected embeddings for a single timestep.
            obs_sequence: Observations [batch, time, obs_dim].
            action_sequence: Actions [batch, time, action_dim].
            mask: Valid timestep mask [batch, time].
            rewards: Trajectory rewards [batch].
            timesteps: Diffusion timesteps [batch, num_diffusion_steps] or None.

        Returns:
            Aggregated trajectory embeddings [batch, embedding_dim].
        """
        batch_size, time_steps = obs_sequence.shape[:2]

        # We'll accumulate embeddings.
        accumulated = None
        valid_counts = torch.zeros(batch_size, 1, device=obs_sequence.device)

        for t in range(time_steps):
            # Skip if all samples are masked at this timestep.
            if mask is not None:
                t_mask = mask[:, t]
                if not t_mask.any():
                    continue
            else:
                t_mask = None

            # Get embedding for this timestep.
            obs_t = obs_sequence[:, t]
            action_t = action_sequence[:, t]
            embedding_t = get_embedding_fn(
                obs_t, action_t, timesteps
            )  # [batch, embed_dim]

            # Apply mask.
            if t_mask is not None:
                embedding_t = embedding_t * t_mask.unsqueeze(-1).float()
                valid_counts += t_mask.unsqueeze(-1).float()
            else:
                valid_counts += 1

            # Accumulate.
            if accumulated is None:
                accumulated = embedding_t
            else:
                accumulated = accumulated + embedding_t

        # Handle edge case where all timesteps are masked.
        if accumulated is None:
            raise ValueError("All timesteps were masked; cannot compute embedding.")

        # Apply reward weighting.
        if rewards is not None:
            accumulated = accumulated * rewards.unsqueeze(-1)

        # Normalize.
        if self.normalize:
            norms = accumulated.norm(dim=1, keepdim=True).clamp(min=self.eps)
            accumulated = accumulated / norms

        return accumulated

    def create_padding_mask(
        self,
        sequence_lengths: Tensor,
        max_length: int,
    ) -> Tensor:
        """Create a boolean mask from sequence lengths.

        Args:
            sequence_lengths: Tensor of shape [batch] with valid lengths.
            max_length: Maximum sequence length (for mask tensor size).

        Returns:
            Boolean tensor [batch, max_length] where True = valid position.
        """
        batch_size = sequence_lengths.shape[0]
        positions = torch.arange(max_length, device=sequence_lengths.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        lengths = sequence_lengths.unsqueeze(1)
        mask = positions < lengths
        return mask
