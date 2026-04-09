"""
Embedding Computer: Processes datasets to generate influence embeddings.

This module handles the data processing loop:
1. Process Training Data: Compute φ_train for all expert demonstrations
2. Process Evaluation Data: Compute φ_test with reward weighting for rollouts

The embeddings are saved as tensors for downstream clustering and attribution.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm


class EmbeddingComputer:
    """Computes and stores influence embeddings for training and evaluation data.

    Handles:
    - Iterating through datasets
    - Computing per-sample embeddings via gradient projection
    - Aggregating trajectories
    - Saving embeddings to disk
    """

    def __init__(
        self,
        model: nn.Module,
        projection_dim: int = 512,
        param_filter: Optional[list] = None,
        num_diffusion_timesteps: int = 10,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        """Initialize EmbeddingComputer.

        Args:
            model: Policy model to compute gradients for.
            projection_dim: Dimension of projected embeddings.
            param_filter: Parameter filter for gradient computation.
            num_diffusion_timesteps: Number of diffusion timesteps to sample per trajectory.
            device: Computation device.
            seed: Random seed for projection matrix.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.projection_dim = projection_dim
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.device = device or next(model.parameters()).device
        self.seed = seed

        # Import here to avoid circular imports.
        from .gradient_projector import GradientProjector
        from .trajectory_aggregator import TrajectoryAggregator

        self.projector = GradientProjector(
            model=model,
            projection_dim=projection_dim,
            param_filter=param_filter,
            device=self.device,
            seed=seed,
        )
        self.aggregator = TrajectoryAggregator(normalize=True)

    def compute_train_embeddings(
        self,
        dataloader: DataLoader,
        loss_fn: Callable,
        noise_scheduler_timesteps: int,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[Tensor, Dict]:
        """Compute embeddings for training demonstrations.

        Args:
            dataloader: DataLoader for training dataset.
            loss_fn: Loss function for gradient computation.
            noise_scheduler_timesteps: Number of noise scheduler timesteps.
            save_path: Optional path to save embeddings.

        Returns:
            Tuple of (embeddings tensor [num_demos, proj_dim], metadata dict).
        """
        self.model.eval()
        all_embeddings = []
        metadata = {"demo_indices": [], "num_samples_per_demo": []}

        running_idx = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(dataloader, desc="Computing train embeddings")
            ):
                batch = self._to_device(batch)
                batch_size = batch["action"].shape[0]

                # Sample diffusion timesteps.
                timesteps = torch.randint(
                    0,
                    noise_scheduler_timesteps,
                    (batch_size, self.num_diffusion_timesteps),
                    device=self.device,
                ).long()
                batch["timesteps"] = timesteps

                # Compute embeddings for this batch.
                embeddings = self._compute_batch_embeddings(batch, loss_fn)
                all_embeddings.append(embeddings.cpu())

                # Track metadata.
                metadata["demo_indices"].extend(
                    range(running_idx, running_idx + batch_size)
                )
                running_idx += batch_size

        # Concatenate all embeddings.
        train_embeddings = torch.cat(all_embeddings, dim=0)
        metadata["num_demos"] = train_embeddings.shape[0]
        metadata["projection_dim"] = self.projection_dim

        self.logger.info(f"Computed {train_embeddings.shape[0]} training embeddings")

        # Save if path provided.
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"embeddings": train_embeddings, "metadata": metadata}, save_path
            )
            self.logger.info(f"Saved training embeddings to {save_path}")

        return train_embeddings, metadata

    def compute_eval_embeddings(
        self,
        dataloader: DataLoader,
        loss_fn: Callable,
        noise_scheduler_timesteps: int,
        rewards: Optional[Tensor] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """Compute embeddings for evaluation rollouts with reward weighting.

        Args:
            dataloader: DataLoader for evaluation dataset.
            loss_fn: Loss function for gradient computation.
            noise_scheduler_timesteps: Number of noise scheduler timesteps.
            rewards: Optional tensor of rewards for each rollout. If None,
                attempts to extract from batch data.
            save_path: Optional path to save embeddings.

        Returns:
            Tuple of (embeddings tensor, rewards tensor, metadata dict).
        """
        self.model.eval()
        all_embeddings = []
        all_rewards = []
        metadata = {"rollout_indices": []}

        running_idx = 0
        reward_idx = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(dataloader, desc="Computing eval embeddings")
            ):
                batch = self._to_device(batch)
                batch_size = batch["action"].shape[0]

                # Sample diffusion timesteps.
                timesteps = torch.randint(
                    0,
                    noise_scheduler_timesteps,
                    (batch_size, self.num_diffusion_timesteps),
                    device=self.device,
                ).long()
                batch["timesteps"] = timesteps

                # Compute embeddings for this batch.
                embeddings = self._compute_batch_embeddings(batch, loss_fn)

                # Get rewards - either from provided tensor or from batch.
                if rewards is not None:
                    batch_rewards = rewards[reward_idx : reward_idx + batch_size]
                    reward_idx += batch_size
                elif "reward" in batch:
                    batch_rewards = batch["reward"]
                elif "success" in batch:
                    # Convert success (0/1) to reward (-1/+1).
                    batch_rewards = batch["success"].float() * 2 - 1
                else:
                    # Default: assume failures have reward -1.
                    self.logger.warning(
                        "No rewards found, assuming all failures (reward=-1)"
                    )
                    batch_rewards = torch.ones(batch_size, device=self.device) * -1

                # Apply reward weighting: φ_test = μ_traj * reward
                # This flips the embedding direction for failures.
                weighted_embeddings = embeddings * batch_rewards.unsqueeze(-1).to(
                    embeddings.device
                )

                # Normalize after reward weighting.
                norms = weighted_embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
                weighted_embeddings = weighted_embeddings / norms

                all_embeddings.append(weighted_embeddings.cpu())
                all_rewards.append(batch_rewards.cpu())

                metadata["rollout_indices"].extend(
                    range(running_idx, running_idx + batch_size)
                )
                running_idx += batch_size

        # Concatenate results.
        eval_embeddings = torch.cat(all_embeddings, dim=0)
        eval_rewards = torch.cat(all_rewards, dim=0)
        metadata["num_rollouts"] = eval_embeddings.shape[0]
        metadata["projection_dim"] = self.projection_dim
        metadata["num_failures"] = (eval_rewards < 0).sum().item()
        metadata["num_successes"] = (eval_rewards > 0).sum().item()

        self.logger.info(
            f"Computed {eval_embeddings.shape[0]} eval embeddings "
            f"({metadata['num_failures']} failures, {metadata['num_successes']} successes)"
        )

        # Save if path provided.
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "embeddings": eval_embeddings,
                    "rewards": eval_rewards,
                    "metadata": metadata,
                },
                save_path,
            )
            self.logger.info(f"Saved eval embeddings to {save_path}")

        return eval_embeddings, eval_rewards, metadata

    def _compute_batch_embeddings(
        self,
        batch: Dict[str, Tensor],
        loss_fn: Callable,
    ) -> Tensor:
        """Compute projected embeddings for a batch using vmap.

        Args:
            batch: Batch dictionary with obs, action, timesteps.
            loss_fn: Loss function for gradient computation.

        Returns:
            Tensor of shape [batch_size, projection_dim].
        """
        batch_size = batch["action"].shape[0]
        timesteps = batch.get("timesteps")
        num_timesteps = timesteps.shape[1] if timesteps is not None else 1

        # Extract batch elements
        action = batch["action"]
        obs = batch["obs"]

        # Create gradient store
        grads = torch.zeros(
            size=(batch_size, self.projector.grad_dim),
            dtype=action.dtype,
            device=action.device,
        )

        # Compute gradients using vmap (following TRAK pattern)
        # Taking gradient wrt weights (second argument, hence argnums=1)
        grads_loss = torch.func.grad(loss_fn, has_aux=False, argnums=1)

        for i_tstep in range(num_timesteps):
            tsteps = timesteps[:, i_tstep : i_tstep + 1]

            # Map over batch dimensions (0 for each batch input, None for model params)
            per_sample_grads = torch.func.vmap(
                grads_loss,
                in_dims=(None, None, None, 0, 0, 0),
                randomness="different",
            )(
                self.model,
                self.projector._func_weights,
                self.projector._func_buffers,
                tsteps,
                action,
                obs,
            )

            # Accumulate and average gradients across timesteps
            self._average_vectorize(per_sample_grads, grads, num_timesteps)

        # Project gradients to lower dimension
        embeddings = grads @ self.projector.projection_matrix

        # Normalize
        norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
        embeddings = embeddings / norms

        return embeddings

    def _average_vectorize(self, g: Dict[str, Tensor], arr: Tensor, num_timesteps: int):
        """Accumulates averaged gradients into array.

        Args:
            g: Dictionary of gradients for each parameter
            arr: Array to accumulate results into
            num_timesteps: Number of timesteps to average over
        """
        pointer = 0
        for param in g.values():
            num_param = param[0].numel()
            arr[:, pointer : pointer + num_param] += (
                param.flatten(start_dim=1).data / num_timesteps
            )
            pointer += num_param

    def _to_device(self, batch: Dict) -> Dict:
        """Move batch to computation device, converting float64 to float32 for MPS."""
        result = {}
        for k, v in batch.items():
            if isinstance(v, Tensor):
                # Convert float64 to float32 for MPS compatibility
                if v.dtype == torch.float64:
                    v = v.float()
                result[k] = v.to(self.device)
            elif isinstance(v, dict):
                result[k] = self._to_device(v)
            else:
                result[k] = v
        return result

    def save_projector(self, path: Union[str, Path]) -> None:
        """Save the gradient projector for later use."""
        self.projector.save(str(path))

    @staticmethod
    def load_embeddings(path: Union[str, Path]) -> Dict:
        """Load saved embeddings from disk."""
        return torch.load(path)
