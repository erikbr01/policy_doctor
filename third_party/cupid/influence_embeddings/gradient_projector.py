"""
Gradient Projector: The core engine for computing influence embeddings.

This module implements the random projection approach (TRAK method) for computing
influence embeddings. It projects high-dimensional gradients onto a fixed random
Gaussian matrix using the Johnson-Lindenstrauss lemma.

The influence embedding for a datapoint is: φ(z) = ∇L(z) @ P
where P is a random projection matrix and ∇L(z) is the gradient of the loss.
"""

import logging
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


class GradientProjector:
    """Computes projected gradient embeddings for influence function approximation.

    Uses random projections (Johnson-Lindenstrauss) to map high-dimensional gradients
    to a lower-dimensional space while preserving distances (and thus influence scores).

    Attributes:
        model: The policy model to compute gradients for.
        projection_dim: Dimension of the projected embedding space.
        projection_matrix: Random Gaussian projection matrix P.
        grad_params: List of parameter names to compute gradients for.
    """

    def __init__(
        self,
        model: nn.Module,
        projection_dim: int = 512,
        param_filter: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        """Initialize the GradientProjector.

        Args:
            model: The policy model (e.g., DiffusionUnetLowdimPolicy).
            projection_dim: Target dimension for projected embeddings.
            param_filter: List of parameter name substrings to include.
                If None, uses last 2 layers (as recommended in spec).
            device: Device for computation. Defaults to model's device.
            seed: Random seed for reproducible projection matrix.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.projection_dim = projection_dim
        self.device = device or next(model.parameters()).device

        # Get parameters to compute gradients for.
        self.grad_params = self._select_parameters(param_filter)
        self.grad_dim = sum(
            p.numel()
            for name, p in model.named_parameters()
            if name in self.grad_params
        )
        self.logger.info(
            f"Selected {len(self.grad_params)} parameters with {self.grad_dim} total elements"
        )

        # Create random projection matrix with fixed seed for reproducibility.
        torch.manual_seed(seed)
        self.projection_matrix = torch.randn(
            self.grad_dim, projection_dim, device=self.device
        ) / (self.grad_dim**0.5)  # Scale for numerical stability

        # Cache functional parameters.
        self._func_weights = {
            k: v for k, v in model.named_parameters() if k in self.grad_params
        }
        self._func_buffers = dict(model.named_buffers())

    def _select_parameters(self, param_filter: Optional[List[str]]) -> List[str]:
        """Select parameters to compute gradients for.

        Args:
            param_filter: List of substrings. Parameters containing any of these
                will be included. If None, selects last 2 layers heuristically.

        Returns:
            Sorted list of parameter names to compute gradients for.
        """
        all_params = list(dict(self.model.named_parameters()).keys())

        if param_filter is not None:
            # Filter by provided substrings.
            selected = [
                name
                for name in all_params
                if any(f in name for f in param_filter) and "dummy" not in name
            ]
        else:
            # Heuristic: select parameters from "up" and "final" modules (last layers).
            # This captures most semantic information while reducing computation.
            default_filters = ["model.up", "model.final", "final_conv"]
            selected = [
                name
                for name in all_params
                if any(f in name for f in default_filters) and "dummy" not in name
            ]

            # Fallback: if no matches, use all parameters.
            if not selected:
                self.logger.warning(
                    "No parameters matched default filters. Using all parameters."
                )
                selected = [name for name in all_params if "dummy" not in name]

        return sorted(selected)

    def compute_embedding(
        self,
        loss_fn: Callable[[Dict[str, Tensor], Dict[str, Tensor]], Tensor],
        batch: Dict[str, Tensor],
    ) -> Tensor:
        """Compute projected gradient embedding for a batch.

        Uses torch.func.vmap and torch.func.grad for efficient per-sample gradients.

        Args:
            loss_fn: Function that computes scalar loss given (weights, batch_element).
                Should be compatible with torch.func.grad.
            batch: Dictionary containing batch data (obs, action, etc.).

        Returns:
            Tensor of shape [batch_size, projection_dim] containing embeddings.
        """
        batch_size = batch["action"].shape[0]

        # Define the per-sample loss function for grad computation.
        def single_sample_loss(weights: Dict[str, Tensor], idx: int) -> Tensor:
            # Extract single sample from batch.
            single_batch = {
                k: v[idx : idx + 1] if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }
            return loss_fn(weights, single_batch)

        # Compute gradients for each sample.
        embeddings = torch.zeros(
            batch_size,
            self.projection_dim,
            device=self.device,
            dtype=batch["action"].dtype,
        )

        # Compute per-sample gradients and project immediately.
        grad_fn = torch.func.grad(single_sample_loss)

        for i in range(batch_size):
            grads = grad_fn(self._func_weights, i)
            # Flatten gradients and project.
            flat_grad = torch.cat([g.flatten() for g in grads.values()])
            embeddings[i] = flat_grad @ self.projection_matrix

        return embeddings

    def compute_embedding_vmapped(
        self,
        loss_fn: Callable[[Dict[str, Tensor], Tensor, Tensor], Tensor],
        obs: Tensor,
        action: Tensor,
        timesteps: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute embeddings using vmap for better efficiency.

        This is the preferred method when the loss function supports vmapping.

        Args:
            loss_fn: Function(weights, buffers, obs, action, timesteps) -> scalar loss.
            obs: Observations tensor [batch, ...].
            action: Actions tensor [batch, ...].
            timesteps: Optional diffusion timesteps [batch, num_timesteps].

        Returns:
            Tensor of shape [batch_size, projection_dim].
        """
        batch_size = obs.shape[0]

        # Create accumulator for gradients across timesteps.
        embeddings = torch.zeros(
            batch_size, self.projection_dim, device=self.device, dtype=obs.dtype
        )

        num_timesteps = 1 if timesteps is None else timesteps.shape[1]

        for t_idx in range(num_timesteps):
            tstep = None if timesteps is None else timesteps[:, t_idx : t_idx + 1]

            # Define loss for gradient computation.
            def per_sample_loss(weights, obs_i, action_i, tstep_i):
                return loss_fn(
                    self.model,
                    weights,
                    self._func_buffers,
                    tstep_i,
                    action_i.unsqueeze(0),
                    obs_i.unsqueeze(0),
                )

            # Compute gradient with respect to weights.
            grad_fn = torch.func.grad(per_sample_loss, argnums=0)

            # vmap over batch dimension.
            if tstep is not None:
                vmapped_grad = torch.func.vmap(
                    grad_fn, in_dims=(None, 0, 0, 0), randomness="different"
                )
                grads = vmapped_grad(self._func_weights, obs, action, tstep)
            else:
                vmapped_grad = torch.func.vmap(
                    grad_fn, in_dims=(None, 0, 0, None), randomness="different"
                )
                grads = vmapped_grad(self._func_weights, obs, action, None)

            # Flatten and project gradients.
            flat_grads = torch.cat(
                [g.flatten(start_dim=1) for g in grads.values()], dim=1
            )
            embeddings += (flat_grads @ self.projection_matrix) / num_timesteps

        return embeddings

    def get_projection_matrix(self) -> Tensor:
        """Return the projection matrix for external use."""
        return self.projection_matrix.clone()

    def save(self, path: str) -> None:
        """Save the projector state (projection matrix and config)."""
        state = {
            "projection_matrix": self.projection_matrix.cpu(),
            "projection_dim": self.projection_dim,
            "grad_dim": self.grad_dim,
            "grad_params": self.grad_params,
        }
        torch.save(state, path)
        self.logger.info(f"Saved GradientProjector state to {path}")

    @classmethod
    def load(cls, path: str, model: nn.Module, device: Optional[torch.device] = None):
        """Load a saved projector state."""
        state = torch.load(path, map_location="cpu")
        instance = cls.__new__(cls)
        instance.logger = logging.getLogger(cls.__name__)
        instance.model = model
        instance.projection_dim = state["projection_dim"]
        instance.grad_dim = state["grad_dim"]
        instance.grad_params = state["grad_params"]
        instance.device = device or next(model.parameters()).device
        instance.projection_matrix = state["projection_matrix"].to(instance.device)
        instance._func_weights = {
            k: v for k, v in model.named_parameters() if k in instance.grad_params
        }
        instance._func_buffers = dict(model.named_buffers())
        return instance
