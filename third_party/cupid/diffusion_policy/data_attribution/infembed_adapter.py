"""Adapter to run InfEmbed's ArnoldiEmbedder on diffusion policy batches.

Provides a wrapper model and dataloader format so InfEmbed (which expects
(model, loss_fn, DataLoader) with batches as (features, labels)) can use
our diffusion loss and batch dicts.
"""

from typing import Any, Dict, Iterator, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from diffusion_policy.common.pytorch_util import dict_apply


class DiffusionLossWrapper(nn.Module):
    """Wraps policy + task so that forward(batch_dict) returns per-example loss (B,).

    InfEmbed expects model(*inputs) to return a tensor; with loss_fn reduction='none'
    we need per-example loss. We compute it by calling task.get_output per sample.
    """

    def __init__(self, policy: nn.Module, task: Any):
        super().__init__()
        self.policy = policy
        self.task = task
        self._is_hybrid = _is_hybrid_task(task)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        B = batch["action"].shape[0]
        num_timesteps = batch["timesteps"].shape[1]
        weights = dict(self.policy.named_parameters())
        buffers = dict(self.policy.named_buffers())
        # Vectorized over diffusion timesteps: one batched forward with (B*T) "virtual"
        # examples (each sample repeated for each of its T timesteps), then mean over T.
        T = num_timesteps
        tsteps = batch["timesteps"].reshape(-1)  # (B*T,)
        action = batch["action"].repeat_interleave(T, dim=0)  # (B*T, Ta, Da)
        if self._is_hybrid:
            obs_dict = batch["obs"]
            assert isinstance(obs_dict, dict), "hybrid policy expects obs as dict"
            obs_keys = list(obs_dict.keys())
            obs = [obs_dict[k].repeat_interleave(T, dim=0) for k in obs_keys]
            loss_bt = self.task.get_output(
                self.policy, weights, buffers, obs_keys, tsteps, action, *obs,
                return_per_example=True,
            )
        else:
            obs = batch["obs"].repeat_interleave(T, dim=0)  # (B*T, To, Do)
            loss_bt = self.task.get_output(
                self.policy, weights, buffers, tsteps, action, obs,
                return_per_example=True,
            )
        # (B*T,) -> (B, T) -> mean over timesteps -> (B,)
        return loss_bt.view(B, T).mean(dim=1)


def _is_hybrid_task(task: Any) -> bool:
    """True if task is hybrid image (obs is dict of tensors)."""
    cls = type(task).__name__
    return "Hybrid" in cls or "Image" in cls


class IdentityLossNone(nn.Module):
    """Loss that returns the first argument (for use with per-example loss output)."""

    reduction = "none"

    def forward(self, input: torch.Tensor, target: Any) -> torch.Tensor:
        return input


def diffusion_batches_for_infembed(
    base_loader: DataLoader,
    device: torch.device,
) -> Iterator[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
    """Yields batches in InfEmbed format: (batch_dict, dummy_labels).

    InfEmbed expects batch[0:-1] = features, batch[-1] = labels. So (batch_dict, labels)
    gives features = (batch_dict,) and model(*features) = model(batch_dict).
    """
    for batch in base_loader:
        batch = dict_apply(batch, lambda x: x.to(device))
        B = batch["action"].shape[0]
        labels = torch.zeros(B, device=device, dtype=torch.float32)
        yield batch, labels


class DiffusionInfembedDataset(torch.utils.data.IterableDataset):
    """IterableDataset that wraps a diffusion DataLoader and yields (batch, labels) for InfEmbed."""

    def __init__(self, base_loader: DataLoader, device: torch.device):
        self.base_loader = base_loader
        self.device = device

    def __iter__(self) -> Iterator[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        return diffusion_batches_for_infembed(self.base_loader, self.device)


def make_infembed_dataloader(
    base_loader: DataLoader,
    device: torch.device,
) -> DataLoader:
    """Build a DataLoader that yields batches in InfEmbed format from a diffusion DataLoader."""
    dataset = DiffusionInfembedDataset(base_loader, device)
    return DataLoader(dataset, batch_size=None, num_workers=0)
