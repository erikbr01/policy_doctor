"""TRAK-based stream scorer: scores new samples against cached finalized features.

Requires the ``cupid`` conda env (imports ``diffusion_policy`` and ``trak``).

Offline artifacts required (produced by ``train_trak_diffusion.py``):
  - ``<trak_save_dir>/metadata.json``           — proj_dim, proj_type
  - ``<trak_save_dir>/<model_id>/features.mmap`` — Hessian-weighted projected train features

Scoring a new sample requires a full forward+backward pass through the policy to
compute the per-sample gradient; no gradient computation is possible from cached
artifacts alone.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

from policy_doctor.monitoring.base import StreamScorer


class TRAKStreamScorer(StreamScorer):
    """Score new test-time samples using pre-computed TRAK train features.

    Parameters
    ----------
    checkpoint:
        Path to the diffusion policy ``.ckpt`` file used during TRAK training.
    trak_save_dir:
        Directory written by ``TRAKer(save_dir=...)``.  Must contain
        ``metadata.json`` and ``<model_id>/features.mmap``.
    model_id:
        Checkpoint index used during ``traker.load_checkpoint(..., model_id=...)``.
        Defaults to 0 (single-checkpoint runs).
    projector_seed:
        The ``projector_seed`` passed to ``TRAKer``.  Defaults to 0 (library default).
    grad_wrt:
        Parameter name prefixes used during TRAK featurization.  Must match the
        ``--model_keys`` argument of ``train_trak_diffusion.py``.  ``None`` means
        all parameters (library default).
    num_diffusion_timesteps:
        Number of diffusion timesteps to average gradients over.  Should match
        ``--num_timesteps`` used during TRAK training.
    loss_fn:
        Loss function name passed to the modelout function (``"square"``, ``"ddpm"``,
        ``"average"``).  Must match ``--loss_fn`` used during TRAK training.
    device:
        Torch device string (e.g. ``"cuda:0"``).
    """

    def __init__(
        self,
        checkpoint: Union[str, Path],
        trak_save_dir: Union[str, Path],
        model_id: int = 0,
        projector_seed: int = 0,
        grad_wrt: Optional[List[str]] = None,
        num_diffusion_timesteps: int = 8,
        loss_fn: str = "square",
        device: str = "cuda:0",
    ) -> None:
        from diffusion_policy.common.trak_util import (
            get_parameter_names,
            get_policy_from_checkpoint,
        )
        from diffusion_policy.data_attribution.gradient_computers import (
            DiffusionHybridImageFunctionalGradientComputer,
            DiffusionLowdimFunctionalGradientComputer,
        )
        from diffusion_policy.data_attribution.modelout_functions import (
            DiffusionHybridImageFunctionalModelOutput,
            DiffusionLowdimFunctionalModelOutput,
        )
        from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
            DiffusionUnetHybridImagePolicy,
        )
        from trak.projectors import BasicProjector, ProjectionType

        self.device = torch.device(device)
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.model_id = model_id

        # --- Load policy ---
        self.policy, cfg = get_policy_from_checkpoint(checkpoint, device=self.device)
        self.policy.eval()
        self._num_train_timesteps = cfg.policy.noise_scheduler.num_train_timesteps

        # --- Resolve grad_wrt and grad_dim ---
        if grad_wrt is not None:
            resolved_grad_wrt = get_parameter_names(self.policy, grad_wrt)
        else:
            resolved_grad_wrt = None

        all_params = dict(self.policy.named_parameters())
        if resolved_grad_wrt is not None:
            param_subset = {k: all_params[k] for k in resolved_grad_wrt if k in all_params}
        else:
            param_subset = all_params
        self.grad_dim = sum(v.numel() for v in param_subset.values())
        self.normalize_factor = math.sqrt(self.grad_dim)

        # --- Load TRAK metadata and features ---
        trak_save_dir = Path(trak_save_dir)
        with open(trak_save_dir / "metadata.json") as f:
            meta = json.load(f)
        self.proj_dim: int = meta["JL dimension"]
        proj_type_str: str = meta["JL matrix type"]

        features_path = trak_save_dir / str(model_id) / "features.mmap"
        if not features_path.exists():
            raise FileNotFoundError(f"TRAK features not found: {features_path}")
        # Load as memmap; cast to float32 for scoring (may be stored as float16)
        raw_features = np.lib.format.open_memmap(features_path, mode="r")
        self._features = raw_features.astype(np.float32)  # (N_train, proj_dim)

        # --- Reconstruct projector with the same seed ---
        proj_type = ProjectionType(proj_type_str)
        self._projector = BasicProjector(
            grad_dim=self.grad_dim,
            proj_dim=self.proj_dim,
            seed=projector_seed,
            proj_type=proj_type,
            device=self.device,
            dtype=torch.float32,
            model_id=model_id,
        )

        # --- Gradient computer ---
        is_hybrid = isinstance(self.policy, DiffusionUnetHybridImagePolicy)
        if is_hybrid:
            task = DiffusionHybridImageFunctionalModelOutput(loss_fn=loss_fn)
            self._grad_computer = DiffusionHybridImageFunctionalGradientComputer(
                model=self.policy,
                task=task,
                grad_dim=self.grad_dim,
                dtype=torch.float32,
                device=self.device,
                grad_wrt=resolved_grad_wrt,
            )
        else:
            task = DiffusionLowdimFunctionalModelOutput(loss_fn=loss_fn)
            self._grad_computer = DiffusionLowdimFunctionalGradientComputer(
                model=self.policy,
                task=task,
                grad_dim=self.grad_dim,
                dtype=torch.float32,
                device=self.device,
                grad_wrt=resolved_grad_wrt,
            )

    def embed(self, batch: dict) -> np.ndarray:
        """Compute the JL-projected gradient for one sample.

        Returns ``(proj_dim,)`` float32 array.
        """
        grads = self._grad_computer.compute_per_sample_grad(batch)  # (1, grad_dim)
        projected = self._projector.project(grads, model_id=self.model_id)  # (1, proj_dim)
        projected = projected / self.normalize_factor
        return projected[0].cpu().float().numpy()

    def score(self, batch: dict) -> np.ndarray:
        """Compute influence scores against all cached training features.

        Returns ``(N_train,)`` float32 array.
        """
        p = self.embed(batch)  # (proj_dim,)
        scores = self._features @ p   # (N_train,)
        return scores.astype(np.float32)
