"""InfEmbed-based stream scorer: embeds new samples using a cached Arnoldi fit.

Requires the ``cupid`` conda env (imports ``diffusion_policy`` and ``infembed``).

Offline artifacts required (produced by ``compute_infembed_embeddings.py``):
  - ``<exp_dir>/infembed_fit.pt``           — fitted Arnoldi eigenvector basis (R matrix)
  - ``<exp_dir>/infembed_embeddings.npz``   — pre-computed demo and rollout embeddings

InfEmbed uses data-adaptive Arnoldi eigenvectors (NOT random JL projection).
The fit file stores scaled top eigenvectors of the Gauss-Newton Hessian of the
training loss; ``predict()`` projects any sample's gradient onto this basis.
Influence between sample A and B ≈ dot product of their embeddings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from policy_doctor.monitoring.base import StreamScorer


def _detect_infembed_layers(policy: "torch.nn.Module", fit_path: str) -> Optional[List[str]]:
    """Return the infembed ``layers`` filter needed to match a saved fit's parameter count.

    When the policy gains new parameters after a fit was computed (e.g. normalizer buffers
    registered as params, ``_dummy_variable`` sentinels), the jacobian ordering diverges
    from the R vectors in the fit. This function loads the fit, counts R parameters, and
    tries common subsets of the policy's parameter list to find a match.

    Returns a list of layer-name prefixes (in the DiffusionLossWrapper namespace, i.e.
    with the ``"policy."`` prefix) or ``None`` if no filter is needed or one can't be found.
    """
    try:
        import dill
        with open(fit_path, "rb") as f:
            fit_results = dill.load(f)
    except Exception:
        return None

    r_shapes = [tuple(t.shape) for t in fit_results.R[0]]
    n_fit = len(r_shapes)

    all_named = list(policy.named_parameters())
    if len(all_named) == n_fit:
        all_shapes = [tuple(p.shape) for _, p in all_named]
        if all_shapes == r_shapes:
            return None  # already matches — no filter needed

    # Try progressively broader prefixes until one matches
    candidate_prefixes: List[Optional[str]] = [
        "model.",
        "obs_encoder.",
        None,  # non-zero params only (last resort)
    ]
    for prefix in candidate_prefixes:
        if prefix is None:
            cands = [(n, p) for n, p in all_named if 0 not in p.shape]
        else:
            cands = [(n, p) for n, p in all_named if n.startswith(prefix)]
        if len(cands) == n_fit:
            cand_shapes = [tuple(p.shape) for _, p in cands]
            if cand_shapes == r_shapes:
                if prefix is not None:
                    return ["policy." + prefix.rstrip(".")]
                # Can't express zero-shape exclusion as a prefix; fall through
    return None


class _SingleBatchDataset(torch.utils.data.IterableDataset):
    """Wraps a single batch dict in an IterableDataset that yields one ``(batch, labels)`` tuple.

    InfEmbed's ``predict()`` iterates over a DataLoader expecting batches of the form
    ``(features, labels)`` where ``features`` is unpacked as ``*features`` into the model.
    We yield ``(batch_dict, dummy_labels)`` so that ``model(batch_dict)`` is called correctly.
    """

    def __init__(self, batch: dict, device: torch.device) -> None:
        self._batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        B = batch["action"].shape[0]
        self._labels = torch.zeros(B, device=device, dtype=torch.float32)

    def __iter__(self) -> Iterator[Tuple[dict, torch.Tensor]]:
        yield self._batch, self._labels


class InfEmbedStreamScorer(StreamScorer):
    """Score new test-time samples using a pre-fitted InfEmbed Arnoldi embedder.

    Parameters
    ----------
    checkpoint:
        Path to the diffusion policy ``.ckpt`` file used during InfEmbed fitting.
    infembed_fit_path:
        Path to ``infembed_fit.pt`` written by ``compute_infembed_embeddings.py``.
    infembed_embeddings_path:
        Path to ``infembed_embeddings.npz``.  Must contain ``demo_embeddings``
        (shape ``(N_demo, proj_dim)``); ``rollout_embeddings`` is used by the
        :class:`~policy_doctor.monitoring.graph_assigner.NearestCentroidAssigner`.
    model_keys:
        Comma-separated parameter name prefixes (same as ``--model_keys`` in the
        original InfEmbed run).  ``None`` means all parameters.
    num_diffusion_timesteps:
        Number of diffusion timesteps to average over (must match ``--num_timesteps``).
    loss_fn:
        Loss function name (must match ``--loss_fn``).
    device:
        Torch device string (e.g. ``"cuda:0"``).
    """

    def __init__(
        self,
        checkpoint: Union[str, Path],
        infembed_fit_path: Union[str, Path],
        infembed_embeddings_path: Union[str, Path],
        model_keys: Optional[str] = None,
        num_diffusion_timesteps: int = 8,
        loss_fn: str = "square",
        device: str = "cuda:0",
    ) -> None:
        from diffusion_policy.common.trak_util import (
            get_parameter_names,
            get_policy_from_checkpoint,
        )
        from diffusion_policy.data_attribution.infembed_adapter import (
            DiffusionLossWrapper,
            IdentityLossNone,
        )
        from diffusion_policy.data_attribution.modelout_functions import (
            DiffusionHybridImageFunctionalModelOutput,
            DiffusionLowdimFunctionalModelOutput,
        )
        from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
            DiffusionUnetHybridImagePolicy,
        )

        try:
            from infembed.embedder._core.arnoldi_embedder import ArnoldiEmbedder
        except ImportError as e:
            raise ImportError(
                "infembed not found. Ensure third_party/infembed is installed or on PYTHONPATH."
            ) from e

        self.device = torch.device(device)
        self.num_diffusion_timesteps = num_diffusion_timesteps

        # --- Load policy ---
        self.policy, cfg = get_policy_from_checkpoint(checkpoint, device=self.device)
        self.policy.eval()
        self._num_train_timesteps = cfg.policy.noise_scheduler.num_train_timesteps

        # --- Resolve model_keys / infembed_layers ---
        key_list: List[str] = [k.strip() for k in model_keys.split(",")] if model_keys else []
        resolved_grad_wrt = get_parameter_names(self.policy, key_list) if key_list else None
        infembed_layers = (
            ["policy." + k.rstrip(".") for k in key_list] if key_list else None
        )

        # --- Auto-detect layer filter when model_keys not specified ---
        # The fit was computed with certain policy parameters; if the policy has since
        # gained extra parameters (e.g. _dummy_variable, normalizer buffers registered as
        # params), the jacobian ordering won't match the R vectors. Detect this by
        # comparing the fit's parameter count to the policy's and try common subsets.
        if not key_list:
            infembed_layers = _detect_infembed_layers(
                self.policy, str(Path(infembed_fit_path))
            )
            if infembed_layers is not None:
                print(f"  [infembed] Auto-detected layer filter: {infembed_layers}")

        # --- Build DiffusionLossWrapper ---
        is_hybrid = isinstance(self.policy, DiffusionUnetHybridImagePolicy)
        if is_hybrid:
            task = DiffusionHybridImageFunctionalModelOutput(loss_fn=loss_fn)
        else:
            task = DiffusionLowdimFunctionalModelOutput(loss_fn=loss_fn)

        wrapper = DiffusionLossWrapper(self.policy, task)
        loss_fn_none = IdentityLossNone()

        # --- Load ArnoldiEmbedder from saved fit ---
        self._embedder = ArnoldiEmbedder(
            model=wrapper,
            loss_fn=loss_fn_none,
            test_loss_fn=loss_fn_none,
            sample_wise_grads_per_batch=False,
            projection_dim=None,   # overridden by the loaded fit
            arnoldi_dim=None,
            projection_on_cpu=True,
            show_progress=False,   # no progress bars for single-sample streaming
            layers=infembed_layers,
        )
        self._embedder.load(str(Path(infembed_fit_path)), projection_on_cpu=True)

        # --- Load pre-computed embeddings ---
        emb_path = Path(infembed_embeddings_path)
        if not emb_path.exists():
            raise FileNotFoundError(f"InfEmbed embeddings not found: {emb_path}")
        data = np.load(emb_path, allow_pickle=False)
        self._demo_embeddings: np.ndarray = data["demo_embeddings"].astype(np.float32)
        self._rollout_embeddings: Optional[np.ndarray] = (
            data["rollout_embeddings"].astype(np.float32)
            if "rollout_embeddings" in data
            else None
        )

    @property
    def demo_embeddings(self) -> np.ndarray:
        """Pre-computed training demo embeddings, shape ``(N_demo, proj_dim)``."""
        return self._demo_embeddings

    @property
    def rollout_embeddings(self) -> Optional[np.ndarray]:
        """Pre-computed rollout embeddings, shape ``(N_rollout, proj_dim)``, or ``None``."""
        return self._rollout_embeddings

    def embed(self, batch: dict) -> np.ndarray:
        """Project a single-sample batch into the InfEmbed embedding space.

        Returns ``(proj_dim,)`` float32 numpy array.
        """
        dataset = _SingleBatchDataset(batch, self.device)
        loader = DataLoader(dataset, batch_size=None, num_workers=0)
        embedding = self._embedder.predict(loader)  # (1, proj_dim) on CPU
        return embedding[0].float().numpy()

    def score(self, batch: dict) -> np.ndarray:
        """Compute influence scores of all demo training samples on ``batch``.

        Score ≈ dot product of ``embed(batch)`` with each training demo embedding.
        Returns ``(N_demo,)`` float32 array; higher = more influential.
        """
        e = self.embed(batch)          # (proj_dim,)
        scores = self._demo_embeddings @ e  # (N_demo,)
        return scores.astype(np.float32)
