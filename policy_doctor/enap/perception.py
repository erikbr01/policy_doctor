"""Visual encoder and HDBSCAN clustering for ENAP perception module.

Implements the Adaptive Symbol Abstraction stage of ENAP:

    observations (o_t = [I_t, p_t])
        → VisualEncoder.forward(images, proprio) → z_t   (128-dim)
        → HDBSCANClusterer.fit_predict(Z)         → c_t  (discrete symbol)

The visual encoder uses a frozen DINOv2 ViT-S/14 backbone (or a lightweight
MLP on pre-extracted features when ``backbone="mlp"``), a spatial-softmax +
MLP visual head, and a Fourier-feature + spectral-norm proprioception head.
Features are concatenated into a single vector ``z_t``.

Cluster centroids ``μ_c`` are stored after fitting and used by the M-step
regularisation loss ``L_center = ‖z_t − μ_{c_t}‖²``.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fourier feature embedding for proprioception
# ---------------------------------------------------------------------------

class FourierFeatureEmbedding(nn.Module):
    """Random Fourier features for continuous proprioception inputs.

    Computes ``[sin(B·x), cos(B·x)]`` where ``B`` is a frozen random Gaussian
    matrix, then applies spectral normalisation to a linear projection.

    Args:
        input_dim: Dimensionality of the proprioception vector.
        output_dim: Output feature dimension (must be even).
        sigma: Standard deviation of the random Gaussian frequencies.
    """

    def __init__(self, input_dim: int, output_dim: int = 64, sigma: float = 1.0) -> None:
        super().__init__()
        assert output_dim % 2 == 0, "output_dim must be even"
        half = output_dim // 2
        B = torch.randn(input_dim, half) * sigma
        self.register_buffer("B", B)  # frozen
        self.proj = nn.utils.spectral_norm(nn.Linear(output_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        z = x @ self.B  # (batch, half)
        features = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)  # (batch, output_dim)
        return self.proj(features)


# ---------------------------------------------------------------------------
# Visual head (Spatial Softmax → MLP)
# ---------------------------------------------------------------------------

class SpatialSoftmax(nn.Module):
    """Spatial softmax over a feature map producing expected (x, y) positions.

    For a feature map of shape ``(B, C, H, W)``, computes the expected
    spatial position for each channel and returns a vector of shape ``(B, 2C)``.
    """

    def __init__(self, height: int, width: int, num_channels: int) -> None:
        super().__init__()
        x_map = torch.linspace(-1, 1, width).unsqueeze(0).expand(height, -1)
        y_map = torch.linspace(-1, 1, height).unsqueeze(1).expand(-1, width)
        # (1, 1, H, W) position grids
        self.register_buffer("x_map", x_map.unsqueeze(0).unsqueeze(0))
        self.register_buffer("y_map", y_map.unsqueeze(0).unsqueeze(0))
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # Softmax over spatial dims
        flat = x.reshape(B, C, -1)  # (B, C, H*W)
        weights = torch.softmax(flat, dim=-1).reshape(B, C, H, W)
        ex = (weights * self.x_map).sum(dim=(-2, -1))  # (B, C)
        ey = (weights * self.y_map).sum(dim=(-2, -1))  # (B, C)
        return torch.cat([ex, ey], dim=-1)  # (B, 2C)


# ---------------------------------------------------------------------------
# Visual Encoder
# ---------------------------------------------------------------------------

class VisualEncoder(nn.Module):
    """Multi-modal encoder: images + proprioception → fused feature vector z_t.

    Supports two backbone modes:

    - ``"dino"`` (default): loads a frozen DINOv2 ViT-S/14 from torch.hub and
      adds a spatial-softmax → MLP visual head on the patch tokens.
    - ``"mlp"`` (lightweight): skips the ViT entirely and treats the image
      input as a pre-extracted flat feature vector, only applying an MLP head.
      Useful for unit tests and for data where DINOv2 features have already
      been extracted.

    Args:
        image_input_dim: Flat image feature dimension (used in ``"mlp"`` mode
            or as the DINOv2 patch-token embedding dim in ``"dino"`` mode).
        proprio_dim: Proprioception input dimensionality.
        output_dim: Final fused feature dimension ``z_t``.
        backbone: ``"dino"`` or ``"mlp"``.
        fourier_output_dim: Proprioception head output size.
    """

    def __init__(
        self,
        image_input_dim: int = 384,
        proprio_dim: int = 7,
        output_dim: int = 128,
        backbone: str = "dino",
        fourier_output_dim: int = 64,
    ) -> None:
        super().__init__()
        self.backbone_mode = backbone
        self._backbone: Optional[nn.Module] = None

        if backbone == "dino":
            # Visual head: DINOv2 patch tokens (384-d for ViT-S) → spatial-softmax not
            # applicable here since we get a 1-D CLS token. We use a 2-layer MLP.
            visual_head_in = image_input_dim  # 384 for ViT-S/14
        else:  # "mlp"
            visual_head_in = image_input_dim

        self.visual_head = nn.Sequential(
            nn.Linear(visual_head_in, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim // 2),
        )

        # Proprioception head
        self.proprio_head = FourierFeatureEmbedding(
            input_dim=proprio_dim,
            output_dim=fourier_output_dim,
        )

        # Fusion MLP: visual (output_dim//2) + proprio (fourier_output_dim) → output_dim
        self.fusion = nn.Sequential(
            nn.Linear(output_dim // 2 + fourier_output_dim, output_dim),
            nn.ReLU(),
        )

        self.output_dim = output_dim

    def _load_dino_backbone(self) -> None:
        """Lazily load and freeze DINOv2 ViT-S/14 from torch.hub."""
        if self._backbone is None:
            self._backbone = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14", pretrained=True
            )
            for p in self._backbone.parameters():
                p.requires_grad_(False)
            self._backbone.eval()

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to visual features.

        Args:
            images: ``(B, C, H, W)`` tensor (dino mode) or ``(B, D)`` flat
                pre-extracted features (mlp mode).

        Returns:
            ``(B, output_dim // 2)`` visual feature vector.
        """
        if self.backbone_mode == "dino":
            self._load_dino_backbone()
            assert self._backbone is not None
            with torch.no_grad():
                cls_token = self._backbone(images)  # (B, 384)
            return self.visual_head(cls_token)
        else:
            return self.visual_head(images)

    def forward(self, images: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Return fused feature vector ``z_t``.

        Args:
            images: ``(B, C, H, W)`` or ``(B, D)`` pre-extracted features.
            proprio: ``(B, proprio_dim)`` proprioception vector.

        Returns:
            ``(B, output_dim)`` fused feature tensor.
        """
        v = self.encode_images(images)     # (B, output_dim // 2)
        p = self.proprio_head(proprio)      # (B, fourier_output_dim)
        return self.fusion(torch.cat([v, p], dim=-1))


# ---------------------------------------------------------------------------
# HDBSCAN clusterer
# ---------------------------------------------------------------------------

class HDBSCANClusterer:
    """Wrapper around ``hdbscan.HDBSCAN`` for ENAP symbol abstraction.

    After :meth:`fit_predict`, the cluster centroids ``μ_c`` are stored in
    :attr:`centroids` and can be used for the M-step regularisation loss
    ``L_center = ‖z_t − μ_{c_t}‖²``.

    Args:
        min_cluster_size: Minimum number of samples in a cluster (HDBSCAN param).
        min_samples: Core point threshold (defaults to ``min_cluster_size``).
        cluster_selection_epsilon: Distance threshold for cluster merging.
    """

    def __init__(
        self,
        min_cluster_size: int = 10,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
    ) -> None:
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self._clusterer = None
        self.centroids: Optional[Dict[int, np.ndarray]] = None
        self.labels_: Optional[np.ndarray] = None

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """Fit HDBSCAN and return per-sample cluster labels.

        Noise points (label = -1) are reassigned to the nearest cluster centroid
        so that every timestep gets a valid symbol.

        Args:
            features: ``(N, D)`` float32 feature array.

        Returns:
            Integer symbol array ``c_t`` of shape ``(N,)`` with values in
            ``{0, 1, ..., K-1}`` — no noise points after reassignment.
        """
        try:
            import hdbscan
        except ImportError as exc:
            raise ImportError(
                "hdbscan is required for ENAPHDBSCANClusterer. "
                "Install with: pip install hdbscan"
            ) from exc

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            prediction_data=True,
        )
        labels = clusterer.fit_predict(features)
        self._clusterer = clusterer

        # Compute centroids for labelled points
        unique_labels = sorted(set(labels) - {-1})
        centroids: Dict[int, np.ndarray] = {}
        for lbl in unique_labels:
            mask = labels == lbl
            centroids[lbl] = features[mask].mean(axis=0)
        self.centroids = centroids

        # Reassign noise points to nearest centroid
        if centroids and np.any(labels == -1):
            centroid_matrix = np.stack([centroids[k] for k in sorted(centroids)])
            centroid_ids = sorted(centroids)
            noise_idx = np.where(labels == -1)[0]
            noise_feats = features[noise_idx]
            dists = np.linalg.norm(
                noise_feats[:, None, :] - centroid_matrix[None, :, :], axis=-1
            )  # (n_noise, K)
            nearest = dists.argmin(axis=-1)
            for i, orig_idx in enumerate(noise_idx):
                labels[orig_idx] = centroid_ids[nearest[i]]

        # Remap labels to 0-indexed contiguous integers
        label_map = {old: new for new, old in enumerate(sorted(set(labels) - {-1}))}
        remapped = np.array([label_map[l] for l in labels], dtype=np.int64)
        # Update centroids with remapped keys
        self.centroids = {label_map[k]: v for k, v in centroids.items()}
        self.labels_ = remapped
        return remapped

    @property
    def num_symbols(self) -> int:
        """Number of distinct symbols (clusters) discovered."""
        if self.centroids is None:
            return 0
        return len(self.centroids)


# ---------------------------------------------------------------------------
# Batch feature extraction (no gradient)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    encoder: VisualEncoder,
    images: torch.Tensor,
    proprio: torch.Tensor,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Extract features from a dataset in batches.

    Args:
        encoder: Trained (or frozen) :class:`VisualEncoder`.
        images: ``(N, ...)`` image tensor.
        proprio: ``(N, proprio_dim)`` proprioception tensor.
        batch_size: Number of samples per forward pass.
        device: Target device (defaults to encoder's device).

    Returns:
        ``(N, output_dim)`` numpy feature array.
    """
    if device is None:
        device = next(encoder.parameters()).device
    encoder.eval()
    N = images.shape[0]
    all_feats = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        imgs_b = images[start:end].to(device)
        prop_b = proprio[start:end].to(device)
        feats = encoder(imgs_b, prop_b).cpu().numpy()
        all_feats.append(feats)
    return np.concatenate(all_feats, axis=0)
