"""
Device resolution for training/eval. Supports CUDA, MPS (Apple Silicon), and CPU.
When config requests cuda but CUDA is not available, falls back to MPS if
available, then CPU.
"""
import torch


def get_device(requested: str) -> torch.device:
    """
    Resolve requested device string to a torch.device, with fallbacks when
    the requested device is not available (e.g. cuda on Mac -> mps or cpu).

    Args:
        requested: Device string, e.g. "cuda:0", "mps", "cpu".

    Returns:
        torch.device to use.
    """
    requested = (requested or "").strip().lower()
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested)
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps":
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested or "cpu")


def _mps_available() -> bool:
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def non_blocking_for(device: torch.device) -> bool:
    """
    Whether to use non_blocking=True when transferring tensors to this device.
    Only CUDA benefits from non_blocking; MPS/CPU should use False to avoid
    transfer quirks.
    """
    return device.type == "cuda"
