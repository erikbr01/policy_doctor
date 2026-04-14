"""Utilities for DistributedDataParallel (DDP) training.

Usage in a workspace's ``run()`` method::

    rank       = int(cfg.training.get("_ddp_rank", 0))
    world_size = int(cfg.training.get("_ddp_world_size", 1))

The dispatcher (train_baseline / train_curated) launches one process per GPU
with ``torch.multiprocessing.spawn``, injecting ``+training._ddp_rank`` and
``+training._ddp_world_size`` into the Hydra overrides before composing the
config.
"""

from __future__ import annotations

import os
import socket
from typing import Any, Callable, Dict


def find_free_port() -> int:
    """Return a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def setup_ddp(rank: int, world_size: int, master_addr: str = "localhost", master_port: int | None = None, backend: str = "nccl") -> None:
    """Initialize the default process group for DDP.

    ``master_port`` is read from ``os.environ["MASTER_PORT"]`` if not given
    explicitly (allows torchrun-style launch as well as manual spawn).
    """
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", master_addr)
    if master_port is not None:
        os.environ["MASTER_PORT"] = str(master_port)
    elif "MASTER_PORT" not in os.environ:
        raise RuntimeError("MASTER_PORT must be set before calling setup_ddp()")

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_ddp() -> None:
    """Destroy the default process group."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def ddp_worker(
    rank: int,
    world_size: int,
    worker_fn: Callable,
    worker_kwargs: Dict[str, Any],
    master_addr: str = "localhost",
    master_port: int | None = None,
    backend: str = "nccl",
) -> None:
    """Generic DDP worker launched by ``torch.multiprocessing.spawn``.

    Injects ``+training._ddp_rank`` and ``+training._ddp_world_size`` into the
    ``overrides`` list inside *worker_kwargs* (mutating a copy), sets up the
    process group, calls *worker_fn*, then cleans up.

    The ``master_port`` is chosen once by the spawning process and passed in via
    ``args``; all ranks share it.
    """
    import torch

    if master_port is None:
        raise ValueError("master_port must be provided to ddp_worker")

    setup_ddp(rank=rank, world_size=world_size, master_addr=master_addr, master_port=master_port, backend=backend)
    torch.cuda.set_device(rank)

    # Inject DDP rank/world-size into the Hydra overrides so the workspace
    # can read them from cfg.training._ddp_rank / cfg.training._ddp_world_size.
    kwargs = dict(worker_kwargs)
    overrides = list(kwargs.get("overrides", []))
    overrides.append(f"+training._ddp_rank={rank}")
    overrides.append(f"+training._ddp_world_size={world_size}")
    kwargs["overrides"] = overrides

    try:
        worker_fn(**kwargs)
    finally:
        cleanup_ddp()


def compile_model(model, **kwargs):
    """Wrap *model* with ``torch.compile`` when available (PyTorch >= 2.0).

    Defaults to ``fullgraph=True, dynamic=False`` for maximum kernel
    specialisation.  Both diffusion policy backbones (TransformerForDiffusion
    and ConditionalUnet1D) trace as a single graph with no breaks, so
    ``fullgraph=True`` is safe.  ``dynamic=False`` lets inductor specialise for
    fixed batch/sequence shapes, which is the normal training regime; a new
    shape triggers recompilation rather than a crash.

    Pass explicit kwargs to override the defaults, e.g.
    ``compile_model(model, dynamic=True)`` if you need variable shapes.

    Logs a warning and returns the model unchanged on older PyTorch builds.
    """
    import torch

    if not hasattr(torch, "compile"):
        import warnings
        warnings.warn(
            "torch.compile is not available (requires PyTorch >= 2.0). "
            "Continuing without compilation.",
            stacklevel=2,
        )
        return model
    kwargs.setdefault("fullgraph", True)
    kwargs.setdefault("dynamic", False)
    return torch.compile(model, **kwargs)


def spawn_ddp(
    worker_fn: Callable,
    worker_kwargs: Dict[str, Any],
    num_gpus: int,
    backend: str = "nccl",
) -> None:
    """Spawn *num_gpus* DDP worker processes via ``torch.multiprocessing.spawn``.

    A free port is chosen once in the calling process and shared with all
    workers so every rank binds to the same rendezvous address.
    """
    import torch.multiprocessing as mp

    port = find_free_port()
    mp.spawn(
        ddp_worker,
        args=(num_gpus, worker_fn, worker_kwargs, "localhost", port, backend),
        nprocs=num_gpus,
        join=True,
    )
