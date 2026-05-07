"""Benchmark / equivalence test for InfEmbed `predict` with torch.compile.

Loads the same wrapper / fit results that compute_infembed_embeddings.py loads
in --predict_only mode, runs `embedder.predict` on a fixed N batches in three
configs (eager, inner_unet compile, wrapper compile), and reports:

  * wall time per config (with and without warmup)
  * per-row cosine similarity / max abs error of compiled vs eager embeddings

All three configs run in the *same process* so RNG state is reset to a fixed
seed before each — that lets us compare the resulting embeddings directly
without worrying about cross-process RNG drift.

Usage (from third_party/cupid):

  python bench_infembed_compile.py \\
      --eval_dir <path> --train_dir <path> --train_ckpt latest \\
      --exp_name <trak exp> --device cuda:1 \\
      --num_batches 30 --warmup_batches 3
"""

from __future__ import annotations

import pathlib
import random
import sys
import time
from typing import Optional

import click
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader

# Make `infembed` importable when run from a worktree without a fresh install.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
INFEMBED_ROOT = PROJECT_ROOT / "third_party" / "infembed"
if INFEMBED_ROOT.exists() and str(INFEMBED_ROOT) not in sys.path:
    sys.path.insert(0, str(INFEMBED_ROOT))

from diffusion_policy.common.device_util import get_device
from diffusion_policy.common.trak_util import (
    get_best_checkpoint,
    get_index_checkpoint,
    get_policy_from_checkpoint,
    get_parameter_names,
)
from diffusion_policy.common.ddp_util import compile_model
from diffusion_policy.data_attribution.infembed_adapter import (
    DiffusionLossWrapper,
    IdentityLossNone,
)

MODELOUT_FN_DIR = "diffusion_policy.data_attribution.modelout_functions"


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _make_train_batches(
    train_set,
    batch_size: int,
    num_train_timesteps: int,
    num_timesteps: int,
    device: torch.device,
    n: int,
    seed: int,
):
    """Materialize the first n InfEmbed-format batches into device memory.

    By materializing eagerly we (a) avoid RNG drift when the same batches are
    iterated multiple times in this process and (b) keep timesteps identical
    across configs (we set the seed before sampling timesteps and reuse the
    resulting tensors).
    """
    _seed_everything(seed)
    loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    batches = []
    for i, raw in enumerate(loader):
        if i >= n:
            break
        raw = {k: v.to(device) for k, v in raw.items()}
        B = raw["action"].shape[0]
        timesteps = torch.randint(
            num_train_timesteps, (B, num_timesteps), device=device
        ).long()
        batch = dict(raw)
        batch["timesteps"] = timesteps
        labels = torch.zeros(B, device=device, dtype=torch.float32)
        batches.append((batch, labels))
    return batches


class _StaticBatchDataset(torch.utils.data.IterableDataset):
    """Yields a fixed list of pre-built (batch_dict, labels) pairs."""

    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        for batch in self.batches:
            yield batch


def _make_loader(batches) -> DataLoader:
    return DataLoader(_StaticBatchDataset(batches), batch_size=None, num_workers=0)


def _time_predict(embedder, batches, device, n_warmup, seed):
    """Run a warmup pass then a timed pass on `batches`.

    Returns (timed_embeddings, warmup_secs, timed_secs).  The timed pass
    re-runs all `batches`; we re-seed before it so any in-graph RNG
    (e.g. noise inside get_output for the wrapper-compile config) is identical
    to a single-run-from-fresh predict.
    """
    warmup_batches = batches[:n_warmup]
    timed_batches = batches  # full N batches, including the warmup ones
    if warmup_batches:
        _seed_everything(seed)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _ = embedder.predict(_make_loader(warmup_batches))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        warmup_s = time.perf_counter() - t0
    else:
        warmup_s = 0.0

    _seed_everything(seed)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    embs = embedder.predict(_make_loader(timed_batches))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timed_s = time.perf_counter() - t0
    return embs.cpu().numpy(), warmup_s, timed_s


def _run_predict(embedder, loader, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    embs = embedder.predict(loader)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    return embs.cpu().numpy(), t1 - t0


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1) + 1e-12
    nb = np.linalg.norm(b, axis=1) + 1e-12
    return (a * b).sum(axis=1) / (na * nb)


def _summarize_diff(name: str, ref: np.ndarray, other: np.ndarray) -> None:
    cos = _cosine_similarity(ref, other)
    abs_err = np.abs(ref - other)
    rel_err = abs_err / (np.abs(ref) + 1e-9)
    print(
        f"  {name}: cos[min={cos.min():.6f} mean={cos.mean():.6f}]  "
        f"abs_err[max={abs_err.max():.3e} mean={abs_err.mean():.3e}]  "
        f"rel_err[max={rel_err.max():.3e}]"
    )


@click.command()
@click.option("--eval_dir", type=str, required=True)
@click.option("--train_dir", type=str, required=True)
@click.option("--train_ckpt", type=str, default="latest")
@click.option("--exp_name", type=str, required=True)
@click.option("--modelout_fn", type=str, default="DiffusionLowdimFunctionalModelOutput")
@click.option("--loss_fn", type=str, default="square")
@click.option("--num_timesteps", type=int, default=64)
@click.option("--batch_size", type=int, default=32)
@click.option("--device", type=str, default="cuda:0")
@click.option("--seed", type=int, default=0)
@click.option("--model_keys", type=str, default="model.")
@click.option("--num_batches", type=int, default=30,
              help="Total batches to feed predict.  Drawn from the train set.")
@click.option("--warmup_batches", type=int, default=3,
              help="Number of batches to use as warmup (timed separately).")
@click.option("--projection_dim", type=int, default=100)
@click.option("--arnoldi_dim", type=int, default=200)
@click.option("--fit_results", type=str, default=None)
@click.option("--tf32", is_flag=True, default=False)
@click.option("--dataset_path", type=str, default=None)
@click.option(
    "--configs",
    type=str,
    default="eager,inner_unet,wrapper",
    help="Comma-separated configs to run.  eager / inner_unet / wrapper / "
         "inner_unet_reduce_overhead.",
)
@click.option(
    "--projection_on_cpu/--projection_on_gpu",
    "projection_on_cpu",
    default=True,
    help="Where the InfEmbed fit_results.R lives.  GPU avoids per-batch "
         "CPU->GPU transfers during predict but uses ~projection_dim*model_size "
         "extra GPU memory.",
)
def main(
    eval_dir, train_dir, train_ckpt, exp_name, modelout_fn, loss_fn,
    num_timesteps, batch_size, device, seed, model_keys,
    num_batches, warmup_batches, projection_dim, arnoldi_dim, fit_results,
    tf32, dataset_path, configs, projection_on_cpu,
):
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    device = get_device(device)
    eval_dir = pathlib.Path(eval_dir)
    train_dir = pathlib.Path(train_dir)
    out_dir = eval_dir / exp_name
    fit_results_path = pathlib.Path(fit_results) if fit_results else (out_dir / "infembed_fit.pt")
    if not fit_results_path.exists():
        raise FileNotFoundError(fit_results_path)

    config_list = [c.strip() for c in configs.split(",") if c.strip()]
    print(f"Configs: {config_list}")
    print(f"Device: {device}, batch_size={batch_size}, num_timesteps={num_timesteps}")
    print(f"num_batches={num_batches}, warmup_batches={warmup_batches}")

    # Load checkpoint and build train_set once; we'll rebuild a wrapper per config.
    checkpoint_dir = train_dir / "checkpoints"
    checkpoints = list(checkpoint_dir.iterdir())
    if train_ckpt == "best":
        checkpoint = get_best_checkpoint(checkpoints)
    elif train_ckpt.isdigit():
        checkpoint = get_index_checkpoint(checkpoints, int(train_ckpt))
    else:
        checkpoint = checkpoint_dir / f"{train_ckpt}.ckpt"

    print(f"Loading checkpoint: {checkpoint.name}")
    policy, cfg = get_policy_from_checkpoint(checkpoint, device=device)
    try:
        from policy_doctor.data.adapters import patch_attribution_dataset_path
        patch_attribution_dataset_path(
            cfg, repo_root=PROJECT_ROOT, dataset_path_override=dataset_path,
        )
    except ImportError:
        pass

    key_list = [k.strip() for k in model_keys.split(",")] if model_keys else []
    infembed_layers = ["policy." + k.rstrip(".") for k in key_list] if key_list else None
    num_train_timesteps = cfg.policy.noise_scheduler.num_train_timesteps
    print(f"infembed_layers: {infembed_layers}, num_train_timesteps={num_train_timesteps}")

    train_set = hydra.utils.instantiate(cfg.task.dataset)
    print(f"Train set size: {len(train_set)}")

    # Pre-build the (timesteps + data) batches once, deterministically.  All
    # configs see the *exact same inputs* so any embedding diff is purely from
    # compile vs eager kernels.
    print(f"Materialising {num_batches} batches with seed={seed}...")
    batches = _make_train_batches(
        train_set, batch_size, num_train_timesteps, num_timesteps, device,
        n=num_batches, seed=seed,
    )
    print(f"  Built {len(batches)} batches.")

    task_cls = hydra.utils.get_class(f"{MODELOUT_FN_DIR}.{modelout_fn}")
    task = task_cls(loss_fn=loss_fn)
    loss_fn_none = IdentityLossNone()

    from infembed.embedder._core.arnoldi_embedder import ArnoldiEmbedder

    results = {}  # name -> (embeddings, total_s, post_warmup_s)

    for cfg_name in config_list:
        print(f"\n=== Config: {cfg_name} ===")
        # Fresh policy load per config: torch.compile mutates the module so we
        # don't want state leakage across runs.
        policy_i, _ = get_policy_from_checkpoint(checkpoint, device=device)
        try:
            patch_attribution_dataset_path(
                cfg, repo_root=PROJECT_ROOT, dataset_path_override=dataset_path,
            )
        except Exception:
            pass
        wrapper = DiffusionLossWrapper(policy_i, task)
        if cfg_name == "eager":
            pass
        elif cfg_name == "inner_unet":
            inner = policy_i.model
            policy_i.model = compile_model(inner, fullgraph=False, dynamic=False)
            print(f"  Compiled inner U-Net ({type(inner).__name__})")
        elif cfg_name == "inner_unet_reduce_overhead":
            inner = policy_i.model
            policy_i.model = compile_model(
                inner, fullgraph=False, dynamic=False, mode="reduce-overhead",
            )
            print(f"  Compiled inner U-Net ({type(inner).__name__}) mode=reduce-overhead")
        elif cfg_name == "wrapper":
            wrapper = compile_model(wrapper, fullgraph=False, dynamic=True)
            print("  Compiled DiffusionLossWrapper (dynamic=True)")
        elif cfg_name == "wrapper_static":
            wrapper = compile_model(wrapper, fullgraph=False, dynamic=False)
            print("  Compiled DiffusionLossWrapper (dynamic=False)")
        else:
            raise ValueError(cfg_name)

        embedder = ArnoldiEmbedder(
            model=wrapper,
            loss_fn=loss_fn_none,
            test_loss_fn=loss_fn_none,
            sample_wise_grads_per_batch=False,
            projection_dim=projection_dim,
            arnoldi_dim=arnoldi_dim,
            projection_on_cpu=projection_on_cpu,
            show_progress=False,
            layers=infembed_layers,
        )
        embedder.load(str(fit_results_path), projection_on_cpu=projection_on_cpu)

        embs, warmup_s, timed_s = _time_predict(
            embedder, batches, device, n_warmup=warmup_batches, seed=seed + 1,
        )
        per_batch = timed_s / num_batches
        print(
            f"  warmup ({warmup_batches} batches): {warmup_s:.2f}s; "
            f"timed ({num_batches} batches): {timed_s:.2f}s ({per_batch:.3f} s/batch)"
        )

        # Phase breakdown: time forward-only and forward+per-sample-grads on a
        # single batch so we know which phase compile actually affects.
        _seed_everything(seed + 2)
        feats, labels = batches[0][0:-1], batches[0][-1]
        params = list(wrapper.parameters())
        if infembed_layers:
            from infembed.embedder._utils.gradient import _extract_parameters_from_layers
            # Same layer module resolution as InfEmbed:
            layer_modules = []
            for name, m in wrapper.named_modules():
                if name in infembed_layers:
                    layer_modules.append(m)
            params = _extract_parameters_from_layers(layer_modules)
        else:
            params = [p for p in wrapper.parameters() if p.requires_grad]
        # Forward only
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(5):
            with torch.enable_grad():
                out = wrapper(*feats)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fwd_s = (time.perf_counter() - t0) / 5
        # Forward + per-sample autograd.grad (the hot loop in predict)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(3):
            with torch.enable_grad():
                out = wrapper(*feats)
                for i in range(out.shape[0]):
                    grads = torch.autograd.grad(
                        outputs=out[i], inputs=tuple(params),
                        grad_outputs=torch.ones_like(out[i]),
                        retain_graph=True, allow_unused=True,
                    )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fwd_grads_s = (time.perf_counter() - t0) / 3
        print(f"  phase: fwd_only={fwd_s:.3f}s/batch  fwd+per_sample_grads={fwd_grads_s:.3f}s/batch")
        results[cfg_name] = (embs, warmup_s, timed_s, per_batch, fwd_s, fwd_grads_s)

    print("\n=== Summary ===")
    print(f"{'config':>30}  {'warmup (s)':>11}  {'timed (s)':>10}  {'s/batch':>9}  {'fwd':>7}  {'fwd+grads':>10}  {'speedup':>8}")
    eager_per_batch = results.get("eager", [None, None, None, None, None, None])[3]
    for cfg_name, (_e, warmup_s, timed_s, per_batch, fwd_s, fwd_grads_s) in results.items():
        speedup = (eager_per_batch / per_batch) if eager_per_batch else float("nan")
        print(f"{cfg_name:>30}  {warmup_s:>11.2f}  {timed_s:>10.2f}  "
              f"{per_batch:>9.3f}  {fwd_s:>7.3f}  {fwd_grads_s:>10.3f}  {speedup:>8.2f}x")

    if "eager" in results:
        eager_embs = results["eager"][0]
        for cfg_name, (embs, *_) in results.items():
            if cfg_name == "eager":
                continue
            print(f"\nEquivalence ({cfg_name} vs eager) on {eager_embs.shape[0]} predict outputs:")
            assert embs.shape == eager_embs.shape, (embs.shape, eager_embs.shape)
            _summarize_diff(cfg_name, eager_embs, embs)


if __name__ == "__main__":
    main()
