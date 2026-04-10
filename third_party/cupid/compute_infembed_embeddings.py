"""
Compute InfEmbed (influence) embeddings for diffusion policy train and rollout samples.

Saves embeddings to eval_dir / <trak_exp_name> / infembed_embeddings.npz for use
in the influence visualizer clustering tab. Run after TRAK (train_trak_diffusion.py)
so that the same eval_dir and experiment layout exist.

Usage (mirrors train_trak.sh / train_trak_diffusion.py):
  python compute_infembed_embeddings.py \\
    --eval_dir <path> --train_dir <path> --train_ckpt latest \\
    --exp_name <same as TRAK exp_name> \\
    --modelout_fn DiffusionLowdimFunctionalModelOutput \\
    --loss_fn square --num_timesteps 64 \\
    [--featurize_holdout] [--batch_size 32] [--device cuda:0] ...

Fit results are saved to eval_dir/<exp>/infembed_fit.pt. During Arnoldi fit,
a partial fit is written after each Arnoldi iteration (overwriting the same file),
and the raw Arnoldi state is saved to infembed_arnoldi_state.pt. If the run is
interrupted during fit, re-run the same command (no flags): the script will
resume from the last saved Arnoldi state. When fit completes, the final fit is
written and the resume state file is removed. If predict then fails, re-run with
--predict_only. If the run failed after demo embeddings
(infembed_embeddings_demo_only.npz) but during rollout, re-run with
--predict_rollout_only.
"""

from __future__ import annotations

import pickle
import sys

try:
    import dill
except ImportError:
    dill = None  # resume state was saved with dill in infembed; use pickle if dill missing
import pathlib
import random
import yaml
from typing import Optional

import click
import hydra
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

# Project root and third_party/infembed (so "import infembed" works without installing)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
INFEMBED_ROOT = PROJECT_ROOT / "third_party" / "infembed"
if INFEMBED_ROOT.exists() and str(INFEMBED_ROOT) not in sys.path:
    sys.path.insert(0, str(INFEMBED_ROOT))

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.device_util import get_device
from diffusion_policy.common.trak_util import (
    get_best_checkpoint,
    get_index_checkpoint,
    get_policy_from_checkpoint,
    get_parameter_names,
)
from diffusion_policy.data_attribution.infembed_adapter import (
    DiffusionLossWrapper,
    IdentityLossNone,
)
from diffusion_policy.dataset.episode_dataset import BatchEpisodeDataset
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
    DiffusionUnetHybridImagePolicy,
)

POLICIES = (DiffusionUnetLowdimPolicy, DiffusionUnetHybridImagePolicy)
MODELOUT_FN_DIR = "diffusion_policy.data_attribution.modelout_functions"


def _find_trak_experiment(eval_dir: pathlib.Path, exp_date: str = "default") -> str:
    """Find TRAK experiment directory name (same logic as influence_visualizer)."""
    pattern = f"{exp_date}_trak_results-*"
    matches = list(eval_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No TRAK experiment found matching {pattern} in {eval_dir}")
    return sorted(matches)[-1].name


def _add_timesteps(batch: dict, num_train_timesteps: int, num_timesteps: int, device: torch.device):
    """Add random timesteps to batch (same as TRAK)."""
    num_samples = batch["action"].shape[0]
    batch = dict(batch)
    batch["timesteps"] = torch.randint(
        num_train_timesteps,
        (num_samples, num_timesteps),
        device=device,
    ).long()
    return batch


class _TrainWithTimestepsIterable(torch.utils.data.IterableDataset):
    """Wraps a DataLoader and adds timesteps to each batch for InfEmbed."""

    def __init__(
        self,
        base_loader: DataLoader,
        num_train_timesteps: int,
        num_timesteps: int,
        device: torch.device,
    ):
        self.base_loader = base_loader
        self.num_train_timesteps = num_train_timesteps
        self.num_timesteps = num_timesteps
        self.device = device

    def __iter__(self):
        for batch in self.base_loader:
            batch = dict_apply(batch, lambda x: x.to(self.device))
            batch = _add_timesteps(
                batch,
                self.num_train_timesteps,
                self.num_timesteps,
                self.device,
            )
            B = batch["action"].shape[0]
            labels = torch.zeros(B, device=self.device, dtype=torch.float32)
            yield batch, labels


def _make_train_infembed_loader(
    train_loader: DataLoader,
    num_train_timesteps: int,
    num_timesteps: int,
    device: torch.device,
) -> DataLoader:
    dataset = _TrainWithTimestepsIterable(
        train_loader, num_train_timesteps, num_timesteps, device
    )
    return DataLoader(dataset, batch_size=None, num_workers=0)


class _RolloutWithTimestepsIterable(torch.utils.data.IterableDataset):
    """Wraps BatchEpisodeDataset and adds timesteps to each batch."""

    def __init__(
        self,
        rollout_dataset: BatchEpisodeDataset,
        num_train_timesteps: int,
        num_timesteps: int,
        device: torch.device,
    ):
        self.rollout_dataset = rollout_dataset
        self.num_train_timesteps = num_train_timesteps
        self.num_timesteps = num_timesteps
        self.device = device

    def __iter__(self):
        for batch in self.rollout_dataset:
            batch = dict_apply(batch, lambda x: x.to(self.device))
            batch = _add_timesteps(
                batch,
                self.num_train_timesteps,
                self.num_timesteps,
                self.device,
            )
            B = batch["action"].shape[0]
            labels = torch.zeros(B, device=self.device, dtype=torch.float32)
            yield batch, labels


def _make_rollout_infembed_loader(
    rollout_dataset: BatchEpisodeDataset,
    num_train_timesteps: int,
    num_timesteps: int,
    device: torch.device,
) -> DataLoader:
    dataset = _RolloutWithTimestepsIterable(
        rollout_dataset, num_train_timesteps, num_timesteps, device
    )
    return DataLoader(dataset, batch_size=None, num_workers=0)


@click.command()
@click.option("--exp_name", type=str, default="auto", help="TRAK experiment name, or 'auto' to detect from eval_dir")
@click.option("--eval_dir", type=str, required=True)
@click.option("--train_dir", type=str, required=True)
@click.option("--train_ckpt", type=str, default="latest")
@click.option("--model_keys", type=str, default=None, help="Comma-separated parameter name prefixes (same as TRAK)")
@click.option("--dataset_path", type=str, default=None,
              help="Override the HDF5 dataset path from the checkpoint config. "
                   "Required when the checkpoint was trained on a different machine or "
                   "the dataset has been moved (MimicGen / RoboCasa / Robomimic).")
@click.option("--modelout_fn", type=str, required=True)
@click.option("--loss_fn", type=str, required=True)
@click.option("--num_timesteps", type=int, required=True)
@click.option("--batch_size", type=int, default=32)
@click.option("--device", type=str, default="cuda:0")
@click.option("--seed", type=int, default=0)
@click.option("--featurize_holdout", is_flag=True, help="Include holdout in demo embeddings (match TRAK)")
@click.option("--projection_dim", type=int, default=100, help="InfEmbed embedding dimension D")
@click.option("--arnoldi_dim", type=int, default=200, help="Arnoldi subspace dimension ( > projection_dim)")
@click.option("--overwrite", is_flag=True, help="Overwrite existing infembed_embeddings.npz")
@click.option(
    "--predict_only",
    is_flag=True,
    help="Skip fit; load saved Arnoldi fit and only run predict (e.g. after a previous run failed during predict).",
)
@click.option(
    "--fit_results",
    type=str,
    default=None,
    help="Path to saved infembed_fit.pt (for --predict_only when the file is in a different dir than eval_dir/<exp>/infembed_fit.pt).",
)
@click.option(
    "--predict_rollout_only",
    is_flag=True,
    help="Resume after demo embeddings: load fit + demo from infembed_fit.pt and infembed_embeddings_demo_only.npz, compute only rollout embeddings, then save full npz.",
)
def main(
    exp_name: str,
    eval_dir: str,
    train_dir: str,
    train_ckpt: str,
    model_keys: Optional[str],
    dataset_path: Optional[str],
    modelout_fn: str,
    loss_fn: str,
    num_timesteps: int,
    batch_size: int,
    device: str,
    seed: int,
    featurize_holdout: bool,
    projection_dim: int,
    arnoldi_dim: int,
    overwrite: bool,
    predict_only: bool,
    fit_results: Optional[str],
    predict_rollout_only: bool,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = get_device(device)
    eval_dir = pathlib.Path(eval_dir)
    train_dir = pathlib.Path(train_dir)

    # Resolve exp_name if not provided (use same as TRAK)
    if exp_name == "auto" or not exp_name:
        exp_name = _find_trak_experiment(eval_dir)
        print(
            f"Using TRAK experiment: {exp_name}\n"
            "  (seed in the name is the TRAK/infembed RNG seed, not the training policy seed; "
            "policy seed is determined by eval_dir/train_dir paths.)"
        )
    out_dir = eval_dir / exp_name
    out_path = out_dir / "infembed_embeddings.npz"
    fit_results_path = pathlib.Path(fit_results) if fit_results else (out_dir / "infembed_fit.pt")
    arnoldi_state_path = out_dir / "infembed_arnoldi_state.pt"
    demo_only_path = out_dir / "infembed_embeddings_demo_only.npz"
    if predict_rollout_only:
        if not fit_results_path.exists():
            raise FileNotFoundError(
                f"Predict-rollout-only requested but fit results not found: {fit_results_path}. "
                "Run a full InfEmbed run first; fit is saved after Arnoldi completes."
            )
        if not demo_only_path.exists():
            raise FileNotFoundError(
                f"Predict-rollout-only requested but demo-only file not found: {demo_only_path}. "
                "Demo embeddings are saved after demo predict; if rollout failed, this file should exist."
            )
    elif predict_only:
        if not fit_results_path.exists():
            raise FileNotFoundError(
                f"Predict-only requested but fit results not found: {fit_results_path}. "
                "Run without --predict_only first to fit and save (fit is saved after Arnoldi completes). "
                "If the fit file exists elsewhere, pass --fit_results <path>."
            )
    else:
        if out_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {out_path}. Use --overwrite to replace."
            )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint and policy (same as train_trak_diffusion)
    checkpoint_dir = train_dir / "checkpoints"
    checkpoints = list(checkpoint_dir.iterdir())
    if train_ckpt == "best":
        checkpoint = get_best_checkpoint(checkpoints)
    elif train_ckpt.isdigit():
        checkpoint = get_index_checkpoint(checkpoints, int(train_ckpt))
    else:
        checkpoint = checkpoint_dir / f"{train_ckpt}.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    policy, cfg = get_policy_from_checkpoint(checkpoint, device=device)
    if not isinstance(policy, POLICIES):
        raise TypeError(f"Unsupported policy: {type(policy)}")

    # Resolve dataset path: patch stale absolute paths and support explicit overrides.
    # Handles robomimic / mimicgen / robocasa HDF5 layouts uniformly.
    try:
        from policy_doctor.data.adapters import patch_attribution_dataset_path
        patch_attribution_dataset_path(
            cfg,
            repo_root=pathlib.Path(__file__).resolve().parent,
            dataset_path_override=dataset_path,
        )
    except ImportError:
        # policy_doctor not on sys.path (standalone cupid usage); skip.
        pass

    # Same parameter set as TRAK: grad_wrt for optional logging; layers for InfEmbed.
    key_list = [k.strip() for k in model_keys.split(",")] if model_keys else []
    grad_wrt = get_parameter_names(policy, key_list) if key_list else None
    # InfEmbed layers: names relative to wrapper; wrapper has .policy, so "model." -> "policy.model"
    infembed_layers = None
    if key_list:
        infembed_layers = ["policy." + k.rstrip(".") for k in key_list]

    num_train_timesteps = cfg.policy.noise_scheduler.num_train_timesteps

    # Task (loss)
    task_cls = hydra.utils.get_class(f"{MODELOUT_FN_DIR}.{modelout_fn}")
    task = task_cls(loss_fn=loss_fn)

    # Wrapper so InfEmbed sees model(batch) -> (B,) per-example loss
    wrapper = DiffusionLossWrapper(policy, task)
    loss_fn_none = IdentityLossNone()

    # Train loader (no shuffle, same order as TRAK)
    train_set = hydra.utils.instantiate(cfg.task.dataset)
    train_set_size = len(train_set)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    train_infembed_loader = _make_train_infembed_loader(
        train_loader, num_train_timesteps, num_timesteps, device
    )

    # Optional holdout (same order as TRAK)
    holdout_loader = None
    holdout_set_size = 0
    if featurize_holdout:
        holdout_set = train_set.get_holdout_dataset()
        if holdout_set is not None and len(holdout_set) > 0:
            holdout_set_size = len(holdout_set)
            holdout_loader = DataLoader(
                holdout_set,
                batch_size=batch_size,
                num_workers=0,
                shuffle=False,
                pin_memory=(device.type == "cuda"),
            )

    # Rollout loader
    # Order preservation: BatchEpisodeDataset iterates episodes in numeric order (ep0, ep1, ...)
    # via EpisodeDataset's _episode_file_sort_key. ArnoldiEmbedder.predict concatenates
    # batch embeddings in dataloader order. Result must match build_rollout_sample_infos
    # (influence visualizer) which uses metadata.yaml episode_lengths in index order.
    rollout_set = BatchEpisodeDataset(
        batch_size=batch_size,
        dataset_path=eval_dir / "episodes",
        exec_horizon=1,
        sample_history=0,
    )
    rollout_set_size = len(rollout_set)

    # Verify rollout order/count matches metadata (same source as influence visualizer)
    metadata_path = eval_dir / "episodes" / "metadata.yaml"
    if metadata_path.exists():
        with open(metadata_path) as f:
            rollout_metadata = yaml.safe_load(f)
        expected_rollout_samples = sum(rollout_metadata.get("episode_lengths", []))
        if rollout_set_size != expected_rollout_samples:
            raise ValueError(
                f"Rollout sample count mismatch: BatchEpisodeDataset has {rollout_set_size}, "
                f"metadata.yaml episode_lengths sum is {expected_rollout_samples}. "
                "Episode file order may not match metadata; check EpisodeDataset sorting."
            )

    rollout_infembed_loader = _make_rollout_infembed_loader(
        rollout_set, num_train_timesteps, num_timesteps, device
    )

    # InfEmbed: same parameter set as TRAK when model_keys given (e.g. model. or obs_encoder.,model.)
    # sample_wise_grads_per_batch=False so we use per-example loss (B,) and loss_fn reduction='none'
    try:
        from infembed.embedder._core.arnoldi_embedder import ArnoldiEmbedder
    except ImportError:
        raise ImportError(
            "InfEmbed not found. Ensure third_party/infembed is installed or on PYTHONPATH."
        )

    embedder = ArnoldiEmbedder(
        model=wrapper,
        loss_fn=loss_fn_none,
        test_loss_fn=loss_fn_none,
        sample_wise_grads_per_batch=False,
        projection_dim=projection_dim,
        arnoldi_dim=arnoldi_dim,
        projection_on_cpu=True,
        show_progress=True,
        layers=infembed_layers,
    )

    saved_fit_path = None
    if predict_only or predict_rollout_only:
        print(f"Loading Arnoldi fit results from {fit_results_path}...")
        embedder.load(str(fit_results_path), projection_on_cpu=True)
    else:
        # Fit on training data only; save after each Arnoldi iteration for recovery/resume
        initial_arnoldi_state = None
        if arnoldi_state_path.exists():
            print(f"Resuming from saved Arnoldi state: {arnoldi_state_path}")
            load_state = dill.load if dill is not None else pickle.load
            with open(arnoldi_state_path, "rb") as f:
                state = load_state(f)
            initial_arnoldi_state = (state["qs"], state["H_filled"])
            n_done = len(initial_arnoldi_state[0]) - 1
            print(f"  Completed {n_done} iterations; will run remaining Arnoldi steps.")
        else:
            print("Fitting ArnoldiEmbedder on training set...")
        embedder.fit(
            train_infembed_loader,
            fit_checkpoint_path=str(fit_results_path.resolve()),
            fit_resume_state_path=str(arnoldi_state_path.resolve()),
            initial_arnoldi_state=initial_arnoldi_state,
        )
        # Save immediately after fit (before predict) so --predict_only works if predict fails later
        fit_results_path_abs = fit_results_path.resolve()
        embedder.save(str(fit_results_path_abs))
        if not fit_results_path_abs.exists():
            raise RuntimeError(
                f"Expected fit results file after save but not found: {fit_results_path_abs}"
            )
        saved_fit_path = fit_results_path_abs
        print(f"Saved fit results to {saved_fit_path}")
        # Remove resume state so next full run starts from scratch
        if arnoldi_state_path.exists():
            arnoldi_state_path.unlink()
            print(f"Removed resume state {arnoldi_state_path.name}")

    if predict_rollout_only:
        # Resume after demo: load saved demo embeddings, compute only rollout, write full npz
        with np.load(demo_only_path, allow_pickle=False) as f:
            demo_embeddings = np.asarray(f["demo_embeddings"])
        print("Computing rollout embeddings (resuming after demo)...")
        rollout_embeddings = embedder.predict(rollout_infembed_loader)
        rollout_embeddings = rollout_embeddings.cpu().numpy()
        assert rollout_embeddings.shape[0] == rollout_set_size, (
            rollout_embeddings.shape[0],
            rollout_set_size,
        )
        np.savez(
            out_path,
            rollout_embeddings=rollout_embeddings,
            demo_embeddings=demo_embeddings,
            embedding_dim=np.int32(projection_dim),
            train_set_size=np.int32(train_set_size),
            test_set_size=np.int32(rollout_set_size),
        )
        print(f"Saved InfEmbed embeddings to {out_path}")
        print(f"  demo_embeddings: {demo_embeddings.shape} (from {demo_only_path.name})")
        print(f"  rollout_embeddings: {rollout_embeddings.shape}")
    else:
        # Predict: create fresh loaders (previous iterables were consumed by fit)
        train_loader_predict = DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=(device.type == "cuda"),
        )
        train_infembed_loader2 = _make_train_infembed_loader(
            train_loader_predict, num_train_timesteps, num_timesteps, device
        )
        try:
            print("Computing demo embeddings (train)...")
            demo_train_embs = embedder.predict(train_infembed_loader2)
            demo_train_embs = demo_train_embs.cpu().numpy()
            assert demo_train_embs.shape[0] == train_set_size, (demo_train_embs.shape[0], train_set_size)

            if holdout_loader is not None and holdout_set_size > 0:
                holdout_infembed_loader = _make_train_infembed_loader(
                    holdout_loader, num_train_timesteps, num_timesteps, device
                )
                print("Computing demo embeddings (holdout)...")
                demo_holdout_embs = embedder.predict(holdout_infembed_loader)
                demo_holdout_embs = demo_holdout_embs.cpu().numpy()
                demo_embeddings = np.concatenate([demo_train_embs, demo_holdout_embs], axis=0)
            else:
                demo_embeddings = demo_train_embs

            # Save partial results (demo only) so we don't lose them if rollout predict fails
            partial_path = out_path.parent / (out_path.stem + "_demo_only.npz")
            np.savez(
                partial_path,
                demo_embeddings=demo_embeddings,
                embedding_dim=np.int32(projection_dim),
                train_set_size=np.int32(train_set_size),
            )
            print(f"Saved demo embeddings (partial) to {partial_path}")

            print("Computing rollout embeddings...")
            rollout_embeddings = embedder.predict(rollout_infembed_loader)
            rollout_embeddings = rollout_embeddings.cpu().numpy()
            assert rollout_embeddings.shape[0] == rollout_set_size, (
                rollout_embeddings.shape[0],
                rollout_set_size,
            )

            np.savez(
                out_path,
                rollout_embeddings=rollout_embeddings,
                demo_embeddings=demo_embeddings,
                embedding_dim=np.int32(projection_dim),
                train_set_size=np.int32(train_set_size),
                test_set_size=np.int32(rollout_set_size),
            )
            print(f"Saved InfEmbed embeddings to {out_path}")
            print(f"  demo_embeddings: {demo_embeddings.shape}")
            print(f"  rollout_embeddings: {rollout_embeddings.shape}")
        except Exception as e:
            if saved_fit_path is not None:
                print(
                    f"Predict failed (fit was already saved). Re-run with:\n"
                    f"  --predict_only --fit_results {saved_fit_path}\n"
                    f"Error: {e}",
                    file=sys.stderr,
                )
            raise


if __name__ == "__main__":
    main()
