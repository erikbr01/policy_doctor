"""
Influence Embeddings Pipeline: Main orchestration script.

This script runs the complete pipeline:
1. Load trained policy and datasets
2. Compute influence embeddings for training data
3. Compute influence embeddings for evaluation rollouts (with reward weighting)
4. Cluster failure embeddings to identify failure modes
5. Attribute failure clusters to training demonstrations
6. Generate and save results

Usage:
    python -m influence_embeddings.pipeline \
        --train_dir /path/to/train \
        --eval_dir /path/to/eval \
        --output_dir /path/to/output \
        --n_clusters 5
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import click
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("InfluenceEmbeddingsPipeline")


@click.command()
@click.option(
    "--train_dir", type=str, required=True, help="Path to training run directory"
)
@click.option(
    "--eval_dir",
    type=str,
    required=True,
    help="Path to evaluation directory with rollouts",
)
@click.option("--output_dir", type=str, required=True, help="Path to save outputs")
@click.option(
    "--train_ckpt",
    type=str,
    default="best",
    help="Checkpoint to use (best, latest, or index)",
)
@click.option(
    "--projection_dim", type=int, default=512, help="Dimension of influence embeddings"
)
@click.option(
    "--n_clusters", type=int, default=5, help="Number of failure mode clusters"
)
@click.option(
    "--top_k",
    type=int,
    default=10,
    help="Number of training demos to attribute per cluster",
)
@click.option(
    "--num_diffusion_timesteps",
    type=int,
    default=10,
    help="Diffusion timesteps per embedding",
)
@click.option(
    "--batch_size", type=int, default=32, help="Batch size for embedding computation"
)
@click.option(
    "--param_filter",
    type=str,
    default=None,
    help="Comma-separated parameter filters (e.g., 'model.up,model.final')",
)
@click.option(
    "--auto_k", is_flag=True, help="Automatically find optimal number of clusters"
)
@click.option("--device", type=str, default="cuda:0", help="Device for computation")
@click.option("--seed", type=int, default=42, help="Random seed")
def main(
    train_dir: str,
    eval_dir: str,
    output_dir: str,
    train_ckpt: str,
    projection_dim: int,
    n_clusters: int,
    top_k: int,
    num_diffusion_timesteps: int,
    batch_size: int,
    param_filter: Optional[str],
    auto_k: bool,
    device: str,
    seed: int,
):
    """Run the complete influence embeddings pipeline."""
    # Set seeds.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Parse param filter.
    param_filter_list = param_filter.split(",") if param_filter else None

    # Create output directory.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)

    logger.info("=" * 60)
    logger.info("INFLUENCE EMBEDDINGS PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Train directory: {train_dir}")
    logger.info(f"Eval directory: {eval_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Projection dimension: {projection_dim}")
    logger.info(f"Number of clusters: {n_clusters}")

    # =========================================================================
    # Phase 1: Load Policy and Datasets
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1: Loading Policy and Datasets")
    logger.info("=" * 60)

    from diffusion_policy.common.trak_util import (
        get_best_checkpoint,
        get_index_checkpoint,
        get_policy_from_checkpoint,
    )
    from diffusion_policy.dataset.episode_dataset import BatchEpisodeDataset

    # Find and load checkpoint.
    checkpoint_dir = Path(train_dir) / "checkpoints"
    checkpoints = list(checkpoint_dir.iterdir())

    if train_ckpt == "best":
        checkpoint = get_best_checkpoint(checkpoints)
    elif train_ckpt.isdigit():
        checkpoint = get_index_checkpoint(checkpoints, int(train_ckpt))
    else:
        checkpoint = checkpoint_dir / f"{train_ckpt}.ckpt"

    logger.info(f"Loading policy from {checkpoint}")
    policy, cfg = get_policy_from_checkpoint(checkpoint, device=device)

    # Load training dataset.
    logger.info("Loading training dataset...")
    train_set = hydra.utils.instantiate(cfg.task.dataset)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )
    logger.info(f"Training set size: {len(train_set)}")

    # Load evaluation dataset.
    logger.info("Loading evaluation dataset...")
    eval_set = BatchEpisodeDataset(
        batch_size=batch_size,
        dataset_path=Path(eval_dir) / "episodes",
        exec_horizon=1,
        sample_history=0,
    )
    logger.info(f"Evaluation set size: {len(eval_set)}")

    # =========================================================================
    # Phase 2: Compute Training Embeddings
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: Computing Training Embeddings")
    logger.info("=" * 60)

    from diffusion_policy.data_attribution.modelout_functions import (
        DiffusionLowdimFunctionalModelOutput,
    )
    from influence_embeddings.embedding_computer import EmbeddingComputer

    # Define cache paths.
    train_embeddings_cache = output_path / "train_embeddings.pt"
    projector_cache = output_path / "gradient_projector.pt"

    # Check if cached training embeddings exist.
    if train_embeddings_cache.exists() and projector_cache.exists():
        logger.info("Loading cached training embeddings...")
        cached_data = torch.load(
            train_embeddings_cache, map_location="cpu", weights_only=False
        )
        train_embeddings = cached_data["embeddings"]
        # Convert to float32 if needed for MPS compatibility
        if train_embeddings.dtype == torch.float64:
            train_embeddings = train_embeddings.float()
        train_embeddings = train_embeddings.to(device)
        train_metadata = cached_data["metadata"]
        logger.info(f"Loaded training embeddings from cache: {train_embeddings.shape}")

        # Load the projector to use for eval embeddings.
        from influence_embeddings.gradient_projector import GradientProjector

        computer = EmbeddingComputer(
            model=policy,
            projection_dim=projection_dim,
            param_filter=param_filter_list,
            num_diffusion_timesteps=num_diffusion_timesteps,
            device=device,
            seed=seed,
        )
        # Replace the projector with the loaded one to ensure consistency.
        computer.projector = GradientProjector.load(
            str(projector_cache), policy, device
        )
        logger.info("Loaded gradient projector from cache")

        # Create loss function for later use.
        loss_fn = DiffusionLowdimFunctionalModelOutput(loss_fn="ddpm")
    else:
        logger.info("Computing training embeddings...")
        # Create embedding computer.
        computer = EmbeddingComputer(
            model=policy,
            projection_dim=projection_dim,
            param_filter=param_filter_list,
            num_diffusion_timesteps=num_diffusion_timesteps,
            device=device,
            seed=seed,
        )

        # Create loss function.
        loss_fn = DiffusionLowdimFunctionalModelOutput(loss_fn="ddpm")

        # Compute training embeddings.
        train_embeddings, train_metadata = computer.compute_train_embeddings(
            dataloader=train_loader,
            loss_fn=loss_fn.get_output,
            noise_scheduler_timesteps=cfg.policy.noise_scheduler.num_train_timesteps,
            save_path=train_embeddings_cache,
        )

        # Save projector for reproducibility.
        computer.save_projector(projector_cache)

    # =========================================================================
    # Phase 3: Compute Evaluation Embeddings
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: Computing Evaluation Embeddings")
    logger.info("=" * 60)

    # Define cache path.
    eval_embeddings_cache = output_path / "eval_embeddings.pt"

    # Check if cached evaluation embeddings exist.
    if eval_embeddings_cache.exists():
        logger.info("Loading cached evaluation embeddings...")
        cached_eval_data = torch.load(
            eval_embeddings_cache, map_location="cpu", weights_only=False
        )
        eval_embeddings = cached_eval_data["embeddings"]
        eval_rewards = cached_eval_data["rewards"]
        # Convert to float32 if needed for MPS compatibility
        if eval_embeddings.dtype == torch.float64:
            eval_embeddings = eval_embeddings.float()
        if eval_rewards.dtype == torch.float64:
            eval_rewards = eval_rewards.float()
        eval_embeddings = eval_embeddings.to(device)
        eval_rewards = eval_rewards.to(device)
        eval_metadata = cached_eval_data["metadata"]
        logger.info(f"Loaded evaluation embeddings from cache: {eval_embeddings.shape}")
    else:
        logger.info("Computing evaluation embeddings...")
        eval_embeddings, eval_rewards, eval_metadata = computer.compute_eval_embeddings(
            dataloader=eval_set,
            loss_fn=loss_fn.get_output,
            noise_scheduler_timesteps=cfg.policy.noise_scheduler.num_train_timesteps,
            save_path=eval_embeddings_cache,
        )

    # =========================================================================
    # Phase 4: Cluster Failure Embeddings
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Phase 4: Clustering Failure Embeddings")
    logger.info("=" * 60)

    from influence_embeddings.failure_clusterer import FailureClusterer

    # Define cache path.
    clusterer_cache = output_path / "failure_clusterer.pt"

    # Check if cached clusterer exists.
    if clusterer_cache.exists():
        logger.info("Loading cached failure clusterer...")
        clusterer = FailureClusterer.load(str(clusterer_cache))
        logger.info(f"Loaded clusterer from cache with {clusterer.n_clusters} clusters")
    else:
        logger.info("Computing failure clustering...")
        # Move embeddings to CPU for clustering (sklearn requires numpy)
        eval_embeddings_cpu = eval_embeddings.cpu()
        eval_rewards_cpu = eval_rewards.cpu()

        # Optionally find optimal k (only if there are failures).
        num_failures = eval_metadata["num_failures"]
        if auto_k and num_failures > 0:
            logger.info("Finding optimal number of clusters...")
            temp_clusterer = FailureClusterer(n_clusters=2, random_state=seed)
            optimal_k, scores = temp_clusterer.find_optimal_k(
                eval_embeddings_cpu,
                eval_rewards_cpu,
                k_range=(2, min(20, num_failures - 1)),
            )
            n_clusters = optimal_k
            logger.info(f"Selected optimal k={n_clusters}")

        # Create and fit clusterer.
        clusterer = FailureClusterer(n_clusters=n_clusters, random_state=seed)
        clusterer.fit(eval_embeddings_cpu, eval_rewards_cpu, filter_failures=True)

        # Save clusterer if clustering was successful.
        if clusterer.labels_ is not None:
            clusterer.save(str(clusterer_cache))

    # Check if clustering was successful (i.e., there were failure embeddings).
    # Initialize variables that may not be set if no failures found
    cluster_summary = None
    attributions = []

    if clusterer.labels_ is None:
        logger.warning(
            "No failure embeddings were found. Skipping clustering and attribution phases."
        )
    else:
        # Get cluster summary.
        cluster_summary = clusterer.get_cluster_summary()
        logger.info(f"Cluster summary: {cluster_summary}")

        # =========================================================================
        # Phase 5: Attribute Failure Clusters to Training Demos
        # =========================================================================
        logger.info("\n" + "=" * 60)
        logger.info("Phase 5: Attributing Failure Clusters to Training Demos")
        logger.info("=" * 60)

        from influence_embeddings.training_attributor import TrainingAttributor

        # Define cache path.
        attributions_cache = output_path / "attributions.pt"

        # Check if cached attributions exist.
        if attributions_cache.exists():
            logger.info("Loading cached attributions...")
            attributions = torch.load(attributions_cache, weights_only=False)
            logger.info(f"Loaded {len(attributions)} attribution results from cache")
        else:
            logger.info("Computing attributions...")
            # Move embeddings to CPU for attribution (uses numpy/sklearn internally)
            train_embeddings_cpu = train_embeddings.cpu()

            # Create attributor.
            attributor = TrainingAttributor(train_embeddings_cpu, top_k=top_k)

            # Get cluster centers and attribute.
            cluster_centers = clusterer.get_cluster_centers()
            attributions = attributor.attribute_all_clusters(
                cluster_centers, top_k=top_k
            )

            # Save attributions.
            attributor.save_attributions(attributions, str(attributions_cache))

        # Generate report (always generate to ensure it's up to date).
        train_embeddings_cpu = train_embeddings.cpu()
        attributor = TrainingAttributor(train_embeddings_cpu, top_k=top_k)
        report = attributor.generate_report(attributions)
        logger.info("\n" + report)

        # Save report to file.
        with open(output_path / "attribution_report.txt", "w") as f:
            f.write(report)

    # =========================================================================
    # Phase 6: Save Final Results
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Phase 6: Saving Final Results")
    logger.info("=" * 60)

    # Compile all results.
    results = {
        "config": {
            "train_dir": train_dir,
            "eval_dir": eval_dir,
            "checkpoint": str(checkpoint),
            "projection_dim": projection_dim,
            "n_clusters": n_clusters,
            "top_k": top_k,
            "num_diffusion_timesteps": num_diffusion_timesteps,
            "param_filter": param_filter_list,
            "seed": seed,
        },
        "train_metadata": train_metadata,
        "eval_metadata": eval_metadata,
        "cluster_summary": cluster_summary,
        "attributions": [
            {
                "cluster_id": a.cluster_id,
                "training_indices": a.training_indices,
                "similarity_scores": a.similarity_scores,
            }
            for a in attributions
        ],
    }
    torch.save(results, output_path / "results.pt")

    logger.info(f"\nAll results saved to {output_path}")
    logger.info("Pipeline complete!")

    return results


def run_pipeline(
    train_dir: str,
    eval_dir: str,
    output_dir: str,
    **kwargs,
) -> Dict[str, Any]:
    """Programmatic entry point for the pipeline.

    Args:
        train_dir: Path to training run directory.
        eval_dir: Path to evaluation directory with rollouts.
        output_dir: Path to save outputs.
        **kwargs: Additional arguments (see main() for options).

    Returns:
        Results dictionary.
    """
    # Set defaults for any missing arguments.
    defaults = {
        "train_ckpt": "best",
        "projection_dim": 512,
        "n_clusters": 5,
        "top_k": 10,
        "num_diffusion_timesteps": 10,
        "batch_size": 32,
        "param_filter": None,
        "auto_k": False,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "seed": 42,
    }
    defaults.update(kwargs)

    # Create a context to invoke the click command.
    ctx = click.Context(main)
    ctx.ensure_object(dict)

    # Run pipeline.
    return ctx.invoke(
        main,
        train_dir=train_dir,
        eval_dir=eval_dir,
        output_dir=output_dir,
        **defaults,
    )


if __name__ == "__main__":
    main()
