"""
Simple Example: Demonstrates the influence embeddings pipeline with synthetic data.

This script creates synthetic data to validate the core pipeline logic:
1. Creates a simple MLP policy
2. Generates synthetic training and evaluation data
3. Computes influence embeddings
4. Clusters failure embeddings
5. Attributes failures to training data

Run with:
    python -m influence_embeddings.example_simple
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleMLPPolicy(nn.Module):
    """Simple MLP policy for testing."""

    def __init__(self, obs_dim: int = 10, action_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def create_synthetic_data(
    n_train: int = 100,
    n_eval: int = 50,
    obs_dim: int = 10,
    action_dim: int = 4,
    failure_rate: float = 0.4,
    seed: int = 42,
):
    """Create synthetic training and evaluation data.

    Creates two "failure modes":
    - Mode A: obs has high values in first 3 dims
    - Mode B: obs has high values in last 3 dims
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Training data (all successes).
    train_obs = torch.randn(n_train, obs_dim)
    train_actions = torch.randn(n_train, action_dim)

    # Evaluation data with failures.
    eval_obs = torch.randn(n_eval, obs_dim)
    eval_actions = torch.randn(n_eval, action_dim)

    # Create failure modes by modifying some observations.
    n_failures = int(n_eval * failure_rate)
    failure_indices = np.random.choice(n_eval, n_failures, replace=False)

    # Half of failures are Mode A (high first dims), half are Mode B (high last dims).
    mode_a_failures = failure_indices[: n_failures // 2]
    mode_b_failures = failure_indices[n_failures // 2 :]

    # Inject failure patterns.
    eval_obs[mode_a_failures, :3] += 3.0  # Mode A
    eval_obs[mode_b_failures, -3:] += 3.0  # Mode B

    # Create rewards: +1 for success, -1 for failure.
    rewards = torch.ones(n_eval)
    rewards[failure_indices] = -1.0

    return {
        "train_obs": train_obs,
        "train_actions": train_actions,
        "eval_obs": eval_obs,
        "eval_actions": eval_actions,
        "rewards": rewards,
        "failure_indices": failure_indices,
        "mode_a_failures": mode_a_failures,
        "mode_b_failures": mode_b_failures,
    }


def run_simple_example(output_dir: str = "./outputs/simple_example"):
    """Run the simplified influence embeddings pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # =========================================================================
    # Step 1: Create model and data
    # =========================================================================
    logger.info("Creating model and synthetic data...")

    obs_dim, action_dim = 10, 4
    model = SimpleMLPPolicy(obs_dim=obs_dim, action_dim=action_dim).to(device)

    # Train the model briefly on synthetic data.
    data = create_synthetic_data(
        n_train=200, n_eval=100, obs_dim=obs_dim, action_dim=action_dim
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_dataset = TensorDataset(data["train_obs"], data["train_actions"])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Quick training loop.
    model.train()
    for epoch in range(10):
        for obs, actions in train_loader:
            obs, actions = obs.to(device), actions.to(device)
            pred = model(obs)
            loss = F.mse_loss(pred, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    logger.info("Model trained.")

    # =========================================================================
    # Step 2: Compute influence embeddings
    # =========================================================================
    logger.info("Computing influence embeddings...")

    from .gradient_projector import GradientProjector

    # Create projector.
    projector = GradientProjector(
        model=model,
        projection_dim=128,
        param_filter=None,  # Use all params for simple model
        device=device,
        seed=42,
    )

    def compute_embeddings_simple(
        obs_batch: torch.Tensor, action_batch: torch.Tensor
    ) -> torch.Tensor:
        """Compute embeddings for a batch using simple autograd."""
        batch_size = obs_batch.shape[0]
        embeddings = torch.zeros(batch_size, projector.projection_dim, device=device)

        for i in range(batch_size):
            obs_i = obs_batch[i : i + 1].to(device).requires_grad_(False)
            action_i = action_batch[i : i + 1].to(device)

            # Forward pass.
            pred = model(obs_i)
            loss = F.mse_loss(pred, action_i)

            # Compute gradients.
            grads = torch.autograd.grad(
                loss,
                list(projector._func_weights.values()),
                create_graph=False,
            )

            # Flatten and project.
            flat_grad = torch.cat([g.flatten() for g in grads])
            embeddings[i] = flat_grad @ projector.projection_matrix

        # Normalize.
        norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
        embeddings = embeddings / norms

        return embeddings

    # Compute training embeddings.
    train_embeddings = compute_embeddings_simple(
        data["train_obs"], data["train_actions"]
    )
    logger.info(f"Computed {train_embeddings.shape[0]} training embeddings")

    # Compute evaluation embeddings with reward weighting.
    eval_embeddings_raw = compute_embeddings_simple(
        data["eval_obs"], data["eval_actions"]
    )
    rewards = data["rewards"].to(device)
    eval_embeddings = eval_embeddings_raw * rewards.unsqueeze(-1)

    # Re-normalize after reward weighting.
    norms = eval_embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
    eval_embeddings = eval_embeddings / norms
    logger.info(f"Computed {eval_embeddings.shape[0]} evaluation embeddings")

    # =========================================================================
    # Step 3: Cluster failure embeddings
    # =========================================================================
    logger.info("Clustering failure embeddings...")

    from .failure_clusterer import FailureClusterer

    clusterer = FailureClusterer(n_clusters=2, random_state=42)  # 2 failure modes
    clusterer.fit(eval_embeddings.cpu(), rewards.cpu(), filter_failures=True)

    cluster_summary = clusterer.get_cluster_summary()
    logger.info(f"Cluster summary: {cluster_summary}")

    # =========================================================================
    # Step 4: Attribute to training data
    # =========================================================================
    logger.info("Attributing failures to training data...")

    from .training_attributor import TrainingAttributor

    attributor = TrainingAttributor(train_embeddings.cpu(), top_k=5)
    cluster_centers = clusterer.get_cluster_centers()
    attributions = attributor.attribute_all_clusters(cluster_centers, top_k=5)

    # Print report.
    report = attributor.generate_report(attributions)
    logger.info("\n" + report)

    # =========================================================================
    # Step 5: Save results
    # =========================================================================
    results = {
        "train_embeddings": train_embeddings.cpu(),
        "eval_embeddings": eval_embeddings.cpu(),
        "rewards": rewards.cpu(),
        "cluster_centers": cluster_centers,
        "cluster_labels": clusterer.labels_,
        "failure_indices": data["failure_indices"],
        "mode_a_failures": data["mode_a_failures"],
        "mode_b_failures": data["mode_b_failures"],
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
    logger.info(f"Results saved to {output_path}")

    # Save report.
    with open(output_path / "report.txt", "w") as f:
        f.write(report)

    logger.info("Simple example complete!")
    return results


if __name__ == "__main__":
    run_simple_example()
