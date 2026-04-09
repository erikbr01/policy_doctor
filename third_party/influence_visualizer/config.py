"""Configuration loading for the influence visualizer.

This module provides YAML-based configuration for the influence visualizer app.
Config files are stored in influence_visualizer/configs/ and define task-specific
settings including data paths, observation keys, and state/action labels.
"""

import pathlib
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class VisualizerConfig:
    """Configuration for the influence visualizer.

    Attributes:
        task: Task identifier (e.g., 'robomimic_lift', 'pusht')
        name: Human-readable task name
        eval_dir: Path to evaluation output (TRAK results + rollout episodes)
        train_dir: Path to training output (checkpoints/)
        train_ckpt: Checkpoint selection ('latest', 'best', or epoch number)
        exp_date: Experiment date prefix for TRAK results
        image_dataset_path: Optional path to image HDF5 (if training used lowdim)
        annotation_file: Path to annotation pickle file for behavior labels
        use_mock: Whether to use mock data for testing
        obs_key: Observation key for images in the dataset
        top_k: Number of top influences to display
        state_labels: Semantic labels for state dimensions
        action_labels: Semantic labels for action dimensions
        seeds: Optional list of seed strings (e.g. ["0", "1", "2"]) for per-seed success rate
        comparison: Optional list of other config names to compare in the Comparison tab
        wandb_project: Optional wandb project for fetching run history (required for Comparison)
        wandb_entity: Optional wandb entity (defaults to API default if unset)
    """

    task: str
    name: str
    eval_dir: Optional[str] = None
    train_dir: Optional[str] = None
    train_ckpt: str = "latest"
    exp_date: str = "default"
    image_dataset_path: Optional[str] = None
    annotation_file: Optional[str] = None
    use_mock: bool = False
    obs_key: str = "agentview_image"
    top_k: int = 10
    lazy_load_images: bool = True  # Use lazy HDF5 loading for large image datasets
    quality_labels: Optional[List[str]] = None  # Quality tier masks to load from HDF5
    state_labels: List[str] = field(default_factory=list)
    action_labels: List[str] = field(default_factory=list)
    seeds: Optional[List[str]] = None  # e.g. ["0", "1", "2"] for per-seed success rate
    comparison: Optional[List[str]] = None  # other config names to compare (e.g. ["square_mh_feb5"])
    wandb_project: Optional[str] = None  # for Comparison tab (fetch test/mean_score over time)
    wandb_entity: Optional[str] = None  # wandb entity (optional)

    def copy(self) -> "VisualizerConfig":
        """Create a copy of this config."""
        return VisualizerConfig(
            task=self.task,
            name=self.name,
            eval_dir=self.eval_dir,
            train_dir=self.train_dir,
            train_ckpt=self.train_ckpt,
            exp_date=self.exp_date,
            image_dataset_path=self.image_dataset_path,
            annotation_file=self.annotation_file,
            use_mock=self.use_mock,
            obs_key=self.obs_key,
            top_k=self.top_k,
            lazy_load_images=self.lazy_load_images,
            quality_labels=list(self.quality_labels) if self.quality_labels else None,
            state_labels=list(self.state_labels),
            action_labels=list(self.action_labels),
            seeds=list(self.seeds) if self.seeds else None,
            comparison=list(self.comparison) if self.comparison else None,
            wandb_project=self.wandb_project,
            wandb_entity=self.wandb_entity,
        )


def get_configs_dir() -> pathlib.Path:
    """Get the configs directory path."""
    return pathlib.Path(__file__).parent / "configs"


def list_configs() -> List[str]:
    """List available config files.

    Returns:
        List of config names (without .yaml extension), sorted alphabetically.
    """
    configs_dir = get_configs_dir()
    if not configs_dir.exists():
        return []
    return sorted([f.stem for f in configs_dir.glob("*.yaml")])


def load_config(config_name: str) -> VisualizerConfig:
    """Load config from YAML file.

    Args:
        config_name: Name of config file (without .yaml extension)

    Returns:
        VisualizerConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_path = get_configs_dir() / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    # Handle null values from YAML
    if data.get("state_labels") is None:
        data["state_labels"] = []
    if data.get("action_labels") is None:
        data["action_labels"] = []

    return VisualizerConfig(**data)


def get_generic_labels(dim: int, prefix: str = "dim") -> List[str]:
    """Generate generic dimension labels.

    Args:
        dim: Number of dimensions
        prefix: Label prefix (default: 'dim')

    Returns:
        List of labels like ['dim_0', 'dim_1', ...]
    """
    return [f"{prefix}_{i}" for i in range(dim)]
