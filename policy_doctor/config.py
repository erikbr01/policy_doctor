"""Configuration loading for policy_doctor.

Config files are stored in the package ``configs/`` directory and define task-specific
settings including data paths and observation keys.
"""

import pathlib
from dataclasses import dataclass
from typing import List, Optional

import yaml


def get_configs_dir() -> pathlib.Path:
    """Return the directory containing task config YAML files."""
    return pathlib.Path(__file__).parent / "configs"


def list_configs() -> List[str]:
    """List available task config names (stems of *.yaml in configs dir)."""
    configs_dir = get_configs_dir()
    if not configs_dir.exists():
        return []
    return sorted(
        p.stem for p in configs_dir.iterdir()
        if p.suffix == ".yaml" and p.name != "config.yaml"
    )


@dataclass
class VisualizerConfig:
    """Configuration for the policy_doctor app."""

    task: str
    name: str
    #: YAML filename stem (e.g. ``transport_mh_jan28``). Used for clustering paths under
    #: ``influence_visualizer/configs/<stem>/clustering/``, not the human-readable ``name``.
    config_stem: str = ""
    eval_dir: Optional[str] = None
    train_dir: Optional[str] = None
    train_ckpt: str = "latest"
    exp_date: str = "default"
    image_dataset_path: Optional[str] = None
    annotation_file: Optional[str] = None
    use_mock: bool = False
    obs_key: str = "agentview_image"
    top_k: int = 10
    lazy_load_images: bool = True

    def copy(self) -> "VisualizerConfig":
        return VisualizerConfig(
            task=self.task,
            name=self.name,
            config_stem=self.config_stem,
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
        )


def load_config(config_name: str) -> VisualizerConfig:
    """Load a task config by name (stem of YAML file)."""
    configs_dir = get_configs_dir()
    path = configs_dir / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return VisualizerConfig(
        task=data.get("task", config_name),
        name=data.get("name", config_name),
        config_stem=config_name,
        eval_dir=data.get("eval_dir"),
        train_dir=data.get("train_dir"),
        train_ckpt=data.get("train_ckpt", "latest"),
        exp_date=data.get("exp_date", "default"),
        image_dataset_path=data.get("image_dataset_path"),
        annotation_file=data.get("annotation_file"),
        use_mock=data.get("use_mock", False),
        obs_key=data.get("obs_key", "agentview_image"),
        top_k=data.get("top_k", 10),
        lazy_load_images=data.get("lazy_load_images", True),
    )
