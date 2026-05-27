"""Build the zarr cache for a merged DROID HDF5 without touching the GPU.

`RobomimicReplayImageDataset.__init__` triggers cache construction the first
time it sees a `.hdf5` whose `.zarr.zip` cache does not yet exist. Running
the full `train.py` would also build the cache, but it loads the policy
onto the GPU and starts compile/train work we don't want at that point.

Use this script to pre-warm caches for other arms in parallel while a
training run is going on a different GPU.

Usage (must run in the `mimicgen_torch2` env — the same env as training):
    conda run -n mimicgen_torch2 python scripts/build_droid_zarr_cache.py \
        /home/erbauer/data/droid_data/kendama_may13_may22_one_grasp_away.hdf5
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = (
    PROJECT_ROOT / "third_party" / "cupid" / "configs" / "image" / "droid"
    / "diffusion_policy_cnn"
)


def main(dataset_path: str) -> None:
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None)
    cfg = compose(
        config_name="config.yaml",
        overrides=[f"++task.dataset.dataset_path={dataset_path}"],
    )
    OmegaConf.resolve(cfg)
    print(f"[build_droid_zarr_cache] dataset_path={dataset_path}")
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    print(f"[build_droid_zarr_cache] done. len(dataset)={len(dataset)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
