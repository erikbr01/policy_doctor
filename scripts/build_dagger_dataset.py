#!/usr/bin/env python
"""Convert DAgger pkl episodes to robomimic HDF5 format for retraining.

Reads collected DAgger episodes (pkl files with per-key obs + acting_agent labels)
and writes to robomimic HDF5 format compatible with RobocasaReplayLowdimDataset.

Two modes:
  1. Extract only human-controlled segments (standard HG-DAgger)
  2. Include full episodes (for offline evaluation)

Usage:
    python scripts/build_dagger_dataset.py \
      --episodes_dir /path/to/dagger/episodes \
      --output_hdf5 /path/to/output.hdf5 \
      --filter_human_only
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def pkl_to_hdf5_demo(
    pkl_path: Path,
    output_group: h5py.Group,
    demo_idx: int,
    obs_keys: list[str],
    filter_human_only: bool = False,
) -> int:
    """Convert one pkl episode to one or more HDF5 demos.

    Parameters
    ----------
    pkl_path : Path
        Path to episode pkl file.
    output_group : h5py.Group
        Parent group in output HDF5 file (data/).
    demo_idx : int
        Starting demo index for naming (demo_N).
    obs_keys : list[str]
        Observation keys to store.
    filter_human_only : bool
        If True, extract only contiguous human-controlled segments.
        If False, store the full episode.

    Returns
    -------
    n_demos : int
        Number of demos written.
    """
    df = pd.read_pickle(pkl_path)

    if len(df) == 0:
        return 0

    n_demos = 0

    if filter_human_only:
        # Extract contiguous human segments
        acting_agents = df["acting_agent"].values
        segments = []
        segment_start = None

        for i, agent in enumerate(acting_agents):
            if agent == "human":
                if segment_start is None:
                    segment_start = i
            else:
                if segment_start is not None:
                    segments.append((segment_start, i))
                    segment_start = None

        if segment_start is not None:
            segments.append((segment_start, len(acting_agents)))

        # Write each segment as a demo
        for seg_start, seg_end in segments:
            seg_df = df.iloc[seg_start:seg_end]
            _write_demo_to_hdf5(
                output_group,
                seg_df,
                demo_idx,
                obs_keys,
            )
            demo_idx += 1
            n_demos += 1

    else:
        # Write entire episode as one demo
        _write_demo_to_hdf5(
            output_group,
            df,
            demo_idx,
            obs_keys,
        )
        n_demos = 1

    return n_demos


def _write_demo_to_hdf5(
    parent_group: h5py.Group,
    df: pd.DataFrame,
    demo_idx: int,
    obs_keys: list[str],
) -> None:
    """Write a DataFrame segment as demo_N group in HDF5."""
    demo_name = f"demo_{demo_idx}"
    demo_group = parent_group.create_group(demo_name)

    n_steps = len(df)

    # Extract and concatenate obs for each key
    obs_group = demo_group.create_group("obs")
    for key in obs_keys:
        obs_array = np.array([row[key] for row in df["obs"]])
        obs_group.create_dataset(key, data=obs_array, compression="gzip")

    # Actions
    actions = np.array([row for row in df["action"]])
    demo_group.create_dataset("actions", data=actions, compression="gzip")

    # Rewards and dones
    rewards = df["reward"].values.astype(np.float32)
    demo_group.create_dataset("rewards", data=rewards, compression="gzip")

    dones = df["done"].values.astype(bool)
    demo_group.create_dataset("dones", data=dones, compression="gzip")

    # Sim states (for deterministic replay)
    if "sim_state" in df.columns and df["sim_state"].iloc[0] is not None:
        states = np.array([row for row in df["sim_state"]])
        demo_group.create_dataset("states", data=states, compression="gzip")

    # Acting agent labels (custom extension to robomimic format)
    if "acting_agent" in df.columns:
        acting_agents = df["acting_agent"].values
        # Encode as: 0 = robot, 1 = human
        agent_codes = np.array([1 if agent == "human" else 0 for agent in acting_agents])
        demo_group.create_dataset("acting_agent", data=agent_codes, compression="gzip")


@click.command()
@click.option(
    "--episodes_dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing DAgger episode pkl files (ep*.pkl)",
)
@click.option(
    "--output_hdf5",
    required=True,
    type=click.Path(),
    help="Output robomimic HDF5 file path",
)
@click.option(
    "--filter_human_only",
    is_flag=True,
    help="Extract only human-controlled segments (standard HG-DAgger)",
)
@click.option(
    "--obs_keys",
    default="object,robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos",
    type=str,
    help="Comma-separated list of observation keys",
)
def main(
    episodes_dir: str,
    output_hdf5: str,
    filter_human_only: bool,
    obs_keys: str,
) -> None:
    """Convert DAgger episodes to robomimic HDF5 format."""

    episodes_dir = Path(episodes_dir)
    output_hdf5 = Path(output_hdf5)
    obs_keys = [k.strip() for k in obs_keys.split(",")]

    # Find all pkl files
    pkl_files = sorted(episodes_dir.glob("ep*.pkl"))
    if not pkl_files:
        print(f"No episode pkl files found in {episodes_dir}")
        return

    print(f"Found {len(pkl_files)} episodes")

    # Create output HDF5
    output_hdf5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(output_hdf5), "w") as f:
        data_group = f.create_group("data")

        demo_idx = 0
        total_demos = 0

        for pkl_path in tqdm(pkl_files, desc="Converting to HDF5"):
            n_demos = pkl_to_hdf5_demo(
                pkl_path,
                data_group,
                demo_idx,
                obs_keys,
                filter_human_only=filter_human_only,
            )
            demo_idx += n_demos
            total_demos += n_demos

        # Store metadata
        f.attrs["total"] = total_demos
        f.attrs["n_demos"] = total_demos
        f.attrs["filter_human_only"] = filter_human_only

    print(f"\nWrote {total_demos} demos to {output_hdf5}")
    print(f"\nTo use for training, merge with original dataset:")
    print(f"  python scripts/convert_rollout_to_robomimic_hdf5.py \\")
    print(f"    --input1 data/robocasa/datasets/kitchen_lowdim_merged.hdf5 \\")
    print(f"    --input2 {output_hdf5} \\")
    print(f"    --output data/robocasa/datasets/kitchen_lowdim_merged_dagger.hdf5")


if __name__ == "__main__":
    main()
