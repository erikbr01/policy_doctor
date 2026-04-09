#!/usr/bin/env python
"""Verify that the influence visualizer's data loading matches TRAK exactly.

This script checks:
1. Sample counts match between visualizer and TRAK
2. Episode boundaries are correct
3. Sample-to-frame mapping is correct by comparing actual data

Usage:
    python influence_visualizer/tests/verify_trak_alignment.py \
        --eval_dir data/outputs/eval_save_episodes/jan17/jan16_train_diffusion_unet_lowdim_lift_mh_0/latest \
        --train_dir data/outputs/train/jan16/jan16_train_diffusion_unet_lowdim_lift_mh_0 \
        --train_ckpt "epoch=0100-test_mean_score=1.000"
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import dill
import hydra
import torch
from torch.utils.data import DataLoader

from diffusion_policy.common.trak_util import get_dataset_metadata
from influence_visualizer.data_loader import (
    load_demo_episodes_from_checkpoint,
    load_influence_matrix,
    load_rollout_episodes,
)


def load_trak_dataset(checkpoint_path: pathlib.Path, include_holdout: bool = False):
    """Load dataset exactly as train_trak_diffusion.py does."""
    payload = torch.load(open(str(checkpoint_path), "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    train_set = hydra.utils.instantiate(cfg.task.dataset)
    train_set_size = len(train_set)

    holdout_set = None
    holdout_set_size = 0
    if include_holdout:
        holdout_set = train_set.get_holdout_dataset()
        holdout_set_size = len(holdout_set)

    return train_set, holdout_set, train_set_size, holdout_set_size, cfg


def verify_sample_counts(
    eval_dir: pathlib.Path,
    checkpoint_path: pathlib.Path,
    include_holdout: bool,
    exp_date: str = "default",
):
    """Verify sample counts match between visualizer and TRAK."""
    print("=" * 60)
    print("1. Verifying sample counts")
    print("=" * 60)

    # Load TRAK metadata
    influence_matrix, trak_train_size, trak_test_size = load_influence_matrix(
        eval_dir=eval_dir,
        exp_date=exp_date,
    )
    print(f"TRAK train set size: {trak_train_size}")
    print(f"TRAK test set size: {trak_test_size}")

    # Load dataset as TRAK does
    train_set, holdout_set, train_size, holdout_size, cfg = load_trak_dataset(
        checkpoint_path, include_holdout
    )
    total_size = train_size + holdout_size
    print(f"Dataset train size: {train_size}")
    print(f"Dataset holdout size: {holdout_size}")
    print(f"Dataset total size: {total_size}")

    # Check match
    if total_size == trak_train_size:
        print("✓ Sample counts MATCH")
        return True, cfg
    else:
        print(f"✗ Sample counts MISMATCH: {total_size} vs {trak_train_size}")
        return False, cfg


def verify_episode_boundaries(
    checkpoint_path: pathlib.Path,
    demo_video_dir: pathlib.Path,
    include_holdout: bool,
):
    """Verify episode boundaries are computed correctly."""
    print("\n" + "=" * 60)
    print("2. Verifying episode boundaries")
    print("=" * 60)

    # Load via visualizer
    (
        episodes,
        sample_to_episode,
        sample_to_timestep,
        vis_train_size,
        vis_holdout_size,
        horizon,
    ) = load_demo_episodes_from_checkpoint(
        checkpoint_path=checkpoint_path,
        demo_video_dir=demo_video_dir,
        include_holdout=include_holdout,
    )

    # Load via TRAK method
    train_set, holdout_set, _, _, cfg = load_trak_dataset(
        checkpoint_path, include_holdout
    )
    trak_metadata = get_dataset_metadata(cfg, train_set)

    print(f"Visualizer: {len(episodes)} episodes, {len(sample_to_episode)} samples")
    print(
        f"TRAK metadata: {trak_metadata['num_eps']} episodes, {trak_metadata['num_samples']} samples"
    )

    # Compare episode lengths
    vis_ep_lens = [ep.num_samples for ep in episodes]
    trak_ep_lens = list(trak_metadata["ep_lens"])

    if include_holdout and holdout_set is not None and len(holdout_set) > 0:
        holdout_metadata = get_dataset_metadata(cfg, holdout_set)
        trak_ep_lens.extend(list(holdout_metadata["ep_lens"]))

    if vis_ep_lens == trak_ep_lens:
        print("✓ Episode lengths MATCH")
        return True
    else:
        print("✗ Episode lengths MISMATCH")
        print(f"  First 5 visualizer: {vis_ep_lens[:5]}")
        print(f"  First 5 TRAK: {trak_ep_lens[:5]}")
        # Find first mismatch
        for i, (v, t) in enumerate(zip(vis_ep_lens, trak_ep_lens)):
            if v != t:
                print(f"  First mismatch at episode {i}: vis={v}, trak={t}")
                break
        return False


def verify_sample_data(
    checkpoint_path: pathlib.Path,
    demo_video_dir: pathlib.Path,
    include_holdout: bool,
    num_samples_to_check: int = 100,
):
    """Verify that sample indices map to the same data."""
    print("\n" + "=" * 60)
    print("3. Verifying sample data alignment")
    print("=" * 60)

    # Load via visualizer
    (
        episodes,
        sample_to_episode,
        sample_to_timestep,
        vis_train_size,
        vis_holdout_size,
        horizon,
    ) = load_demo_episodes_from_checkpoint(
        checkpoint_path=checkpoint_path,
        demo_video_dir=demo_video_dir,
        include_holdout=include_holdout,
    )

    # Load dataset
    train_set, holdout_set, _, _, cfg = load_trak_dataset(
        checkpoint_path, include_holdout
    )

    # Check random samples
    total_samples = len(sample_to_episode)
    np.random.seed(42)
    sample_indices = np.random.choice(
        total_samples, min(num_samples_to_check, total_samples), replace=False
    )
    sample_indices = np.sort(sample_indices)

    print(f"Checking {len(sample_indices)} random samples...")

    mismatches = 0
    for idx in sample_indices:
        # Get visualizer's mapping
        vis_ep_idx = sample_to_episode[idx]
        vis_frame_idx = sample_to_timestep[idx]
        vis_orig_ep_idx = episodes[vis_ep_idx].index

        # Get data from dataset at this index
        dataset_sample = train_set[idx]

        # The dataset returns a sequence, we want to verify the frame index makes sense
        # For SequenceSampler, the action at sample idx corresponds to a specific sequence
        # The frame_idx should be the buffer_start_idx relative to episode start

        # Get the sampler indices for this sample
        sampler_indices = train_set.sampler.indices[idx]
        buffer_start, buffer_end, sample_start, sample_end = sampler_indices

        # The buffer_start is the global index in the replay buffer
        # We need to convert to episode-relative index
        episode_ends = train_set.replay_buffer.episode_ends[:]

        # Find which episode this buffer_start belongs to
        ep_start = 0
        for ep_idx, ep_end in enumerate(episode_ends):
            if buffer_start < ep_end:
                if ep_idx > 0:
                    ep_start = episode_ends[ep_idx - 1]
                break

        expected_frame_idx = buffer_start - ep_start

        if vis_frame_idx != expected_frame_idx:
            if mismatches < 5:  # Only print first 5 mismatches
                print(
                    f"  Sample {idx}: vis_frame={vis_frame_idx}, expected={expected_frame_idx}"
                )
            mismatches += 1

    if mismatches == 0:
        print(f"✓ All {len(sample_indices)} samples have correct frame indices")
        return True
    else:
        print(
            f"✗ {mismatches}/{len(sample_indices)} samples have incorrect frame indices"
        )
        return False


def verify_rollout_alignment(eval_dir: pathlib.Path):
    """Verify rollout episode loading."""
    print("\n" + "=" * 60)
    print("4. Verifying rollout episode alignment")
    print("=" * 60)

    # Load rollout episodes
    (
        rollout_episodes,
        rollout_sample_to_episode,
        rollout_sample_to_timestep,
        rollout_video_dir,
    ) = load_rollout_episodes(eval_dir)

    # Load TRAK test size
    influence_matrix, _, trak_test_size = load_influence_matrix(eval_dir)

    total_rollout_samples = sum(ep.num_samples for ep in rollout_episodes)

    print(f"Rollout episodes: {len(rollout_episodes)}")
    print(f"Rollout samples: {total_rollout_samples}")
    print(f"TRAK test size: {trak_test_size}")

    if total_rollout_samples == trak_test_size:
        print("✓ Rollout sample counts MATCH")
        return True
    else:
        print(f"✗ Rollout sample counts MISMATCH")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify TRAK alignment")
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--train_ckpt", type=str, default="latest")
    parser.add_argument("--demo_video_dir", type=str, default=None)
    parser.add_argument("--include_holdout", type=bool, default=False)
    parser.add_argument("--exp_date", type=str, default="default")
    args = parser.parse_args()

    eval_dir = pathlib.Path(args.eval_dir)
    train_dir = pathlib.Path(args.train_dir)

    # Find checkpoint
    checkpoint_dir = train_dir / "checkpoints"
    if args.train_ckpt == "latest":
        checkpoint_path = checkpoint_dir / "latest.ckpt"
    else:
        checkpoint_path = checkpoint_dir / f"{args.train_ckpt}.ckpt"

    demo_video_dir = (
        pathlib.Path(args.demo_video_dir) if args.demo_video_dir else eval_dir / "media"
    )

    print(f"Eval dir: {eval_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Demo video dir: {demo_video_dir}")
    print(f"Include holdout: {args.include_holdout}")
    print()

    # Run verifications
    results = []

    # 1. Sample counts
    ok, cfg = verify_sample_counts(
        eval_dir, checkpoint_path, args.include_holdout, args.exp_date
    )
    results.append(("Sample counts", ok))

    # 2. Episode boundaries
    ok = verify_episode_boundaries(
        checkpoint_path, demo_video_dir, args.include_holdout
    )
    results.append(("Episode boundaries", ok))

    # 3. Sample data alignment
    ok = verify_sample_data(checkpoint_path, demo_video_dir, args.include_holdout)
    results.append(("Sample data", ok))

    # 4. Rollout alignment
    ok = verify_rollout_alignment(eval_dir)
    results.append(("Rollout alignment", ok))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_ok = True
    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {name}: {status}")
        all_ok = all_ok and ok

    print()
    if all_ok:
        print("All verifications PASSED!")
        return 0
    else:
        print("Some verifications FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
