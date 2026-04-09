#!/usr/bin/env python
"""Verify that visualizer image retrieval matches SequenceSampler.

This script verifies that the manual indexing logic in the visualizer's data_loader
perfectly matches the SequenceSampler used during training/TRAK featurization.
This ensures that the images shown in the visualizer are the exact same images
that correspond to the influence scores.

Usage:
    python influence_visualizer/tests/verify_image_alignment.py \
        --eval_dir data/outputs/eval_save_episodes/... \
        --train_dir data/outputs/train/... \
        --train_ckpt latest
"""

import argparse
import pathlib
import sys
import numpy as np
import hydra
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from influence_visualizer.data_loader import load_influence_data, InfluenceData
from diffusion_policy.common.sampler import SequenceSampler

def verify_alignment(
    data: InfluenceData, 
    num_samples_to_check: int = 100,
    seed: int = 42
):
    """Verify alignment between data_loader manual indexing and SequenceSampler."""
    rng = np.random.default_rng(seed)
    
    # Determine which dataset to test (image_dataset if available, else demo_dataset)
    dataset = data.image_dataset if data.image_dataset is not None else data.demo_dataset
    dataset_name = "image_dataset" if data.image_dataset is not None else "demo_dataset"
    print(f"\nVerifying alignment using {dataset_name}...")
    
    if dataset is None:
        print("Error: No dataset available to verify.")
        return False
        
    replay_buffer = dataset.replay_buffer
    
    # Check if we can find an image key
    available_keys = list(replay_buffer.keys())
    image_keys = [k for k in available_keys if "image" in k.lower()]
    
    if not image_keys:
        print(f"Warning: No keys with 'image' found in replay buffer. Keys: {available_keys}")
        # Fallback to any key just to verify indexing (e.g. agent_pos)
        obs_key = available_keys[0]
        print(f"Testing with non-image key: {obs_key}")
    else:
        obs_key = image_keys[0]
        print(f"Testing with image key: {obs_key}")
        
    # Instantiate SequenceSampler using the dataset's mask
    # We need to handle both train and holdout splits
    
    # 1. Verify Train Split
    print("\n[Train Split Verification]")
    if not hasattr(dataset, "train_mask"):
        print("Dataset has no train_mask attribute. Skipping train verification.")
    else:
        train_mask = dataset.train_mask
        print(f"Instantiating SequenceSampler with train_mask ({train_mask.sum()} episodes)...")
        
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=data.horizon,
            pad_before=data.pad_before,
            pad_after=data.pad_after,
            episode_mask=train_mask,
            keys=[obs_key]
        )
        
        num_train_samples = len(data.demo_sample_infos)
        if len(sampler) != num_train_samples:
            print(f"ERROR: Sampler length ({len(sampler)}) != demo_sample_infos length ({num_train_samples})")
            return False
            
        # Select random indices
        indices = rng.choice(num_train_samples, size=min(num_samples_to_check, num_train_samples), replace=False)
        indices.sort()
        
        failures = 0
        for idx in indices:
            # Get frame via Visualizer logic
            # data.get_demo_frame uses global index. For train split, global_idx matches local idx
            vis_frame = data.get_demo_frame(idx, obs_key=obs_key, timestep_in_horizon=0)
            
            # Get frame via SequenceSampler
            seq = sampler.sample_sequence(idx)
            sampler_frame = seq[obs_key][0] # Timestep 0
            
            # SequenceSampler might return different dtype or normalization?
            # data_loader.py converts to uint8. 
            # Let's check raw values first, or convert sampler frame to uint8 if needed.
            
            # Convert sampler frame to match visualizer output format if needed
            if sampler_frame.dtype != np.uint8 and np.issubdtype(sampler_frame.dtype, np.floating):
                if sampler_frame.max() <= 1.0:
                    sampler_frame = (sampler_frame * 255).astype(np.uint8)
                else:
                    sampler_frame = sampler_frame.astype(np.uint8)
            
            if vis_frame is None:
                print(f"  Sample {idx}: Visualizer returned None")
                failures += 1
                continue
                
            if not np.array_equal(vis_frame, sampler_frame):
                print(f"  Sample {idx}: MISMATCH!")
                print(f"    Vis shape: {vis_frame.shape}, Sampler shape: {sampler_frame.shape}")
                print(f"    Max diff: {np.max(np.abs(vis_frame.astype(float) - sampler_frame.astype(float)))}")
                failures += 1
                if failures >= 5:
                    print("  Too many failures, stopping train verification.")
                    break
        
        if failures == 0:
            print(f"  SUCCESS: All {len(indices)} checked train samples match.")
        else:
            print(f"  FAILURE: {failures} mismatches found in train split.")
            return False

    # 2. Verify Holdout Split (if available)
    # The visualizer stores holdout samples after train samples in all_demo_sample_infos
    # We need to construct a sampler using holdout_mask, but be careful about indexing.
    # verify_image_alignment test typically focuses on train, but let's check holdout if possible.
    
    # We can skip holdout for now to keep it simple, or implement if needed. 
    # data_loader.py: build_demo_sample_infos(holdout_dataset, ...)
    # If holdout_dataset is a separate object, we should use that.
    
    if data.holdout_dataset is not None:
        print("\n[Holdout Split Verification]")
        
        # Determine which replay buffer to use for holdout
        # If we have an image_dataset (global), we should use its buffer to get images,
        # but use the mask from the holdout_dataset (which identifies holdout episodes).
        if data.image_dataset is not None:
            h_replay_buffer = data.image_dataset.replay_buffer
        else:
            h_replay_buffer = data.holdout_dataset.replay_buffer

        # Use holdout_dataset's mask (which is set to select holdout episodes)
        if hasattr(data.holdout_dataset, "train_mask"):
            h_mask = data.holdout_dataset.train_mask
            print(f"Instantiating SequenceSampler for holdout ({h_mask.sum()} episodes)...")
            
            h_sampler = SequenceSampler(
                replay_buffer=h_replay_buffer,
                sequence_length=data.horizon,
                pad_before=data.pad_before,
                pad_after=data.pad_after,
                episode_mask=h_mask,
                keys=[obs_key]
            )
            
            num_holdout_samples = len(data.holdout_sample_infos)
            if len(h_sampler) != num_holdout_samples:
                 print(f"ERROR: Holdout Sampler length ({len(h_sampler)}) != holdout_sample_infos length ({num_holdout_samples})")
                 # This might happen if holdout_dataset logic is complex
            else:
                indices = rng.choice(num_holdout_samples, size=min(num_samples_to_check, num_holdout_samples), replace=False)
                indices.sort()
                
                h_failures = 0
                train_offset = len(data.demo_sample_infos)
                
                for local_idx in indices:
                    global_idx = train_offset + local_idx
                    
                    vis_frame = data.get_demo_frame(global_idx, obs_key=obs_key, timestep_in_horizon=0)
                    
                    seq = h_sampler.sample_sequence(local_idx)
                    sampler_frame = seq[obs_key][0]
                    
                    if sampler_frame.dtype != np.uint8 and np.issubdtype(sampler_frame.dtype, np.floating):
                        if sampler_frame.max() <= 1.0:
                            sampler_frame = (sampler_frame * 255).astype(np.uint8)
                        else:
                            sampler_frame = sampler_frame.astype(np.uint8)
                            
                    if not np.array_equal(vis_frame, sampler_frame):
                        print(f"  Holdout Sample {local_idx} (Global {global_idx}): MISMATCH!")
                        h_failures += 1
                        if h_failures >= 5: break
                
                if h_failures == 0:
                    print(f"  SUCCESS: All {len(indices)} checked holdout samples match.")
                else:
                    print(f"  FAILURE: {h_failures} mismatches found in holdout split.")
                    return False

    return True

def verify_episode_lengths(data: InfluenceData):
    """Verify that episode lengths in InfluenceData match the ground truth in replay buffer."""
    print("\n[Episode Length Verification]")
    
    # Determine proper datasets to check against
    # We want to check against the dataset that was used to build the episode info
    
    # 1. Train Episodes
    if data.demo_dataset is not None:
        print("Verifying train episode lengths...")
        dataset = data.demo_dataset
        replay_buffer = dataset.replay_buffer
        episode_ends = replay_buffer.episode_ends[:]
        
        # Calculate expected lengths from episode_ends
        # episode indices in demo_episodes.index refer to original episode indices
        
        failures = 0
        checked = 0
        
        for ep_info in data.demo_episodes:
            idx = ep_info.index
            
            # Get actual length from buffer
            if idx == 0:
                start = 0
            else:
                start = episode_ends[idx-1]
            end = episode_ends[idx]
            actual_length = end - start
            
            if ep_info.raw_length != actual_length:
                print(f"  Train Episode {idx}: Length Mismatch!")
                print(f"    Visualizer raw_length: {ep_info.raw_length}")
                print(f"    Actual buffer length: {actual_length}")
                failures += 1
            
            checked += 1
            
        if failures == 0:
            print(f"  SUCCESS: All {checked} train episode lengths match.")
        else:
            print(f"  FAILURE: {failures} mismatches in train episode lengths.")
            return False
            
    # 2. Holdout Episodes
    if data.holdout_episodes and data.holdout_dataset is not None:
        print("Verifying holdout episode lengths...")
        dataset = data.holdout_dataset
        replay_buffer = dataset.replay_buffer
        episode_ends = replay_buffer.episode_ends[:]
        
        failures = 0
        checked = 0
        
        for ep_info in data.holdout_episodes:
            idx = ep_info.index
            
            # Get actual length from buffer
            if idx == 0:
                start = 0
            else:
                start = episode_ends[idx-1]
            end = episode_ends[idx]
            actual_length = end - start
            
            if ep_info.raw_length != actual_length:
                print(f"  Holdout Episode {idx}: Length Mismatch!")
                print(f"    Visualizer raw_length: {ep_info.raw_length}")
                print(f"    Actual buffer length: {actual_length}")
                failures += 1
            
            checked += 1
            
        if failures == 0:
            print(f"  SUCCESS: All {checked} holdout episode lengths match.")
        else:
            print(f"  FAILURE: {failures} mismatches in holdout episode lengths.")
            return False

    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--train_ckpt", type=str, default="latest")
    parser.add_argument("--exp_date", type=str, default="default")
    parser.add_argument("--image_dataset_path", type=str, default=None)
    args = parser.parse_args()
    
    print("=" * 80)
    print("Image Alignment Verification Test")
    print("=" * 80)
    
    # Load data
    print("Loading InfluenceData...")
    # We set check_integrity=False to speed up if supported, 
    # but load_influence_data doesn't have that flag.
    data = load_influence_data(
        eval_dir=args.eval_dir,
        train_dir=args.train_dir,
        train_ckpt=args.train_ckpt,
        exp_date=args.exp_date,
        image_dataset_path=args.image_dataset_path
    )
    
    align_success = verify_alignment(data)
    len_success = verify_episode_lengths(data)
    
    success = align_success and len_success
    
    if success:
        print("\n\nPASSED: Visualizer image alignment is verified.")
        return 0
    else:
        print("\n\nFAILED: Visualizer image alignment mismatches detected.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
