#!/usr/bin/env python
"""Test that the influence visualizer's score aggregation matches CUPID's aggregated scores.

This script verifies that when we load the raw TRAK influence matrix via the visualizer
and aggregate it using the same functions as CUPID (eval_demonstration_scores.py),
we get identical results to what CUPID stored in its pickle files.

This tests for potential indexing bugs in the visualizer's data loading pipeline.

Usage:
    python influence_visualizer/tests/test_score_aggregation.py \
        --eval_dir data/outputs/eval_save_episodes/jan17/jan16_train_diffusion_unet_lowdim_lift_mh_0/latest \
        --train_dir data/outputs/train/jan16/jan16_train_diffusion_unet_lowdim_lift_mh_0 \
        --train_ckpt latest \
        --result_date 25.03.03 \
        --exp_date default
"""

import argparse
import pathlib
import pickle
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import dill
import hydra
import torch

from diffusion_policy.common import error_util, results_util, trak_util
from diffusion_policy.dataset.episode_dataset import BatchEpisodeDataset
from influence_visualizer.data_loader import (
    InfluenceData,
    find_trak_experiment,
    load_influence_data,
    load_influence_matrix,
)

# Aggregation functions used by CUPID
AGGR_FNS: Dict[str, Callable[[np.ndarray, bool], float]] = {
    "mean_of_mean": error_util.mean_of_mean_influence,
    "mean_of_mean_success": error_util.mean_of_mean_influence_success,
    "sum_of_sum": error_util.sum_of_sum_influence,
    "sum_of_sum_success": error_util.sum_of_sum_influence_success,
    "min_of_max": error_util.min_of_max_influence,
    "max_of_min": error_util.max_of_min_influence,
}

METRICS = ["succ", "fail", "net"]


class TestResult:
    """Container for test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: List[Tuple[str, str, str]] = []  # (test_name, expected, actual)

    def record_pass(self, name: str):
        self.passed += 1
        print(f"  [PASS] {name}")

    def record_fail(self, name: str, expected: Any, actual: Any, detail: str = ""):
        self.failed += 1
        msg = f"Expected: {expected}, Got: {actual}"
        if detail:
            msg += f" ({detail})"
        self.errors.append((name, str(expected), str(actual)))
        print(f"  [FAIL] {name}: {msg}")

    def summary(self) -> str:
        total = self.passed + self.failed
        return f"\nResults: {self.passed}/{total} tests passed"


def load_cupid_scores(
    eval_dir: pathlib.Path,
    exp_name: str = "demonstration_scores",
) -> Optional[Dict[str, Any]]:
    """Load CUPID's pre-computed aggregated scores from pickle files."""
    scores_dir = eval_dir / exp_name

    if not scores_dir.exists():
        print(f"CUPID scores directory not found: {scores_dir}")
        return None

    cupid_scores = {}

    # Load online_trak_influence pickle (contains aggregated trajectory scores)
    trak_influence_path = scores_dir / "online_trak_influence.pkl"
    if trak_influence_path.exists():
        with open(trak_influence_path, "rb") as f:
            cupid_scores["online_trak_influence"] = pickle.load(f)
        print(f"Loaded CUPID online_trak_influence from {trak_influence_path}")
    else:
        print(f"CUPID online_trak_influence not found at {trak_influence_path}")

    return cupid_scores


def compute_visualizer_aggregated_scores(
    data: InfluenceData,
    train_set_metadata: Dict[str, Any],
    holdout_set_metadata: Dict[str, Any],
    test_set_metadata: Dict[str, Any],
    split: str = "train",
    return_dtype: type = np.float32,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute aggregated scores using the visualizer's loaded data.

    This replicates the logic from eval_demonstration_scores.py:online_trak_influence_routine
    but uses the influence matrix loaded via the visualizer.
    """
    # Select the appropriate demo metadata and influence matrix slice
    if split == "train":
        demo_set_metadata = train_set_metadata
        # For train split, use only the train portion of the influence matrix
        test_demo_scores = data.influence_matrix[:, : train_set_metadata["num_samples"]]
    elif split == "holdout":
        demo_set_metadata = holdout_set_metadata
        # For holdout split, use only the holdout portion
        test_demo_scores = data.influence_matrix[:, train_set_metadata["num_samples"] :]
    else:
        raise ValueError(f"Split must be 'train' or 'holdout', got {split}")

    print(f"  Influence matrix slice shape for {split}: {test_demo_scores.shape}")
    print(
        f"  Expected shape: ({test_set_metadata['num_samples']}, {demo_set_metadata['num_samples']})"
    )

    results = {}

    for aggr_fn_name, aggr_fn in AGGR_FNS.items():
        # Aggregate sample-level scores to trajectory-level
        test_demo_traj_scores = error_util.pairwise_sample_to_trajectory_scores(
            pairwise_sample_scores=test_demo_scores,
            num_test_eps=test_set_metadata["num_eps"],
            num_train_eps=demo_set_metadata["num_eps"],
            test_ep_idxs=test_set_metadata["ep_idxs"],
            train_ep_idxs=demo_set_metadata["ep_idxs"],
            test_ep_lens=test_set_metadata["ep_lens"],
            train_ep_lens=demo_set_metadata["ep_lens"],
            success_mask=test_set_metadata["success_mask"],
            aggr_fn=aggr_fn,
            return_dtype=return_dtype,
        )

        for metric in METRICS:
            # Compute demo quality scores (same as CUPID)
            # CUPID uses "all" rollouts by default
            rollout_idx = np.arange(test_set_metadata["num_eps"])

            demo_quality_scores = error_util.compute_demo_quality_scores(
                traj_scores=test_demo_traj_scores[rollout_idx, :],
                success_mask=test_set_metadata["success_mask"][rollout_idx],
                metric=metric,
            )

            exp_key = results_util.get_online_trak_influence_exp_key(
                aggr_fn=aggr_fn_name,
                metric=metric,
                num_rollouts="all",
            )
            results[exp_key] = demo_quality_scores

    return results


def compare_scores(
    visualizer_scores: Dict[str, np.ndarray],
    cupid_scores: Dict[str, np.ndarray],
    result: TestResult,
    split: str,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Compare visualizer and CUPID scores."""
    print(f"\n  Comparing {split} split scores:")

    # Find common keys
    vis_keys = set(visualizer_scores.keys())
    cupid_keys = set(cupid_scores.keys())
    common_keys = vis_keys & cupid_keys

    if not common_keys:
        print(f"    No common keys found!")
        print(f"    Visualizer keys: {sorted(vis_keys)[:5]}...")
        print(f"    CUPID keys: {sorted(cupid_keys)[:5]}...")
        result.record_fail(
            f"{split}_common_keys",
            "at least 1 common key",
            "0 common keys",
        )
        return

    print(f"    Found {len(common_keys)} common experiment keys")

    for key in sorted(common_keys):
        vis_score = visualizer_scores[key]
        cupid_score = cupid_scores[key]

        test_name = f"{split}_{key}"

        # Handle None values
        if vis_score is None and cupid_score is None:
            result.record_pass(test_name)
            continue
        elif vis_score is None or cupid_score is None:
            result.record_fail(
                test_name,
                f"both None or both array",
                f"vis={type(vis_score)}, cupid={type(cupid_score)}",
            )
            continue

        # Check shapes match
        if vis_score.shape != cupid_score.shape:
            result.record_fail(
                f"{test_name}_shape",
                cupid_score.shape,
                vis_score.shape,
            )
            continue

        # Check values match
        if np.allclose(vis_score, cupid_score, rtol=rtol, atol=atol):
            result.record_pass(test_name)
        else:
            # Find the maximum difference
            max_diff = np.max(np.abs(vis_score - cupid_score))
            mean_diff = np.mean(np.abs(vis_score - cupid_score))

            # Check if it's an indexing issue (values shuffled)
            vis_sorted = np.sort(vis_score)
            cupid_sorted = np.sort(cupid_score)
            sorted_match = np.allclose(vis_sorted, cupid_sorted, rtol=rtol, atol=atol)

            detail = f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
            if sorted_match:
                detail += " [VALUES MATCH WHEN SORTED - POSSIBLE INDEXING BUG]"

            result.record_fail(
                test_name,
                f"arrays equal (rtol={rtol})",
                f"arrays differ",
                detail,
            )

            # Print first few differences for debugging
            diff_mask = ~np.isclose(vis_score, cupid_score, rtol=rtol, atol=atol)
            diff_indices = np.where(diff_mask)[0][:5]
            if len(diff_indices) > 0:
                print(f"      First differing indices: {diff_indices}")
                for idx in diff_indices:
                    print(
                        f"        idx={idx}: vis={vis_score[idx]:.6f}, cupid={cupid_score[idx]:.6f}"
                    )


def verify_metadata_alignment(
    vis_train_metadata: Dict[str, Any],
    vis_holdout_metadata: Dict[str, Any],
    vis_test_metadata: Dict[str, Any],
    cupid_train_metadata: Dict[str, Any],
    cupid_holdout_metadata: Dict[str, Any],
    cupid_test_metadata: Dict[str, Any],
    result: TestResult,
) -> bool:
    """Verify that metadata from visualizer and CUPID match."""
    print("\n  Verifying metadata alignment:")

    all_match = True

    # Check train metadata
    for key in ["num_eps", "num_samples"]:
        vis_val = vis_train_metadata.get(key)
        cupid_val = cupid_train_metadata.get(key)
        if vis_val == cupid_val:
            result.record_pass(f"train_{key}")
        else:
            result.record_fail(f"train_{key}", cupid_val, vis_val)
            all_match = False

    # Check holdout metadata
    for key in ["num_eps", "num_samples"]:
        vis_val = vis_holdout_metadata.get(key)
        cupid_val = cupid_holdout_metadata.get(key)
        if vis_val == cupid_val:
            result.record_pass(f"holdout_{key}")
        else:
            result.record_fail(f"holdout_{key}", cupid_val, vis_val)
            all_match = False

    # Check test metadata
    for key in ["num_eps", "num_samples"]:
        vis_val = vis_test_metadata.get(key)
        cupid_val = cupid_test_metadata.get(key)
        if vis_val == cupid_val:
            result.record_pass(f"test_{key}")
        else:
            result.record_fail(f"test_{key}", cupid_val, vis_val)
            all_match = False

    # Check episode lengths match
    if np.array_equal(vis_train_metadata["ep_lens"], cupid_train_metadata["ep_lens"]):
        result.record_pass("train_ep_lens")
    else:
        result.record_fail(
            "train_ep_lens",
            f"array of {len(cupid_train_metadata['ep_lens'])} lens",
            f"array of {len(vis_train_metadata['ep_lens'])} lens",
        )
        all_match = False

    # Check success mask for test set
    if "success_mask" in vis_test_metadata and "success_mask" in cupid_test_metadata:
        if np.array_equal(
            vis_test_metadata["success_mask"], cupid_test_metadata["success_mask"]
        ):
            result.record_pass("test_success_mask")
        else:
            result.record_fail(
                "test_success_mask",
                f"mask with {cupid_test_metadata['success_mask'].sum()} successes",
                f"mask with {vis_test_metadata['success_mask'].sum()} successes",
            )
            all_match = False

    return all_match


def build_metadata_from_visualizer(
    data: InfluenceData,
    cfg: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Build metadata dictionaries from visualizer's loaded data.

    This replicates the metadata structure used by CUPID.
    """
    from diffusion_policy.dataset.episode_dataset_utils import ep_lens_to_idxs

    # Train set metadata
    train_ep_lens = np.array([ep.num_samples for ep in data.demo_episodes])
    train_metadata = {
        "num_eps": len(data.demo_episodes),
        "num_samples": len(data.demo_sample_infos),
        "ep_lens": train_ep_lens,
        "ep_idxs": ep_lens_to_idxs(train_ep_lens),
    }

    # Holdout set metadata
    holdout_ep_lens = np.array([ep.num_samples for ep in data.holdout_episodes])
    holdout_metadata = {
        "num_eps": len(data.holdout_episodes),
        "num_samples": len(data.holdout_sample_infos),
        "ep_lens": holdout_ep_lens,
        "ep_idxs": ep_lens_to_idxs(holdout_ep_lens) if len(holdout_ep_lens) > 0 else [],
    }

    # Test set metadata (rollouts)
    test_ep_lens = np.array([ep.num_samples for ep in data.rollout_episodes])
    test_metadata = {
        "num_eps": len(data.rollout_episodes),
        "num_samples": len(data.rollout_sample_infos),
        "ep_lens": test_ep_lens,
        "ep_idxs": ep_lens_to_idxs(test_ep_lens),
        "success_mask": np.array(
            [ep.success for ep in data.rollout_episodes], dtype=bool
        ),
    }

    return train_metadata, holdout_metadata, test_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Compare visualizer aggregated scores with CUPID scores"
    )
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--train_ckpt", type=str, default="latest")
    parser.add_argument("--exp_date", type=str, default="default")
    parser.add_argument("--result_date", type=str, default="25.03.03")
    parser.add_argument("--scores_exp_name", type=str, default="demonstration_scores")
    parser.add_argument("--include_holdout", type=bool, default=True)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    eval_dir = pathlib.Path(args.eval_dir)
    train_dir = pathlib.Path(args.train_dir)

    print("=" * 70)
    print("Score Aggregation Alignment Test")
    print("=" * 70)
    print(f"Eval dir: {eval_dir}")
    print(f"Train dir: {train_dir}")
    print(f"Checkpoint: {args.train_ckpt}")
    print(f"TRAK exp_date: {args.exp_date}")
    print(f"Results exp_date: {args.result_date}")
    print()

    result = TestResult()

    # =========================================================================
    # Step 1: Load CUPID's pre-computed scores
    # =========================================================================
    print("Step 1: Loading CUPID's pre-computed aggregated scores...")
    cupid_scores = load_cupid_scores(eval_dir, args.scores_exp_name)

    if cupid_scores is None or "online_trak_influence" not in cupid_scores:
        print("ERROR: Could not load CUPID scores. Cannot run comparison.")
        print("Make sure eval_demonstration_scores.py has been run with:")
        print("  --eval_online_trak_influence True")
        return 1

    # =========================================================================
    # Step 2: Load influence data via visualizer
    # =========================================================================
    print("\nStep 2: Loading influence data via visualizer...")
    data = load_influence_data(
        eval_dir=str(eval_dir),
        train_dir=str(train_dir),
        train_ckpt=args.train_ckpt,
        exp_date=args.exp_date,
        include_holdout=args.include_holdout,
    )

    print(f"  Influence matrix shape: {data.influence_matrix.shape}")
    print(f"  Demo episodes (train): {len(data.demo_episodes)}")
    print(f"  Demo samples (train): {len(data.demo_sample_infos)}")
    print(f"  Holdout episodes: {len(data.holdout_episodes)}")
    print(f"  Holdout samples: {len(data.holdout_sample_infos)}")
    print(f"  Rollout episodes: {len(data.rollout_episodes)}")
    print(f"  Rollout samples: {len(data.rollout_sample_infos)}")

    # =========================================================================
    # Step 3: Load CUPID's metadata for comparison
    # =========================================================================
    print("\nStep 3: Loading CUPID metadata for comparison...")

    # Find checkpoint
    checkpoint_dir = train_dir / "checkpoints"
    if args.train_ckpt == "latest":
        checkpoint_path = checkpoint_dir / "latest.ckpt"
    else:
        checkpoint_path = checkpoint_dir / f"{args.train_ckpt}.ckpt"

    # Load policy and config (same as CUPID does)
    policy, cfg = trak_util.get_policy_from_checkpoint(checkpoint_path, device="cpu")

    # Load datasets and metadata (same as CUPID does)
    train_set = hydra.utils.instantiate(cfg.task.dataset)
    cupid_train_metadata = trak_util.get_dataset_metadata(cfg, train_set)

    holdout_set = train_set.get_holdout_dataset()
    cupid_holdout_metadata = trak_util.get_dataset_metadata(cfg, holdout_set)

    test_set = BatchEpisodeDataset(
        batch_size=1,
        dataset_path=eval_dir / "episodes",
        exec_horizon=1,
        sample_history=0,
    )
    cupid_test_metadata = trak_util.get_dataset_metadata(cfg, test_set)

    print(
        f"  CUPID train: {cupid_train_metadata['num_eps']} eps, {cupid_train_metadata['num_samples']} samples"
    )
    print(
        f"  CUPID holdout: {cupid_holdout_metadata['num_eps']} eps, {cupid_holdout_metadata['num_samples']} samples"
    )
    print(
        f"  CUPID test: {cupid_test_metadata['num_eps']} eps, {cupid_test_metadata['num_samples']} samples"
    )

    # =========================================================================
    # Step 4: Build metadata from visualizer
    # =========================================================================
    print("\nStep 4: Building metadata from visualizer's loaded data...")
    vis_train_metadata, vis_holdout_metadata, vis_test_metadata = (
        build_metadata_from_visualizer(data, cfg)
    )

    print(
        f"  Visualizer train: {vis_train_metadata['num_eps']} eps, {vis_train_metadata['num_samples']} samples"
    )
    print(
        f"  Visualizer holdout: {vis_holdout_metadata['num_eps']} eps, {vis_holdout_metadata['num_samples']} samples"
    )
    print(
        f"  Visualizer test: {vis_test_metadata['num_eps']} eps, {vis_test_metadata['num_samples']} samples"
    )

    # =========================================================================
    # Step 5: Verify metadata alignment
    # =========================================================================
    print("\nStep 5: Verifying metadata alignment...")
    metadata_ok = verify_metadata_alignment(
        vis_train_metadata,
        vis_holdout_metadata,
        vis_test_metadata,
        cupid_train_metadata,
        cupid_holdout_metadata,
        cupid_test_metadata,
        result,
    )

    if not metadata_ok:
        print("\nWARNING: Metadata mismatch detected! Score comparison may be invalid.")

    # =========================================================================
    # Step 6: Compute aggregated scores using visualizer's data
    # =========================================================================
    print("\nStep 6: Computing aggregated scores using visualizer's data...")

    # Use CUPID metadata to ensure consistency
    print("  Computing train split scores...")
    vis_train_scores = compute_visualizer_aggregated_scores(
        data=data,
        train_set_metadata=cupid_train_metadata,
        holdout_set_metadata=cupid_holdout_metadata,
        test_set_metadata=cupid_test_metadata,
        split="train",
    )

    vis_holdout_scores = {}
    if cupid_holdout_metadata["num_samples"] > 0 and args.include_holdout:
        print("  Computing holdout split scores...")
        vis_holdout_scores = compute_visualizer_aggregated_scores(
            data=data,
            train_set_metadata=cupid_train_metadata,
            holdout_set_metadata=cupid_holdout_metadata,
            test_set_metadata=cupid_test_metadata,
            split="holdout",
        )

    # =========================================================================
    # Step 7: Compare scores
    # =========================================================================
    print("\nStep 7: Comparing aggregated scores...")

    cupid_trak = cupid_scores["online_trak_influence"]

    # Compare train scores
    if "train" in cupid_trak:
        compare_scores(
            vis_train_scores,
            cupid_trak["train"],
            result,
            split="train",
            rtol=args.rtol,
            atol=args.atol,
        )
    else:
        print("  CUPID train scores not found, skipping train comparison")

    # Compare holdout scores
    if "holdout" in cupid_trak and vis_holdout_scores:
        compare_scores(
            vis_holdout_scores,
            cupid_trak["holdout"],
            result,
            split="holdout",
            rtol=args.rtol,
            atol=args.atol,
        )
    else:
        print("  Holdout scores not available, skipping holdout comparison")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(result.summary())

    if result.failed > 0:
        print("\nFailed tests:")
        for name, expected, actual in result.errors:
            print(f"  - {name}")
        print("\nPossible causes of failure:")
        print("  1. Indexing bug in visualizer's data loading")
        print(
            "  2. Different seed/mask used during TRAK featurization vs visualization"
        )
        print("  3. Episode order mismatch between datasets")
        print("  4. Numeric precision differences")
        return 1

    print("\nAll tests passed! Visualizer aggregation matches CUPID.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
