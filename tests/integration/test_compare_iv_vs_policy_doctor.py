"""
Step-by-step comparison: influence_visualizer (IV) vs policy_doctor (PD).

Loads the same data and clustering, runs each pipeline stage in both codebases,
and compares outputs. Use to find where IV and PD diverge.
"""

import unittest
from pathlib import Path

import numpy as np

from policy_doctor.paths import REPO_ROOT, iv_task_configs_base

_REPO_ROOT = REPO_ROOT
_IV_CFG = iv_task_configs_base()

_CLUSTERING_DIR = (
    _IV_CFG
    / "transport_mh_jan28"
    / "clustering"
    / "sliding_window_rollout_kmeans_k15_2026_03_05"
)

# Params matching reference test_advantage_selection
GAMMA = 0.99
REWARD_SUCCESS = 1.0
REWARD_FAILURE = -1.0
REWARD_END = 0.0
ADVANTAGE_THRESHOLD = 0.1
WINDOW_WIDTH = 5
AGGREGATION_METHOD = "mean"
SELECTION_PERCENTILE = 99.0


def _require_clustering():
    if not _CLUSTERING_DIR.is_dir() or not (_CLUSTERING_DIR / "manifest.yaml").exists():
        raise unittest.SkipTest(f"Clustering folder not found: {_CLUSTERING_DIR}")


def _load_data_and_clustering():
    """Load IV data and clustering from the known folder. Returns (data, cluster_labels, metadata)."""
    try:
        from influence_visualizer.clustering_results import load_clustering_result_from_path
        from influence_visualizer.data_loader import load_influence_data
    except ImportError:
        raise unittest.SkipTest("influence_visualizer not installed; comparison tests require it")
    import yaml

    cfg_path = _IV_CFG / "transport_mh_jan28.yaml"
    if not cfg_path.exists():
        raise unittest.SkipTest(f"Task config not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    eval_dir = cfg.get("eval_dir")
    train_dir = cfg.get("train_dir")
    if not eval_dir or not train_dir:
        raise unittest.SkipTest("transport_mh_jan28 config missing eval_dir/train_dir")
    data = load_influence_data(
        eval_dir=str(_REPO_ROOT / eval_dir),
        train_dir=str(_REPO_ROOT / train_dir),
        train_ckpt=cfg.get("train_ckpt", "latest"),
        exp_date=cfg.get("exp_date", "default"),
        include_holdout=True,
        image_dataset_path=cfg.get("image_dataset_path"),
        lazy_load_images=cfg.get("lazy_load_images", True),
        quality_labels=cfg.get("quality_labels"),
    )
    cluster_labels, metadata, _ = load_clustering_result_from_path(_CLUSTERING_DIR)
    return data, cluster_labels, metadata


class TestCompareIVvsPolicyDoctor(unittest.TestCase):
    """Compare IV and PD outputs at each pipeline step."""

    @classmethod
    def setUpClass(cls):
        _require_clustering()
        cls.data, cls.cluster_labels, cls.metadata = _load_data_and_clustering()
        cls.num_train = len(cls.data.demo_sample_infos)
        cls.num_holdout = len(cls.data.holdout_sample_infos)

    def test_step1_advantages_match(self):
        """Step 1: Behavior graph + advantages — same inputs, compare advantage arrays."""
        from influence_visualizer.behavior_value_loader import get_behavior_graph_and_slice_values as iv_adv
        from policy_doctor.behaviors.behavior_values import get_behavior_graph_and_slice_values as pd_adv

        iv_graph, iv_values, iv_q, iv_adv_arr = iv_adv(
            self.cluster_labels,
            self.metadata,
            gamma=GAMMA,
            reward_success=REWARD_SUCCESS,
            reward_failure=REWARD_FAILURE,
            reward_end=REWARD_END,
        )
        pd_graph, pd_values, pd_q, pd_adv_arr = pd_adv(
            self.cluster_labels,
            self.metadata,
            gamma=GAMMA,
            reward_success=REWARD_SUCCESS,
            reward_failure=REWARD_FAILURE,
            reward_end=REWARD_END,
        )

        self.assertEqual(iv_adv_arr.shape, pd_adv_arr.shape, "advantage shape")
        n = iv_adv_arr.size
        # Compare element-wise (allow small numerical difference)
        iv_finite = np.isfinite(iv_adv_arr)
        pd_finite = np.isfinite(pd_adv_arr)
        self.assertEqual(iv_finite.sum(), pd_finite.sum(), "finite count")
        mask = iv_finite & pd_finite
        if mask.sum() > 0:
            diff = np.abs(iv_adv_arr[mask].astype(np.float64) - pd_adv_arr[mask].astype(np.float64))
            max_diff = float(np.max(diff))
            self.assertLess(max_diff, 1e-5, f"max advantage diff {max_diff}")

        # Count selected (IV mask: valid = cluster_labels != -1 only)
        valid_iv = self.cluster_labels != -1
        valid_pd = np.isfinite(pd_adv_arr) & (self.cluster_labels >= 0)
        sel_iv = np.sum(valid_iv & (iv_adv_arr >= ADVANTAGE_THRESHOLD))
        sel_pd = np.sum(valid_pd & (pd_adv_arr >= ADVANTAGE_THRESHOLD))
        self.assertEqual(int(sel_iv), int(sel_pd), f"selected count (adv >= {ADVANTAGE_THRESHOLD}): IV={sel_iv} PD={sel_pd}")

    def test_step2_rollout_slices_match(self):
        """Step 2: Selected indices -> rollout slice dicts; compare count and first entries."""
        from influence_visualizer.behavior_value_loader import (
            get_behavior_graph_and_slice_values as iv_adv,
            slice_indices_to_rollout_slices as iv_slices,
        )
        from policy_doctor.behaviors.behavior_values import (
            get_behavior_graph_and_slice_values as pd_adv,
            slice_indices_to_rollout_slices as pd_slices,
        )
        from policy_doctor.data.structures import EpisodeInfo

        _, _, _, iv_adv_arr = iv_adv(
            self.cluster_labels, self.metadata,
            gamma=GAMMA, reward_success=REWARD_SUCCESS,
            reward_failure=REWARD_FAILURE, reward_end=REWARD_END,
        )
        _, _, _, pd_adv_arr = pd_adv(
            self.cluster_labels, self.metadata,
            gamma=GAMMA, reward_success=REWARD_SUCCESS,
            reward_failure=REWARD_FAILURE, reward_end=REWARD_END,
        )

        valid_iv = self.cluster_labels != -1
        valid_pd = np.isfinite(pd_adv_arr) & (self.cluster_labels >= 0)
        indices_iv = np.where(valid_iv & (iv_adv_arr >= ADVANTAGE_THRESHOLD))[0]
        indices_pd = np.where(valid_pd & (pd_adv_arr >= ADVANTAGE_THRESHOLD))[0]

        iv_rollout = iv_slices(self.metadata, self.data, self.cluster_labels, indices_iv)
        # PD needs list of EpisodeInfo (same as IV's rollout_episodes)
        def to_ep(ep):
            return EpisodeInfo(
                index=ep.index,
                num_samples=ep.num_samples,
                sample_start_idx=ep.sample_start_idx,
                sample_end_idx=ep.sample_end_idx,
                success=getattr(ep, "success", None),
                raw_length=getattr(ep, "raw_length", None) or ep.num_samples,
            )
        pd_rollout_eps = [to_ep(ep) for ep in self.data.rollout_episodes]
        pd_rollout = pd_slices(self.metadata, pd_rollout_eps, self.cluster_labels, indices_pd)

        self.assertEqual(len(iv_rollout), len(pd_rollout), "rollout slice count")
        # Compare first 5 (rollout_idx, start, end)
        for i in range(min(5, len(iv_rollout), len(pd_rollout))):
            a, b = iv_rollout[i], pd_rollout[i]
            self.assertEqual(a["rollout_idx"], b["rollout_idx"], f"slice[{i}] rollout_idx")
            self.assertEqual(a["start"], b["start"], f"slice[{i}] start")
            self.assertEqual(a["end"], b["end"], f"slice[{i}] end")

    def test_step3_slice_search_one_slice_scores_match(self):
        """Step 3: For the first rollout slice, compare aggregated scores from IV vs PD."""
        import numpy as np
        from influence_visualizer.render_learning import _run_slice_search_one as iv_search_one
        from influence_visualizer.behavior_value_loader import (
            get_behavior_graph_and_slice_values as iv_adv,
            slice_indices_to_rollout_slices as iv_slices,
        )
        from policy_doctor.data.structures import GlobalInfluenceMatrix, EpisodeInfo
        from policy_doctor.data.aggregation import sliding_window_aggregate_left_aligned

        _, _, _, iv_adv_arr = iv_adv(
            self.cluster_labels, self.metadata,
            gamma=GAMMA, reward_success=REWARD_SUCCESS,
            reward_failure=REWARD_FAILURE, reward_end=REWARD_END,
        )
        valid = self.cluster_labels != -1
        indices = np.where(valid & (iv_adv_arr >= ADVANTAGE_THRESHOLD))[0]
        iv_rollout = iv_slices(self.metadata, self.data, self.cluster_labels, indices)
        if not iv_rollout:
            self.skipTest("No rollout slices")
        first_slice = iv_rollout[0]

        # IV: one slice
        _, _, _, iv_candidates = iv_search_one(
            self.data,
            first_slice,
            demo_split="holdout",
            window_width=WINDOW_WIDTH,
            aggregation_method=AGGREGATION_METHOD,
            per_slice_top_k=100,
            ascending=False,
            use_all_demos_per_slice=True,
        )

        # PD: build global matrix and compute scores for same slice
        def to_ep(ep):
            return EpisodeInfo(
                index=ep.index, num_samples=ep.num_samples,
                sample_start_idx=ep.sample_start_idx, sample_end_idx=ep.sample_end_idx,
                success=getattr(ep, "success", None),
                raw_length=getattr(ep, "raw_length", None) or ep.num_samples,
            )
        all_demo = [to_ep(ep) for ep in self.data.all_demo_episodes]
        rollout_eps = [to_ep(ep) for ep in self.data.rollout_episodes]
        global_matrix = GlobalInfluenceMatrix(
            self.data.influence_matrix,
            rollout_eps,
            all_demo,
        )
        ro_ep = first_slice["rollout_ep"]
        start, end = first_slice["start"], first_slice["end"]
        r_lo = ro_ep.sample_start_idx + start
        r_hi = ro_ep.sample_start_idx + end + 1
        d_lo = self.num_train
        d_hi = self.num_train + self.num_holdout
        block = global_matrix.get_slice(r_lo, r_hi, d_lo, d_hi)
        pd_aggregated = sliding_window_aggregate_left_aligned(
            block, window_width=WINDOW_WIDTH, kind=AGGREGATION_METHOD, pad_mode="edge",
        )

        # IV candidates are for holdout only; scores in same order as sorted_indices
        self.assertEqual(len(pd_aggregated), self.num_holdout, "PD aggregated length = num_holdout")
        self.assertEqual(len(iv_candidates), self.num_holdout, "IV candidates length = num_holdout (use_all_demos)")

        # IV returns candidates sorted by score desc; scores are sorted_scores
        iv_scores = np.array([c["score"] for c in iv_candidates], dtype=np.float64)
        # PD aggregated is in demo index order; sort descending to match IV order of candidates
        pd_sorted_idx = np.argsort(pd_aggregated)[::-1]
        pd_scores_ordered = pd_aggregated[pd_sorted_idx]

        self.assertEqual(len(iv_scores), len(pd_scores_ordered))
        diff = np.abs(iv_scores.astype(np.float64) - pd_scores_ordered.astype(np.float64))
        max_diff = float(np.max(diff))
        self.assertLess(max_diff, 1e-4, f"max score diff {max_diff}")

    def test_step4_percentile_and_resolve_one_slice(self):
        """Step 4: Per-slice percentile + resolve for first slice; compare (ep_idx, start, end)."""
        import numpy as np
        from influence_visualizer.render_learning import (
            _run_slice_search_one as iv_search_one,
            _per_slice_percentile_selection as iv_percentile,
            _resolve_candidates_to_unique_slices as iv_resolve,
            _build_local_sample_lookup,
        )
        from influence_visualizer.render_heatmaps import get_split_data
        from influence_visualizer.behavior_value_loader import (
            get_behavior_graph_and_slice_values as iv_adv,
            slice_indices_to_rollout_slices as iv_slices,
        )
        from policy_doctor.curation.attribution import (
            per_slice_percentile_selection as pd_percentile,
            resolve_candidates_to_demo_slices as pd_resolve,
        )
        from policy_doctor.data.structures import EpisodeInfo

        _, _, _, iv_adv_arr = iv_adv(
            self.cluster_labels, self.metadata,
            gamma=GAMMA, reward_success=REWARD_SUCCESS,
            reward_failure=REWARD_FAILURE, reward_end=REWARD_END,
        )
        valid = self.cluster_labels != -1
        indices = np.where(valid & (iv_adv_arr >= ADVANTAGE_THRESHOLD))[0]
        iv_rollout = iv_slices(self.metadata, self.data, self.cluster_labels, indices)
        if not iv_rollout:
            self.skipTest("No rollout slices")
        first_slice = iv_rollout[0]
        _, _, _, iv_candidates = iv_search_one(
            self.data, first_slice,
            demo_split="holdout", window_width=WINDOW_WIDTH,
            aggregation_method=AGGREGATION_METHOD, per_slice_top_k=100,
            ascending=False, use_all_demos_per_slice=True,
        )
        # Single-slice percentile
        iv_selected = iv_percentile([iv_candidates], SELECTION_PERCENTILE)
        _, demo_episodes, ep_idxs, _ = get_split_data(self.data, "holdout")
        lookup = _build_local_sample_lookup(
            self.data, "holdout", ep_idxs, demo_episodes, WINDOW_WIDTH,
        )
        iv_resolved = iv_resolve(
            iv_selected, self.data, "holdout", demo_episodes, ep_idxs, WINDOW_WIDTH,
        )
        iv_tuples = sorted((r["episode"].index, r["demo_start"], r["demo_end"]) for r in iv_resolved)

        # PD: same candidates (we already verified scores match), so build PD-style candidates
        pd_candidates = [{"local_sample_idx": c["local_sample_idx"], "score": c["score"]} for c in iv_candidates]
        pd_selected = pd_percentile([pd_candidates], SELECTION_PERCENTILE)
        holdout_eps = [
            EpisodeInfo(
                index=ep.index, num_samples=ep.num_samples,
                sample_start_idx=ep.sample_start_idx, sample_end_idx=ep.sample_end_idx,
                raw_length=getattr(ep, "raw_length", None) or ep.num_samples,
            )
            for ep in self.data.holdout_episodes
        ]
        pd_resolved = pd_resolve(
            pd_selected,
            self.data.holdout_sample_infos,
            holdout_eps,
            WINDOW_WIDTH,
        )
        pd_tuples = sorted(pd_resolved)

        self.assertEqual(iv_tuples, pd_tuples, "resolved (episode_idx, start, end) match")

    def test_step5_full_pipeline_iv_and_pd_match(self):
        """Step 5: Run full pipeline in both IV and PD; assert they produce the same unique demo slice count.

        Slow (~5 min): slice search over all rollout slices. With current data/clustering, both produce
        ~19k slices; reference test_advantage_selection (1170) was created under different conditions."""
        from influence_visualizer.behavior_value_loader import (
            get_behavior_graph_and_slice_values as iv_adv,
            slice_indices_to_rollout_slices as iv_slices,
        )
        from influence_visualizer.render_learning import (
            _run_slice_search,
            _per_slice_percentile_selection as iv_percentile,
            _resolve_candidates_to_unique_slices as iv_resolve,
        )
        from influence_visualizer.render_heatmaps import get_split_data
        from policy_doctor.data.structures import GlobalInfluenceMatrix, EpisodeInfo
        from policy_doctor.behaviors.behavior_values import (
            get_behavior_graph_and_slice_values as pd_adv,
            slice_indices_to_rollout_slices as pd_slices,
        )
        from policy_doctor.curation.attribution import (
            run_slice_search,
            per_slice_percentile_selection as pd_percentile,
            resolve_candidates_to_demo_slices as pd_resolve,
        )

        def to_ep(ep):
            return EpisodeInfo(
                index=ep.index,
                num_samples=ep.num_samples,
                sample_start_idx=ep.sample_start_idx,
                sample_end_idx=ep.sample_end_idx,
                success=getattr(ep, "success", None),
                raw_length=getattr(ep, "raw_length", None) or ep.num_samples,
            )

        # Shared: advantage and selected indices (use IV so rollout_slices match)
        _, _, _, iv_adv_arr = iv_adv(
            self.cluster_labels, self.metadata,
            gamma=GAMMA, reward_success=REWARD_SUCCESS,
            reward_failure=REWARD_FAILURE, reward_end=REWARD_END,
        )
        valid = self.cluster_labels != -1
        indices = np.where(valid & (iv_adv_arr >= ADVANTAGE_THRESHOLD))[0]
        iv_rollout_slices = iv_slices(self.metadata, self.data, self.cluster_labels, indices)
        if not iv_rollout_slices:
            self.skipTest("No rollout slices")

        # IV full path
        _, per_slice_candidates_iv = _run_slice_search(
            self.data,
            iv_rollout_slices,
            demo_split="holdout",
            window_width=WINDOW_WIDTH,
            aggregation_method=AGGREGATION_METHOD,
            per_slice_top_k=100,
            ascending=False,
            use_all_demos_per_slice=True,
        )
        raw_selection_iv = iv_percentile(per_slice_candidates_iv, SELECTION_PERCENTILE)
        _, demo_episodes, ep_idxs, _ = get_split_data(self.data, "holdout")
        iv_resolved = iv_resolve(
            raw_selection_iv, self.data, "holdout", demo_episodes, ep_idxs, WINDOW_WIDTH,
        )
        iv_unique_count = len(iv_resolved)

        # PD full path (same rollout slices, same params)
        rollout_eps_pd = [to_ep(ep) for ep in self.data.rollout_episodes]
        all_demo_pd = [to_ep(ep) for ep in self.data.all_demo_episodes]
        global_matrix = GlobalInfluenceMatrix(
            self.data.influence_matrix, rollout_eps_pd, all_demo_pd,
        )
        pd_rollout_slices = pd_slices(
            self.metadata, rollout_eps_pd, self.cluster_labels, indices,
        )
        _, per_slice_candidates_pd = run_slice_search(
            global_matrix,
            pd_rollout_slices,
            all_demo_pd,
            window_width_demo=WINDOW_WIDTH,
            per_slice_top_k=100,
            ascending=False,
            demo_start_idx=self.num_train,
            demo_end_idx=self.num_train + self.num_holdout,
            use_all_demos_per_slice=True,
            aggregation_method=AGGREGATION_METHOD,
        )
        raw_selection_pd = pd_percentile(per_slice_candidates_pd, SELECTION_PERCENTILE)
        holdout_eps_pd = [to_ep(ep) for ep in self.data.holdout_episodes]
        pd_resolved = pd_resolve(
            raw_selection_pd,
            self.data.holdout_sample_infos,
            holdout_eps_pd,
            WINDOW_WIDTH,
        )
        pd_unique_count = len(pd_resolved)

        self.assertEqual(
            pd_unique_count,
            iv_unique_count,
            f"PD full pipeline produced {pd_unique_count} unique demo slices; "
            f"IV produced {iv_unique_count}. They must match.",
        )
