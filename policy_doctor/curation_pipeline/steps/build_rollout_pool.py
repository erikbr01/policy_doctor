"""Pipeline step: build the E2 rollout pool from existing eval_save_episodes output.

The pool is the set of base-policy rollouts the VLM reasons over. This step:

  1. Reads the eval_save_episodes ``episodes_dir`` (per-rollout pkl + metadata.yaml).
  2. Renders one storyboard composite per rollout (4 frames in 2x2 grid) under
     ``<step_dir>/storyboards/{rollout_id}.png``.
  3. Optionally classifies each rollout against the existing behavior graph and
     writes its cluster path under ``<step_dir>/cluster_paths.json``.
  4. Writes a frozen ``rollouts.jsonl`` index suitable for the proposal_server.

Videos are not re-rendered — the eval pipeline already produces them; we just
record sidecar paths to them when present.

Config keys (under ``e2_rollout_pool``):
  episodes_dir         path; required (resolved relative to ``repo_root``)
  classify             bool; default true (skip if no clustering result)
  clustering_dir       path; resolved from ``run_clustering`` result if not set
  storyboard_n_frames  int;  default 4
  storyboard_size      [int, int]; default [256, 256]
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep


class BuildRolloutPoolStep(PipelineStep[Dict[str, Any]]):
    name = "build_rollout_pool"

    def save(self, result: Dict[str, Any]) -> None:
        self.step_dir.mkdir(parents=True, exist_ok=True)
        with open(self.step_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        (self.step_dir / "done").touch()

    def load(self) -> Optional[Dict[str, Any]]:
        p = self.step_dir / "result.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return None

    def compute(self) -> Dict[str, Any]:
        import yaml

        from policy_doctor.vlm.proposals.pool import (
            episode_idx_to_rollout_id,
            RolloutPool,
        )

        cfg = self.cfg
        rp_cfg = OmegaConf.select(cfg, "e2_rollout_pool") or {}
        rp_cfg = OmegaConf.to_container(rp_cfg, resolve=True) or {}

        episodes_dir = rp_cfg.get("episodes_dir") or OmegaConf.select(cfg, "e2_proposals.pool_episodes_dir")
        if not episodes_dir:
            raise ValueError(
                "build_rollout_pool: set e2_rollout_pool.episodes_dir or "
                "e2_proposals.pool_episodes_dir"
            )
        episodes_dir = pathlib.Path(episodes_dir)
        if not episodes_dir.is_absolute():
            episodes_dir = self.repo_root / episodes_dir

        classify = bool(rp_cfg.get("classify", True))
        clustering_dir = rp_cfg.get("clustering_dir")

        storyboard_n_frames = int(rp_cfg.get("storyboard_n_frames", 4))
        storyboard_size = tuple(rp_cfg.get("storyboard_size", [256, 256]))

        self.step_dir.mkdir(parents=True, exist_ok=True)
        sb_dir = self.step_dir / "storyboards"
        sb_dir.mkdir(exist_ok=True)

        # 1. Build pool
        pool = RolloutPool.from_episodes_dir(episodes_dir, storyboard_dir=sb_dir)

        # 2. Render storyboards (skip ones that already exist)
        rendered = self._render_storyboards(
            pool, n_frames=storyboard_n_frames, size=storyboard_size
        )

        # 3. Classify against the saved graph if available
        cluster_paths: Dict[int, List[int]] = {}
        if classify:
            try:
                cluster_paths = self._compute_cluster_paths(pool, clustering_dir)
                with open(self.step_dir / "cluster_paths.json", "w") as f:
                    json.dump(cluster_paths, f, indent=2)
            except Exception as e:
                print(f"[build_rollout_pool] classify skipped: {e}")

        # 4. Frozen rollouts.jsonl index
        with open(self.step_dir / "rollouts.jsonl", "w") as f:
            for entry in pool.entries:
                row = {
                    "rollout_id": entry.rollout_id,
                    "episode_idx": entry.episode_idx,
                    "episode_pkl": str(entry.episode_pkl),
                    "length": entry.length,
                    "success": entry.success,
                    "storyboard_path": str(entry.storyboard_path) if entry.storyboard_path else None,
                    "cluster_path": cluster_paths.get(entry.episode_idx),
                }
                f.write(json.dumps(row) + "\n")

        return {
            "episodes_dir": str(episodes_dir),
            "n_rollouts": len(pool),
            "n_storyboards_rendered": rendered,
            "cluster_paths_present": bool(cluster_paths),
            "step_dir": str(self.step_dir),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _render_storyboards(self, pool, *, n_frames: int, size: tuple) -> int:
        """Best-effort storyboard rendering. If a rollout has no recorded camera
        frames in its pkl, we skip it and let downstream code fall back to a
        text-only description."""
        from PIL import Image
        from policy_doctor.vlm.storyboard import make_storyboard

        n_done = 0
        for entry in pool.entries:
            if entry.storyboard_path is None or entry.storyboard_path.exists():
                continue
            frames = self._load_episode_frames(entry.episode_pkl, n_frames=n_frames)
            if not frames:
                continue
            composite = make_storyboard(frames, max_frames=n_frames)
            composite = composite.resize(size, Image.LANCZOS) if composite.size != size else composite
            composite.save(entry.storyboard_path)
            n_done += 1
        return n_done

    def _load_episode_frames(self, pkl_path: pathlib.Path, *, n_frames: int) -> list:
        """Read ``frames`` (list of HxWx3 uint8) from the pkl when available.
        eval_save_episodes stores per-step ``image`` keys for image-obs runs;
        for low-dim runs we fall back to no frames (storyboard skipped)."""
        import pandas as pd
        import numpy as np

        try:
            df = pd.read_pickle(str(pkl_path))
        except Exception:
            return []
        if "frame" in df.columns:
            arrs = df["frame"].to_list()
        elif "image" in df.columns:
            arrs = df["image"].to_list()
        else:
            return []
        if not arrs:
            return []
        from PIL import Image
        n = len(arrs)
        idxs = [int(i * (n - 1) / max(1, n_frames - 1)) for i in range(n_frames)]
        return [Image.fromarray(np.asarray(arrs[i]).astype("uint8")) for i in idxs]

    def _compute_cluster_paths(self, pool, clustering_dir: Optional[str]) -> Dict[int, List[int]]:
        """Classify every rollout in the pool against the saved BehaviorGraph.

        Reuses :class:`TrajectoryClassifier`. Each rollout's cluster path is the
        run-length-collapsed sequence of cluster ids assigned to its timesteps.
        """
        from policy_doctor.data.clustering_loader import load_clustering_result_from_path
        from policy_doctor.behaviors.behavior_graph import BehaviorGraph
        from policy_doctor.vlm.proposals.adherence import classify_demo_pkl
        from policy_doctor.monitoring.trajectory_classifier import TrajectoryClassifier

        if not clustering_dir:
            cd = OmegaConf.select(self.cfg, "e2_proposals.clustering_dir")
            if cd:
                clustering_dir = cd
        if not clustering_dir:
            run_clust_result = self.parent_run_dir / "run_clustering" / "result.json"
            if run_clust_result.exists():
                with open(run_clust_result) as f:
                    res = json.load(f)
                clustering_dir = res.get("clustering_dir")
        if not clustering_dir:
            raise FileNotFoundError(
                "no clustering_dir; set e2_rollout_pool.clustering_dir or run "
                "run_clustering before this step"
            )

        # The pool-level cluster paths can also be read straight from the
        # clustering metadata when level=='rollout' — no policy checkpoint required.
        labels, metadata, manifest = load_clustering_result_from_path(pathlib.Path(clustering_dir))
        graph = BehaviorGraph.from_cluster_assignments(
            labels, metadata, level=manifest.get("level", "rollout")
        )

        # Per-rollout collapsed path from clustering metadata directly.
        from collections import defaultdict
        ep_seqs = defaultdict(list)
        for i, meta in enumerate(metadata):
            label = int(labels[i])
            if label == -1:
                continue
            ep_seqs[meta.get("rollout_idx", meta.get("demo_idx", -1))].append(
                (meta.get("timestep", meta.get("window_start", 0)), label)
            )
        out: Dict[int, List[int]] = {}
        for ep_idx, seq in ep_seqs.items():
            seq.sort(key=lambda x: x[0])
            collapsed: List[int] = []
            for _, label in seq:
                if not collapsed or collapsed[-1] != label:
                    collapsed.append(label)
            out[int(ep_idx)] = collapsed
        return out
