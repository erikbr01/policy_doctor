"""Pipeline step: score adherence for all E2 demonstrations in a run dir.

Reads:
  - <run_dir>/proposals/{condition}/selected_run.json    (request lists per condition)
  - <run_dir>/demonstrations/{request_id}/ep0000.pkl     (operator's demo pkl)
  - the clustering result + policy checkpoint (for the TrajectoryClassifier)

Writes per-run under ``score_adherence_e2/``:
  - per_demo_scores.jsonl
  - filtered_demos.jsonl
  - filter_summary.json

Config (under ``e2_score``):
  e2_run_dir         path; required
  clustering_dir     path; required
  checkpoint         path; required
  infembed_fit       path; required
  infembed_npz       path; required
  weights            inherited from e2_proposals.adherence.weights
  filter_threshold   inherited from e2_proposals.adherence.filter_threshold
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep


class ScoreAdherenceE2Step(PipelineStep[Dict[str, Any]]):
    name = "score_adherence_e2"

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
        from policy_doctor.monitoring.trajectory_classifier import TrajectoryClassifier
        from policy_doctor.vlm.proposals.adherence import score_batch_to_jsonl
        from policy_doctor.vlm.proposals.pool import RolloutPool
        from policy_doctor.vlm.proposals.request import DemonstrationRequest

        cfg = self.cfg
        sc = OmegaConf.select(cfg, "e2_score") or {}
        sc = OmegaConf.to_container(sc, resolve=True) or {}

        e2_run_dir = pathlib.Path(sc["e2_run_dir"])
        clustering_dir = pathlib.Path(sc["clustering_dir"])
        checkpoint = sc["checkpoint"]
        infembed_fit = sc["infembed_fit"]
        infembed_npz = sc["infembed_npz"]

        adherence_cfg = OmegaConf.select(cfg, "e2_proposals.adherence") or {}
        weights = OmegaConf.to_container(adherence_cfg, resolve=True).get("weights", None) if adherence_cfg else None
        filter_threshold = float(
            OmegaConf.select(cfg, "e2_proposals.adherence.filter_threshold") or 0.6
        )

        episodes_dir = pathlib.Path(
            OmegaConf.select(cfg, "e2_proposals.pool_episodes_dir") or sc["pool_episodes_dir"]
        )
        pool = RolloutPool.from_episodes_dir(episodes_dir)

        # Collect (request, demo_pkl) pairs across both conditions
        pairs: List[Tuple[DemonstrationRequest, pathlib.Path]] = []
        reference_paths: Dict[str, List[int]] = {}

        for condition_dir in (e2_run_dir / "proposals").iterdir():
            if not condition_dir.is_dir():
                continue
            sel_path = condition_dir / "selected_run.json"
            if not sel_path.exists():
                continue
            with open(sel_path) as f:
                selected = json.load(f)
            for entry in selected.get("requests", []):
                req = DemonstrationRequest.from_dict(entry)
                demo_dir = e2_run_dir / "demonstrations" / req.request_id
                demo_path = demo_dir / "demo.hdf5"
                if not demo_path.exists():
                    cands = sorted(demo_dir.glob("*.hdf5")) or sorted(demo_dir.glob("ep*.pkl"))
                    if cands:
                        demo_path = cands[-1]
                if demo_path.exists():
                    pairs.append((req, demo_path))

        # Per-rollout cluster path lookup for alternative_strategy scoring
        cp_path = e2_run_dir / "build_rollout_pool" / "cluster_paths.json"
        if not cp_path.exists():
            cp_path = self.parent_run_dir / "build_rollout_pool" / "cluster_paths.json"
        if cp_path.exists():
            with open(cp_path) as f:
                ep_to_path = json.load(f)
            for entry in pool.entries:
                p = ep_to_path.get(str(entry.episode_idx))
                if p:
                    reference_paths[entry.rollout_id] = p

        classifier = TrajectoryClassifier.from_checkpoint(
            checkpoint=checkpoint,
            infembed_fit_path=infembed_fit,
            infembed_embeddings_path=infembed_npz,
            clustering_dir=str(clustering_dir),
            mode="demo",
            episodes_dir=str(episodes_dir),
        )

        out_dir = self.step_dir
        summary = score_batch_to_jsonl(
            pairs=pairs,
            classifier=classifier,
            output_dir=out_dir,
            reference_cluster_paths=reference_paths,
            reference_pkl_resolver=lambda rid: pool.by_id(rid).episode_pkl,
            weights=weights,
            filter_threshold=filter_threshold,
        )
        return summary
