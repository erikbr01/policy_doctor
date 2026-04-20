"""Run Extended L* (custom/modified variant) — pipeline step.

This is the *modified* ENAP E-step variant.  It loads GRU hidden states
``h_t`` from :class:`train_enap_rnn_custom` and symbol assignments ``c_t``
from :class:`train_enap_perception`, then runs the
:class:`~policy_doctor.enap.extended_l_star.ExtendedLStar` algorithm to
extract a PMM.

Unlike the faithful variant (:class:`extract_enap_graph`), the Extended L*
here is our custom implementation that works directly on pre-computed ``h_t``
embeddings rather than recomputing them from a loaded RNN checkpoint.

Saved outputs (in ``step_dir/``):
- ``pmm.json``             — serialised PMM (ExtendedLStar format)
- ``node_assignments.npy`` — ``(N,)`` per-timestep node IDs
- ``actions.npy``          — ``(N, action_dim)`` actions
- ``metadata.json``        — per-timestep metadata
- ``result.json`` / ``done``
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep


class ExtractENAPGraphCustomStep(PipelineStep[Dict[str, Any]]):
    """Run custom Extended L* on (h_t, c_t) to build a PMM.

    Config keys (all under ``graph_building.enap``):
    - ``tau_sim``: cosine similarity threshold (default 0.7)
    - ``max_iterations``: L* iteration cap (default 50)
    - ``min_edge_count``: minimum edge count to keep (default 2)
    - ``level``: behaviour granularity (default ``"rollout"``)
    """

    name = "extract_enap_graph_custom"

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
        from policy_doctor.enap.extended_l_star import ExtendedLStar
        from policy_doctor.curation_pipeline.steps.train_enap_rnn_custom import (
            TrainENAPRNNCustomStep,
        )
        from policy_doctor.curation_pipeline.steps.train_enap_perception import (
            TrainENAPPerceptionStep,
        )

        cfg = self.cfg
        enap_cfg = OmegaConf.select(cfg, "graph_building.enap") or {}

        rnn_prior = TrainENAPRNNCustomStep(cfg, self.run_dir).load()
        if not rnn_prior:
            raise RuntimeError(
                "extract_enap_graph_custom: train_enap_rnn_custom has not been completed."
            )
        perception_prior = TrainENAPPerceptionStep(cfg, self.run_dir).load()
        if not perception_prior:
            raise RuntimeError(
                "extract_enap_graph_custom: train_enap_perception has not been completed."
            )

        rnn_custom_dir = self.run_dir / "train_enap_rnn_custom"
        perception_dir = self.run_dir / "train_enap_perception"

        if self.dry_run:
            print("  [dry_run] ExtractENAPGraphCustomStep: would run ExtendedLStar")
            return {"dry_run": True}

        h_all = np.load(rnn_custom_dir / "hidden_states.npy")
        symbols_all = np.load(perception_dir / "symbol_assignments.npy")
        actions_all = np.load(perception_dir / "actions.npy")
        with open(perception_dir / "metadata.json") as f:
            metadata: List[Dict] = json.load(f)

        print(
            f"  Hidden states: {h_all.shape}, "
            f"symbols: {symbols_all.shape}, actions: {actions_all.shape}"
        )

        tau_sim = float(OmegaConf.select(enap_cfg, "tau_sim") or 0.7)
        max_iterations = int(OmegaConf.select(enap_cfg, "max_iterations") or 50)
        min_edge_count = int(OmegaConf.select(enap_cfg, "min_edge_count") or 2)
        level = str(
            OmegaConf.select(enap_cfg, "level")
            or OmegaConf.select(cfg, "clustering_level")
            or "rollout"
        )

        print(f"  ExtendedLStar: tau_sim={tau_sim}, max_iter={max_iterations}")
        lstar = ExtendedLStar(
            h_embeddings=h_all,
            symbols=symbols_all,
            actions=actions_all,
            metadata=metadata,
            tau_sim=tau_sim,
            level=level,
            max_iterations=max_iterations,
            min_edge_count=min_edge_count,
        )
        pmm, node_assignments = lstar.build_graph()

        n_nodes = len(pmm.nodes)
        n_edges = sum(len(n.outgoing) for n in pmm.nodes.values())
        print(f"  PMM: {n_nodes} nodes, {n_edges} edges, start={pmm.start_node_id}")

        self.step_dir.mkdir(parents=True, exist_ok=True)
        with open(self.step_dir / "pmm.json", "w") as f:
            json.dump(pmm.to_dict(), f, indent=2)
        np.save(self.step_dir / "node_assignments.npy", node_assignments)
        np.save(self.step_dir / "actions.npy", actions_all)
        with open(self.step_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        return {
            "num_nodes": int(n_nodes),
            "num_edges": int(n_edges),
            "start_node_id": int(pmm.start_node_id),
            "num_timesteps": int(len(node_assignments)),
            "level": level,
            "tau_sim": tau_sim,
        }
