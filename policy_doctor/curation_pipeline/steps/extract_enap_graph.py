"""Run Extended L* algorithm to extract a PMM graph — pipeline step.

Implements **Stage 3** of the ENAP E-step:

    h_t + c_t → ExtendedLStar → PMM  (Probabilistic Mealy Machine)

Loads the RNN hidden states (h_t) from ``train_enap_rnn`` and the symbol
assignments (c_t) from ``train_enap_perception``, runs the Extended L*
algorithm to discover a compact Probabilistic Mealy Machine, then persists
the results so that ``build_behavior_graph`` can convert the PMM into the
shared :class:`~policy_doctor.behaviors.behavior_graph.BehaviorGraph` format.

Saved outputs (in ``step_dir/``):
- ``pmm.json``             — serialised PMM (nodes + edges + metadata)
- ``node_assignments.npy`` — ``(N,)`` per-timestep L*-derived node IDs
- ``actions.npy``          — ``(N, action_dim)`` actions (re-saved for convenience)
- ``metadata.json``        — per-timestep metadata (re-saved for convenience)
- ``result.json`` / ``done``
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep


class ExtractENAPGraphStep(PipelineStep[Dict[str, Any]]):
    """Run Extended L* on (h_t, c_t) sequences to build a PMM.

    Config keys consumed (all under ``graph_building.enap``):
    - ``tau_sim``: cosine similarity threshold for L* membership queries
      (default 0.7; higher = tighter clusters, more nodes)
    - ``max_iterations``: L* iteration limit (default 50)
    - ``min_edge_count``: minimum transition count to keep an edge (default 2)
    - ``level``: behaviour granularity passed to BehaviorGraph (default ``"rollout"``)
    """

    name = "extract_enap_graph"

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
        from policy_doctor.curation_pipeline.steps.train_enap_rnn import TrainENAPRNNStep
        from policy_doctor.curation_pipeline.steps.train_enap_perception import (
            TrainENAPPerceptionStep,
        )

        cfg = self.cfg
        enap_cfg = OmegaConf.select(cfg, "graph_building.enap") or {}

        # --- Verify prior steps ---
        rnn_prior = TrainENAPRNNStep(cfg, self.run_dir).load()
        if not rnn_prior:
            raise RuntimeError(
                "extract_enap_graph: train_enap_rnn has not been completed yet."
            )
        perception_prior = TrainENAPPerceptionStep(cfg, self.run_dir).load()
        if not perception_prior:
            raise RuntimeError(
                "extract_enap_graph: train_enap_perception has not been completed yet."
            )

        rnn_dir = self.run_dir / "train_enap_rnn"
        perception_dir = self.run_dir / "train_enap_perception"

        if self.dry_run:
            print("  [dry_run] ExtractENAPGraphStep: would run ExtendedLStar")
            return {"dry_run": True}

        # --- Load data ---
        h_all = np.load(rnn_dir / "hidden_states.npy")           # (N, hidden_dim)
        symbols_all = np.load(perception_dir / "symbol_assignments.npy")  # (N,)
        actions_all = np.load(perception_dir / "actions.npy")    # (N, action_dim)
        with open(perception_dir / "metadata.json") as f:
            metadata: List[Dict] = json.load(f)

        print(
            f"  Hidden states: {h_all.shape}, "
            f"symbols: {symbols_all.shape}, "
            f"actions: {actions_all.shape}"
        )

        # --- Config ---
        tau_sim = float(OmegaConf.select(enap_cfg, "tau_sim") or 0.7)
        max_iterations = int(OmegaConf.select(enap_cfg, "max_iterations") or 50)
        min_edge_count = int(OmegaConf.select(enap_cfg, "min_edge_count") or 2)
        level = str(
            OmegaConf.select(enap_cfg, "level")
            or OmegaConf.select(cfg, "clustering_level")
            or "rollout"
        )

        # --- Run Extended L* ---
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

        print(
            f"  PMM: {len(pmm.nodes)} nodes, "
            f"start_node={pmm.start_node_id}"
        )

        # --- Persist ---
        self.step_dir.mkdir(parents=True, exist_ok=True)
        pmm_dict = pmm.to_dict()
        with open(self.step_dir / "pmm.json", "w") as f:
            json.dump(pmm_dict, f, indent=2)
        np.save(self.step_dir / "node_assignments.npy", node_assignments)
        # Re-save actions and metadata so build_behavior_graph only needs
        # to look in extract_enap_graph's step_dir.
        np.save(self.step_dir / "actions.npy", actions_all)
        with open(self.step_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        num_edges = sum(len(n.outgoing) for n in pmm.nodes.values())

        return {
            "num_nodes": int(len(pmm.nodes)),
            "num_edges": int(num_edges),
            "start_node_id": int(pmm.start_node_id),
            "num_timesteps": int(len(node_assignments)),
            "level": level,
            "tau_sim": tau_sim,
        }
