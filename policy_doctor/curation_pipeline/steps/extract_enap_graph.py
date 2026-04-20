"""Run Extended L* algorithm via the faithful PMM class — pipeline step.

Implements **Stage 3** of the ENAP E-step:

    trajectories + RNN checkpoint → PMM.learn_pmm() → PMM topology

Loads the pretrained RNN checkpoint from ``train_enap_rnn`` and the symbol /
action data from ``train_enap_perception``, converts them to the trajectory
format expected by :class:`~policy_doctor.enap.pmm.PMM`, runs
``learn_pmm()`` (faithful port of the ENAP repo's Extended L* algorithm),
then serialises the result so ``build_behavior_graph`` and
``train_enap_residual`` can consume it.

Saved outputs (in ``step_dir/``):
- ``pmm.pkl``              — serialised :class:`PMM` object (pickle)
- ``pmm.json``             — JSON summary of PMM topology
- ``node_assignments.npy`` — ``(N,)`` per-timestep PMM state IDs
- ``actions.npy``          — ``(N, action_dim)`` actions (re-saved)
- ``metadata.json``        — per-timestep metadata (re-saved)
- ``cluster_centers.npy``  — ``(num_symbols, feat_dim)`` HDBSCAN centroids
- ``result.json`` / ``done``
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep


class ExtractENAPGraphStep(PipelineStep[Dict[str, Any]]):
    """Run PMM.learn_pmm() on rollout trajectories to extract a PMM.

    Config keys consumed (all under ``graph_building.enap``):
    - ``cos_tau_row``: cosine similarity threshold for L* (default 0.6)
    - ``error_threshold``: action-distance EQ threshold (default 0.3)
    - ``max_iterations``: L* iteration cap (default 20)
    - ``stabil_required``: consecutive stable passes before convergence (default 2)
    - ``min_edge_count``: minimum replay samples to keep an edge (default 1)
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
        from policy_doctor.enap.pmm import PMM
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
            print("  [dry_run] ExtractENAPGraphStep: would run PMM.learn_pmm()")
            return {"dry_run": True}

        # --- Load data ---
        symbols_all = np.load(perception_dir / "symbol_assignments.npy")  # (N,) int
        actions_all = np.load(perception_dir / "actions.npy")             # (N, a_dim)
        features_all = np.load(perception_dir / "feature_embeddings.npy") # (N, feat_dim)
        with open(perception_dir / "metadata.json") as f:
            metadata: List[Dict] = json.load(f)

        num_symbols = int(perception_prior["num_symbols"])
        action_dim = int(actions_all.shape[1] if actions_all.ndim > 1 else 1)
        feat_dim = int(features_all.shape[1])

        print(
            f"  Data: {len(metadata)} timesteps, "
            f"action_dim={action_dim}, num_symbols={num_symbols}, feat_dim={feat_dim}"
        )

        # --- Compute cluster centers (mean feature per symbol) ---
        cluster_centers = np.zeros((num_symbols, feat_dim), dtype=np.float32)
        counts = np.zeros(num_symbols, dtype=np.int64)
        for i, sym in enumerate(symbols_all):
            cluster_centers[int(sym)] += features_all[i]
            counts[int(sym)] += 1
        for c in range(num_symbols):
            if counts[c] > 0:
                cluster_centers[c] /= counts[c]

        # --- Convert to PMM trajectory format ---
        # trajectory_batch: [[{'action': (a_dim,), 'state': (s_dim,) one-hot}, ...], ...]
        trajectory_batch: List[List[Dict]] = []
        current_ep: Optional[int] = None
        ep_steps: List[Dict] = []

        def _flush(ep_steps):
            if ep_steps:
                trajectory_batch.append(list(ep_steps))

        for i, meta in enumerate(metadata):
            ep_idx = meta["rollout_idx"]
            if current_ep is None:
                current_ep = ep_idx
            if ep_idx != current_ep:
                _flush(ep_steps)
                ep_steps = []
                current_ep = ep_idx

            act = actions_all[i]
            if act.ndim == 0:
                act = act.reshape(1)

            sym = int(symbols_all[i])
            state_onehot = np.zeros(num_symbols, dtype=np.float32)
            state_onehot[sym] = 1.0

            ep_steps.append({"action": act.astype(np.float32), "state": state_onehot})

        _flush(ep_steps)

        print(f"  Trajectory batch: {len(trajectory_batch)} episodes")

        # --- Config ---
        cos_tau_row = float(OmegaConf.select(enap_cfg, "cos_tau_row") or 0.6)
        error_threshold = float(OmegaConf.select(enap_cfg, "error_threshold") or 0.3)
        max_iterations = int(OmegaConf.select(enap_cfg, "max_iterations") or 20)
        stabil_required = int(OmegaConf.select(enap_cfg, "stabil_required") or 2)
        level = str(
            OmegaConf.select(enap_cfg, "level")
            or OmegaConf.select(cfg, "clustering_level")
            or "rollout"
        )

        rnn_weights_path = rnn_prior.get(
            "checkpoint_path",
            str(rnn_dir / "pretrain_checkpoint.pt"),
        )

        # --- Run PMM.learn_pmm ---
        print(
            f"  PMM: cos_tau_row={cos_tau_row}, error_threshold={error_threshold}, "
            f"max_iter={max_iterations}"
        )
        pmm_obj = PMM(
            cos_tau_row=cos_tau_row,
            error_threshold=error_threshold,
            max_inner_iters=max_iterations,
            stabil_required=stabil_required,
        )
        pmm_obj.learn_pmm(
            trajectory_batch=trajectory_batch,
            rnn_weights_path=rnn_weights_path,
            cluster_centers=cluster_centers,
        )

        n_nodes = len(pmm_obj.pmm["Q"])
        n_edges = sum(len(d) for d in pmm_obj.pmm["delta"].values())
        print(f"  PMM: {n_nodes} nodes, {n_edges} edges")

        # --- Extract per-timestep node assignments from replay cache ---
        step_cache = pmm_obj._step_cache or {}
        # step_cache[(epi, t)] = (q_src, x, q_dst)
        # Flatten in episode order: node assignment at t = q_src (state that handled t)
        node_assignments = np.full(len(metadata), 0, dtype=np.int64)
        offset = 0
        ep_lengths: List[int] = []
        for ep in pmm_obj.episodes:
            ep_lengths.append(ep["S"].shape[0])

        for ep_idx, ep_len in enumerate(ep_lengths):
            for t in range(ep_len):
                flat_idx = offset + t
                if flat_idx < len(node_assignments):
                    edge = step_cache.get((ep_idx, t))
                    if edge is not None:
                        node_assignments[flat_idx] = edge[0]  # q_src
            offset += ep_len

        # --- Persist ---
        self.step_dir.mkdir(parents=True, exist_ok=True)
        pmm_obj.save_pmm(str(self.step_dir / "pmm.pkl"))

        pmm_json = pmm_obj.to_json_serializable()
        with open(self.step_dir / "pmm.json", "w") as f:
            json.dump(pmm_json, f, indent=2)

        np.save(self.step_dir / "node_assignments.npy", node_assignments)
        np.save(self.step_dir / "actions.npy", actions_all)
        np.save(self.step_dir / "cluster_centers.npy", cluster_centers)
        with open(self.step_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        return {
            "num_nodes": int(n_nodes),
            "num_edges": int(n_edges),
            "num_timesteps": int(len(node_assignments)),
            "num_episodes": len(trajectory_batch),
            "level": level,
            "cos_tau_row": cos_tau_row,
        }
