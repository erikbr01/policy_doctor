"""Build a BehaviorGraph from clustering or ENAP outputs — pipeline step."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import OmegaConf

from policy_doctor.behaviors.behavior_graph import BehaviorGraph
from policy_doctor.curation_pipeline.base_step import PipelineStep


class BuildBehaviorGraphStep(PipelineStep[Dict[str, Any]]):
    """Build and persist a :class:`BehaviorGraph` from the selected graph-building method.

    Dispatches on ``cfg.graph_building.method`` (default: ``"cupid"``):

    - **cupid**: loads ``run_clustering`` output and calls
      :meth:`BehaviorGraph.from_cluster_assignments`.
    - **enap**: loads ``extract_enap_graph`` output and calls
      :meth:`BehaviorGraph.from_enap_assignments`.

    Saves three files to ``step_dir/``:

    - ``behavior_graph.json``  — full graph serialised via
      :meth:`BehaviorGraph.to_dict`.
    - ``node_assignments.npy`` — per-timestep integer node IDs (shape ``(N,)``).
      For cupid these are the KMeans cluster labels; for ENAP the L*-derived
      node IDs.
    - ``metadata.json``        — per-timestep metadata list (same format as the
      clustering step metadata).

    The step result (``result.json``) records the paths to those files plus
    summary statistics so downstream steps can locate them quickly.

    Downstream steps should use :meth:`load_graph` to retrieve
    ``(BehaviorGraph, node_assignments, metadata)`` in one call rather than
    reading the files directly.
    """

    name = "build_behavior_graph"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Public helper
    # ------------------------------------------------------------------

    def load_graph(self) -> Optional[Tuple[BehaviorGraph, np.ndarray, List[Dict]]]:
        """Return ``(BehaviorGraph, node_assignments, metadata)`` or ``None``.

        Loads the graph and per-timestep data saved by a prior :meth:`run` call.
        Returns ``None`` if the step has not been completed yet.
        """
        result = self.load()
        if not result:
            return None
        graph_path = self.step_dir / "behavior_graph.json"
        na_path = self.step_dir / "node_assignments.npy"
        meta_path = self.step_dir / "metadata.json"
        if not (graph_path.exists() and na_path.exists() and meta_path.exists()):
            return None
        with open(graph_path) as f:
            graph = BehaviorGraph.from_dict(json.load(f))
        node_assignments = np.load(na_path)
        with open(meta_path) as f:
            metadata = json.load(f)
        return graph, node_assignments, metadata

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute(self) -> Dict[str, Any]:
        method = OmegaConf.select(self.cfg, "graph_building.method") or "cupid"
        if method == "cupid":
            return self._build_cupid()
        elif method == "enap":
            return self._build_enap()
        elif method == "enap_custom":
            return self._build_enap_custom()
        else:
            raise ValueError(
                f"Unknown graph_building.method: {method!r}. "
                "Expected 'cupid', 'enap', or 'enap_custom'."
            )

    # ------------------------------------------------------------------
    # CuPID path
    # ------------------------------------------------------------------

    def _build_cupid(self) -> Dict[str, Any]:
        from policy_doctor.curation_pipeline.steps.run_clustering import RunClusteringStep
        from policy_doctor.data.clustering_loader import load_clustering_result_from_path

        self.step_dir.mkdir(parents=True, exist_ok=True)

        # Resolve clustering directory: prefer explicit cfg.clustering_dir,
        # otherwise pull from run_clustering result.
        explicit_dir = OmegaConf.select(self.cfg, "clustering_dir")
        if explicit_dir:
            cdir = pathlib.Path(str(explicit_dir))
            if not cdir.is_absolute():
                cdir = (self.repo_root / cdir).resolve()
            clustering_dirs = {"_single": str(cdir)}
        else:
            prior = RunClusteringStep(self.cfg, self.run_dir).load()
            if not prior or not prior.get("clustering_dirs"):
                raise RuntimeError(
                    "build_behavior_graph (cupid): no clustering_dir set and "
                    "run_clustering has not been completed yet."
                )
            clustering_dirs: Dict[str, str] = {
                str(k): str(v) for k, v in prior["clustering_dirs"].items()
            }

        # For cupid we persist one graph per seed.  The canonical result is
        # the first seed (or the only seed when clustering_dir is set
        # explicitly).  Downstream steps that need per-seed graphs can load
        # multiple results.
        results_by_seed: Dict[str, Dict[str, Any]] = {}
        for seed_key, raw_path in clustering_dirs.items():
            cdir = pathlib.Path(raw_path)
            if not cdir.is_absolute():
                cdir = (self.repo_root / cdir).resolve()

            if self.dry_run:
                print(f"  [dry_run] BuildBehaviorGraphStep cupid seed={seed_key} path={cdir}")
                results_by_seed[seed_key] = {"dry_run": True}
                continue

            cluster_labels, metadata, manifest = load_clustering_result_from_path(cdir)
            level = manifest.get("level") or OmegaConf.select(self.cfg, "clustering_level") or "rollout"

            graph = BehaviorGraph.from_cluster_assignments(
                cluster_labels, metadata, level=str(level)
            )
            results_by_seed[seed_key] = self._persist_graph(
                graph=graph,
                node_assignments=cluster_labels,
                metadata=metadata,
                seed_key=seed_key,
            )

        return {
            "builder": "cupid",
            "seeds": results_by_seed,
            # Expose first seed's paths at the top level for single-seed convenience
            **self._first_seed_paths(results_by_seed),
        }

    # ------------------------------------------------------------------
    # ENAP path
    # ------------------------------------------------------------------

    def _build_enap(self) -> Dict[str, Any]:
        from policy_doctor.curation_pipeline.steps.extract_enap_graph import ExtractENAPGraphStep

        self.step_dir.mkdir(parents=True, exist_ok=True)

        prior = ExtractENAPGraphStep(self.cfg, self.run_dir).load()
        if not prior:
            raise RuntimeError(
                "build_behavior_graph (enap): extract_enap_graph has not "
                "been completed yet."
            )

        enap_step_dir = self.run_dir / "extract_enap_graph"
        na_path = enap_step_dir / "node_assignments.npy"
        actions_path = enap_step_dir / "actions.npy"
        meta_path = enap_step_dir / "metadata.json"
        pmm_path = enap_step_dir / "pmm.json"

        if self.dry_run:
            print("[dry_run] BuildBehaviorGraphStep enap: would load extract_enap_graph artifacts")
            return {"builder": "enap", "dry_run": True}

        node_assignments = np.load(na_path)
        actions = np.load(actions_path)
        with open(meta_path) as f:
            metadata = json.load(f)
        with open(pmm_path) as f:
            pmm_dict = json.load(f)

        level = prior.get("level") or OmegaConf.select(self.cfg, "graph_building.enap.level") or "rollout"

        # Convert PMM edge dict to the format expected by from_enap_assignments.
        # PMM edges use int-string keys; convert back to int.
        pmm_edges: Dict[int, Dict[int, Dict[str, Any]]] = {}
        for src_s, tgts in pmm_dict.get("edges", {}).items():
            src = int(src_s)
            pmm_edges[src] = {}
            for tgt_s, edata in tgts.items():
                pmm_edges[src][int(tgt_s)] = edata

        graph = BehaviorGraph.from_enap_assignments(
            node_assignments=node_assignments,
            actions=actions,
            metadata=metadata,
            level=str(level),
            pmm_edges=pmm_edges,
        )
        seed_result = self._persist_graph(
            graph=graph,
            node_assignments=node_assignments,
            metadata=metadata,
            seed_key="_enap",
        )
        return {
            "builder": "enap",
            "seeds": {"_enap": seed_result},
            **self._first_seed_paths({"_enap": seed_result}),
        }

    # ------------------------------------------------------------------
    # ENAP custom path (GRU + ExtendedLStar)
    # ------------------------------------------------------------------

    def _build_enap_custom(self) -> Dict[str, Any]:
        from policy_doctor.curation_pipeline.steps.extract_enap_graph_custom import (
            ExtractENAPGraphCustomStep,
        )

        self.step_dir.mkdir(parents=True, exist_ok=True)

        prior = ExtractENAPGraphCustomStep(self.cfg, self.run_dir).load()
        if not prior:
            raise RuntimeError(
                "build_behavior_graph (enap_custom): extract_enap_graph_custom "
                "has not been completed yet."
            )

        custom_dir = self.run_dir / "extract_enap_graph_custom"
        na_path = custom_dir / "node_assignments.npy"
        actions_path = custom_dir / "actions.npy"
        meta_path = custom_dir / "metadata.json"
        pmm_path = custom_dir / "pmm.json"

        if self.dry_run:
            print("[dry_run] BuildBehaviorGraphStep enap_custom")
            return {"builder": "enap_custom", "dry_run": True}

        node_assignments = np.load(na_path)
        actions = np.load(actions_path)
        with open(meta_path) as f:
            metadata = json.load(f)
        with open(pmm_path) as f:
            pmm_dict = json.load(f)

        level = prior.get("level") or OmegaConf.select(self.cfg, "graph_building.enap.level") or "rollout"

        # ExtendedLStar PMM format: {"nodes": {nid: {"outgoing": {sym: edge}}}}
        pmm_edges: Dict[int, Dict[int, Any]] = {}
        for nid_s, node_d in pmm_dict.get("nodes", {}).items():
            src = int(nid_s)
            pmm_edges[src] = {}
            for sym_s, edge_d in node_d.get("outgoing", {}).items():
                tgt = int(edge_d["target_id"])
                pmm_edges[src][tgt] = {
                    "input_symbol": edge_d.get("input_symbol"),
                    "next_input_set": edge_d.get("next_input_set"),
                }

        graph = BehaviorGraph.from_enap_assignments(
            node_assignments=node_assignments,
            actions=actions,
            metadata=metadata,
            level=str(level),
            pmm_edges=pmm_edges,
        )
        seed_result = self._persist_graph(
            graph=graph,
            node_assignments=node_assignments,
            metadata=metadata,
            seed_key="_enap_custom",
        )
        return {
            "builder": "enap_custom",
            "seeds": {"_enap_custom": seed_result},
            **self._first_seed_paths({"_enap_custom": seed_result}),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _persist_graph(
        self,
        graph: BehaviorGraph,
        node_assignments: np.ndarray,
        metadata: List[Dict],
        seed_key: str,
    ) -> Dict[str, Any]:
        """Write graph + node_assignments + metadata files; return paths dict."""
        suffix = "" if seed_key in ("_single", "_enap") else f"_seed{seed_key}"
        graph_fname = f"behavior_graph{suffix}.json"
        na_fname = f"node_assignments{suffix}.npy"
        meta_fname = f"metadata{suffix}.json"

        graph_path = self.step_dir / graph_fname
        na_path = self.step_dir / na_fname
        meta_path = self.step_dir / meta_fname

        with open(graph_path, "w") as f:
            json.dump(graph.to_dict(), f, indent=2)
        np.save(na_path, node_assignments)
        with open(meta_path, "w") as f:
            json.dump(metadata, f)

        return {
            "graph_path": str(graph_path.relative_to(self.step_dir)),
            "node_assignments_path": str(na_path.relative_to(self.step_dir)),
            "metadata_path": str(meta_path.relative_to(self.step_dir)),
            "builder": graph.builder,
            "level": graph.level,
            "num_cluster_nodes": len(graph.cluster_nodes),
            "num_episodes": graph.num_episodes,
        }

    @staticmethod
    def _first_seed_paths(results_by_seed: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Return the first seed's path keys at the top level."""
        if not results_by_seed:
            return {}
        first = next(iter(results_by_seed.values()))
        if first.get("dry_run"):
            return {}
        return {
            "graph_path": first.get("graph_path"),
            "node_assignments_path": first.get("node_assignments_path"),
            "metadata_path": first.get("metadata_path"),
            "builder": first.get("builder"),
            "level": first.get("level"),
        }
