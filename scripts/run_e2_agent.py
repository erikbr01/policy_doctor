"""Run agent sessions for one or more E2 conditions.

Hydra entry point. Loads the same configs/e2 layout the proposal server uses,
plus the agentic.* block. Writes traces and submitted-request artefacts to
``<run_dir>/agent_sessions/<condition>/seed_<n>/``.

Typical invocation::

    python scripts/run_e2_agent.py \\
        --config-path ../policy_doctor/configs --config-name e2/defaults \\
        e2_proposals.pool_episodes_dir=/path/to/eval \\
        e2_proposals.clustering_dir=/path/to/clustering \\
        e2_proposals.run_dir=/tmp/e2_agent_run \\
        e2_proposals.conditions=[A_G,A_NG]

For Tier 0 (CI smoke):

    python scripts/run_e2_agent.py +experiment=e2_smoke_tier0
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO / "third_party" / "cupid"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@hydra.main(
    version_base=None,
    config_path=str(_REPO / "policy_doctor" / "configs"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    e2_cfg = OmegaConf.select(cfg, "e2_proposals")
    if e2_cfg is None:
        raise RuntimeError(
            "missing e2_proposals config; pass +experiment=e2_smoke_tier0 (or similar)"
        )

    if str(OmegaConf.select(e2_cfg, "mode") or "agentic") != "agentic":
        raise RuntimeError(
            "run_e2_agent expects mode=agentic; for the legacy one-shot path use the proposal server"
        )

    _run(e2_cfg)


def _run(e2_cfg: DictConfig) -> None:
    from policy_doctor.behaviors.behavior_graph import BehaviorGraph
    from policy_doctor.data.clustering_loader import load_clustering_result_from_path
    from policy_doctor.vlm import get_vlm_backend
    from policy_doctor.vlm.proposals.agents.budget import BudgetConfig
    from policy_doctor.vlm.proposals.agents.run import (
        AgentRunConfig,
        run_condition,
        write_pre_registration,
    )
    from policy_doctor.vlm.proposals.pool import RolloutPool

    run_dir = Path(OmegaConf.select(e2_cfg, "run_dir"))
    run_dir.mkdir(parents=True, exist_ok=True)

    pool_episodes_dir = Path(OmegaConf.select(e2_cfg, "pool_episodes_dir"))
    clustering_dir = Path(OmegaConf.select(e2_cfg, "clustering_dir"))
    storyboard_dir = OmegaConf.select(e2_cfg, "storyboard_dir")
    video_dir = OmegaConf.select(e2_cfg, "video_dir")
    raw_states_dir = OmegaConf.select(e2_cfg, "raw_states_dir")

    pool = RolloutPool.from_episodes_dir(
        pool_episodes_dir,
        storyboard_dir=Path(storyboard_dir) if storyboard_dir else None,
        video_dir=Path(video_dir) if video_dir else None,
    )

    labels, metadata, manifest = load_clustering_result_from_path(clustering_dir)
    graph = BehaviorGraph.from_cluster_assignments(
        labels, metadata, level=manifest.get("level", "rollout")
    )

    agentic_cfg = OmegaConf.select(e2_cfg, "agentic")
    backend_name = str(OmegaConf.select(agentic_cfg, "agent_backend") or "mock")
    backend_params = OmegaConf.to_container(
        OmegaConf.select(agentic_cfg, "agent_backend_params") or {},
        resolve=True,
    ) or {}
    backend = get_vlm_backend(backend_name, backend_params)

    budget_dict = OmegaConf.to_container(
        OmegaConf.select(agentic_cfg, "budget") or {},
        resolve=True,
    ) or {}
    budget_config = BudgetConfig.from_dict(budget_dict)

    run_cfg = AgentRunConfig(
        backend=backend,
        budget_config=budget_config,
        max_turns=int(OmegaConf.select(agentic_cfg, "max_turns") or 100),
        temperature=float(OmegaConf.select(agentic_cfg, "temperature") or 0.3),
        max_tokens=int(OmegaConf.select(agentic_cfg, "max_tokens") or 4096),
        n_sessions=int(OmegaConf.select(agentic_cfg, "n_sessions_per_condition") or 3),
        base_seed=int(OmegaConf.select(e2_cfg, "base_seed") or 0),
        kinematic_summary_strategy=str(
            OmegaConf.select(agentic_cfg, "kinematic_summary_strategy") or "raw_states"
        ),
        cache_enabled=bool(
            OmegaConf.select(agentic_cfg, "cache_enabled")
            if OmegaConf.select(agentic_cfg, "cache_enabled") is not None
            else True
        ),
    )

    conditions = list(OmegaConf.select(e2_cfg, "conditions") or ["A_G", "A_NG"])
    out_root = run_dir / "agent_sessions"

    if bool(OmegaConf.select(e2_cfg, "pre_register")):
        write_pre_registration(
            run_dir / "pre_registration.yaml",
            cfg=OmegaConf.to_container(e2_cfg, resolve=True) or {},
            conditions=conditions,
        )

    task_hint = str(OmegaConf.select(e2_cfg, "task_hint") or "")
    aggregation_method = str(OmegaConf.select(e2_cfg, "aggregation") or "best_consistency_run")

    from policy_doctor.vlm.proposals.agents.aggregate import (
        aggregate_agent_sessions,
        write_aggregate_artefacts,
    )

    for cond in conditions:
        outputs = run_condition(
            condition=cond,
            cfg=run_cfg,
            graph=graph,
            pool=pool,
            base_out_dir=out_root,
            cluster_labels=labels,
            cluster_metadata=metadata,
            raw_states_dir=Path(raw_states_dir) if raw_states_dir else None,
            storyboards_dir=Path(storyboard_dir) if storyboard_dir else None,
            videos_dir=Path(video_dir) if video_dir else None,
            task_hint=task_hint,
        )
        # Per-condition summary at the top level for quick auditing.
        summary = {
            "condition": outputs.condition,
            "seeds": outputs.seeds,
            "n_sessions": len(outputs.session_results),
            "stop_reasons": [r.stop_reason for r in outputs.session_results],
            "n_submitted_per_session": [len(r.submitted_requests) for r in outputs.session_results],
            "errors": [r.error for r in outputs.session_results if r.error],
        }
        import json as _json

        (out_root / cond / "summary.json").write_text(_json.dumps(summary, indent=2))

        # Aggregate the N sessions into one strategy and write
        # proposals/<cond>/{selected_run,consistency_metrics,union_run}.json.
        # This is the artefact downstream score_adherence_e2 reads.
        agg = aggregate_agent_sessions(outputs.session_results, method=aggregation_method)
        proposals_dir = run_dir / "proposals" / outputs.condition
        write_aggregate_artefacts(agg, out_dir=proposals_dir)


if __name__ == "__main__":
    main()
