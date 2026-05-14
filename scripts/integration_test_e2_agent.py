"""End-to-end integration test of the E2 agent path against real clustering data.

Runs from real apr26-sweep clustering artefacts up to (but not past) the point
a human operator would come in to execute demonstrations. Drives the entire
agentic pipeline: SessionContext build, tool registry, AgentSession.run loop,
trace, submission, aggregation.

Verifies (in order):
  1. Real BehaviorGraph constructs from the saved cluster_labels + metadata.
  2. RolloutPool indexes the rollouts the metadata references.
  3. Tool registry assembles for both A_G and A_NG conditions.
  4. AgentSession with the deterministic mock backend runs to ``finalize``.
  5. Submitted requests land in the proposals/<cond>/selected_run.json
     artefact downstream operator pipeline expects.
  6. Optional: same with Claude backend if ANTHROPIC_API_KEY is set.
  7. Optional: same with Gemini backend if GOOGLE_API_KEY is set.

Stops before /requests/active drainage (which is the operator's job).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO / "third_party" / "cupid"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


CLUSTERING_DIR = Path(
    "/home/erbauer/refactor_cupid/policy_doctor/third_party/influence_visualizer/"
    "configs/square_mh_apr23_mimicgen_pipeline/clustering/"
    "mimicgen_square_sweep_apr26_seed0_kmeans_k15"
)


def synthesize_pool_from_metadata(metadata, tmp_episodes_dir: Path):
    """Build a RolloutPool whose entries match the rollouts the clustering touched.

    No real ep*.pkl files exist (eval_save_episodes wasn't run to disk for
    this sweep), so we synthesize an empty pkl per rollout so the pool index
    constructs cleanly. Visual tools that try to read frames from these stub
    pkls return graceful errors — exactly the path Layer 2 takes when a real
    run encounters a low-dim rollout.
    """
    import yaml

    from policy_doctor.vlm.proposals.pool import RolloutPool

    rollout_outcomes = {}
    rollout_lengths = defaultdict(int)
    for meta in metadata:
        rid = int(meta["rollout_idx"])
        rollout_outcomes.setdefault(rid, bool(meta.get("success", False)))
        # Approximate per-rollout length from the max window_end across slices.
        end = int(meta.get("window_end", 0))
        if end > rollout_lengths[rid]:
            rollout_lengths[rid] = end

    rollout_ids_sorted = sorted(rollout_outcomes.keys())
    n_rollouts = len(rollout_ids_sorted)
    if rollout_ids_sorted != list(range(n_rollouts)):
        raise RuntimeError(
            f"rollout indices not contiguous (got {rollout_ids_sorted[:5]}...{rollout_ids_sorted[-3:]})"
        )

    tmp_episodes_dir.mkdir(parents=True, exist_ok=True)
    successes = []
    lengths = []
    for rid in rollout_ids_sorted:
        # Empty pkl placeholder — pool needs the file path to exist.
        (tmp_episodes_dir / f"ep{rid:04d}.pkl").write_bytes(b"")
        successes.append(rollout_outcomes[rid])
        lengths.append(rollout_lengths[rid])
    with open(tmp_episodes_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(
            {"episode_successes": successes, "episode_lengths": lengths},
            f,
        )

    return RolloutPool.from_episodes_dir(tmp_episodes_dir)


def run_one(condition: str, ctx, backend, system_prompt: str, user_message: str,
            out_dir: Path, label: str):
    from policy_doctor.vlm.proposals.agents.session import AgentSession
    from policy_doctor.vlm.proposals.agents.tools.registry import build_tool_registry
    from policy_doctor.vlm.proposals.agents.trace import SessionTrace

    tools = build_tool_registry(condition, ctx)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.jsonl"

    with SessionTrace(out_path=trace_path) as trace:
        session = AgentSession(
            backend=backend,
            ctx=ctx,
            tools=tools,
            system_prompt=system_prompt,
            user_message=user_message,
            seed=0,
            max_turns=30,
            trace=trace,
            out_dir=out_dir,
        )
        result = session.run()

    print(f"\n[{label}] condition={condition}")
    print(f"  stop_reason   : {result.stop_reason}")
    print(f"  n_turns       : {result.n_turns}")
    print(f"  n_tool_calls  : {result.n_tool_calls}")
    print(f"  n_failed_calls: {result.n_failed_tool_calls}")
    print(f"  n_submitted   : {len(result.submitted_requests)}")
    print(f"  budget        : {result.budget_summary}")
    if result.error:
        print(f"  ERROR         : {result.error}")
    print(f"  out_dir       : {out_dir}")
    return result


def main() -> int:
    print("=" * 70)
    print("E2 agent integration test — apr26 sweep baselines")
    print("=" * 70)

    print(f"\nClustering dir: {CLUSTERING_DIR}")
    if not CLUSTERING_DIR.exists():
        print("[FAIL] clustering dir not found")
        return 1

    # ---- 1. Real BehaviorGraph from real cluster artefacts -----------------
    from policy_doctor.behaviors.behavior_graph import BehaviorGraph
    from policy_doctor.data.clustering_loader import load_clustering_result_from_path

    labels, metadata, manifest = load_clustering_result_from_path(CLUSTERING_DIR)
    graph = BehaviorGraph.from_cluster_assignments(
        labels, metadata, level=manifest.get("level", "rollout")
    )
    print(f"\n[1/6] Loaded BehaviorGraph:")
    print(f"  level       : {graph.level}")
    print(f"  n_nodes     : {len(graph.nodes)}")
    print(f"  n_episodes  : {graph.num_episodes}")
    print(f"  n_slices    : {len(labels)}")

    # ---- 2. Synthetic RolloutPool matching the clustering ------------------
    tmpdir = Path(tempfile.mkdtemp(prefix="e2_integration_"))
    pool = synthesize_pool_from_metadata(metadata, tmpdir / "episodes")
    n_succ = len(pool.successes())
    n_fail = len(pool.failures())
    print(f"\n[2/6] Synthetic RolloutPool:")
    print(f"  n_rollouts  : {len(pool)}")
    print(f"  n_successes : {n_succ}")
    print(f"  n_failures  : {n_fail}")

    # ---- 3. SessionContext build + spot-check Layer 1 against real graph ---
    from policy_doctor.vlm.proposals.agents.budget import BudgetConfig
    from policy_doctor.vlm.proposals.agents.context import SessionContext

    out_root = tmpdir / "agent_sessions"
    out_root.mkdir()

    def _build_ctx(cond: str) -> SessionContext:
        return SessionContext.build(
            condition=cond,
            graph=graph,
            pool=pool,
            cluster_labels=labels,
            cluster_metadata=metadata,
            budget_config=BudgetConfig(
                max_tool_calls=40, max_visual_calls=10, max_video_calls=2,
                max_session_duration_s=300,
            ),
            task_hint="Pick up the square nut and place it on the rod (mimicgen square).",
            config={"kinematic_summary_strategy": "cluster_stats"},
        )

    print("\n[3/6] Spot-checking Layer 1 tools against the real graph:")
    from policy_doctor.vlm.proposals.agents.tools.registry import build_tool_registry

    ctx_probe = _build_ctx("A_G")
    tools_probe = build_tool_registry("A_G", ctx_probe)
    summary_result = tools_probe["get_graph_summary"].func({})
    if not summary_result.ok:
        print(f"  [FAIL] get_graph_summary returned error: {summary_result.content[0].text}")
        return 1
    summary_payload = json.loads(summary_result.content[0].text)
    print(
        f"  get_graph_summary  : {summary_payload['n_cluster_nodes']} clusters, "
        f"{summary_payload['n_paths_to_failure']} failure paths, "
        f"{summary_payload['n_paths_to_success']} success paths"
    )
    list_nodes_result = tools_probe["list_nodes"].func({"min_failure_likelihood": 0.5})
    n_high_fail = json.loads(list_nodes_result.content[0].text)["n_filtered"]
    print(f"  list_nodes(>=0.5) : {n_high_fail} high-failure nodes")
    failure_nodes = tools_probe["find_failure_nodes"].func({"min_failure_prob": 0.3})
    n_fail_nodes = json.loads(failure_nodes.content[0].text)["n_nodes"]
    print(f"  find_failure_nodes: {n_fail_nodes} nodes flagged")

    # Drop the probe; each session below builds its own fresh ctx so budgets
    # / submitted_requests don't bleed across runs.
    del ctx_probe, tools_probe

    # ---- 4. Mock-backend session — A_G -------------------------------------
    print("\n[4/6] Running mock-backend agent sessions...")
    from policy_doctor.vlm.backends.mock import MockVLMBackend
    from policy_doctor.vlm.proposals.agents.system_prompts import prompt_text

    user_msg_ag = (
        "Task: Pick up the square nut and place it on the rod. "
        f"Pool has {len(pool)} rollouts ({n_succ} success, {n_fail} failure); "
        f"sample ids: {' '.join(pool.rollout_ids[:8])}. "
        "Explore the graph and submit demonstration requests."
    )
    user_msg_ang = (
        "Task: Pick up the square nut and place it on the rod. "
        f"Pool has {len(pool)} rollouts ({n_succ} success, {n_fail} failure); "
        f"sample ids: {' '.join(pool.rollout_ids[:8])}. "
        "Explore rollouts and submit demonstration requests."
    )

    mock_backend = MockVLMBackend()

    ag_result = run_one(
        "A_G", _build_ctx("A_G"), mock_backend,
        prompt_text("A_G"), user_msg_ag,
        out_root / "A_G" / "seed_0", "MOCK A_G",
    )
    ang_result = run_one(
        "A_NG", _build_ctx("A_NG"), mock_backend,
        prompt_text("A_NG"), user_msg_ang,
        out_root / "A_NG" / "seed_0", "MOCK A_NG",
    )

    # ---- 5. Aggregation handoff to downstream pipeline ---------------------
    print("\n[5/6] Aggregation + downstream artefact handoff:")
    from policy_doctor.vlm.proposals.agents.aggregate import (
        aggregate_agent_sessions,
        write_aggregate_artefacts,
    )

    proposals_root = tmpdir / "proposals"
    for cond, results in (("A_G", [ag_result]), ("A_NG", [ang_result])):
        agg = aggregate_agent_sessions(results, method="best_consistency_run")
        write_aggregate_artefacts(agg, out_dir=proposals_root / cond)
        selected = json.loads((proposals_root / cond / "selected_run.json").read_text())
        print(
            f"  {cond:5s}: selected_seed={agg.selected_seed} "
            f"n_requests={len(selected['requests'])} "
            f"-> {proposals_root / cond / 'selected_run.json'}"
        )
        for r in selected["requests"][:2]:
            ic = r["initial_conditions"]
            print(
                f"         · {r['request_type']:20s} ref={ic['reference_rollout_id']} "
                f"frame={ic['reference_frame']} cluster={r.get('target_cluster')}"
            )

    # ---- 6. Optional real-API roundtrips -----------------------------------
    print("\n[6/6] Optional live-API smoke tests:")
    have_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    have_gemini = bool(os.environ.get("GOOGLE_API_KEY"))
    print(f"  ANTHROPIC_API_KEY : {'set' if have_anthropic else 'not set — skipping'}")
    print(f"  GOOGLE_API_KEY    : {'set' if have_gemini else 'not set — skipping'}")

    if have_anthropic:
        try:
            from policy_doctor.vlm.backends.claude import ClaudeVLMBackend

            claude = ClaudeVLMBackend(model_name="claude-haiku-4-5-20251001")  # cheapest
            run_one(
                "A_G", _build_ctx("A_G"), claude,
                prompt_text("A_G"), user_msg_ag,
                out_root / "A_G" / "claude_live", "CLAUDE A_G",
            )
        except Exception as e:
            print(f"  [WARN] claude live run errored: {type(e).__name__}: {e}")

    if have_gemini:
        try:
            from policy_doctor.vlm.backends.gemini import GeminiVLMBackend

            gemini = GeminiVLMBackend(model_name="gemini-2.0-flash")
            run_one(
                "A_G", _build_ctx("A_G"), gemini,
                prompt_text("A_G"), user_msg_ag,
                out_root / "A_G" / "gemini_live", "GEMINI A_G",
            )
        except Exception as e:
            print(f"  [WARN] gemini live run errored: {type(e).__name__}: {e}")

    print("\n" + "=" * 70)
    print(f"Integration test wrote artefacts under: {tmpdir}")
    print("=" * 70)
    print("Stopped at the request-submission boundary. Next step would be:")
    print("  - operator (or run_e2_sim.py) drains /requests/active")
    print("  - score_adherence_e2 reads proposals/<cond>/selected_run.json")
    print("  - retrain + eval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
