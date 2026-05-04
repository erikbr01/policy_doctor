"""End-to-end E2 agent run on transport_mh r512 seed0 with Qwen3-VL-32B (4-bit).

Drives one A_G session against the rebuilt seed-0 clustering and the matching
eval-pool episodes. Persists the session dir under /tmp/e2_runs and renders an
HTML report.

Usage::

    python scripts/run_e2_agent_transport_mh.py \\
        --clustering_dir /tmp/transport_mh_seed0_r512_clustering \\
        --episodes_dir <eval episodes dir> \\
        --condition A_G --seed 0

Defaults assume the rebuilt clustering at /tmp/transport_mh_seed0_r512_clustering.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO / "third_party" / "cupid"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


_DEFAULT_CLUSTERING = Path("/tmp/transport_mh_seed0_r512_clustering")
_DEFAULT_EPISODES = Path(
    "/mnt/ssdB/erik/cupid_data/outputs/eval_save_episodes/mar27/"
    "mar27_train_diffusion_unet_lowdim_transport_mh_0_r512x512/latest/episodes"
)


def _build_graph_and_pool(clustering_dir: Path, episodes_dir: Path):
    import yaml
    import numpy as np

    from policy_doctor.behaviors.behavior_graph import BehaviorGraph
    from policy_doctor.vlm.proposals.pool import RolloutPool

    cluster_labels = np.load(clustering_dir / "cluster_labels.npy")
    metadata = json.loads((clustering_dir / "metadata.json").read_text())
    manifest = yaml.safe_load((clustering_dir / "manifest.yaml").read_text())

    pool = RolloutPool.from_episodes_dir(episodes_dir)

    # Build a behavior graph from the per-slice cluster labels + metadata.
    # The slice metadata carries rollout_idx + window bounds + success.
    graph = BehaviorGraph.from_cluster_assignments(
        cluster_labels=cluster_labels,
        metadata=metadata,
    )
    return graph, pool, cluster_labels, metadata, manifest


def _make_backend(backend: str, model_id: str, max_new_tokens: int):
    """Build a VLMBackend that implements ``chat_with_tools``.

    Note: Qwen2/3-VL backends do NOT implement chat_with_tools yet — they
    cover one-shot ``classify_slice`` calls only. Until Qwen function-calling
    is wired into the agent loop, a hosted model is required for A_G/A_NG.
    """
    if backend == "gemini":
        import os
        from policy_doctor.vlm.backends.gemini import GeminiVLMBackend

        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            # Try the project-local .env so you don't have to export by hand.
            env = Path(__file__).resolve().parents[1] / ".env"
            if env.exists():
                for line in env.read_text().splitlines():
                    if line.startswith("GEMINI_API_KEY="):
                        os.environ["GEMINI_API_KEY"] = line.split("=", 1)[1]
        return GeminiVLMBackend(model_name=model_id, max_output_tokens=max_new_tokens)

    if backend == "claude":
        from policy_doctor.vlm.backends.claude import ClaudeVLMBackend

        return ClaudeVLMBackend(model_name=model_id, max_tokens=max_new_tokens)

    if backend == "mock":
        from policy_doctor.vlm.backends.mock import MockVLMBackend

        return MockVLMBackend()

    raise ValueError(
        f"unsupported backend {backend!r}. Use 'gemini', 'claude', or 'mock'. "
        "Qwen-as-agent requires implementing chat_with_tools — currently unsupported."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clustering_dir", type=Path, default=_DEFAULT_CLUSTERING)
    ap.add_argument("--episodes_dir", type=Path, default=_DEFAULT_EPISODES)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--condition", choices=["A_G", "A_NG"], default="A_G")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["gemini", "claude", "mock"], default="gemini")
    ap.add_argument("--model_id", default="gemini-2.5-flash")
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--max_tool_calls", type=int, default=25)
    ap.add_argument("--max_visual_calls", type=int, default=18)
    ap.add_argument("--max_session_duration_s", type=int, default=1800)
    ap.add_argument("--max_turns", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--task_hint", default=(
        "robomimic transport_mh: two Franka Panda arms cooperatively move a hammer "
        "from a starting bin to a goal bin. Each arm has a parallel-jaw gripper. "
        "The transfer involves arm-0 picking up the hammer from one bin, then "
        "handing off (or placing) so arm-1 can move it to the goal."
    ))
    ap.add_argument("--no_render", action="store_true")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = Path(tempfile.mkdtemp(prefix="e2_run_qwen32b_"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] out_dir={args.out_dir}", flush=True)

    print(f"[run] building graph + pool from {args.clustering_dir}", flush=True)
    graph, pool, cluster_labels, metadata, _ = _build_graph_and_pool(
        args.clustering_dir, args.episodes_dir,
    )
    print(f"[run] pool: {len(pool)} rollouts; graph: {len(graph.nodes)} nodes; slices: {len(metadata)}", flush=True)

    print(f"[run] loading backend={args.backend} model={args.model_id}", flush=True)
    backend = _make_backend(args.backend, args.model_id, args.max_new_tokens)

    from policy_doctor.vlm.proposals.agents.budget import BudgetConfig
    from policy_doctor.vlm.proposals.agents.run import run_one_session

    bc = BudgetConfig(
        max_tool_calls=args.max_tool_calls,
        max_visual_calls=args.max_visual_calls,
        max_video_calls=0,
        max_session_duration_s=args.max_session_duration_s,
    )

    storyboard_cfg = {
        "n_frames": 5,
        "pad_before": 12,
        "pad_after": 12,
        "target_size": (1024, 1024),
        "cameras": None,  # auto-pick: agentview + every wrist cam available
    }

    print(f"[run] starting {args.condition} session (seed={args.seed})", flush=True)
    result = run_one_session(
        condition=args.condition,
        seed=args.seed,
        backend=backend,
        graph=graph,
        pool=pool,
        out_dir=args.out_dir,
        budget_config=bc,
        max_turns=args.max_turns,
        temperature=args.temperature,
        cluster_labels=cluster_labels,
        cluster_metadata=metadata,
        task_hint=args.task_hint,
        storyboard=storyboard_cfg,
    )

    print(
        f"[run] done: stop={result.stop_reason} n_submitted={len(result.submitted_requests)} "
        f"n_turns={result.budget_summary.get('n_tool_calls', '?')}",
        flush=True,
    )
    print(f"[run] artifacts in {args.out_dir}", flush=True)

    if not args.no_render:
        from scripts.render_agent_session import render

        report_path = args.out_dir / "session_report.html"
        render(args.out_dir, pool, out_path=report_path,
               title=f"{args.condition} seed={args.seed} ({args.model_id})")
        print(f"[run] wrote {report_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
