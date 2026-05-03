"""Shared fixtures for agent-tool tests.

Builds a small synthetic graph (3 cluster nodes), a small rollout pool with
mocked sim states, and a SessionContext suitable for exercising every Layer
1, 2, 3, 4 tool without touching disk-heavy infrastructure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from policy_doctor.behaviors.behavior_graph import BehaviorGraph
from policy_doctor.vlm.proposals.agents.budget import BudgetConfig
from policy_doctor.vlm.proposals.agents.context import SessionContext
from policy_doctor.vlm.proposals.pool import RolloutEntry, RolloutPool


def make_fixture_pool(tmpdir: Path) -> RolloutPool:
    """Create a 6-rollout pool: 3 success, 3 failure, no real pkls."""
    entries: List[RolloutEntry] = []
    for i in range(6):
        rid = f"r{i:04d}"
        entries.append(
            RolloutEntry(
                rollout_id=rid,
                episode_idx=i,
                episode_pkl=tmpdir / f"ep{i:04d}.pkl",
                length=20 + i,
                success=(i < 3),
                cluster_path=[0, 1, 2] if i < 3 else [0, 1, 4],  # 4 means "to-failure" passthrough
            )
        )
    # We tweak: success rollouts go START -> 0 -> 1 -> 2 -> SUCCESS,
    # failure rollouts go         START -> 0 -> 1 -> 4 -> FAILURE.
    # Cluster ids 0, 1, 2 are real; "4" appears in failure cluster_path so
    # tests can confirm the path-edge counting works.
    return RolloutPool(episodes_dir=tmpdir, entries=entries)


def make_fixture_labels_metadata():
    """Synthesize per-slice cluster_labels + metadata.

    Layout: 6 rollouts × 3 slices, slices ordered by window_start, cluster
    membership matches the cluster_paths above.
    """
    labels = []
    metadata: List[Dict[str, Any]] = []
    success_path = [0, 1, 2]
    failure_path = [0, 1, 4]
    for ep_idx in range(6):
        path = success_path if ep_idx < 3 else failure_path
        for j, c in enumerate(path):
            labels.append(c)
            metadata.append({
                "rollout_idx": ep_idx,
                "window_start": j * 5,
                "window_end": (j + 1) * 5,
                "success": (ep_idx < 3),
            })
    return np.asarray(labels, dtype=np.int64), metadata


def make_fixture_graph() -> BehaviorGraph:
    labels, metadata = make_fixture_labels_metadata()
    return BehaviorGraph.from_cluster_assignments(labels, metadata, level="rollout")


def build_fixture_context(tmpdir: Path, *, condition: str = "A_G") -> SessionContext:
    pool = make_fixture_pool(tmpdir)
    graph = make_fixture_graph()
    labels, metadata = make_fixture_labels_metadata()
    centroids = np.array(
        [
            [0.0, 0.0],   # c0
            [1.0, 0.0],   # c1
            [2.0, 0.0],   # c2
            [0.0, 0.0],   # c3 (placeholder)
            [-1.0, 0.0],  # c4
        ],
        dtype=float,
    )
    ctx = SessionContext.build(
        condition=condition,
        graph=graph,
        pool=pool,
        cluster_labels=labels,
        cluster_metadata=metadata,
        cluster_centroids=centroids,
        task_hint="Pick up the green cube and place it on the platform.",
        budget_config=BudgetConfig(max_tool_calls=80, max_visual_calls=30, max_video_calls=5),
    )
    return ctx
