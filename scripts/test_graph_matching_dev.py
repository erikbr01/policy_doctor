"""Quick dev script for testing graph matching on real clustering data.

Usage (from worktree root):
    conda activate policy_doctor
    python scripts/test_graph_matching_dev.py

By default matches the apr26 variance-sweep seed0 vs seed1 clusterings as a
proxy for two flywheel iterations.  Pass --dirs to override.

For actual flywheel data, point at two run_clustering/ dirs:
    python scripts/test_graph_matching_dev.py \\
        --dirs <run_dir>/mimicgen_flywheel/bg_div/iter_0/run_clustering/<name> \\
               <run_dir>/mimicgen_flywheel/bg_div/iter_1/run_clustering/<name>
"""

import argparse
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------

_DEFAULT_CLUSTERING_ROOT = (
    Path(__file__).parent.parent.parent
    / "third_party/influence_visualizer/configs/square_mh_apr23_mimicgen_pipeline/clustering"
)

_DEFAULT_DIRS = [
    _DEFAULT_CLUSTERING_ROOT / "mimicgen_square_sweep_apr26_seed0_kmeans_k15",
    _DEFAULT_CLUSTERING_ROOT / "mimicgen_square_sweep_apr26_seed1_kmeans_k15",
]


def load_clustering(path: Path):
    labels = np.load(path / "cluster_labels.npy")
    with open(path / "metadata.json") as f:
        metadata = json.load(f)
    return labels, metadata


def build_graph(labels, metadata):
    from policy_doctor.behaviors.behavior_graph import BehaviorGraph
    return BehaviorGraph.from_cluster_assignments(labels, metadata, level="rollout")


def print_graph_summary(name: str, graph, values: dict):
    from policy_doctor.behaviors.behavior_graph import SUCCESS_NODE_ID, FAILURE_NODE_ID
    print(f"\n{'─'*60}")
    print(f"  {name}  ({graph.num_episodes} episodes, {len(graph.cluster_nodes)} cluster nodes)")
    print(f"{'─'*60}")
    for nid in sorted(graph.cluster_nodes):
        node = graph.nodes[nid]
        p_s = graph.transition_probs.get(nid, {}).get(SUCCESS_NODE_ID, 0.0)
        p_f = graph.transition_probs.get(nid, {}).get(FAILURE_NODE_ID, 0.0)
        v = values.get(nid, 0.0)
        print(
            f"  node {nid:2d}  ts={node.num_timesteps:5d}  ep={node.num_episodes:3d}"
            f"  V={v:+.3f}  P(→S)={p_s:.2f}  P(→F)={p_f:.2f}"
        )


def print_result(result, graph_a, graph_b, values_a, values_b, name_a: str, name_b: str):
    from policy_doctor.behaviors.behavior_graph import SUCCESS_NODE_ID, FAILURE_NODE_ID

    def node_summary(graph, nid, values):
        if nid is None:
            return "(none)"
        node = graph.nodes[nid]
        p_s = graph.transition_probs.get(nid, {}).get(SUCCESS_NODE_ID, 0.0)
        v = values.get(nid, 0.0)
        return f"node {nid:2d} (ts={node.num_timesteps}, V={v:+.3f}, P→S={p_s:.2f})"

    print(f"\n{'═'*60}")
    print(f"  Matches: {name_a}  ←→  {name_b}  [{result.method}]")
    print(f"  {len(result.matches)} matched  |  "
          f"{len(result.unmatched_a)} unmatched in A  |  "
          f"{len(result.unmatched_b)} unmatched in B")
    print(f"{'─'*60}")
    for m in sorted(result.matches, key=lambda x: x.distance):
        a_str = node_summary(graph_a, m.node_id_a, values_a)
        b_str = node_summary(graph_b, m.node_id_b, values_b)
        print(f"  d={m.distance:.4f}  {a_str}  →  {b_str}")
    if result.unmatched_a:
        print(f"\n  Unmatched in {name_a}: {sorted(result.unmatched_a)}")
    if result.unmatched_b:
        print(f"  Unmatched in {name_b}: {sorted(result.unmatched_b)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs=2, metavar="DIR",
                        help="Two clustering result dirs to match (default: apr26 seed0 vs seed1)")
    parser.add_argument("--methods", nargs="+",
                        default=["structural", "temporal", "combined"],
                        choices=["structural", "temporal", "combined", "procrustes"],
                        help="Matching methods to compare")
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--n_bins", type=int, default=10)
    args = parser.parse_args()

    dirs = [Path(d) for d in args.dirs] if args.dirs else _DEFAULT_DIRS
    for d in dirs:
        if not d.exists():
            raise FileNotFoundError(
                f"Clustering dir not found: {d}\n"
                "Run the pipeline first, or pass --dirs explicitly."
            )

    print(f"Loading A: {dirs[0].name}")
    labels_a, meta_a = load_clustering(dirs[0])
    print(f"Loading B: {dirs[1].name}")
    labels_b, meta_b = load_clustering(dirs[1])

    graph_a = build_graph(labels_a, meta_a)
    graph_b = build_graph(labels_b, meta_b)
    values_a = graph_a.compute_values()
    values_b = graph_b.compute_values()

    print_graph_summary(dirs[0].name, graph_a, values_a)
    print_graph_summary(dirs[1].name, graph_b, values_b)

    from policy_doctor.behaviors.graph_matching import match_graphs, build_tracking_chains

    results = {}
    for method in args.methods:
        extra = {}
        if method in ("procrustes",):
            extra["clustering_dir_a"] = dirs[0]
            extra["clustering_dir_b"] = dirs[1]
        result = match_graphs(
            graph_a, meta_a, labels_a,
            graph_b, meta_b, labels_b,
            method=method,
            ratio=args.ratio,
            n_bins=args.n_bins,
            **extra,
        )
        results[method] = result
        print_result(result, graph_a, graph_b, values_a, values_b, dirs[0].name, dirs[1].name)

    # Cross-method agreement: how often do all methods agree on the same pair?
    if len(results) > 1:
        method_matches = {
            m: {(r.node_id_a, r.node_id_b) for r in res.matches}
            for m, res in results.items()
        }
        agreed = set.intersection(*method_matches.values())
        print(f"\n{'═'*60}")
        print(f"  Cross-method agreement: {len(agreed)} pairs agreed by all methods")
        for pair in sorted(agreed):
            print(f"    {pair[0]} ↔ {pair[1]}")

    # Show tracking chains for a 2-iteration sequence
    first_method = args.methods[0]
    chains = build_tracking_chains(
        [results[first_method]], [dirs[0].name, dirs[1].name]
    )
    print(f"\n  Tracking chains ({first_method}):")
    for chain in chains:
        a, b = chain[0], chain[1]
        tag = "NEW" if a is None else ("LOST" if b is None else "")
        print(f"    {str(a):>6} → {str(b):<6}  {tag}")


if __name__ == "__main__":
    main()
