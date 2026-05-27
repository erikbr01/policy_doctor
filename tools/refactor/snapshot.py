"""Golden-snapshot tool for the architecture refactor.

Captures deterministic outputs from three core code paths so we can detect
regressions during the refactor (REFACTOR_PLAN.md §5).

The three anchors:
  1. clustering         — policy_doctor.behaviors.clustering.run_clustering
  2. behavior_graph     — policy_doctor.behaviors.behavior_graph.BehaviorGraph.from_cluster_assignments
  3. influence_slice    — policy_doctor.data.structures.GlobalInfluenceMatrix.get_slice

(A fourth anchor, MimicGen seed selection, is deferred — see TODO at bottom.)

Usage:
    python -m tools.refactor.snapshot write        # generate goldens
    python -m tools.refactor.snapshot verify       # assert current code reproduces them
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# All anchor inputs are seeded deterministically from this constant.
_SEED = 42


# ---------------------------------------------------------------------------
# Anchor 1: clustering
# ---------------------------------------------------------------------------


def _clustering_inputs() -> Dict[str, np.ndarray]:
    """Deterministic synthetic embeddings exercising run_clustering.

    Uses GaussianMixture + PCA (both deterministic with fixed seed) so the
    output is reproducible across machines and library versions that preserve
    numerical determinism.
    """
    rng = np.random.default_rng(_SEED)
    # Three well-separated Gaussian blobs in 5-D, 60 samples total.
    blobs = [
        rng.normal(loc=mean, scale=0.4, size=(20, 5))
        for mean in (np.zeros(5), np.ones(5) * 3.0, np.ones(5) * -3.0)
    ]
    embeddings = np.concatenate(blobs, axis=0).astype(np.float32)
    return {"embeddings": embeddings}


def _clustering_compute(inputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
    from policy_doctor.behaviors.clustering import run_clustering

    labels, coords, metrics = run_clustering(
        inputs["embeddings"],
        method="gaussian_mixture",
        dim_reduce="pca",
        n_components_2d=2,
        normalize="none",
        n_components=3,
        random_state=_SEED,
    )
    return {"labels": labels, "coords": coords, "metrics": metrics}


def _clustering_save(out: Dict[str, Any], dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    np.save(dst / "labels.npy", out["labels"])
    np.save(dst / "coords.npy", out["coords"])
    (dst / "metrics.json").write_text(json.dumps(out["metrics"], sort_keys=True, indent=2))


def _clustering_load(dst: Path) -> Dict[str, Any]:
    return {
        "labels": np.load(dst / "labels.npy"),
        "coords": np.load(dst / "coords.npy"),
        "metrics": json.loads((dst / "metrics.json").read_text()),
    }


def _clustering_compare(actual: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    """Return list of failure messages (empty list = pass)."""
    failures: List[str] = []
    if not np.array_equal(actual["labels"], expected["labels"]):
        failures.append(
            f"clustering.labels differ: {np.flatnonzero(actual['labels'] != expected['labels']).tolist()[:10]}"
        )
    # PCA orientation is sign-arbitrary across some sklearn versions; we
    # pin the array bit-for-bit and trust that sklearn preserves it within
    # one version. If this becomes flaky, switch to allclose with abs().
    if not np.array_equal(actual["coords"], expected["coords"]):
        max_abs = float(np.max(np.abs(actual["coords"] - expected["coords"])))
        failures.append(f"clustering.coords differ (max |Δ| = {max_abs:.6g})")
    if actual["metrics"] != expected["metrics"]:
        failures.append(
            f"clustering.metrics differ: actual={actual['metrics']} expected={expected['metrics']}"
        )
    return failures


# ---------------------------------------------------------------------------
# Anchor 2: behavior graph
# ---------------------------------------------------------------------------


def _behavior_graph_inputs() -> Dict[str, Any]:
    """Deterministic labels + episode metadata for behavior-graph construction.

    Six episodes, 10 timesteps each. Half succeed, half fail. Labels follow a
    pattern that produces several distinct collapsed sequences.
    """
    rng = np.random.default_rng(_SEED + 1)
    num_episodes = 6
    steps_per_episode = 10
    labels: List[int] = []
    metadata: List[Dict[str, Any]] = []
    for ep in range(num_episodes):
        # Each episode follows pattern A-A-B-B-C-C-A-A-D-D plus per-episode jitter.
        base = np.array([0, 0, 1, 1, 2, 2, 0, 0, 3, 3], dtype=np.int32)
        jitter = rng.integers(low=0, high=2, size=steps_per_episode)
        ep_labels = np.where(jitter == 1, np.roll(base, 1), base)
        success = bool(ep < num_episodes // 2)
        for t, lab in enumerate(ep_labels.tolist()):
            labels.append(int(lab))
            metadata.append({"rollout_idx": ep, "timestep": t, "success": success})
    return {"labels": np.asarray(labels, dtype=np.int32), "metadata": metadata}


def _behavior_graph_compute(inputs: Dict[str, Any]) -> Dict[str, Any]:
    from policy_doctor.behaviors.behavior_graph import BehaviorGraph

    graph = BehaviorGraph.from_cluster_assignments(
        inputs["labels"], inputs["metadata"], level="rollout"
    )
    return {"graph": graph}


def _graph_to_canonical_json(graph: Any) -> Dict[str, Any]:
    """Stable JSON-serializable representation of a BehaviorGraph.

    All dict keys converted to strings, all nested dicts sorted. Round-trip
    safe under json.dumps(sort_keys=True).
    """
    nodes = {
        str(node_id): {
            "cluster_id": node.cluster_id,
            "name": node.name,
            "num_timesteps": node.num_timesteps,
            "num_episodes": node.num_episodes,
            "episode_indices": sorted(int(i) for i in node.episode_indices),
        }
        for node_id, node in sorted(graph.nodes.items())
    }
    counts = {
        str(src): {str(dst): int(c) for dst, c in sorted(targets.items())}
        for src, targets in sorted(graph.transition_counts.items())
    }
    probs = {
        str(src): {str(dst): float(p) for dst, p in sorted(targets.items())}
        for src, targets in sorted(graph.transition_probs.items())
    }
    return {
        "nodes": nodes,
        "transition_counts": counts,
        "transition_probs": probs,
        "num_episodes": int(graph.num_episodes),
        "level": str(graph.level),
    }


def _behavior_graph_save(out: Dict[str, Any], dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    payload = _graph_to_canonical_json(out["graph"])
    (dst / "graph.json").write_text(json.dumps(payload, sort_keys=True, indent=2))


def _behavior_graph_load(dst: Path) -> Dict[str, Any]:
    return {"graph_json": json.loads((dst / "graph.json").read_text())}


def _behavior_graph_compare(actual: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    actual_json = _graph_to_canonical_json(actual["graph"])
    expected_json = expected["graph_json"]
    failures: List[str] = []
    if actual_json != expected_json:
        # Pinpoint the first differing top-level key for a useful message.
        for key in sorted(set(actual_json) | set(expected_json)):
            if actual_json.get(key) != expected_json.get(key):
                failures.append(f"behavior_graph.{key} differs")
                break
    return failures


# ---------------------------------------------------------------------------
# Anchor 3: influence matrix slicing
# ---------------------------------------------------------------------------


def _influence_slice_inputs() -> Dict[str, Any]:
    """Deterministic synthetic influence matrix and a slice request."""
    rng = np.random.default_rng(_SEED + 2)
    matrix = rng.normal(size=(30, 50)).astype(np.float32)
    # Three rollout episodes of 10 samples each; four demo episodes of 12-13.
    rollout_eps = [(0, 10), (10, 20), (20, 30)]
    demo_eps = [(0, 12), (12, 25), (25, 37), (37, 50)]
    slice_request = {"r_lo": 5, "r_hi": 22, "d_lo": 8, "d_hi": 40}
    return {
        "matrix": matrix,
        "rollout_eps": rollout_eps,
        "demo_eps": demo_eps,
        "slice": slice_request,
    }


def _influence_slice_compute(inputs: Dict[str, Any]) -> Dict[str, Any]:
    from policy_doctor.data.structures import GlobalInfluenceMatrix, EpisodeInfo

    rollout_eps = [
        EpisodeInfo(index=i, num_samples=hi - lo, sample_start_idx=lo, sample_end_idx=hi)
        for i, (lo, hi) in enumerate(inputs["rollout_eps"])
    ]
    demo_eps = [
        EpisodeInfo(index=i, num_samples=hi - lo, sample_start_idx=lo, sample_end_idx=hi)
        for i, (lo, hi) in enumerate(inputs["demo_eps"])
    ]
    gim = GlobalInfluenceMatrix(inputs["matrix"], rollout_eps, demo_eps)
    s = inputs["slice"]
    block = gim.get_slice(s["r_lo"], s["r_hi"], s["d_lo"], s["d_hi"])

    # Also pin two derived quantities to catch backing-store bugs.
    local = gim.get_local_matrix(rollout_trajectory_idx=1, demo_trajectory_idx=2)
    local_agg_demo = local.aggregate(axis=1, agg_fn="sum")
    global_agg_total = gim.aggregate(axis=None, agg_fn="mean")

    return {
        "slice": block,
        "local_aggregate_demo_axis": local_agg_demo,
        "global_aggregate_mean": global_agg_total,
    }


def _influence_slice_save(out: Dict[str, Any], dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    np.save(dst / "slice.npy", out["slice"])
    np.save(dst / "local_aggregate_demo_axis.npy", out["local_aggregate_demo_axis"])
    np.save(dst / "global_aggregate_mean.npy", out["global_aggregate_mean"])


def _influence_slice_load(dst: Path) -> Dict[str, Any]:
    return {
        "slice": np.load(dst / "slice.npy"),
        "local_aggregate_demo_axis": np.load(dst / "local_aggregate_demo_axis.npy"),
        "global_aggregate_mean": np.load(dst / "global_aggregate_mean.npy"),
    }


def _influence_slice_compare(actual: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    failures: List[str] = []
    for key in ("slice", "local_aggregate_demo_axis", "global_aggregate_mean"):
        if not np.array_equal(actual[key], expected[key]):
            max_abs = float(np.max(np.abs(actual[key] - expected[key])))
            failures.append(f"influence_slice.{key} differs (max |Δ| = {max_abs:.6g})")
    return failures


# ---------------------------------------------------------------------------
# Anchor registry and driver
# ---------------------------------------------------------------------------


_ANCHORS: Tuple[Tuple[str, Any, Any, Any, Any, Any], ...] = (
    (
        "clustering",
        _clustering_inputs,
        _clustering_compute,
        _clustering_save,
        _clustering_load,
        _clustering_compare,
    ),
    (
        "behavior_graph",
        _behavior_graph_inputs,
        _behavior_graph_compute,
        _behavior_graph_save,
        _behavior_graph_load,
        _behavior_graph_compare,
    ),
    (
        "influence_slice",
        _influence_slice_inputs,
        _influence_slice_compute,
        _influence_slice_save,
        _influence_slice_load,
        _influence_slice_compare,
    ),
)


def write_all(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {"seed": _SEED, "anchors": []}
    for name, inputs_fn, compute_fn, save_fn, _load_fn, _cmp_fn in _ANCHORS:
        print(f"[{name}] computing…", flush=True)
        outputs = compute_fn(inputs_fn())
        save_fn(outputs, out_dir / name)
        manifest["anchors"].append(name)
        print(f"[{name}] wrote {out_dir / name}", flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2))


def verify_all(out_dir: Path) -> int:
    failures: List[str] = []
    for name, inputs_fn, compute_fn, _save_fn, load_fn, cmp_fn in _ANCHORS:
        print(f"[{name}] verifying…", flush=True)
        actual = compute_fn(inputs_fn())
        expected = load_fn(out_dir / name)
        anchor_failures = cmp_fn(actual, expected)
        if anchor_failures:
            failures.extend(f"  - {f}" for f in anchor_failures)
            print(f"[{name}] FAIL ({len(anchor_failures)} difference(s))", flush=True)
        else:
            print(f"[{name}] ok", flush=True)
    if failures:
        print("\nVerification FAILED:", flush=True)
        for f in failures:
            print(f, flush=True)
        return 1
    print("\nAll anchors verified.", flush=True)
    return 0


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("write", "verify"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "tests" / "golden",
        help="Golden-fixture directory (default: <repo>/tests/golden)",
    )
    args = parser.parse_args(argv)
    if args.mode == "write":
        write_all(args.out)
        return 0
    return verify_all(args.out)


# TODO(phase-0-followup): Add a fourth anchor for MimicGen seed selection
# (BehaviorGraphPathHeuristic, DiversitySelectionHeuristic, RandomSelectionHeuristic).
# Requires synthesizing a rollouts HDF5 matching MimicGenSeedTrajectory.from_rollout_hdf5's
# expected schema (state/action/reward/done datasets + per-demo success attrs).

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
