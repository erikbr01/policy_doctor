"""Automated adherence scoring for VLM-proposed demonstrations (Experiment E2).

Three axes, weighted by ``ADHERENCE_WEIGHTS`` (initial_condition=0.25, cluster=0.5,
success=0.25), with a filter threshold of 0.6 on the overall weighted score.
Prohibition adherence is intentionally omitted from scoring; the ``prohibitions``
field on :class:`DemonstrationRequest` is operator-facing free text only.

The cluster axis is the load-bearing check: it classifies the new demonstration
against the saved behavior graph (same UMAP/k-means used during construction)
and verifies the resulting cluster path matches what the VLM requested. See
``E2_vlm_proposals.md`` sections 6.1-6.3 for the full design.
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from policy_doctor.vlm.proposals.init_state import verify_sim_state_replays
from policy_doctor.vlm.proposals.request import DemonstrationRequest

if TYPE_CHECKING:
    from policy_doctor.monitoring import TrajectoryClassifier


DEFAULT_WEIGHTS: Dict[str, float] = {
    "initial_condition": 0.25,
    "cluster": 0.5,
    "success": 0.25,
}
DEFAULT_FILTER_THRESHOLD: float = 0.6
_AXIS_NAMES: Tuple[str, ...] = ("initial_condition", "cluster", "success")


@dataclass
class AdherenceAxisScore:
    name: str
    score: float
    description: str
    evidence: Optional[Dict[str, Any]] = None


@dataclass
class AdherenceScore:
    request_id: str
    source_condition: Optional[str]
    axes: Dict[str, AdherenceAxisScore]
    overall: float
    passed_filter: bool
    weights: Dict[str, float]
    filter_threshold: float
    request_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Cluster path extraction
# ---------------------------------------------------------------------------


def classify_demo_pkl(
    classifier: "TrajectoryClassifier",
    pkl_path: Path,
) -> List[int]:
    """Classify a demo trajectory file and return the collapsed cluster path.

    Supports eval / DAgger pkl files and robomimic-compatible HDF5 files.
    Per-timestep cluster ids are pulled from
    ``MonitorResult.assignment.cluster_id``; ``-1`` (HDBSCAN noise) and rows
    whose assignment is ``None`` are excluded before the run-length collapse.
    """
    pkl_path = Path(pkl_path)
    if pkl_path.suffix in {".hdf5", ".h5"}:
        import h5py

        with h5py.File(pkl_path, "r") as f:
            demo = _first_hdf5_demo_group(f)
            results = classifier.classify_demo_from_hdf5(demo)
    else:
        df = pd.read_pickle(str(pkl_path))
        results = classifier.classify_episode_from_pkl(df)
    return _collapse_to_path(results)


def _first_hdf5_demo_group(hdf5_file):
    keys = sorted(k for k in hdf5_file["data"].keys() if k.startswith("demo_"))
    if not keys:
        raise KeyError("HDF5 file contains no data/demo_* groups")
    return hdf5_file["data"][keys[0]]


def _collapse_to_path(results: List[Tuple[int, Any]]) -> List[int]:
    path: List[int] = []
    last: Optional[int] = None
    for _t, mr in results:
        a = getattr(mr, "assignment", None)
        if a is None:
            continue
        cid = int(a.cluster_id)
        if cid < 0:
            continue
        if cid != last:
            path.append(cid)
            last = cid
    return path


# ---------------------------------------------------------------------------
# Per-axis scoring
# ---------------------------------------------------------------------------


def _score_initial_condition(
    request: DemonstrationRequest,
    demo_pkl: Path,
    reference_pkl_resolver: Optional[Callable[[str], Path]],
) -> AdherenceAxisScore:
    if reference_pkl_resolver is None:
        return AdherenceAxisScore(
            name="initial_condition",
            score=1.0,
            description="no resolver provided; sim init_state reset is bit-exact",
            evidence=None,
        )
    try:
        ref_pkl = Path(reference_pkl_resolver(request.initial_conditions.reference_rollout_id))
    except Exception as e:
        return AdherenceAxisScore(
            name="initial_condition",
            score=0.0,
            description=f"resolver failed: {type(e).__name__}: {e}",
            evidence={"reference_rollout_id": request.initial_conditions.reference_rollout_id},
        )

    if Path(demo_pkl).suffix in {".hdf5", ".h5"}:
        import h5py

        with h5py.File(demo_pkl, "r") as f:
            demo_first_state = np.asarray(_first_hdf5_demo_group(f)["states"][0], dtype=np.float64)
    else:
        demo_df = pd.read_pickle(str(demo_pkl))
        demo_first_state = np.asarray(demo_df.iloc[0]["sim_state"], dtype=np.float64)
    expected_frame = int(request.initial_conditions.reference_frame)

    try:
        ok = verify_sim_state_replays(
            ref_pkl,
            demo_first_state,
            expected_frame,
            atol=1e-6,
        )
    except (IndexError, KeyError, FileNotFoundError) as e:
        return AdherenceAxisScore(
            name="initial_condition",
            score=0.0,
            description=f"sim_state replay verification raised {type(e).__name__}: {e}",
            evidence={
                "reference_rollout_id": request.initial_conditions.reference_rollout_id,
                "reference_frame": expected_frame,
            },
        )

    return AdherenceAxisScore(
        name="initial_condition",
        score=1.0 if ok else 0.0,
        description=("sim_state matches reference frame within atol=1e-6"
                     if ok else "sim_state mismatch with reference frame"),
        evidence={
            "reference_rollout_id": request.initial_conditions.reference_rollout_id,
            "reference_frame": expected_frame,
            "matches": bool(ok),
        },
    )


def _jaccard_similarity(a: List[int], b: List[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _score_cluster(
    request: DemonstrationRequest,
    demo_path: List[int],
    reference_cluster_path: Optional[List[int]],
) -> AdherenceAxisScore:
    rt = request.request_type
    target = request.target_cluster

    if rt == "full_trajectory":
        if target is None:
            return AdherenceAxisScore(
                name="cluster",
                score=0.0,
                description="full_trajectory request has no target_cluster",
                evidence={"demo_path": demo_path},
            )
        in_path = int(target) in demo_path
        return AdherenceAxisScore(
            name="cluster",
            score=1.0 if in_path else 0.0,
            description=("target cluster present in demo path"
                         if in_path else "target cluster absent from demo path"),
            evidence={"demo_path": demo_path, "target": int(target), "in_path": in_path},
        )

    if rt == "alternative_strategy":
        if reference_cluster_path is None:
            return AdherenceAxisScore(
                name="cluster",
                score=0.0,
                description="alternative_strategy request requires reference_cluster_path",
                evidence={"demo_path": demo_path},
            )
        sim = _jaccard_similarity(demo_path, list(reference_cluster_path))
        score = max(0.0, min(1.0, 1.0 - sim))
        return AdherenceAxisScore(
            name="cluster",
            score=float(score),
            description=f"alternative_strategy: 1 - jaccard(demo, reference) = {score:.3f}",
            evidence={
                "demo_path": demo_path,
                "reference_path": list(reference_cluster_path),
                "jaccard": sim,
            },
        )

    return AdherenceAxisScore(
        name="cluster",
        score=0.0,
        description=f"unknown request_type {rt!r}",
        evidence={"demo_path": demo_path},
    )


def _score_cluster_recovery(
    request: DemonstrationRequest,
    demo_path: List[int],
    success: bool,
) -> AdherenceAxisScore:
    target = request.target_cluster
    if target is None:
        return AdherenceAxisScore(
            name="cluster",
            score=0.0,
            description="recovery request has no target_cluster",
            evidence={"demo_path": demo_path, "success": bool(success)},
        )
    starts_in_target = bool(demo_path) and demo_path[0] == int(target)
    if starts_in_target and success:
        score, desc = 1.0, "starts in target cluster and reaches success"
    elif starts_in_target or success:
        score, desc = 0.5, "partial recovery: only one of {start-in-target, success}"
    else:
        score, desc = 0.0, "neither starts in target cluster nor reaches success"
    return AdherenceAxisScore(
        name="cluster",
        score=score,
        description=desc,
        evidence={
            "demo_path": demo_path,
            "target": int(target),
            "starts_in_target": bool(starts_in_target),
            "success": bool(success),
        },
    )


# ---------------------------------------------------------------------------
# Success extraction
# ---------------------------------------------------------------------------


def _resolve_success(
    demo_pkl: Path,
    success_arg: Optional[bool],
) -> Tuple[bool, str]:
    if success_arg is not None:
        return bool(success_arg), "from caller"
    demo_pkl = Path(demo_pkl)
    if demo_pkl.suffix in {".hdf5", ".h5"}:
        import h5py

        with h5py.File(demo_pkl, "r") as f:
            demo = _first_hdf5_demo_group(f)
            if "success" in demo.attrs:
                return bool(demo.attrs["success"]), "from HDF5 demo attr 'success'"
            if "dones" in demo and len(demo["dones"]) > 0:
                return bool(demo["dones"][-1]), "from HDF5 final done"
    df = pd.read_pickle(str(demo_pkl))
    if "success" in df.columns:
        return bool(df["success"].iloc[-1]), "from demo_pkl 'success' column"
    attrs = getattr(df, "attrs", None) or {}
    if "success" in attrs:
        return bool(attrs["success"]), "from demo_pkl df.attrs['success']"
    warnings.warn(
        f"adherence: demo {demo_pkl} has no 'success' column or attr; defaulting to False",
        RuntimeWarning,
        stacklevel=3,
    )
    return False, "default False (no success info on demo)"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_request_adherence(
    request: DemonstrationRequest,
    demo_pkl: Path,
    classifier: "TrajectoryClassifier",
    *,
    reference_cluster_path: Optional[List[int]] = None,
    success: Optional[bool] = None,
    weights: Optional[Dict[str, float]] = None,
    filter_threshold: float = DEFAULT_FILTER_THRESHOLD,
    reference_pkl_resolver: Optional[Callable[[str], Path]] = None,
) -> AdherenceScore:
    """Score one demonstration against its request on the three E2 axes.

    See module docstring for axis definitions and weights. ``success`` defaults
    to a fallback chain that reads the demo pkl when the caller did not supply
    an explicit outcome.
    """
    demo_pkl = Path(demo_pkl)
    weights = dict(weights) if weights is not None else dict(DEFAULT_WEIGHTS)

    success_value, success_source = _resolve_success(demo_pkl, success)

    axis_init = _score_initial_condition(request, demo_pkl, reference_pkl_resolver)

    demo_path = classify_demo_pkl(classifier, demo_pkl)
    if request.request_type == "recovery":
        axis_cluster = _score_cluster_recovery(request, demo_path, success_value)
    else:
        axis_cluster = _score_cluster(request, demo_path, reference_cluster_path)

    axis_success = AdherenceAxisScore(
        name="success",
        score=1.0 if success_value else 0.0,
        description=f"task success={success_value} ({success_source})",
        evidence={"success": bool(success_value), "source": success_source},
    )

    axes = {
        "initial_condition": axis_init,
        "cluster": axis_cluster,
        "success": axis_success,
    }
    overall = float(sum(weights[k] * axes[k].score for k in _AXIS_NAMES))
    return AdherenceScore(
        request_id=request.request_id,
        source_condition=request.source_condition,
        axes=axes,
        overall=overall,
        passed_filter=overall >= filter_threshold,
        weights=weights,
        filter_threshold=filter_threshold,
        request_type=request.request_type,
    )


# ---------------------------------------------------------------------------
# Batch scoring + JSONL output
# ---------------------------------------------------------------------------


def _score_record_to_dict(score: AdherenceScore, demo_pkl: Path) -> Dict[str, Any]:
    rec = score.to_dict()
    rec["demo_pkl"] = str(demo_pkl)
    return rec


def _bucket_summary(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n_total": 0, "n_passed": 0})
    for r in rows:
        k = r.get(key)
        if k is None:
            k = "unknown"
        buckets[str(k)]["n_total"] += 1
        if r.get("passed_filter"):
            buckets[str(k)]["n_passed"] += 1
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in buckets.items():
        n_total = v["n_total"]
        n_passed = v["n_passed"]
        out[k] = {
            "n_total": n_total,
            "n_passed": n_passed,
            "n_failed": n_total - n_passed,
            "pass_rate": (n_passed / n_total) if n_total else 0.0,
        }
    return out


def score_batch_to_jsonl(
    pairs: List[Tuple[DemonstrationRequest, Path]],
    classifier: "TrajectoryClassifier",
    output_dir: Path,
    *,
    reference_cluster_paths: Optional[Dict[str, List[int]]] = None,
    weights: Optional[Dict[str, float]] = None,
    filter_threshold: float = DEFAULT_FILTER_THRESHOLD,
    reference_pkl_resolver: Optional[Callable[[str], Path]] = None,
) -> Dict[str, Any]:
    """Score every (request, demo_pkl) pair and write E2 adherence artifacts.

    Writes ``per_demo_scores.jsonl`` (all demos), ``filtered_demos.jsonl``
    (subset with ``overall >= filter_threshold``), and ``filter_summary.json``
    (totals and per-condition / per-request_type breakdowns).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = dict(weights) if weights is not None else dict(DEFAULT_WEIGHTS)
    reference_cluster_paths = reference_cluster_paths or {}

    per_demo_path = output_dir / "per_demo_scores.jsonl"
    filtered_path = output_dir / "filtered_demos.jsonl"
    summary_path = output_dir / "filter_summary.json"

    rows: List[Dict[str, Any]] = []
    with per_demo_path.open("w") as f_all, filtered_path.open("w") as f_pass:
        for request, demo_pkl in pairs:
            ref_path = None
            if request.request_type == "alternative_strategy":
                ref_path = reference_cluster_paths.get(
                    request.initial_conditions.reference_rollout_id
                )
            score = score_request_adherence(
                request,
                demo_pkl,
                classifier,
                reference_cluster_path=ref_path,
                weights=weights,
                filter_threshold=filter_threshold,
                reference_pkl_resolver=reference_pkl_resolver,
            )
            rec = _score_record_to_dict(score, Path(demo_pkl))
            rows.append(rec)
            line = json.dumps(rec, default=str) + "\n"
            f_all.write(line)
            if score.passed_filter:
                f_pass.write(line)

    n_total = len(rows)
    n_passed = sum(1 for r in rows if r.get("passed_filter"))
    summary: Dict[str, Any] = {
        "n_total": n_total,
        "n_passed": n_passed,
        "n_failed": n_total - n_passed,
        "pass_rate": (n_passed / n_total) if n_total else 0.0,
        "by_condition": _bucket_summary(rows, "source_condition"),
        "by_request_type": _bucket_summary(rows, "request_type"),
        "weights": weights,
        "filter_threshold": filter_threshold,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


__all__ = [
    "AdherenceAxisScore",
    "AdherenceScore",
    "DEFAULT_FILTER_THRESHOLD",
    "DEFAULT_WEIGHTS",
    "classify_demo_pkl",
    "score_batch_to_jsonl",
    "score_request_adherence",
]
