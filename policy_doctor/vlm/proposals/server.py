"""HTTP proposal server for Experiment E2.

Owns the rollout pool, behavior graph, request queue, chat history, and adherence
scoring for one E2 run. Streamlit console + sim runner are both stateless HTTP
clients of this server.

Endpoints (all JSON unless noted):

  GET  /health                       liveness + capability flags
  GET  /pool                         RolloutPool index
  GET  /graph                        BehaviorGraph artefact paths + node summary
  POST /propose                      generate a fresh batch (cold conversation)
                                     body: {"condition": "graph"|"outcome_only", ...}
  POST /chat                         single chat turn against the proposal model
                                     body: {"text": "...", "condition": ...}
                                     405 when chat_enabled=false in config
  GET  /requests                     full queue
  GET  /requests/active              the next pending request, OPERATOR-VIEW only
                                     (target_cluster + source_condition stripped)
  POST /requests/{id}/result         attach a demo pkl + success flag to a request
                                     body: {"demo_pkl": "...", "success": bool}
  GET  /adherence/{id}               cached adherence score for a completed request
  GET  /adherence                    all adherence scores

Wire format: JSON bodies, no raw bytes (all heavy data lives on disk).

Start:
    python -m policy_doctor.envs.proposal_server \
        --config policy_doctor/configs/e2/defaults.yaml \
        --port 5003

Sim runner pulls /requests/active and POSTs /requests/{id}/result; Streamlit
console reads /requests, /pool, /graph and POSTs /chat or /propose.
"""

from __future__ import annotations

import argparse
import json
import logging
import secrets
import shutil
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


_ROOT = Path(__file__).resolve().parent.parent.parent
_CUPID = _ROOT / "third_party" / "cupid"
for _p in [str(_ROOT), str(_CUPID)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------


@dataclass
class _RequestRecord:
    """Server-side record per request. The condition + target_cluster live HERE
    and are stripped before any operator-facing endpoint returns the request."""

    request: Any                          # DemonstrationRequest
    status: str = "pending"               # pending | active | completed | failed
    queue_position: int = -1
    demo_pkl: Optional[Path] = None
    success: Optional[bool] = None
    adherence: Optional[Any] = None       # AdherenceScore once scored
    error: Optional[str] = None


@dataclass
class _ServerState:
    """All mutable state for the proposal server. Guarded by ``lock``."""

    run_dir: Path
    pool: Any                            # RolloutPool
    behavior_graph: Any                  # BehaviorGraph
    classifier: Any = None               # TrajectoryClassifier (for adherence)
    reference_cluster_paths: Dict[str, List[int]] = field(default_factory=dict)
    # Per-slice cluster labels + metadata used by Layer 2 agent tools.
    cluster_labels: Any = None
    cluster_metadata: Any = None

    chat_enabled: bool = False

    # Queue: list of request_ids in operator presentation order (shuffled across conditions)
    queue: List[str] = field(default_factory=list)
    requests: Dict[str, _RequestRecord] = field(default_factory=dict)
    active_idx: int = 0

    chat_history: List[Dict[str, Any]] = field(default_factory=list)

    # VLM lifecycle: if 'unload', the backend is moved to CPU between /propose calls
    vlm_lifecycle: str = "persistent"      # persistent | unload

    lock: threading.RLock = field(default_factory=threading.RLock)


# Build at startup; mutated by handlers under .lock
_STATE: Optional[_ServerState] = None
_BACKEND: Any = None                       # VLMBackend instance (proposal generator)
_AGENT_BACKEND: Any = None                 # VLMBackend instance for the agentic loop
_BACKEND_DEVICE: str = "cpu"               # for unload mode
_CFG: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> float:
    return time.time()


def _shuffle_queue(state: _ServerState, seed: Optional[int] = None) -> None:
    """Shuffle the queue across conditions in place.

    Uses ``secrets`` when seed is None, ``random.Random(seed)`` otherwise so the
    presentation order is reproducible if pre-registered.
    """
    if not state.queue:
        return
    if seed is None:
        rnd = secrets.SystemRandom()
    else:
        import random
        rnd = random.Random(seed)
    rnd.shuffle(state.queue)
    for i, rid in enumerate(state.queue):
        state.requests[rid].queue_position = i


def _move_backend(target_device: str) -> None:
    """Move the VLM backend's torch model between cpu and gpu, if applicable."""
    global _BACKEND_DEVICE
    if target_device == _BACKEND_DEVICE:
        return
    model = getattr(_BACKEND, "_model", None) or getattr(_BACKEND, "model", None)
    if model is not None and hasattr(model, "to"):
        try:
            model.to(target_device)
            _BACKEND_DEVICE = target_device
        except Exception as e:
            logging.warning("backend .to(%s) failed: %s", target_device, e)


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------


def _make_app():
    from flask import Flask, jsonify, request

    app = Flask(__name__)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    # ---- introspection ---------------------------------------------------

    @app.get("/health")
    def health():
        with _STATE.lock:
            return jsonify(
                status="ok",
                run_dir=str(_STATE.run_dir),
                chat_enabled=_STATE.chat_enabled,
                vlm_lifecycle=_STATE.vlm_lifecycle,
                backend_device=_BACKEND_DEVICE,
                n_requests=len(_STATE.requests),
                n_pending=sum(1 for r in _STATE.requests.values() if r.status == "pending"),
                pool_size=len(_STATE.pool) if _STATE.pool is not None else 0,
            )

    @app.get("/pool")
    def get_pool():
        with _STATE.lock:
            return jsonify(_STATE.pool.to_index_dict())

    @app.get("/graph")
    def get_graph():
        with _STATE.lock:
            g = _STATE.behavior_graph
            return jsonify(
                level=getattr(g, "level", "rollout"),
                num_episodes=getattr(g, "num_episodes", 0),
                node_summary={
                    int(nid): {
                        "name": node.name,
                        "n_episodes": node.num_episodes,
                        "n_timesteps": node.num_timesteps,
                    }
                    for nid, node in g.nodes.items()
                },
            )

    @app.get("/storyboards/<sid>")
    def get_storyboard(sid):
        """Render a slice or rollout id to a JPEG. Used by the operator console
        to show the imagery the agent grounded its request on (the request's
        ``reference_storyboard_ids``). Slice ids are ``{rid}::{start}::{end}``;
        rollout ids are bare. The endpoint deliberately does NOT distinguish
        the two condition-side names — both render the same way."""
        import io
        from flask import Response, abort

        from policy_doctor.vlm.proposals.agents.tools.access import (
            _render_slice_storyboard,
            parse_slice_id,
        )

        with _STATE.lock:
            pool = _STATE.pool
        if pool is None:
            abort(503, "pool unavailable")

        parsed = parse_slice_id(sid)
        try:
            if parsed is not None:
                rid, start, end = parsed
                entry = pool.by_id(rid)
                img = _render_slice_storyboard(entry.episode_pkl, start, end)
            else:
                entry = pool.by_id(sid)
                img = _render_slice_storyboard(entry.episode_pkl, 0, max(entry.length - 1, 0))
        except KeyError:
            abort(404, f"unknown id {sid}")
        if img is None:
            abort(500, f"could not render {sid}")

        max_dim = 384
        img = img.convert("RGB")
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            img = img.resize(
                (int(img.size[0] * ratio), int(img.size[1] * ratio)),
            )
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return Response(buf.getvalue(), mimetype="image/jpeg")

    # ---- proposal generation --------------------------------------------

    @app.post("/propose")
    def propose():
        from policy_doctor.vlm.proposals.propose import generate_proposals

        body = request.get_json(force=True, silent=True) or {}
        condition = body.get("condition")
        if condition not in ("graph", "outcome_only"):
            return jsonify(error="condition must be 'graph' or 'outcome_only'"), 400

        n_per_type = body.get("n_requests_per_type") or _CFG.get(
            "n_requests_per_type", {"full_trajectory": 4, "recovery": 2, "alternative_strategy": 2}
        )
        n_repetitions = int(body.get("n_repetitions") or _CFG.get("n_repetitions", 2))
        max_retries = int(body.get("max_retries") or _CFG.get("max_retries", 3))
        temperature = float(body.get("temperature") or _CFG.get("temperature", 0.3))
        seed = body.get("seed") if "seed" in body else _CFG.get("base_seed", 42)
        task_hint = body.get("task_hint") or _CFG.get("task_hint", "")
        graph_repr_name = body.get("graph_representation") or _CFG.get(
            "graph_representation", "combined"
        )
        builder_name = body.get("vlm_input_builder") or None  # None → default per-condition

        if _STATE.vlm_lifecycle == "unload":
            target_dev = _CFG.get("vlm_device", "cuda:0")
            _move_backend(target_dev)

        out_dir = _STATE.run_dir / "proposals" / condition
        try:
            result = generate_proposals(
                backend=_BACKEND,
                pool=_STATE.pool,
                behavior_graph=_STATE.behavior_graph,
                condition=condition,
                n_requests_per_type=n_per_type,
                output_dir=out_dir,
                n_repetitions=n_repetitions,
                temperature=temperature,
                base_seed=seed,
                max_retries=max_retries,
                task_hint=task_hint,
                graph_representation_name=graph_repr_name,
                vlm_input_builder_name=builder_name,
            )
        finally:
            if _STATE.vlm_lifecycle == "unload":
                _move_backend("cpu")

        with _STATE.lock:
            for req in result.selected_requests:
                rec = _RequestRecord(request=req)
                _STATE.requests[req.request_id] = rec
                _STATE.queue.append(req.request_id)
            _shuffle_queue(_STATE, seed=_CFG.get("queue_shuffle_seed"))

        return jsonify(
            condition=condition,
            n_added=len(result.selected_requests),
            consistency_metrics=result.consistency_metrics,
            output_dir=str(out_dir),
        )

    # ---- chat (gated by chat_enabled) -----------------------------------

    @app.post("/chat")
    def chat():
        if not _STATE.chat_enabled:
            return (
                jsonify(error="chat is disabled by server config (chat_enabled=false)"),
                405,
            )
        body = request.get_json(force=True, silent=True) or {}
        text = body.get("text") or ""
        if not text:
            return jsonify(error="text is required"), 400

        from policy_doctor.vlm.proposals.vlm_input.base import Message

        with _STATE.lock:
            history = list(_STATE.chat_history)
        msgs = [
            Message(role=h["role"], text_blocks=[h["text"]])
            for h in history
        ] + [Message(role="user", text_blocks=[text])]

        if _STATE.vlm_lifecycle == "unload":
            _move_backend(_CFG.get("vlm_device", "cuda:0"))
        try:
            if hasattr(_BACKEND, "generate"):
                reply = _BACKEND.generate(msgs, temperature=0.5, seed=None)
            else:
                reply = _BACKEND.describe_slice(
                    images=[], system_prompt=None, user_prompt=text
                )
        finally:
            if _STATE.vlm_lifecycle == "unload":
                _move_backend("cpu")

        ts = _now()
        with _STATE.lock:
            _STATE.chat_history.append({"role": "user", "text": text, "ts": ts})
            _STATE.chat_history.append({"role": "assistant", "text": reply, "ts": _now()})
            _persist_chat()

        return jsonify(role="assistant", text=reply)

    # ---- request queue ---------------------------------------------------

    @app.get("/requests")
    def list_requests():
        with _STATE.lock:
            return jsonify(
                queue=[
                    {
                        "request_id": rid,
                        "status": _STATE.requests[rid].status,
                        "queue_position": _STATE.requests[rid].queue_position,
                    }
                    for rid in _STATE.queue
                ]
            )

    @app.get("/requests/active")
    def get_active():
        with _STATE.lock:
            for rid in _STATE.queue:
                rec = _STATE.requests[rid]
                if rec.status == "pending":
                    rec.status = "active"
                    return jsonify(rec.request.to_operator_dict())
            return jsonify({}), 204  # no pending requests left

    @app.post("/requests/<rid>/result")
    def post_result(rid):
        body = request.get_json(force=True, silent=True) or {}
        demo_pkl = body.get("demo_pkl")
        success = body.get("success")
        if not demo_pkl:
            return jsonify(error="demo_pkl is required"), 400
        with _STATE.lock:
            if rid not in _STATE.requests:
                return jsonify(error=f"unknown request_id {rid}"), 404
            rec = _STATE.requests[rid]
            rec.demo_pkl = Path(demo_pkl)
            rec.success = bool(success) if success is not None else None
            rec.status = "completed"
        _maybe_score(rid)
        with _STATE.lock:
            adh = rec.adherence
        return jsonify(
            request_id=rid,
            scored=adh is not None,
            success=rec.success,
            initial_condition_score=(adh.axes["initial_condition"].score if adh else None),
            cluster_score=(adh.axes["cluster"].score if adh else None),
            success_score=(adh.axes["success"].score if adh else None),
            overall=(adh.overall if adh else None),
            passed_filter=(adh.passed_filter if adh else None),
        )

    @app.get("/adherence/<rid>")
    def get_adherence(rid):
        with _STATE.lock:
            if rid not in _STATE.requests:
                return jsonify(error=f"unknown request_id {rid}"), 404
            rec = _STATE.requests[rid]
            adh = rec.adherence
        if adh is None:
            return jsonify(scored=False), 200
        return jsonify(_adherence_to_jsonable(adh))

    @app.get("/adherence")
    def list_adherence():
        with _STATE.lock:
            out = []
            for rid in _STATE.queue:
                rec = _STATE.requests[rid]
                if rec.adherence is None:
                    continue
                out.append(_adherence_to_jsonable(rec.adherence))
        return jsonify(scores=out)

    # ---- human-condition request submission ----------------------------

    @app.post("/human_request")
    def human_request():
        """Enqueue one DemonstrationRequest submitted by an operator (H_NG / H_G).

        Body: ``{"condition": "H_NG"|"H_G", "request": {...}, "rationale": "..."}``.
        The request is validated against the same denylist + rollout-id check
        the agent path uses, so leaks and bad references are rejected uniformly.
        """
        from policy_doctor.vlm.proposals.request import (
            DemonstrationRequest,
            RequestValidationError,
            validate_request,
        )

        body = request.get_json(force=True, silent=True) or {}
        cond = str(body.get("condition") or "")
        req_dict = body.get("request") or {}
        rationale = str(body.get("rationale") or "").strip()

        if cond not in ("H_NG", "H_G"):
            return jsonify(error="condition must be 'H_NG' or 'H_G'"), 400
        if not rationale:
            return jsonify(error="rationale is required"), 400

        try:
            req = DemonstrationRequest.from_dict(req_dict)
            req.source_condition = cond
            validate_request(req, allowed_rollout_ids=set(_STATE.pool.rollout_ids))
        except (KeyError, TypeError, RequestValidationError) as e:
            return jsonify(error=f"invalid request: {e}"), 400

        with _STATE.lock:
            _STATE.requests[req.request_id] = _RequestRecord(request=req)
            _STATE.queue.append(req.request_id)
            _shuffle_queue(_STATE, seed=_CFG.get("queue_shuffle_seed"))

        # Persist rationale alongside the operator session log so the human and
        # agent paths produce comparable artefacts.
        log_path = _STATE.run_dir / "human_sessions" / cond / "submissions.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "request_id": req.request_id,
                "rationale": rationale,
                "request": req.to_dict(),
                "ts": _now(),
            }, default=str) + "\n")

        return jsonify(request_id=req.request_id, n_in_queue=len(_STATE.queue))

    # ---- agentic proposal session ---------------------------------------

    @app.post("/agent_session")
    def agent_session():
        """Run one agent session and enqueue its submitted requests.

        Body: ``{"condition": "A_G"|"A_NG"|"H_G"|"H_NG", "seed": int}``.
        Returns trace artefact paths and a session summary. Heavy work runs
        synchronously inside the request handler — the operator GUI shows a
        spinner. The session may take several minutes for real backends.
        """
        body = request.get_json(force=True, silent=True) or {}
        cond = str(body.get("condition") or "")
        seed = int(body.get("seed", 0))
        if not cond:
            return jsonify(error="condition is required"), 400
        if _AGENT_BACKEND is None:
            return jsonify(error="agent backend not configured on this server"), 400

        from policy_doctor.vlm.proposals.agents.budget import BudgetConfig
        from policy_doctor.vlm.proposals.agents.run import run_one_session

        budget = BudgetConfig.from_dict(_CFG.get("agentic_budget") or {})
        out_dir = _STATE.run_dir / "agent_sessions" / cond / f"seed_{seed}"

        result = run_one_session(
            condition=cond,
            seed=seed,
            backend=_AGENT_BACKEND,
            graph=_STATE.behavior_graph,
            pool=_STATE.pool,
            out_dir=out_dir,
            budget_config=budget,
            max_turns=int(_CFG.get("agentic_max_turns", 100)),
            temperature=float(_CFG.get("agentic_temperature", 0.3)),
            max_tokens=int(_CFG.get("agentic_max_tokens", 4096)),
            cluster_labels=_STATE.cluster_labels,
            cluster_metadata=_STATE.cluster_metadata,
            raw_states_dir=_CFG.get("raw_states_dir"),
            storyboards_dir=_CFG.get("storyboard_dir"),
            videos_dir=_CFG.get("video_dir"),
            task_hint=_CFG.get("task_hint", ""),
            kinematic_summary_strategy=_CFG.get("agentic_kin_strategy", "raw_states"),
            cache_enabled=bool(_CFG.get("agentic_cache_enabled", True)),
        )

        # Enqueue the agent's submitted requests on the existing operator queue.
        with _STATE.lock:
            from policy_doctor.vlm.proposals.request import DemonstrationRequest

            n_added = 0
            for sr in result.submitted_requests:
                req_dict = sr["request"] if isinstance(sr, dict) else sr
                req = DemonstrationRequest.from_dict(req_dict)
                if req.request_id in _STATE.requests:
                    continue
                _STATE.requests[req.request_id] = _RequestRecord(request=req)
                _STATE.queue.append(req.request_id)
                n_added += 1
            _shuffle_queue(_STATE, seed=_CFG.get("queue_shuffle_seed"))

        return jsonify(
            condition=cond,
            seed=seed,
            stop_reason=result.stop_reason,
            n_submitted=len(result.submitted_requests),
            n_enqueued=n_added,
            out_dir=str(out_dir),
            budget_summary=result.budget_summary,
        )

    return app


# ---------------------------------------------------------------------------
# Adherence scoring shim
# ---------------------------------------------------------------------------


def _maybe_score(rid: str) -> None:
    """Run adherence scoring on a completed request, if classifier is available."""
    with _STATE.lock:
        rec = _STATE.requests.get(rid)
        if rec is None or rec.adherence is not None:
            return
        if _STATE.classifier is None or rec.demo_pkl is None:
            return
    try:
        from policy_doctor.vlm.proposals.adherence import score_request_adherence
        adh = score_request_adherence(
            request=rec.request,
            demo_pkl=rec.demo_pkl,
            classifier=_STATE.classifier,
            reference_cluster_path=_STATE.reference_cluster_paths.get(
                rec.request.initial_conditions.reference_rollout_id
            ),
            success=rec.success,
            reference_pkl_resolver=lambda rid: _STATE.pool.by_id(rid).episode_pkl,
        )
    except Exception as e:
        logging.exception("adherence scoring failed for %s", rid)
        with _STATE.lock:
            rec.error = f"{type(e).__name__}: {e}"
        return
    with _STATE.lock:
        rec.adherence = adh


def _adherence_to_jsonable(adh) -> Dict[str, Any]:
    return {
        "request_id": adh.request_id,
        "source_condition": adh.source_condition,
        "axes": {
            name: {
                "score": axis.score,
                "description": axis.description,
                "evidence": axis.evidence,
            }
            for name, axis in adh.axes.items()
        },
        "overall": adh.overall,
        "passed_filter": adh.passed_filter,
        "weights": adh.weights,
        "filter_threshold": adh.filter_threshold,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _persist_chat() -> None:
    """Append-only chat log under run_dir/chat.jsonl."""
    if _STATE.chat_history:
        path = _STATE.run_dir / "chat.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for entry in _STATE.chat_history:
                f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def boot(
    *,
    run_dir: Path,
    pool_episodes_dir: Path,
    clustering_dir: Path,
    chat_enabled: bool = False,
    vlm_backend_name: str = "mock",
    vlm_backend_params: Optional[Dict[str, Any]] = None,
    vlm_lifecycle: str = "persistent",
    storyboard_dir: Optional[Path] = None,
    video_dir: Optional[Path] = None,
    classifier_kwargs: Optional[Dict[str, Any]] = None,
    # ---- agentic mode ---------------------------------------------------
    agent_backend_name: Optional[str] = None,
    agent_backend_params: Optional[Dict[str, Any]] = None,
    agentic_cfg: Optional[Dict[str, Any]] = None,
    raw_states_dir: Optional[Path] = None,
    task_hint: Optional[str] = None,
) -> None:
    """Build server state. Call before running the Flask app."""
    global _STATE, _BACKEND, _AGENT_BACKEND, _BACKEND_DEVICE, _CFG

    from policy_doctor.behaviors.behavior_graph import BehaviorGraph
    from policy_doctor.data.clustering_loader import load_clustering_result_from_path
    from policy_doctor.vlm import get_vlm_backend
    from policy_doctor.vlm.proposals.pool import RolloutPool

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    pool = RolloutPool.from_episodes_dir(
        Path(pool_episodes_dir),
        storyboard_dir=Path(storyboard_dir) if storyboard_dir else None,
        video_dir=Path(video_dir) if video_dir else None,
    )

    labels, metadata, manifest = load_clustering_result_from_path(Path(clustering_dir))
    graph = BehaviorGraph.from_cluster_assignments(
        labels, metadata, level=manifest.get("level", "rollout")
    )

    backend = get_vlm_backend(vlm_backend_name, vlm_backend_params or {})
    _BACKEND = backend
    _BACKEND_DEVICE = (
        (vlm_backend_params or {}).get("device", "cpu") if vlm_lifecycle == "persistent" else "cpu"
    )

    if agent_backend_name:
        _AGENT_BACKEND = get_vlm_backend(agent_backend_name, agent_backend_params or {})

    classifier = None
    if classifier_kwargs:
        try:
            from policy_doctor.monitoring.trajectory_classifier import TrajectoryClassifier
            classifier = TrajectoryClassifier.from_checkpoint(**classifier_kwargs)
        except Exception as e:
            logging.warning("Could not build TrajectoryClassifier: %s — adherence disabled", e)

    _STATE = _ServerState(
        run_dir=run_dir,
        pool=pool,
        behavior_graph=graph,
        classifier=classifier,
        chat_enabled=chat_enabled,
        vlm_lifecycle=vlm_lifecycle,
        cluster_labels=labels,
        cluster_metadata=metadata,
    )
    agentic_cfg = agentic_cfg or {}
    _CFG = {
        "vlm_device": (vlm_backend_params or {}).get("device", "cuda:0"),
        "raw_states_dir": Path(raw_states_dir) if raw_states_dir else None,
        "storyboard_dir": Path(storyboard_dir) if storyboard_dir else None,
        "video_dir": Path(video_dir) if video_dir else None,
        "task_hint": task_hint or "",
        "agentic_budget": agentic_cfg.get("budget"),
        "agentic_max_turns": agentic_cfg.get("max_turns", 100),
        "agentic_temperature": agentic_cfg.get("temperature", 0.3),
        "agentic_max_tokens": agentic_cfg.get("max_tokens", 4096),
        "agentic_kin_strategy": agentic_cfg.get("kinematic_summary_strategy", "raw_states"),
        "agentic_cache_enabled": agentic_cfg.get("cache_enabled", True),
    }


def serve(host: str = "127.0.0.1", port: int = 5003) -> None:
    app = _make_app()
    app.run(host=host, port=port, threaded=True, use_reloader=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2 proposal HTTP server")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=5003)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    cfg = _read_yaml(args.config)
    agentic = cfg.get("agentic") or {}
    boot(
        run_dir=Path(cfg["run_dir"]),
        pool_episodes_dir=Path(cfg["pool_episodes_dir"]),
        clustering_dir=Path(cfg["clustering_dir"]),
        chat_enabled=bool(cfg.get("chat_enabled", False)),
        vlm_backend_name=cfg.get("vlm_backend", "mock"),
        vlm_backend_params=cfg.get("vlm_backend_params") or {},
        vlm_lifecycle=cfg.get("vlm_lifecycle", "persistent"),
        storyboard_dir=cfg.get("storyboard_dir"),
        video_dir=cfg.get("video_dir"),
        classifier_kwargs=cfg.get("classifier_kwargs"),
        agent_backend_name=agentic.get("agent_backend"),
        agent_backend_params=agentic.get("agent_backend_params"),
        agentic_cfg=agentic,
        raw_states_dir=cfg.get("raw_states_dir"),
        task_hint=cfg.get("task_hint"),
    )
    serve(host=args.host, port=args.port)
