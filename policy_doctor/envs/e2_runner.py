"""E2 sim runner shim: pulls a DemonstrationRequest, executes one demo, posts back.

Wraps the existing :class:`policy_doctor.envs.RobomimicDAggerRunner`. The runner
itself is unchanged; this script handles:

  - polling ``GET /requests/active`` on the proposal server
  - resetting the sim env to the request's reference rollout ``init_state``
    (frame 0 → rollout start; >0 → mid-rollout state via :mod:`init_state`)
  - executing one episode with the operator
  - posting the saved demo pkl back via ``POST /requests/{id}/result``

CRITICAL: nothing in this script may print or log the request's
``source_condition`` or ``target_cluster``; those fields aren't sent in the
operator-view payload anyway, but if you read the raw request elsewhere don't
echo them.

Run from third_party/cupid/ in the cupid env:

    python ../../scripts/run_e2_sim.py task=square_mh \
        train_dir=/path/to/train_dir \
        proposal_server=http://localhost:5003 \
        viz_url=http://localhost:5002 \
        output_dir=/tmp/e2_demos
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import requests


_ROOT = Path(__file__).resolve().parent.parent.parent
_CUPID = _ROOT / "third_party" / "cupid"
for _p in [str(_ROOT), str(_CUPID)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Server client
# ---------------------------------------------------------------------------


class ProposalServerClient:
    """Thin sync HTTP client; no condition-leaking logging."""

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")

    def health(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.url}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def get_active(self) -> Optional[Dict[str, Any]]:
        resp = requests.get(f"{self.url}/requests/active", timeout=10)
        if resp.status_code == 204:
            return None
        resp.raise_for_status()
        return resp.json() or None

    def post_result(self, request_id: str, *, demo_pkl: Path, success: bool) -> Dict[str, Any]:
        resp = requests.post(
            f"{self.url}/requests/{request_id}/result",
            json={"demo_pkl": str(demo_pkl), "success": bool(success)},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Init-state extraction (cupid env can read pickles via pandas just fine)
# ---------------------------------------------------------------------------


def _resolve_reference_pkl(server_pool_index: Dict[str, Any], rollout_id: str) -> Path:
    for entry in server_pool_index.get("rollouts", []):
        if entry["rollout_id"] == rollout_id:
            episodes_dir = Path(server_pool_index["episodes_dir"])
            stem = f"ep{entry['episode_idx']:04d}"
            unsuffixed = episodes_dir / f"{stem}.pkl"
            if unsuffixed.exists():
                return unsuffixed
            matches = sorted(episodes_dir.glob(f"{stem}_*.pkl"))
            if matches:
                return matches[0]
            return unsuffixed
    raise KeyError(f"rollout_id {rollout_id!r} not in server pool")


def _init_state_for_request(
    server_url: str,
    request: Dict[str, Any],
) -> Optional[np.ndarray]:
    """Pull the reference rollout pkl path from the server's pool index, then
    extract the recorded ``sim_state`` at the request's reference_frame.

    Returns ``None`` when the referenced pkl was produced by
    ``eval_save_episodes`` (which does NOT record ``sim_state``) — in that
    case the caller falls back to ``env.reset()`` which gives a
    rollout-start state. Mid-rollout init (recovery, alternative_strategy)
    requires DAgger-saved pkls, which carry the per-step ``sim_state``.
    """
    from policy_doctor.vlm.proposals.init_state import extract_sim_state_at_frame

    resp = requests.get(f"{server_url.rstrip('/')}/pool", timeout=10)
    resp.raise_for_status()
    pool_index = resp.json()
    ic = request["initial_conditions"]
    pkl = _resolve_reference_pkl(pool_index, ic["reference_rollout_id"])
    try:
        return extract_sim_state_at_frame(pkl, int(ic.get("reference_frame", 0)))
    except (KeyError, AttributeError, FileNotFoundError):
        # ``sim_state`` column missing — eval pkls don't always record it.
        return None


# ---------------------------------------------------------------------------
# Public entry point — invoked by scripts/run_e2_sim.py
# ---------------------------------------------------------------------------


def run_e2_session(
    *,
    proposal_server_url: str,
    output_dir: Path,
    task: str,
    train_dir: Optional[str] = None,
    train_ckpt: str = "best",
    dataset_path: Optional[str] = None,
    device: str = "auto",
    viz_url: Optional[str] = None,
    dagger_config: str = "keyboard_default",
    max_demos: Optional[int] = None,
    poll_interval_s: float = 2.0,
) -> int:
    """Process the proposal server's queue until empty (or max_demos reached).

    Returns the number of completed demonstrations. Each demo is saved as
    ``<output_dir>/<request_id>/ep0000.pkl`` — note the ``request_id`` directory
    name is the OPAQUE request id from the schema, never a condition string.
    """
    from policy_doctor.paths import REPO_ROOT
    from policy_doctor.envs import RobomimicDAggerEnv, RobomimicDAggerRunner
    from policy_doctor.envs.dagger_config import (
        create_intervention_device,
        load_dagger_config,
    )
    from policy_doctor.envs.policy_wrappers import BarePolicy
    from policy_doctor.envs.intervention_device import HTTPInterventionDevice
    from policy_doctor.envs.visualization import DAggerVisualizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = ProposalServerClient(proposal_server_url)
    health = client.health()
    print(f"[e2_runner] connected: pending={health['n_pending']} pool={health['pool_size']}")

    # Resolve dataset for env construction. Same task table as run_dagger.py.
    from scripts.run_dagger import TASK_CONFIG, auto_device, resolve_checkpoint  # type: ignore

    task_cfg = TASK_CONFIG.get(task)
    if task_cfg is None:
        raise ValueError(f"Unknown task {task!r}; choices: {list(TASK_CONFIG)}")
    if dataset_path is None:
        dataset_path = task_cfg["dataset_path"]
    obs_keys = task_cfg["obs_keys"]
    if device == "auto":
        device = auto_device()
    print(f"[e2_runner] device={device}")

    # Load policy
    policy_wrapped = None
    classifier_n_obs_steps = 2
    classifier_n_action_steps = 8
    abs_action = False
    rotation_transformer = None
    if train_dir:
        import dill, hydra, torch
        from omegaconf import OmegaConf as _OC

        ckpt = resolve_checkpoint(train_dir, train_ckpt)
        print(f"[e2_runner] checkpoint={ckpt}")
        payload = torch.load(open(str(ckpt), "rb"), pickle_module=dill)
        cfg_ckpt = payload["cfg"]
        cls = hydra.utils.get_class(cfg_ckpt._target_)
        ws = cls(cfg_ckpt, output_dir=str(output_dir))
        ws.load_payload(payload, exclude_keys=None, include_keys=None)
        raw_policy = ws.ema_model if getattr(cfg_ckpt.training, "use_ema", False) else ws.model
        raw_policy.to(device).eval()
        policy_wrapped = BarePolicy(raw_policy)
        abs_action = bool(_OC.select(cfg_ckpt, "task.dataset.abs_action") or False)
        classifier_n_obs_steps = int(_OC.select(cfg_ckpt, "n_obs_steps") or 2)
        classifier_n_action_steps = int(_OC.select(cfg_ckpt, "n_action_steps") or 8)

    if abs_action:
        from diffusion_policy.model.common.rotation_transformer import RotationTransformer
        rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")

    def convert_action(action_10d):
        if rotation_transformer is not None:
            pos = action_10d[..., :3]
            rot = action_10d[..., 3:9]
            gripper = action_10d[..., [9]]
            return np.concatenate(
                [pos, rotation_transformer.inverse(rot), gripper], axis=-1
            )
        return action_10d[..., :7]

    # Build env once — we'll mutate init_state per request.
    from robomimic.utils.env_utils import EnvUtils
    from robomimic.utils.file_utils import FileUtils
    from robomimic.utils.obs_utils import ObsUtils
    from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper

    env_meta = FileUtils.get_env_metadata_from_dataset(str(dataset_path))
    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": obs_keys})
    if abs_action:
        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
    robomimic_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=False,
    )
    lowdim_wrapper = RobomimicLowdimWrapper(env=robomimic_env, obs_keys=obs_keys, init_state=None)

    dagger_cfg = load_dagger_config(dagger_config)
    if viz_url:
        intervention_device = HTTPInterventionDevice(server_url=viz_url)
    else:
        intervention_device = create_intervention_device(dagger_cfg)

    visualizer = None
    viz_cfg = dagger_cfg.get("visualization", {})
    if viz_url or viz_cfg.get("enabled", True):
        try:
            kw = dict(
                camera_names=viz_cfg.get("camera_names", ["agentview"]),
                figsize=tuple(viz_cfg.get("figsize", [8, 5])),
            )
            if viz_url:
                kw["server_url"] = viz_url
            visualizer = DAggerVisualizer(**kw)
        except Exception as e:
            print(f"[e2_runner] visualizer setup failed: {e}")

    n_done = 0
    while max_demos is None or n_done < max_demos:
        active = client.get_active()
        if active is None:
            print("[e2_runner] queue empty — exiting")
            break

        rid = active["request_id"]
        # Per-request demo dir uses ONLY the opaque id; never the condition.
        per_req_dir = output_dir / rid
        per_req_dir.mkdir(parents=True, exist_ok=True)
        with open(per_req_dir / "request.json", "w") as f:
            json.dump(active, f, indent=2)

        # Reset sim to the request's reference state. None = fall back to
        # env.reset() (rollout-start). Recovery / alternative_strategy
        # requests prefer mid-rollout state but degrade gracefully when the
        # source pkls don't carry sim_state (typical for eval_save_episodes
        # outputs); the operator sees the request prose either way.
        init_state = _init_state_for_request(proposal_server_url, active)
        lowdim_wrapper.init_state = init_state
        if init_state is None and active.get("request_type") != "full_trajectory":
            print(
                f"[e2_runner] note: {active['request_type']} request {rid} "
                "has no sim_state in source pkl — starting from env.reset() "
                "(use DAgger-saved pkls for true mid-rollout init)"
            )

        env = RobomimicDAggerEnv(
            inner_env=lowdim_wrapper, obs_keys=obs_keys, output_dir=per_req_dir
        )

        runner = RobomimicDAggerRunner(
            monitored_policy=policy_wrapped,
            env=env,
            intervention_device=intervention_device,
            n_obs_steps=classifier_n_obs_steps,
            n_action_steps=classifier_n_action_steps,
            max_steps=500,
            output_dir=per_req_dir,
            visualizer=visualizer,
            action_transform=convert_action,
        )

        # Operator-facing display
        print(f"\n[e2_runner] === request {rid} ({active['request_type']}) ===")
        print(f"[e2_runner] target_behavior: {active['target_behavior']}")
        if active.get("prohibitions"):
            print(f"[e2_runner] prohibitions: {active['prohibitions']}")
        print(f"[e2_runner] success_criterion: {active.get('success_criterion','task_success')}")

        records = runner.run(1)
        if not records:
            print(f"[e2_runner] runner produced no record for {rid}")
            continue
        success = bool(records[0].success)

        demo_pkl = per_req_dir / "ep0000.pkl"
        if not demo_pkl.exists():
            cands = sorted(per_req_dir.glob("ep*.pkl"))
            demo_pkl = cands[-1] if cands else demo_pkl

        result = client.post_result(rid, demo_pkl=demo_pkl, success=success)
        n_done += 1
        # Operator-visible feedback only: success + overall, never per-axis breakdown.
        print(
            f"[e2_runner] request {rid}: success={success} "
            f"overall={result.get('overall')} passed_filter={result.get('passed_filter')}"
        )

        # Sleep briefly between demos so the operator can prepare.
        time.sleep(poll_interval_s)

    if visualizer:
        visualizer.close()
    return n_done
