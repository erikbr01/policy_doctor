"""Policy backends for DROID real-robot inference.

All backends share the same submit / get / poll interface so DROIDInferenceRunner
is backend-agnostic.  The contract matches the existing PolicyClient from
policy_doctor/envs/policy_server.py.

Available backends:
  WebSocketPolicy  — openpi WebSocket server (pi0 / pi0.5)
  HttpPolicy       — Flask HTTP server (policy_server.py + diffusion_policy ckpts)
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class PolicyBase(ABC):
    """Uniform inference interface: submit (non-blocking) → get (blocking)."""

    @abstractmethod
    def reset(self) -> None:
        """Called at the start of each episode."""

    @abstractmethod
    def submit(self, obs_dict: dict, instruction: str = "") -> None:
        """Kick off inference asynchronously. Does not block."""

    @abstractmethod
    def get(self) -> np.ndarray:
        """Block until the last submitted inference completes.

        Returns
        -------
        chunk : np.ndarray, shape (T, action_dim)
        """

    @abstractmethod
    def poll(self) -> Optional[np.ndarray]:
        """Return action chunk if inference is already done, else None."""

    @property
    @abstractmethod
    def pending(self) -> bool:
        """True while a submitted call has not yet been collected."""


# ---------------------------------------------------------------------------
# WebSocket backend (openpi / pi0 / pi0.5)
# ---------------------------------------------------------------------------

class WebSocketPolicy(PolicyBase):
    """openpi WebSocket policy client with async submit/get.

    The WebSocket call is blocking I/O, so we run it in a background
    ThreadPoolExecutor (one worker) so the control loop stays live.

    Parameters
    ----------
    host : str
    port : int
    external_camera : str
        Which external image to feed: 'ext1' or 'ext2' (matches left/right).
    wrist_key : str
        Key in the obs_dict that holds the wrist image.
    ext_key : str
        Key in the obs_dict for the chosen exterior image.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        external_camera: str = "ext1",
        wrist_key: str = "wrist_image",
        ext_key: Optional[str] = None,
    ) -> None:
        self._host = host
        self._port = port
        self._ext_key = ext_key or (
            "exterior_image_1_left" if external_camera == "ext1" else "exterior_image_2_left"
        )
        self._wrist_key = wrist_key
        self._executor: Optional[ThreadPoolExecutor] = None
        self._future = None

    # ------------------------------------------------------------------

    def _get_client(self):
        from openpi_client import websocket_client_policy
        if not hasattr(self, "_client"):
            self._client = websocket_client_policy.WebsocketClientPolicy(self._host, self._port)
        return self._client

    def reset(self) -> None:
        self._future = None

    def submit(self, obs_dict: dict, instruction: str = "") -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="droid_ws_policy")
        request = self._build_request(obs_dict, instruction)
        self._future = self._executor.submit(self._infer, request)

    def get(self) -> np.ndarray:
        if self._future is None:
            raise RuntimeError("call submit() before get()")
        chunk = self._future.result()  # re-raises if inference failed
        self._future = None
        return chunk

    def poll(self) -> Optional[np.ndarray]:
        if self._future is None or not self._future.done():
            return None
        return self.get()

    @property
    def pending(self) -> bool:
        return self._future is not None and not self._future.done()

    # ------------------------------------------------------------------

    def _build_request(self, obs_dict: dict, instruction: str) -> dict:
        from openpi_client import image_tools

        ext_img = obs_dict.get(self._ext_key)
        wrist_img = obs_dict.get(self._wrist_key)

        req: dict = {
            "observation/joint_position": np.asarray(obs_dict["joint_position"], dtype=np.float32),
            "observation/gripper_position": np.asarray(obs_dict["gripper_position"], dtype=np.float32),
            "prompt": instruction,
        }
        if ext_img is not None:
            req["observation/exterior_image_1_left"] = image_tools.resize_with_pad(ext_img, 224, 224)
        if wrist_img is not None:
            req["observation/wrist_image_left"] = image_tools.resize_with_pad(wrist_img, 224, 224)
        return req

    def _infer(self, request: dict) -> np.ndarray:
        result = self._get_client().infer(request)
        return np.asarray(result["actions"], dtype=np.float32)  # (T, action_dim)


# ---------------------------------------------------------------------------
# HTTP backend (policy_server.py + diffusion_policy checkpoints)
# ---------------------------------------------------------------------------

class HttpPolicy(PolicyBase):
    """Thin wrapper around the existing PolicyClient from policy_server.py.

    Delegates submit / get / poll directly; no new logic needed.

    NOTE: submit() builds a state-only observation (joint_position + gripper_position)
    and ignores image keys.  This is appropriate for diffusion_policy low-dim
    checkpoints served by policy_server.py.  Do NOT use HttpPolicy with image
    policies — use WebSocketPolicy with an openpi server instead.
    """

    def __init__(self, url: str = "http://127.0.0.1:5001") -> None:
        from policy_doctor.envs.policy_server import PolicyClient

        self._client = PolicyClient(url=url)

    def reset(self) -> None:
        self._client.reset()

    def submit(self, obs_dict: dict, instruction: str = "") -> None:
        # PolicyClient.submit() expects {"obs": array}; build that from obs_dict.
        state = np.concatenate([
            obs_dict.get("joint_position", np.zeros(7)),
            obs_dict.get("gripper_position", np.zeros(1)),
        ]).astype(np.float32)
        self._client.submit({"obs": state[None]})  # (1, obs_dim)

    def get(self) -> np.ndarray:
        return self._client.get()  # (T, action_dim)

    def poll(self) -> Optional[np.ndarray]:
        return self._client.poll()

    @property
    def pending(self) -> bool:
        return self._client.pending
