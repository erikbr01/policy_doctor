"""HTTP policy server: loads the policy and serves inference via Flask.

Wire format: raw numpy bytes (application/octet-stream) using np.save/np.load.

Start the server in one terminal:
    python -m policy_doctor.envs.policy_server \
        --checkpoint /path/to/epoch=0950-....ckpt \
        --device mps \
        --port 5001

Then use PolicyClient in the DAgger runner:
    client = PolicyClient(url="http://localhost:5001")
    client.submit(obs_dict)            # non-blocking, fires HTTP in background thread
    chunk = client.get()               # blocks until response (usually already ready)
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def _build_app(checkpoint: str, device: str):
    """Build and return the Flask app with the policy loaded."""
    import warnings
    warnings.filterwarnings("ignore")

    import dill
    import hydra
    import torch
    from flask import Flask, request

    print(f"[server] loading checkpoint: {checkpoint}", flush=True)
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    ws = cls(cfg, output_dir="/tmp/_policy_server_ws")
    ws.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = ws.ema_model if getattr(cfg.training, "use_ema", False) else ws.model
    policy.to(device)
    policy.eval()
    print(f"[server] ready on {device}", flush=True)

    app = Flask(__name__)

    @app.get("/health")
    def health():
        return {"status": "ok", "device": device}

    @app.post("/infer")
    def infer():
        # Receive obs array as raw numpy bytes
        obs = np.load(io.BytesIO(request.data))          # shape: (1, n_obs, obs_dim)
        obs_t = torch.from_numpy(obs).float().to(device)
        with torch.no_grad():
            result = policy.predict_action({"obs": obs_t})
        action = result["action"]
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()       # (1, n_action_steps, action_dim)
        buf = io.BytesIO()
        np.save(buf, action)
        return buf.getvalue(), 200, {"Content-Type": "application/octet-stream"}

    return app


def serve(checkpoint: str, device: str, host: str = "127.0.0.1", port: int = 5001):
    app = _build_app(checkpoint, device)
    # Use threaded=False so Flask doesn't spawn its own threads (keeps MPS on one thread)
    app.run(host=host, port=port, threaded=False, use_reloader=False)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class PolicyClient:
    """Non-blocking HTTP client for the policy server.

    Uses a single background thread for the HTTP request so the main thread
    (sim + viz) stays live.  HTTP is pure network I/O — safe to thread even
    on MPS because the main process never touches the GPU.

    Compatible with the MonitoredPolicy interface so the runner works with both.
    """

    episode_results: list

    def __init__(self, url: str = "http://127.0.0.1:5001") -> None:
        self._url = url.rstrip("/")
        self._future = None
        self._executor = None
        self.episode_results = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy_http")

    def reset(self) -> None:
        self.episode_results = []
        self._future = None

    def stop(self) -> None:
        if self._executor:
            self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Inference interface
    # ------------------------------------------------------------------

    def submit(self, obs_dict: dict) -> None:
        """Fire-and-forget: send obs to server in background thread."""
        if self._executor is None:
            self.start()
        obs = obs_dict["obs"]
        if hasattr(obs, "detach"):
            obs = obs.detach().cpu().numpy()
        buf = io.BytesIO()
        np.save(buf, obs)
        data = buf.getvalue()
        self._future = self._executor.submit(self._post, data)

    def poll(self) -> Optional[np.ndarray]:
        """Return action chunk if ready, else None."""
        if self._future is None or not self._future.done():
            return None
        return self._collect()

    def get(self) -> np.ndarray:
        """Block until the server responds."""
        if self._future is None:
            raise RuntimeError("call submit() before get()")
        return self._collect()

    @property
    def pending(self) -> bool:
        return self._future is not None and not self._future.done()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _post(self, data: bytes) -> np.ndarray:
        import requests
        resp = requests.post(
            f"{self._url}/infer",
            data=data,
            headers={"Content-Type": "application/octet-stream"},
            timeout=30,
        )
        resp.raise_for_status()
        return np.load(io.BytesIO(resp.content))  # (1, n_action_steps, action_dim)

    def _collect(self) -> np.ndarray:
        result = self._future.result()  # re-raises if HTTP failed
        self._future = None
        return result[0]  # drop batch dim → (n_action_steps, action_dim)

    # ------------------------------------------------------------------
    # MonitoredPolicy shim (blocking path used when no client is wired)
    # ------------------------------------------------------------------

    def predict_action(self, obs_dict: dict) -> dict:
        self.submit(obs_dict)
        chunk = self.get()
        return {"action": chunk[None]}  # re-add batch dim


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Make policy_doctor and diffusion_policy importable
    _root = Path(__file__).resolve().parent.parent.parent
    _cupid = _root / "third_party" / "cupid"
    for p in [str(_root), str(_cupid)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    parser = argparse.ArgumentParser(description="HTTP policy inference server")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    serve(args.checkpoint, args.device, args.host, args.port)
