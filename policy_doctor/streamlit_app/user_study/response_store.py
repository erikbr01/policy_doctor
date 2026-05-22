"""Response storage abstraction for the user study.

Auto-selects backend based on ``SURVEY_GCS_BUCKET`` env var:
- Set: uses Google Cloud Storage (responses as blobs under ``survey_responses/``)
- Unset: falls back to a local JSON directory

Usage::

    from policy_doctor.streamlit_app.user_study.response_store import get_store
    store = get_store(local_fallback_dir)
    response_id = store.save(result_dict)   # returns a string ID
    all_responses = store.list_all()        # list[dict]
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path


def _response_filename(response: dict) -> str:
    ts = response.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
    group = str(response.get("group", "X")).lower()
    pid = str(response.get("participant_id", "anon")).replace(" ", "_")[:32]
    return f"group_{group}_{ts}_{pid}.json"


class LocalResponseStore:
    def __init__(self, base_dir: Path | str):
        self._dir = Path(base_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, response: dict) -> str:
        fname = _response_filename(response)
        path = self._dir / fname
        with open(path, "w") as f:
            json.dump(response, f, indent=2)
        return fname

    def list_all(self) -> list[dict]:
        results = []
        if not self._dir.exists():
            return results
        for p in sorted(self._dir.glob("*.json")):
            try:
                with open(p) as f:
                    results.append(json.load(f))
            except Exception:
                pass
        return results


class GCSResponseStore:
    def __init__(self, bucket_name: str, prefix: str = "survey_responses/"):
        from google.cloud import storage  # lazy — not a hard dep in dev
        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket_name)
        self._prefix = prefix.rstrip("/") + "/"

    def save(self, response: dict) -> str:
        blob_name = self._prefix + _response_filename(response)
        blob = self._bucket.blob(blob_name)
        blob.upload_from_string(
            json.dumps(response, indent=2),
            content_type="application/json",
        )
        return blob_name

    def list_all(self) -> list[dict]:
        results = []
        for blob in self._bucket.list_blobs(prefix=self._prefix):
            if not blob.name.endswith(".json"):
                continue
            try:
                results.append(json.loads(blob.download_as_text()))
            except Exception:
                pass
        return results


def get_store(local_dir: Path | str) -> LocalResponseStore | GCSResponseStore:
    """Return the appropriate store based on env config."""
    bucket = os.environ.get("SURVEY_GCS_BUCKET", "").strip()
    if bucket:
        return GCSResponseStore(bucket)
    return LocalResponseStore(local_dir)
