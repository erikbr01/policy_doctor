"""Push an offline agent-session's submissions onto the proposal server's queue.

Thin CLI wrapper around POST /requests/import_session — saves you from
remembering the JSON body shape. Use after ``run_e2_agent_transport_mh.py``
has written a session dir, before the sim runner drains the queue.

Usage::

    python scripts/enqueue_session.py /tmp/e2_session_seed0
    python scripts/enqueue_session.py /tmp/e2_session_seed0 --server http://localhost:5003
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_dir", type=Path, help="Path to an agent session dir (must contain submitted_requests.json)")
    ap.add_argument("--server", default="http://127.0.0.1:5003", help="proposal server URL")
    args = ap.parse_args()

    sub = args.session_dir / "submitted_requests.json"
    if not sub.exists():
        print(f"error: no submitted_requests.json in {args.session_dir}", file=sys.stderr)
        return 1

    resp = requests.post(
        f"{args.server.rstrip('/')}/requests/import_session",
        json={"session_dir": str(args.session_dir)},
        timeout=60,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"error: server returned {resp.status_code}: {resp.text}", file=sys.stderr)
        return 1

    body = resp.json()
    print(f"added {body['n_added']}/{body['n_added'] + body['n_skipped']} requests "
          f"(queue size: {body['n_in_queue']})")
    if body.get("skipped"):
        print("skipped:")
        for s in body["skipped"]:
            print(f"  [{s['index']}] {s['reason']}")
    if body.get("added"):
        print("added request_ids:", " ".join(body["added"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
