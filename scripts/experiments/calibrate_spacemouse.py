#!/usr/bin/env python3
"""Interactive SpaceMouse calibration for DAgger (hidapi + same decode as viz_server).

Uses the same HID decode as ``policy_doctor.envs.viz_server``: translation ``tx,ty,tz``,
wire-order rotation ``r0,r1,r2`` (bytes 7–8, 9–10, 11–12), then ``rotation_mix`` in
task YAML maps ``[r0,r1,r2]`` → OSC roll/pitch/yaw. Walks through translation / rotation
/ buttons and checks that the expected channels dominate each gesture.

conda may buffer stdout — run unbuffered, for example::

  PYTHONUNBUFFERED=1 conda run -n cupid python -u scripts/experiments/calibrate_spacemouse.py

From repo root (or set ``PYTHONPATH`` to the project root). Press Ctrl+C to abort.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from policy_doctor.spacemouse_hid import decode_spacemouse_motion_report


def _force_terminal_output() -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
            sys.stderr.reconfigure(line_buffering=True)
        except Exception:
            pass


def decode_motion_report(data: Any) -> tuple[float, float, float, float, float, float]:
    """Desk-frame motion (report id 1), same as ``decode_spacemouse_motion_report`` — used by wizard."""
    seq = list(data)
    return decode_spacemouse_motion_report(seq)


def _poll_motion_metrics(
    dev,
    duration_s: float,
    *,
    progress_cb: Callable[[dict[str, float], int], None] | None = None,
) -> tuple[dict[str, float], dict[str, float], int]:
    """While polling ``duration_s``, track max |axis| and signed sum per channel."""
    keys = ("x", "y", "z", "r0", "r1", "r2")
    max_abs = {k: 0.0 for k in keys}
    sum_val = {k: 0.0 for k in keys}
    n_motion = 0
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end:
        data = dev.read(64)
        if data and len(data) >= 13 and data[0] == 1:
            x, y, z, r0, r1, r2 = decode_motion_report(data)
            vals = {"x": x, "y": y, "z": z, "r0": r0, "r1": r1, "r2": r2}
            for k in keys:
                v = vals[k]
                max_abs[k] = max(max_abs[k], abs(v))
                sum_val[k] += v
            n_motion += 1
            if progress_cb and n_motion % 25 == 0:
                progress_cb(max_abs, n_motion)
        else:
            time.sleep(0.001)
    return max_abs, sum_val, n_motion


def _dominates(
    max_abs: dict[str, float],
    primary: str,
    others: list[str],
    *,
    ratio: float,
    min_peak: float,
) -> bool:
    if max_abs[primary] < min_peak:
        return False
    for o in others:
        if max_abs[o] * ratio > max_abs[primary]:
            return False
    return True


def _wait_enter(prompt: str) -> None:
    input(f"{prompt}  [Enter when ready] ")
    print(flush=True)


def _print_motion_probe() -> None:
    try:
        import hid
    except ImportError:
        return
    print("\nHID devices (3Dconnexion / SpaceMouse hints):", flush=True)
    any_line = False
    for d in hid.enumerate():
        vid = int(d.get("vendor_id", 0))
        pid = int(d.get("product_id", 0))
        man = (d.get("manufacturer_string") or "") or ""
        prod = (d.get("product_string") or "") or ""
        if vid == 0x256F or "3dconnexion" in man.lower() or "space" in prod.lower():
            any_line = True
            print(
                f"  vid=0x{vid:04x} pid=0x{pid:04x}  ({vid},{pid})  {prod!r}",
                flush=True,
            )
    if not any_line:
        print("  (none matching — plug receiver / USB cable)", flush=True)


def live_mode(dev, hz: float) -> None:
    dt = 1.0 / max(hz, 1.0)
    print(f"\nLive SpaceMouse decode @ {hz:.1f} Hz. Ctrl+C exits.\n", flush=True)
    try:
        while True:
            data = dev.read(64)
            if not data:
                time.sleep(dt)
                continue
            rid = data[0]
            if rid == 1 and len(data) >= 13:
                x, y, z, r0, r1, r2 = decode_motion_report(data)
                print(
                    f"motion  xyz=({x:+.4f},{y:+.4f},{z:+.4f})  "
                    f"r012=({r0:+.4f},{r1:+.4f},{r2:+.4f})",
                    flush=True,
                )
            elif rid == 3 and len(data) >= 2:
                print(f"buttons raw={bytes(data).hex()}  mask={data[1] & 3}", flush=True)
            elif rid != 1:
                print(f"report_id={rid} len={len(data)} raw={bytes(data).hex()}", flush=True)
            else:
                time.sleep(dt)
    except KeyboardInterrupt:
        print("\nDone.", flush=True)


def run_wizard(dev, *, sample_s: float, min_peak: float, dominance_ratio: float) -> int:
    print("\n=== Guided SpaceMouse check (viz_server HID decode) ===\n", flush=True)
    print(
        "  Rotation is wire-order: r0=bytes (7,8), r1=(9,10), r2=(11,12); "
        "task ``rotation_mix`` maps [r0,r1,r2] → OSC roll/pitch/yaw.\n",
        flush=True,
    )
    trans = ("x", "y", "z")
    rot = ("r0", "r1", "r2")

    steps: list[tuple[str, str, str, list[str]]] = [
        (
            "Translation — push cap toward your monitor (forward).",
            "y",
            "translation Y",
            ["x", "z"],
        ),
        (
            "Translation — slide cap left (your left).",
            "x",
            "translation X",
            ["y", "z"],
        ),
        (
            "Translation — lift the cap straight up.",
            "z",
            "translation Z",
            ["x", "y"],
        ),
        (
            "Rotation — yaw: twist CLOCKWISE as seen from above "
            "(front lip of the cap moves toward your RIGHT; like steering right).",
            "r2",
            "r2 (bytes 11–12)",
            ["r0", "r1"],
        ),
        (
            "Rotation — nod: FRONT edge of the cap tilts DOWN toward the desk.",
            "r0",
            "r0 (bytes 7–8)",
            ["r1", "r2"],
        ),
        (
            "Rotation — tilt: LEFT edge of the cap goes DOWN toward the desk.",
            "r1",
            "r1 (bytes 9–10)",
            ["r0", "r2"],
        ),
    ]

    for title, primary, label, _competitors in steps:
        pool = trans if primary in trans else rot
        others = [k for k in pool if k != primary]

        while True:
            print(f"\n── {title} ──", flush=True)
            print(
                f"  For ~{sample_s:.0f}s we sample motion reports; "
                f"**{label}** should dominate (min peak {min_peak:g}).",
                flush=True,
            )
            _wait_enter("Rest your hand off the cap, then perform ONLY this motion.")

            def _cb(ma: dict[str, float], n: int) -> None:
                print(f"  … {n} motion samples  max|axes| xyz=({ma['x']:.3f},{ma['y']:.3f},{ma['z']:.3f})", flush=True)

            max_abs, sum_val, n_motion = _poll_motion_metrics(
                dev, sample_s, progress_cb=_cb if sample_s >= 3.0 else None
            )
            print(
                f"  Received {n_motion} motion report(s).  max|.| "
                f"x={max_abs['x']:.4f} y={max_abs['y']:.4f} z={max_abs['z']:.4f}  "
                f"r0={max_abs['r0']:.4f} r1={max_abs['r1']:.4f} r2={max_abs['r2']:.4f}",
                flush=True,
            )

            if n_motion < 5:
                print(
                    "  **Too few samples** — move more decisively or increase --sample-seconds.",
                    flush=True,
                )
                if input("  Retry this step? [Y/n] ").strip().lower() in ("", "y", "yes"):
                    continue
                return 1

            ok = _dominates(max_abs, primary, others, ratio=dominance_ratio, min_peak=min_peak)
            # Cross-family: translation step shouldn't be dominated by huge accidental rotation
            if primary in trans:
                ok = ok and max(max_abs["r0"], max_abs["r1"], max_abs["r2"]) <= max_abs[primary] * 2.5
            else:
                ok = ok and max(max_abs["x"], max_abs["y"], max_abs["z"]) <= max_abs[primary] * 2.5

            if ok:
                sign = "+" if sum_val[primary] >= 0 else "-"
                print(f"  OK — strongest motion on **{label}** (net sign {sign}).", flush=True)
                break

            print(
                "  **Check failed** — wrong axis mix or motion too small. "
                "Try isolating that DOF more carefully.",
                flush=True,
            )
            if input("  Retry this step? [Y/n] ").strip().lower() not in ("", "y", "yes"):
                return 1

    print("\n── Buttons ──", flush=True)
    print(
        "  SpaceMouse has two buttons. We'll watch HID report id 3 (same as viz_server).",
        flush=True,
    )

    def wait_button(which: str, bit: int) -> bool:
        print(f"\n  Press and release **{which} button**.", flush=True)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            data = dev.read(64)
            if (
                data
                and len(data) >= 2
                and data[0] == 3
                and (int(data[1]) & bit) != 0
            ):
                print(f"  OK — saw button report id 3 with bit {bit} set.", flush=True)
                return True
            time.sleep(0.01)
        print("  Timeout — no matching button report.", flush=True)
        return False

    if not wait_button("LEFT", 1):
        return 1
    if not wait_button("RIGHT", 2):
        return 1

    print("\n=== All checks passed ===\n", flush=True)
    print(
        "Note: in ``viz_server``, SpaceMouse motion updates the arm only while "
        "**human control is ON** (right SpaceMouse button or Space in the OpenCV window).\n",
        flush=True,
    )
    return 0


def main() -> int:
    _force_terminal_output()
    parser = argparse.ArgumentParser(description="Calibrate / verify SpaceMouse HID streaming for DAgger.")
    parser.add_argument("--live", action="store_true", help="stream decoded motion + button reports")
    parser.add_argument("--hz", type=float, default=15.0, help="poll pace for --live when idle (default 15)")
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=5.0,
        help="per-step motion sampling window in wizard mode (default 5)",
    )
    parser.add_argument(
        "--min-peak",
        type=float,
        default=0.04,
        help="minimum max-|axis| on the primary DOF to count as motion (default 0.04)",
    )
    parser.add_argument(
        "--dominance-ratio",
        type=float,
        default=1.35,
        help="primary axis max-abs must exceed others by this factor (default 1.35)",
    )
    parser.add_argument(
        "--spacemouse-vid",
        type=lambda s: int(s, 0),
        default=None,
        metavar="VID",
        help="USB vendor id (hex ok), e.g. 0x256f — use with --spacemouse-pid",
    )
    parser.add_argument(
        "--spacemouse-pid",
        type=lambda s: int(s, 0),
        default=None,
        metavar="PID",
        help="USB product id (hex ok) — use with --spacemouse-vid",
    )
    args = parser.parse_args()
    if (args.spacemouse_vid is None) ^ (args.spacemouse_pid is None):
        print("ERROR: pass both --spacemouse-vid and --spacemouse-pid, or neither.", flush=True)
        return 1

    print("calibrate_spacemouse: starting…", flush=True)

    try:
        from policy_doctor.spacemouse_hid import (
            dedupe_usb_pairs,
            spacemouse_usb_pairs_with_override,
            try_open_first_spacemouse,
        )
    except ImportError as e:
        print(f"ERROR: cannot import policy_doctor.spacemouse_hid ({e})", flush=True)
        return 1

    try:
        import hid  # noqa: F401
    except ImportError:
        print("ERROR: hidapi not installed (pip install hidapi)", flush=True)
        return 1

    pairs = spacemouse_usb_pairs_with_override(args.spacemouse_vid, args.spacemouse_pid)
    try:
        opened = try_open_first_spacemouse(pairs)
    except ImportError:
        print("ERROR: hidapi not installed (pip install hidapi)", flush=True)
        return 1

    if opened is None:
        print("ERROR: could not open SpaceMouse on tried USB pairs:", flush=True)
        for v, p in dedupe_usb_pairs(list(pairs)):
            print(f"  0x{v:04x}:0x{p:04x}", flush=True)
        _print_motion_probe()
        return 1

    dev, vid, pid = opened
    print(f"\nOpened SpaceMouse  usb 0x{vid:04x}:0x{pid:04x}  ({vid},{pid})\n", flush=True)

    try:
        if args.live:
            live_mode(dev, args.hz)
            return 0
        code = run_wizard(
            dev,
            sample_s=args.sample_seconds,
            min_peak=args.min_peak,
            dominance_ratio=args.dominance_ratio,
        )
        if code == 0:
            print("Paste under ``spacemouse:`` in dagger YAML if you use non-default USB IDs:\n", flush=True)
            print(f"  vendor_id: {vid}", flush=True)
            print(f"  product_id: {pid}", flush=True)
            print(flush=True)
        return code
    finally:
        fn = getattr(dev, "close", None)
        if callable(fn):
            fn()


if __name__ == "__main__":
    raise SystemExit(main())
