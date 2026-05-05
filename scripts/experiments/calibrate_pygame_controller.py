#!/usr/bin/env python3
"""Interactive pygame joystick calibration for policy_doctor DAgger (raw button indices).

conda often buffers stdout until exit — run unbuffered, for example::

  PYTHONUNBUFFERED=1 conda run -n cupid python -u scripts/experiments/calibrate_pygame_controller.py

Press Ctrl+C to abort.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def _force_terminal_output() -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
            sys.stderr.reconfigure(line_buffering=True)
        except Exception:
            pass


def _drain_pygame_events(pg) -> None:
    pg.event.pump()


def _wait_all_released(pg, joy, hint_every_s: float = 2.0) -> None:
    n = joy.get_numbuttons()
    t0 = time.monotonic()
    last_hint = t0
    print("  Waiting until ALL buttons are released…", flush=True)
    while True:
        _drain_pygame_events(pg)
        if not any(joy.get_button(i) for i in range(n)):
            return
        now = time.monotonic()
        if now - last_hint >= hint_every_s:
            pressed = [i for i in range(n) if joy.get_button(i)]
            print(
                f"  …still held (indices {pressed}). Release every button to continue.",
                flush=True,
            )
            last_hint = now
        time.sleep(0.02)


def wait_single_button_press(pg, joy, label: str) -> int:
    print(f"\n── {label} ──", flush=True)
    print("  Release every button, then press ONLY that control once.", flush=True)
    _wait_all_released(pg, joy)
    print("  Listening…", flush=True)
    n = joy.get_numbuttons()
    while True:
        _drain_pygame_events(pg)
        for i in range(n):
            if joy.get_button(i):
                bid = i
                while joy.get_button(bid):
                    _drain_pygame_events(pg)
                    time.sleep(0.015)
                print(f"  → joystick button index: {bid}", flush=True)
                return bid
        time.sleep(0.015)


def live_mode(pg, joy, hz: float) -> None:
    dt = 1.0 / max(hz, 1.0)
    n_btn = joy.get_numbuttons()
    n_ax = joy.get_numaxes()
    print(
        f"\nLive ({joy.get_name()!r})  buttons 0..{n_btn - 1}  axes 0..{n_ax - 1}. Ctrl+C exits.\n",
        flush=True,
    )
    try:
        while True:
            _drain_pygame_events(pg)
            pressed = [i for i in range(n_btn) if joy.get_button(i)]
            axes = [f"a{i}:{joy.get_axis(i):+.3f}" for i in range(n_ax)]
            line = "buttons " + str(pressed) + " | " + " ".join(axes)
            print(line[:200], flush=True)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nDone.", flush=True)


def wizard(pg, joy) -> dict[str, int]:
    print("\n=== Guided calibration (raw pygame joystick indices) ===\n", flush=True)
    targets = [
        ("Gripper CLOSE", "Top LEFT bumper (L1 / LB)"),
        ("Gripper OPEN", "Top RIGHT bumper (R1 / RB)"),
        ("Scene RESET", "e.g. click RIGHT stick (R3), or Share"),
        ("Human / robot TOGGLE", "Options / Start / Menu"),
    ]
    keys = ["button_gripper_close", "button_gripper_open", "button_reset", "button_toggle"]
    out: dict[str, int] = {}
    for key, (title, desc) in zip(keys, targets):
        out[key] = wait_single_button_press(pg, joy, f"{title}: {desc}")
    return out


def main() -> int:
    _force_terminal_output()
    parser = argparse.ArgumentParser(description="Calibrate pygame joystick button indices for DAgger.")
    parser.add_argument("--device-index", type=int, default=0, help="joystick index (default 0)")
    parser.add_argument("--live", action="store_true", help="print pressed buttons + axes each line")
    parser.add_argument("--hz", type=float, default=10.0, help="poll rate for --live (default 10)")
    args = parser.parse_args()

    print("calibrate_pygame_controller: starting…", flush=True)

    try:
        import pygame
    except ImportError:
        print("ERROR: pygame not installed (pip install pygame)", flush=True)
        return 1

    print("Initializing pygame…", flush=True)
    pygame.init()
    pygame.joystick.init()

    n = pygame.joystick.get_count()
    print(f"Detected {n} joystick device(s).", flush=True)
    if n <= 0:
        print("ERROR: No joystick. Connect a controller.", flush=True)
        return 1
    if args.device_index >= n:
        print(f"ERROR: device-index {args.device_index} invalid (use 0..{n - 1}).", flush=True)
        return 1

    joy = pygame.joystick.Joystick(args.device_index)
    joy.init()
    print(f"\nUsing [{args.device_index}] {joy.get_name()!r}", flush=True)
    print(
        f"  buttons={joy.get_numbuttons()} axes={joy.get_numaxes()} hats={joy.get_numhats()}",
        flush=True,
    )

    try:
        if args.live:
            live_mode(pygame, joy, args.hz)
            return 0
        mapping = wizard(pygame, joy)
        print("\n=== Paste under pygame: in your dagger YAML ===\n", flush=True)
        print("  controller_layout: ps4  # or xbox — must match how you use presets\n", flush=True)
        for k, v in mapping.items():
            print(f"  {k}: {v}", flush=True)
        print(flush=True)
        return 0
    finally:
        try:
            joy.quit()
        except Exception:
            pass
        pygame.quit()


if __name__ == "__main__":
    raise SystemExit(main())
