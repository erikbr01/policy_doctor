#!/usr/bin/env python3
"""Quick headless check for the mimicgen conda stack (no X11 / pynput).

``third_party/mimicgen/.../demo_random_action.py`` imports keyboard helpers and needs
``DISPLAY``. Use this script on SSH or batch nodes instead.

Examples::

    MUJOCO_GL=egl python scripts/mimicgen_headless_smoke_test.py
    MUJOCO_GL=osmesa python scripts/mimicgen_headless_smoke_test.py
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="Lift", help="robosuite environment name")
    parser.add_argument("--robots", default="Panda", help="robot(s) spec for suite.make")
    parser.add_argument(
        "--mujoco-gl",
        default=None,
        help="Set MUJOCO_GL for this run (default: existing env or 'egl')",
    )
    parser.add_argument("--steps", type=int, default=5, help="random-action steps after reset")
    args = parser.parse_args()

    gl = args.mujoco_gl or os.environ.get("MUJOCO_GL", "egl")
    os.environ["MUJOCO_GL"] = gl

    import numpy as np
    import robosuite as suite

    import mimicgen  # noqa: F401 — register MimicGen robosuite envs

    from robosuite.controllers import load_controller_config

    try:
        env = suite.make(
            env_name=args.env,
            robots=args.robots,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            control_freq=20,
            ignore_done=True,
            controller_configs=load_controller_config(default_controller="OSC_POSE"),
        )
    except Exception as e:
        print(f"suite.make failed (MUJOCO_GL={gl}): {e}", file=sys.stderr)
        if gl.lower() != "osmesa":
            print(
                "Try: MUJOCO_GL=osmesa python scripts/mimicgen_headless_smoke_test.py",
                file=sys.stderr,
            )
        return 1

    try:
        env.reset()
        low, high = env.action_spec
        for _ in range(args.steps):
            env.step(np.random.uniform(low, high))
    finally:
        env.close()

    print(f"OK ({args.env}, {args.steps} steps, MUJOCO_GL={gl})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
