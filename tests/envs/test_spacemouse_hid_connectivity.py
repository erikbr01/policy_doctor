"""Integration check for 3Dconnexion SpaceMouse via hidapi.

Without hardware or ``hidapi``, tests **skip** so CI stays green.

Environment variables
---------------------
``SPACEMOUSE_INTEGRATION=1``
    Fail (instead of skip) when hidapi is missing or no device / no stream.

``SPACEMOUSE_STREAM_TIMEOUT_SEC`` (default ``10``)
    Seconds to wait for motion HID reports after opening the device.

``SPACEMOUSE_MIN_MOTION_REPORTS`` (default ``5``)
    Minimum number of report-id ``1`` (motion) packets required to treat streaming as OK.

``SPACEMOUSE_STREAM_DEBUG=1``
    Print each decoded motion sample to stderr (translation + rotation axes).

Run::

    PYTHONPATH=. python -m unittest tests.envs.test_spacemouse_hid_connectivity -v

Strict local check (must plug in device and **move the puck** during the window)::

    SPACEMOUSE_INTEGRATION=1 SPACEMOUSE_STREAM_DEBUG=1 \\
      PYTHONPATH=. python -m unittest tests.envs.test_spacemouse_hid_connectivity -v
"""

from __future__ import annotations

import os
import sys
import time
import unittest

from policy_doctor.spacemouse_hid import decode_spacemouse_motion_report


def _integration_required() -> bool:
    return os.environ.get("SPACEMOUSE_INTEGRATION", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _stream_debug() -> bool:
    return os.environ.get("SPACEMOUSE_STREAM_DEBUG", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _decode_motion_report(data) -> tuple[float, float, float, float, float, float]:
    """Decode motion report id 1 (desk frame)."""
    return decode_spacemouse_motion_report(list(data))


def _close_hid_device(dev) -> None:
    fn = getattr(dev, "close", None)
    if callable(fn):
        fn()


class TestSpaceMouseHidConnectivity(unittest.TestCase):
    def test_spacemouse_opens_and_streams_motion(self) -> None:
        try:
            from policy_doctor.spacemouse_hid import try_open_first_spacemouse
        except ImportError as e:
            self.skipTest(f"policy_doctor import failed: {e}")

        strict = _integration_required()
        timeout = float(os.environ.get("SPACEMOUSE_STREAM_TIMEOUT_SEC", "10"))
        min_motion = int(os.environ.get("SPACEMOUSE_MIN_MOTION_REPORTS", "5"))
        debug = _stream_debug()

        try:
            opened = try_open_first_spacemouse()
        except ImportError:
            if strict:
                self.fail("hidapi not installed (pip install hidapi)")
            self.skipTest("hidapi not installed (pip install hidapi)")

        if opened is None:
            msg = (
                "No SpaceMouse on default USB VID/PID list — plug USB receiver/cable "
                "or set SPACEMOUSE_INTEGRATION=1 after fixing IDs."
            )
            if strict:
                self.fail(msg)
            self.skipTest(msg)

        dev, vid, pid = opened
        self.assertEqual(vid, 0x256F)
        self.assertGreater(pid, 0)

        motion_reports = 0
        button_reports = 0
        deadline = time.monotonic() + max(0.5, timeout)

        try:
            while time.monotonic() < deadline:
                data = dev.read(64)
                if not data:
                    time.sleep(0.002)
                    continue
                rid = data[0]
                if rid == 1:
                    if len(data) < 13:
                        time.sleep(0.002)
                        continue
                    motion_reports += 1
                    if debug:
                        try:
                            x, y, z, ro, pi, ya = _decode_motion_report(data)
                            print(
                                f"[spacemouse stream] motion#{motion_reports}  "
                                f"xyz=({x:+.4f},{y:+.4f},{z:+.4f})  "
                                f"rpy=({ro:+.4f},{pi:+.4f},{ya:+.4f})",
                                file=sys.stderr,
                                flush=True,
                            )
                        except ValueError as e:
                            print(
                                f"[spacemouse stream] bad motion packet {data!r}: {e}",
                                file=sys.stderr,
                                flush=True,
                            )
                    if motion_reports >= min_motion:
                        break
                elif rid == 3 and len(data) >= 2:
                    button_reports += 1
                    if debug:
                        print(
                            f"[spacemouse stream] buttons payload={bytes(data).hex()}",
                            file=sys.stderr,
                            flush=True,
                        )
                else:
                    if debug:
                        print(
                            f"[spacemouse stream] report_id={rid} len={len(data)} "
                            f"raw={bytes(data).hex()}",
                            file=sys.stderr,
                            flush=True,
                        )

            if motion_reports < min_motion:
                msg = (
                    f"Only {motion_reports} motion report(s) in {timeout:g}s "
                    f"(need {min_motion}). Move / rotate the SpaceMouse cap during the test."
                )
                if strict:
                    self.fail(msg)
                self.skipTest(msg)

            self.assertGreaterEqual(
                motion_reports,
                min_motion,
                msg="streaming verification should have reached minimum motion reports",
            )
            if debug and button_reports:
                print(
                    f"[spacemouse stream] also saw {button_reports} button report(s)",
                    file=sys.stderr,
                    flush=True,
                )
        finally:
            _close_hid_device(dev)
