"""USB HID helpers for 3Dconnexion SpaceMouse / SpaceNavigator devices.

Kept free of ``gym`` / sim imports so integration tests can import without the full env stack.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

# 3Dconnexion USB vendor is always 0x256F (9583). Product IDs vary by model / wireless receiver.
SPACEMOUSE_USB_PAIRS_DEFAULT: tuple[tuple[int, int], ...] = (
    (9583, 50741),  # 0xC635 wired SpaceMouse Compact / SpaceNavigator (common)
    (9583, 50735),  # 0xC62F wireless USB receiver
    (9583, 50734),  # 0xC62E wireless (USB / some firmware paths)
    (9583, 50746),  # 0xC63A SpaceMouse Wireless (newer)
)


def dedupe_usb_pairs(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for vid, pid in pairs:
        if (vid, pid) not in seen:
            seen.add((vid, pid))
            out.append((vid, pid))
    return out


def spacemouse_usb_pairs_with_override(
    override_vid: Optional[int],
    override_pid: Optional[int],
) -> list[tuple[int, int]]:
    """Built-in pairs with optional user VID/PID tried first."""
    pairs = list(SPACEMOUSE_USB_PAIRS_DEFAULT)
    if override_vid is not None and override_pid is not None:
        pairs.insert(0, (override_vid, override_pid))
    return dedupe_usb_pairs(pairs)


def try_open_first_spacemouse(
    usb_pairs: Optional[list[tuple[int, int]]] = None,
) -> Optional[tuple[Any, int, int]]:
    """Open the first matching SpaceMouse HID device.

    Returns ``(hid.device, vendor_id, product_id)`` or ``None``. Caller must
    ``close()`` the device when done.

    Raises
    ------
    ImportError
        If ``hidapi`` is not installed (``pip install hidapi``).

    Parameters
    ----------
    usb_pairs :
        VID/PID pairs to try (after dedupe). Defaults to :data:`SPACEMOUSE_USB_PAIRS_DEFAULT`.
    """
    import hid

    pairs = dedupe_usb_pairs(list(usb_pairs or list(SPACEMOUSE_USB_PAIRS_DEFAULT)))
    for vendor_id, product_id in pairs:
        try:
            candidate = hid.device()
            candidate.open(vendor_id, product_id)
            candidate.set_nonblocking(True)
            return candidate, vendor_id, product_id
        except Exception:
            continue
    return None


def decode_spacemouse_motion_report(data: Sequence[int]) -> tuple[float, float, float, float, float, float]:
    """Decode HID report id 1 — translation desk frame + **wire-order** rotation words only.

    Translation (unchanged scaling / sign for ``tz``):

    - ``+tx``: bytes ``(1,2)`` — slide cap toward user's left
    - ``+ty``: bytes ``(3,4)`` — push cap toward the monitor
    - ``+tz``: ``-`` bytes ``(5,6)`` — lift cap upward (same convention as calibration script)

    Rotation: **no OSC / robosuite semantics** — only int16 pairs in report order:

    - ``r0``: bytes ``(7,8)``
    - ``r1``: bytes ``(9,10)``
    - ``r2``: bytes ``(11,12)``

    Map ``[r0, r1, r2]`` → arm roll / pitch / yaw with ``rotation_mix`` in dagger YAML
    (``apply_spacemouse_spatial_mapping``).

    Raises
    ------
    ValueError
        If ``data`` is too short or not a motion report.
    """
    if len(data) < 13:
        raise ValueError("SpaceMouse motion report must be at least 13 bytes")
    if data[0] != 1:
        raise ValueError(f"expected report id 1, got {data[0]}")

    def conv(b1: int, b2: int) -> float:
        v = (b2 << 8) | b1
        if v >= 32768:
            v -= 65536
        return float(v) / 350.0

    tx = conv(data[1], data[2])
    ty = conv(data[3], data[4])
    tz = -conv(data[5], data[6])
    r0 = conv(data[7], data[8])
    r1 = conv(data[9], data[10])
    r2 = conv(data[11], data[12])
    return tx, ty, tz, r0, r1, r2


def apply_spacemouse_spatial_mapping(
    tx: float,
    ty: float,
    tz: float,
    r0: float,
    r1: float,
    r2: float,
    translation_mix: np.ndarray,
    rotation_mix: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Apply task YAML ``spacemouse.spatial_mapping`` 3×3 mixes.

    ``translation_mix`` maps desk ``[tx, ty, tz]`` → arm translation deltas.
    ``rotation_mix`` maps **wire-order** ``[r0, r1, r2]`` from ``decode_spacemouse_motion_report``
    → OSC ``[roll, pitch, yaw]`` (same order as action indices 3–5).
    """
    t = translation_mix @ np.array([tx, ty, tz], dtype=np.float64)
    r = rotation_mix @ np.array([r0, r1, r2], dtype=np.float64)
    return float(t[0]), float(t[1]), float(t[2]), float(r[0]), float(r[1]), float(r[2])
