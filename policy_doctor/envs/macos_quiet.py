"""Reduce noisy macOS / pygame output when OpenCV and pygame both load SDL2."""

from __future__ import annotations

import re
import sys
import warnings

_OBJC_DUP_LINE = re.compile(
    r"^objc\[\d+\]: Class .+ is implemented in both .+ and .+\)\. This may cause",
)

_suppressed_stderr_installed = False


class _LineFilterStderr:
    """Drop whole lines matching a pattern; pass everything else to the real stream."""

    def __init__(self, underlying):
        self._u = underlying
        self._buf = ""

    def write(self, s):  # noqa: ANN001 — match TextIO API
        if not isinstance(s, str):
            try:
                return int(self._u.write(s))
            except Exception:
                return 0
        self._buf += s
        while True:
            nl = self._buf.find("\n")
            if nl < 0:
                break
            line = self._buf[: nl + 1]
            self._buf = self._buf[nl + 1 :]
            if _OBJC_DUP_LINE.match(line):
                continue
            self._u.write(line)
        return len(s)

    def flush(self) -> None:
        if self._buf:
            if not _OBJC_DUP_LINE.match(self._buf):
                self._u.write(self._buf)
            self._buf = ""
        self._u.flush()

    def __getattr__(self, name: str):
        return getattr(self._u, name)


def install_macos_sdl_noise_suppression() -> None:
    """Silence ObjC duplicate-class spam (two libSDL2) and pygame setuptools noise."""
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API",
        category=UserWarning,
        module="pygame.pkgdata",
    )
    global _suppressed_stderr_installed
    if sys.platform != "darwin" or _suppressed_stderr_installed:
        return
    if sys.stderr is None:
        return
    try:
        sys.stderr = _LineFilterStderr(sys.stderr)
    except Exception:
        return
    _suppressed_stderr_installed = True
