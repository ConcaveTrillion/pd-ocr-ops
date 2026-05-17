"""Desktop shortcut stubs — Phase 4 implementation deferred."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pd_ocr_ops.suite.types import InstalledApp

_PLATFORM_MESSAGES = {
    "linux": (
        "Desktop shortcut install not yet implemented on linux "
        "(deferred to Phase 4 of the cross-cut design)"
    ),
    "darwin": (
        "Desktop shortcut install not yet implemented on macOS "
        "(deferred to Phase 4 of the cross-cut design)"
    ),
    "win32": (
        "Desktop shortcut install not yet implemented on Windows "
        "(deferred to Phase 4 of the cross-cut design)"
    ),
}


def install_shortcut(app: InstalledApp) -> None:
    """TODO Phase 4: write .desktop (Linux) / .app (macOS) / .lnk (Windows).

    For Phase 1 this is a stub — each pd-* app's CLI flag exists so the
    surface is wired when the platform code lands.
    """
    plat = sys.platform
    msg = _PLATFORM_MESSAGES.get(plat, f"unsupported platform: {plat}")
    raise NotImplementedError(msg)


def remove_shortcut(app_id: str) -> None:
    """TODO Phase 4: remove .desktop (Linux) / .app (macOS) / .lnk (Windows)."""
    plat = sys.platform
    msg = _PLATFORM_MESSAGES.get(plat, f"unsupported platform: {plat}")
    raise NotImplementedError(msg)
