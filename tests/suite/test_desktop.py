import sys
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from pd_ocr_ops.suite.desktop import install_shortcut, remove_shortcut
from pd_ocr_ops.suite.types import InstalledApp

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_installed() -> InstalledApp:
    return InstalledApp(
        app_id="pd-app-a",
        package="pd_app_a",
        version="1.0.0",
        binary="/usr/bin/python3",
        default_port=8001,
        icon="test",
        display_name="pd-app-a",
        registered_at=_NOW,
    )


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_install_shortcut_raises_not_implemented_on_platform(platform):
    with patch.object(sys, "platform", platform):
        with pytest.raises(NotImplementedError) as exc_info:
            install_shortcut(_make_installed())
    msg = str(exc_info.value).lower()
    # darwin maps to "macos" in the message; win32 maps to "windows"
    platform_aliases = {"darwin": "macos", "win32": "windows", "linux": "linux"}
    expected_platform_str = platform_aliases.get(platform, platform)
    assert expected_platform_str in msg or platform in msg
    assert "phase 4" in msg or "deferred" in msg


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_remove_shortcut_raises_not_implemented_on_each_platform(platform):
    with patch.object(sys, "platform", platform):
        with pytest.raises(NotImplementedError):
            remove_shortcut("pd-app-a")


def test_install_shortcut_signature_accepts_installed_app():
    with pytest.raises(NotImplementedError):
        install_shortcut(_make_installed())


def test_install_shortcut_unknown_platform_raises_generic():
    with patch.object(sys, "platform", "aix"):
        with pytest.raises(NotImplementedError) as exc_info:
            install_shortcut(_make_installed())
    assert "unsupported platform" in str(exc_info.value)
