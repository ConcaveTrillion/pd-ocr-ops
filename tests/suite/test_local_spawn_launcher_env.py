from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from pdomain_ops.suite.sibling_spawn import LocalSpawnLauncher
from pdomain_ops.suite.types import InstalledApp

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _make_installed(port: int = 8001) -> InstalledApp:
    return InstalledApp(
        app_id="pd-app-a",
        package="pd_app_a",
        version="1.0.0",
        binary="/usr/bin/python3",
        default_port=port,
        icon="test",
        display_name="pd-app-a",
        registered_at=_NOW,
    )


def _make_mock_popen(pid: int = 12345):
    mock_proc = MagicMock()
    mock_proc.pid = pid
    return mock_proc


async def _mock_get_fail(*args, **kwargs):
    raise httpx.ConnectError("refused")


@pytest.mark.asyncio
async def test_spawn_inherits_pd_suite_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PD_SUITE_DATA_DIR", str(tmp_path))
    app = _make_installed()
    launcher = LocalSpawnLauncher()

    call_count = [0]

    async def mock_get(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise httpx.ConnectError("refused")
        resp = MagicMock()
        resp.status_code = 200
        return resp

    captured_env = {}

    def spy_popen(cmd, **kwargs):
        captured_env.update(kwargs.get("env", {}))
        return _make_mock_popen()

    with (
        patch("httpx.AsyncClient.get", side_effect=mock_get),
        patch("subprocess.Popen", side_effect=spy_popen),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await launcher.launch(app)

    assert "PD_SUITE_DATA_DIR" in captured_env
    assert captured_env["PD_SUITE_DATA_DIR"] == str(tmp_path)


@pytest.mark.asyncio
async def test_spawn_does_not_inherit_arbitrary_parent_env(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "hunter2")
    app = _make_installed()
    launcher = LocalSpawnLauncher()

    call_count = [0]

    async def mock_get(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise httpx.ConnectError("refused")
        resp = MagicMock()
        resp.status_code = 200
        return resp

    captured_env = {}

    def spy_popen(cmd, **kwargs):
        captured_env.update(kwargs.get("env", {}))
        return _make_mock_popen()

    with (
        patch("httpx.AsyncClient.get", side_effect=mock_get),
        patch("subprocess.Popen", side_effect=spy_popen),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await launcher.launch(app)

    assert "SECRET_KEY" not in captured_env


@pytest.mark.asyncio
async def test_spawn_cwd_is_home_dir():
    from pathlib import Path

    app = _make_installed()
    launcher = LocalSpawnLauncher()

    call_count = [0]

    async def mock_get(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise httpx.ConnectError("refused")
        resp = MagicMock()
        resp.status_code = 200
        return resp

    captured_cwd = {}

    def spy_popen(cmd, **kwargs):
        captured_cwd["cwd"] = kwargs.get("cwd")
        return _make_mock_popen()

    with (
        patch("httpx.AsyncClient.get", side_effect=mock_get),
        patch("subprocess.Popen", side_effect=spy_popen),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await launcher.launch(app)

    assert captured_cwd["cwd"] == Path.home()
