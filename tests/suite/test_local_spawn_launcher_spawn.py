from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from pdomain_ops.suite.sibling_spawn import (
    LaunchResultOpened,
    LaunchTimeoutError,
    LocalSpawnLauncher,
)
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


@pytest.mark.asyncio
async def test_launch_spawns_when_healthz_fails():
    app = _make_installed(8001)
    launcher = LocalSpawnLauncher()
    mock_proc = _make_mock_popen(12345)

    call_count = [0]

    async def mock_get(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise httpx.ConnectError("refused")
        resp = MagicMock()
        resp.status_code = 200
        return resp

    with (
        patch("httpx.AsyncClient.get", side_effect=mock_get),
        patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await launcher.launch(app)

    assert isinstance(result, LaunchResultOpened)
    assert result.spawned is True
    assert result.pid == 12345
    mock_popen.assert_called_once()
    call_args = mock_popen.call_args[0][0]
    assert "/usr/bin/python3" in call_args
    assert "--port" in call_args
    assert "8001" in call_args


@pytest.mark.asyncio
async def test_launch_polls_until_ready():
    app = _make_installed(8001)
    launcher = LocalSpawnLauncher()
    mock_proc = _make_mock_popen()

    call_count = [0]

    async def mock_get(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] < 4:
            raise httpx.ConnectError("refused")
        resp = MagicMock()
        resp.status_code = 200
        return resp

    with (
        patch("httpx.AsyncClient.get", side_effect=mock_get),
        patch("subprocess.Popen", return_value=mock_proc),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await launcher.launch(app)

    assert call_count[0] == 4


@pytest.mark.asyncio
async def test_launch_times_out():
    app = _make_installed(8001)
    launcher = LocalSpawnLauncher(timeout_s=0.001)
    mock_proc = _make_mock_popen()

    async def mock_get(*args, **kwargs):
        raise httpx.ConnectError("refused")

    with (
        patch("httpx.AsyncClient.get", side_effect=mock_get),
        patch("subprocess.Popen", return_value=mock_proc),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        with pytest.raises(LaunchTimeoutError):
            await launcher.launch(app)

        # Process should NOT be terminated on timeout
        mock_proc.terminate.assert_not_called()


@pytest.mark.asyncio
async def test_launch_returns_pid_from_popen():
    app = _make_installed(8001)
    launcher = LocalSpawnLauncher()
    mock_proc = _make_mock_popen(pid=99999)

    call_count = [0]

    async def mock_get(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise httpx.ConnectError("refused")
        resp = MagicMock()
        resp.status_code = 200
        return resp

    with (
        patch("httpx.AsyncClient.get", side_effect=mock_get),
        patch("subprocess.Popen", return_value=mock_proc),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await launcher.launch(app)

    assert result.pid == 99999
