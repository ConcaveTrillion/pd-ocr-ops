from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pd_ocr_ops.suite.sibling_spawn import LaunchResultOpened, LocalSpawnLauncher
from pd_ocr_ops.suite.types import InstalledApp

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


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


@pytest.mark.asyncio
async def test_launch_returns_opened_no_spawn_when_healthz_passes():
    app = _make_installed(8001)
    launcher = LocalSpawnLauncher()

    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        result = await launcher.launch(app)

    assert isinstance(result, LaunchResultOpened)
    assert result.spawned is False
    assert result.pid is None
    assert "8001" in result.url


@pytest.mark.asyncio
async def test_already_running_does_not_call_subprocess_popen():
    app = _make_installed(8001)
    launcher = LocalSpawnLauncher()

    mock_response = MagicMock()
    mock_response.status_code = 200

    with (
        patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_get.return_value = mock_response
        await launcher.launch(app)

    mock_popen.assert_not_called()
