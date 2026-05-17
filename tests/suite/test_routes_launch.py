from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from pd_ocr_ops.suite.routes import mount_routes
from pd_ocr_ops.suite.sibling_spawn import LaunchResultOpened
from pd_ocr_ops.suite.types import InstalledApp, SuiteAdapters

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_installed(app_id: str, enabled: bool = True) -> InstalledApp:
    return InstalledApp(
        app_id=app_id,
        package=app_id.replace("-", "_"),
        version="1.0.0",
        binary="/usr/bin/python3",
        default_port=8001,
        icon="test",
        display_name=app_id,
        enabled=enabled,
        registered_at=_NOW,
    )


class _SpyLauncher:
    def __init__(self, result: LaunchResultOpened):
        self._result = result
        self.launched_apps = []

    async def launch(self, app):
        self.launched_apps.append(app)
        return self._result


class _FakeRegistry:
    def __init__(self, apps):
        self._apps = apps

    def list_installed(self):
        return self._apps

    def register(self, app):
        pass

    def unregister(self, app_id):
        pass


class _FakePrefs:
    def read(self):
        from pd_ocr_ops.suite.types import UIPrefs

        return UIPrefs()

    def write_common(self, common):
        pass

    def write_app(self, app_id, payload):
        pass


class _FakeAuth:
    async def authenticate(self, request):
        from pd_ocr_ops.suite.auth import Identity

        return Identity(user_id="local", display_name="Local User")

    async def is_authenticated(self, request):
        return True


class _FakeStorage:
    def read(self, key):
        return b""

    def write(self, key, data):
        pass

    def exists(self, key):
        return False

    def delete(self, key):
        pass

    def list_prefix(self, prefix):
        return []


def _make_adapters(apps=None, launcher=None):
    if launcher is None:
        launcher = _SpyLauncher(LaunchResultOpened(url="http://localhost:8001", spawned=False))
    return SuiteAdapters(
        registry=_FakeRegistry(apps or []),
        prefs=_FakePrefs(),
        launcher=launcher,
        auth=_FakeAuth(),
        storage=_FakeStorage(),
    )


def test_launch_unknown_app_returns_404():
    app = FastAPI()
    mount_routes(app, adapters=_make_adapters([]))
    client = TestClient(app)
    resp = client.post("/api/suite/launch?app_id=pd-unknown")
    assert resp.status_code == 404
    assert "unknown app" in resp.json()["detail"]


def test_launch_disabled_app_returns_409():
    installed = _make_installed("pd-app-a", enabled=False)
    app = FastAPI()
    mount_routes(app, adapters=_make_adapters([installed]))
    client = TestClient(app)
    resp = client.post("/api/suite/launch?app_id=pd-app-a")
    assert resp.status_code == 409


def test_launch_calls_launcher_with_app():
    installed = _make_installed("pd-app-a")
    spy = _SpyLauncher(LaunchResultOpened(url="http://localhost:8001", spawned=True, pid=12345))
    app = FastAPI()
    mount_routes(app, adapters=_make_adapters([installed], launcher=spy))
    client = TestClient(app)
    resp = client.post("/api/suite/launch?app_id=pd-app-a")
    assert resp.status_code == 200
    assert len(spy.launched_apps) == 1
    assert spy.launched_apps[0].app_id == "pd-app-a"


def test_launch_already_running_no_spawn():
    installed = _make_installed("pd-app-a")
    spy = _SpyLauncher(LaunchResultOpened(url="http://localhost:8001", spawned=False))
    app = FastAPI()
    mount_routes(app, adapters=_make_adapters([installed], launcher=spy))
    client = TestClient(app)
    resp = client.post("/api/suite/launch?app_id=pd-app-a")
    assert resp.status_code == 200
    data = resp.json()
    assert data["spawned"] is False
