from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from pd_ocr_ops.suite.routes import mount_routes
from pd_ocr_ops.suite.types import InstalledApp, SuiteAdapters

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_installed(app_id: str) -> InstalledApp:
    return InstalledApp(
        app_id=app_id,
        package=app_id.replace("-", "_"),
        version="1.0.0",
        binary="/usr/bin/python3",
        default_port=8001,
        icon="test",
        display_name=app_id,
        registered_at=_NOW,
    )


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


class _FakeLauncher:
    async def launch(self, app):
        from pd_ocr_ops.suite.sibling_spawn import LaunchResultOpened

        return LaunchResultOpened(url="http://localhost:8001", spawned=False)


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


def _make_adapters(apps=None):
    return SuiteAdapters(
        registry=_FakeRegistry(apps or []),
        prefs=_FakePrefs(),
        launcher=_FakeLauncher(),
        auth=_FakeAuth(),
        storage=_FakeStorage(),
    )


def test_get_installed_returns_registry_list():
    apps = [_make_installed("pd-app-a"), _make_installed("pd-app-b")]
    app = FastAPI()
    mount_routes(app, adapters=_make_adapters(apps))
    client = TestClient(app)
    resp = client.get("/api/suite/installed")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2


def test_get_installed_serializes_full_shape():
    apps = [_make_installed("pd-app-a")]
    app = FastAPI()
    mount_routes(app, adapters=_make_adapters(apps))
    client = TestClient(app)
    resp = client.get("/api/suite/installed")
    data = resp.json()
    item = data[0]
    for field in [
        "app_id",
        "version",
        "binary",
        "default_port",
        "icon",
        "display_name",
        "enabled",
        "registered_at",
    ]:
        assert field in item, f"Missing field: {field}"


def test_get_installed_empty():
    app = FastAPI()
    mount_routes(app, adapters=_make_adapters([]))
    client = TestClient(app)
    resp = client.get("/api/suite/installed")
    assert resp.status_code == 200
    assert resp.json() == []
