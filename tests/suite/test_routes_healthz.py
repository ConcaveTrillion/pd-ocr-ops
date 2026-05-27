"""Tests for GET /healthz centralized health endpoint (Task 4.7)."""

from __future__ import annotations

import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from pdomain_ops.suite.routes import mount_routes
from pdomain_ops.suite.types import InstalledApp, SuiteAdapters

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_installed(
    app_id: str = "test-app",
    version: str = "0.1.0",
) -> InstalledApp:
    return InstalledApp(
        app_id=app_id,
        package=app_id.replace("-", "_"),
        version=version,
        binary="/usr/bin/python3",
        default_port=8000,
        icon="test",
        display_name=app_id,
    )


class _FakeRegistry:
    def __init__(self, apps=None):
        self._apps = apps or []

    def list_installed(self):
        return self._apps

    def register(self, app):
        pass

    def unregister(self, app_id):
        pass


class _FakePrefs:
    def read(self):
        from pdomain_ops.suite.types import UIPrefs

        return UIPrefs()

    def write_common(self, common):
        pass

    def write_app(self, app_id, payload):
        pass


class _FakeLauncher:
    async def launch(self, app):
        from pdomain_ops.suite.sibling_spawn import LaunchResultOpened

        return LaunchResultOpened(url="http://localhost:8001", spawned=False)


class _FakeAuth:
    async def authenticate(self, request):
        from pdomain_ops.suite.auth import Identity

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


def _make_adapters(suite_app: InstalledApp | None = None) -> SuiteAdapters:
    return SuiteAdapters(
        registry=_FakeRegistry(),
        prefs=_FakePrefs(),
        launcher=_FakeLauncher(),
        auth=_FakeAuth(),
        storage=_FakeStorage(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_healthz_returns_ok_with_metadata():
    """GET /healthz returns 200 + JSON with the four required fields."""
    suite_app = _make_installed(app_id="test-app", version="0.1.0")
    app = FastAPI()
    mount_routes(app, adapters=_make_adapters(), suite_app=suite_app)
    client = TestClient(app)

    resp = client.get("/healthz")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["app_id"] == "test-app"
    assert data["version"] == "0.1.0"
    assert "uptime_s" in data
    assert isinstance(data["uptime_s"], int | float)


def test_healthz_uptime_increases():
    """Second GET /healthz has a higher or equal uptime_s than the first."""
    suite_app = _make_installed()
    app = FastAPI()
    mount_routes(app, adapters=_make_adapters(), suite_app=suite_app)
    client = TestClient(app)

    resp1 = client.get("/healthz")
    time.sleep(0.05)  # small delay to let uptime_s tick
    resp2 = client.get("/healthz")

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp2.json()["uptime_s"] >= resp1.json()["uptime_s"]


def test_healthz_no_auth_required():
    """GET /healthz is publicly callable without credentials."""
    suite_app = _make_installed()
    app = FastAPI()
    mount_routes(app, adapters=_make_adapters(), suite_app=suite_app)
    client = TestClient(app)

    # No auth headers — must still return 200
    resp = client.get("/healthz")
    assert resp.status_code == 200


def test_healthz_mounted_without_suite_app_returns_unknown():
    """When suite_app is not supplied, /healthz returns unknown placeholders."""
    app = FastAPI()
    mount_routes(app, adapters=_make_adapters())
    client = TestClient(app)

    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["app_id"] == "unknown"
    assert data["version"] == "unknown"
