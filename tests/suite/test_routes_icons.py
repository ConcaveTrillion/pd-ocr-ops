from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from pdomain_ops.suite.routes import mount_routes
from pdomain_ops.suite.types import InstalledApp, SuiteAdapters

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


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


def _make_installed_with_icon(tmp_path, app_id: str = "pd-app-a", size: int = 128) -> InstalledApp:
    """Create a fake installed app with a real icon file."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    binary = bin_dir / app_id
    binary.write_bytes(b"#!/bin/sh\n")
    # The route resolves: binary.parent.parent / "icons" / f"{size}x{size}.png"
    icons_dir = tmp_path / "icons"
    icons_dir.mkdir(exist_ok=True)
    icon_file = icons_dir / f"{size}x{size}.png"
    icon_file.write_bytes(b"\x89PNG\r\n")  # minimal PNG header
    return InstalledApp(
        app_id=app_id,
        package=app_id.replace("-", "_"),
        version="1.0.0",
        binary=str(binary),
        default_port=8001,
        icon="test",
        display_name=app_id,
        registered_at=_NOW,
    )


def _make_adapters(apps=None):
    return SuiteAdapters(
        registry=_FakeRegistry(apps or []),
        prefs=_FakePrefs(),
        launcher=_FakeLauncher(),
        auth=_FakeAuth(),
        storage=_FakeStorage(),
    )


def test_get_icon_returns_png_for_known_app(tmp_path):
    installed = _make_installed_with_icon(tmp_path, "pd-app-a", 128)
    app = FastAPI()
    mount_routes(app, adapters=_make_adapters([installed]))
    client = TestClient(app)
    resp = client.get("/api/icons/128?app_id=pd-app-a")
    assert resp.status_code == 200
    assert "image/png" in resp.headers["content-type"]


def test_get_icon_missing_returns_404(tmp_path):
    # App without icon dir
    binary = tmp_path / "bin" / "pd-app-a"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"#!/bin/sh\n")
    installed = InstalledApp(
        app_id="pd-app-a",
        package="pd_app_a",
        version="1.0.0",
        binary=str(binary),
        default_port=8001,
        icon="test",
        display_name="pd-app-a",
        registered_at=_NOW,
    )
    app = FastAPI()
    mount_routes(app, adapters=_make_adapters([installed]))
    client = TestClient(app)
    resp = client.get("/api/icons/128?app_id=pd-app-a")
    assert resp.status_code == 404


def test_get_icon_unsupported_size_returns_400(tmp_path):
    installed = _make_installed_with_icon(tmp_path)
    app = FastAPI()
    mount_routes(app, adapters=_make_adapters([installed]))
    client = TestClient(app)
    resp = client.get("/api/icons/999?app_id=pd-app-a")
    assert resp.status_code == 400
