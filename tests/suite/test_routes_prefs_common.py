from fastapi import FastAPI
from fastapi.testclient import TestClient

from pdomain_ops.suite.routes import mount_routes
from pdomain_ops.suite.types import CommonUIPrefs, SuiteAdapters, UIPrefs


class _SpyPrefs:
    def __init__(self, initial_prefs=None):
        self._prefs = initial_prefs or UIPrefs()
        self.write_common_calls = []
        self.write_app_calls = []

    def read(self):
        return self._prefs

    def write_common(self, common):
        self.write_common_calls.append(common)

    def write_app(self, app_id, payload):
        self.write_app_calls.append((app_id, payload))


class _FakeRegistry:
    def list_installed(self):
        return []

    def register(self, app):
        pass

    def unregister(self, app_id):
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


def _make_adapters(prefs=None):
    return SuiteAdapters(
        registry=_FakeRegistry(),
        prefs=prefs or _SpyPrefs(),
        launcher=_FakeLauncher(),
        auth=_FakeAuth(),
        storage=_FakeStorage(),
    )


def test_get_prefs_returns_full_shape():
    prefs_with_apps = UIPrefs(apps={"pd-app-a": {"setting": "value"}})
    spy = _SpyPrefs(prefs_with_apps)
    app = FastAPI()
    mount_routes(app, adapters=_make_adapters(spy))
    client = TestClient(app)
    resp = client.get("/api/suite/prefs")
    assert resp.status_code == 200
    data = resp.json()
    assert "common" in data
    assert "apps" in data
    assert data["apps"]["pd-app-a"]["setting"] == "value"


def test_put_common_invokes_write_common():
    spy = _SpyPrefs()
    app = FastAPI()
    mount_routes(app, adapters=_make_adapters(spy))
    client = TestClient(app)
    payload = {
        "theme": "light",
        "density": "compact",
        "accent": "#ff0000",
        "font_size_base": 14,
        "layer_colors": {
            "word": "#4a9eff",
            "line": "#ff9f4a",
            "para": "#4aff9f",
            "block": "#ff4a9f",
        },
    }
    resp = client.put("/api/suite/prefs/common", json=payload)
    assert resp.status_code == 204
    assert len(spy.write_common_calls) == 1
    assert isinstance(spy.write_common_calls[0], CommonUIPrefs)
    assert spy.write_common_calls[0].theme == "light"


def test_put_common_validates_payload():
    spy = _SpyPrefs()
    app = FastAPI()
    mount_routes(app, adapters=_make_adapters(spy))
    client = TestClient(app)
    resp = client.put("/api/suite/prefs/common", json={"accent": "not-a-hex-color"})
    assert resp.status_code == 422
