from fastapi import FastAPI
from fastapi.testclient import TestClient

from pd_ocr_ops.suite.routes import mount_routes
from pd_ocr_ops.suite.types import SuiteAdapters


class _SpyPrefs:
    def __init__(self):
        self.write_app_calls = []

    def read(self):
        from pd_ocr_ops.suite.types import UIPrefs

        return UIPrefs()

    def write_common(self, common):
        pass

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


def _make_adapters(prefs=None):
    return SuiteAdapters(
        registry=_FakeRegistry(),
        prefs=prefs or _SpyPrefs(),
        launcher=_FakeLauncher(),
        auth=_FakeAuth(),
        storage=_FakeStorage(),
    )


def test_put_app_invokes_write_app():
    spy = _SpyPrefs()
    app = FastAPI()
    mount_routes(app, adapters=_make_adapters(spy))
    client = TestClient(app)
    payload = {"show_match_diff_default": "fuzzy-and-mismatch"}
    resp = client.put("/api/suite/prefs/apps/pd-ocr-labeler-spa", json=payload)
    assert resp.status_code == 204
    assert len(spy.write_app_calls) == 1
    app_id, recorded_payload = spy.write_app_calls[0]
    assert app_id == "pd-ocr-labeler-spa"
    assert recorded_payload == payload


def test_put_app_unknown_app_id_still_writes():
    spy = _SpyPrefs()
    app = FastAPI()
    mount_routes(app, adapters=_make_adapters(spy))
    client = TestClient(app)
    resp = client.put("/api/suite/prefs/apps/pd-never-heard-of-it", json={"key": "val"})
    assert resp.status_code == 204
    assert len(spy.write_app_calls) == 1
    assert spy.write_app_calls[0][0] == "pd-never-heard-of-it"
