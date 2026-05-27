import threading
from datetime import UTC, datetime
from unittest.mock import patch

import filelock

from pdomain_ops.suite.registry import LocalTomlSuiteRegistry
from pdomain_ops.suite.types import InstalledApp

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _make_app(app_id: str, version: str = "1.0.0") -> InstalledApp:
    return InstalledApp(
        app_id=app_id,
        package=app_id.replace("-", "_"),
        version=version,
        binary="/usr/bin/python3",  # real path that exists on CI
        default_port=8001,
        icon="test",
        display_name=app_id,
        registered_at=_NOW,
    )


def test_register_writes_new_entry(tmp_path):
    toml_file = tmp_path / "installed.toml"
    registry = LocalTomlSuiteRegistry(root=toml_file)
    app = _make_app("pd-test-app")
    registry.register(app)
    apps = registry.list_installed()
    assert len(apps) == 1
    assert apps[0].app_id == "pd-test-app"


def test_register_refreshes_existing_entry(tmp_path):
    toml_file = tmp_path / "installed.toml"
    registry = LocalTomlSuiteRegistry(root=toml_file)
    registry.register(_make_app("pd-test-app", "1.0.0"))
    registry.register(_make_app("pd-test-app", "2.0.0"))
    apps = registry.list_installed()
    assert len(apps) == 1
    assert apps[0].version == "2.0.0"


def test_register_preserves_other_apps(tmp_path):
    toml_file = tmp_path / "installed.toml"
    registry = LocalTomlSuiteRegistry(root=toml_file)
    registry.register(_make_app("pd-app-a"))
    registry.register(_make_app("pd-app-b"))
    apps = registry.list_installed()
    app_ids = {a.app_id for a in apps}
    assert "pd-app-a" in app_ids
    assert "pd-app-b" in app_ids


def test_unregister_removes_block(tmp_path):
    toml_file = tmp_path / "installed.toml"
    registry = LocalTomlSuiteRegistry(root=toml_file)
    registry.register(_make_app("pd-test-app"))
    registry.unregister("pd-test-app")
    apps = registry.list_installed()
    assert apps == []


def test_unregister_missing_app_is_noop(tmp_path):
    toml_file = tmp_path / "installed.toml"
    registry = LocalTomlSuiteRegistry(root=toml_file)
    # Should not raise
    registry.unregister("nonexistent-app")


def test_write_uses_filelock(tmp_path):
    toml_file = tmp_path / "installed.toml"
    registry = LocalTomlSuiteRegistry(root=toml_file)
    app = _make_app("pd-test-app")

    lock_entered = []
    original_enter = filelock.FileLock.__enter__

    def spy_enter(self):
        lock_entered.append(True)
        return original_enter(self)

    with patch.object(filelock.FileLock, "__enter__", spy_enter):
        registry.register(app)

    assert len(lock_entered) > 0


def test_concurrent_registers_serialize(tmp_path):
    toml_file = tmp_path / "installed.toml"
    registry = LocalTomlSuiteRegistry(root=toml_file)

    app_a = _make_app("pd-app-a")
    app_b = _make_app("pd-app-b")

    errors = []

    def register_a():
        try:
            registry.register(app_a)
        except Exception as e:
            errors.append(e)

    def register_b():
        try:
            registry.register(app_b)
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=register_a)
    t2 = threading.Thread(target=register_b)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors
    apps = registry.list_installed()
    app_ids = {a.app_id for a in apps}
    assert "pd-app-a" in app_ids
    assert "pd-app-b" in app_ids
