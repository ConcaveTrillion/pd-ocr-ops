import json
import threading
from unittest.mock import patch

import filelock

from pdomain_ops.suite.prefs import LocalFilePrefs
from pdomain_ops.suite.types import CommonUIPrefs, LayerColors


def _make_common(**kwargs) -> CommonUIPrefs:
    defaults = {
        "theme": "dark",
        "density": "normal",
        "accent": "#d6925a",
        "font_size_base": 12,
        "layer_colors": LayerColors(),
    }
    defaults.update(kwargs)
    return CommonUIPrefs(**defaults)


def test_write_common_creates_file_if_missing(tmp_path):
    prefs_file = tmp_path / "ui-prefs.json"
    writer = LocalFilePrefs(root=prefs_file)
    common = _make_common(theme="light")
    writer.write_common(common)
    assert prefs_file.exists()
    data = json.loads(prefs_file.read_text())
    assert data["common"]["theme"] == "light"


def test_write_common_preserves_apps_section(tmp_path):
    prefs_file = tmp_path / "ui-prefs.json"
    writer = LocalFilePrefs(root=prefs_file)
    # Pre-populate with apps data
    writer.write_app("pd-app-a", {"setting": "value"})
    # Now write common
    writer.write_common(_make_common(theme="light"))
    data = json.loads(prefs_file.read_text())
    assert data["apps"]["pd-app-a"]["setting"] == "value"
    assert data["common"]["theme"] == "light"


def test_write_app_creates_apps_section_if_missing(tmp_path):
    prefs_file = tmp_path / "ui-prefs.json"
    writer = LocalFilePrefs(root=prefs_file)
    writer.write_app("pd-test-app", {"key": "val"})
    data = json.loads(prefs_file.read_text())
    assert data["apps"]["pd-test-app"]["key"] == "val"


def test_write_app_replaces_only_that_app(tmp_path):
    prefs_file = tmp_path / "ui-prefs.json"
    writer = LocalFilePrefs(root=prefs_file)
    writer.write_app("pd-app-a", {"a_key": "a_val"})
    writer.write_app("pd-app-b", {"b_key": "b_val"})
    writer.write_app("pd-app-a", {"a_key": "updated"})
    data = json.loads(prefs_file.read_text())
    assert data["apps"]["pd-app-a"]["a_key"] == "updated"
    assert data["apps"]["pd-app-b"]["b_key"] == "b_val"


def test_write_uses_filelock(tmp_path):
    prefs_file = tmp_path / "ui-prefs.json"
    writer = LocalFilePrefs(root=prefs_file)
    lock_entered = []
    original_enter = filelock.FileLock.__enter__

    def spy_enter(self):
        lock_entered.append(True)
        return original_enter(self)

    with patch.object(filelock.FileLock, "__enter__", spy_enter):
        writer.write_common(_make_common())

    assert len(lock_entered) > 0


def test_concurrent_writes_serialize(tmp_path):
    prefs_file = tmp_path / "ui-prefs.json"
    writer = LocalFilePrefs(root=prefs_file)
    errors = []

    def write_labeler():
        try:
            writer.write_app("labeler", {"key": "val_labeler"})
        except Exception as e:
            errors.append(e)

    def write_pgdp():
        try:
            writer.write_app("pgdp", {"key": "val_pgdp"})
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=write_labeler)
    t2 = threading.Thread(target=write_pgdp)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors
    data = json.loads(prefs_file.read_text())
    assert "labeler" in data["apps"]
    assert "pgdp" in data["apps"]
