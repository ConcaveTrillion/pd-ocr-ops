from unittest.mock import patch

import filelock

from pd_ocr_ops.suite.registry import LocalTomlSuiteRegistry

_SPEC_EXAMPLE_TOML = """
[apps.pd-ocr-labeler-spa]
app_id = "pd-ocr-labeler-spa"
package = "pd_ocr_labeler_spa"
version = "0.4.2"
binary = "/usr/bin/python3"
default_port = 8001
icon = "labeler"
display_name = "OCR Labeler"
enabled = true
registered_at = "2026-01-01T00:00:00+00:00"
"""


def test_read_empty_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PD_SUITE_DATA_DIR", str(tmp_path))
    registry = LocalTomlSuiteRegistry(root=tmp_path / "installed.toml")
    assert registry.list_installed() == []


def test_read_parses_spec_example(tmp_path, monkeypatch):
    monkeypatch.setenv("PD_SUITE_DATA_DIR", str(tmp_path))
    toml_file = tmp_path / "installed.toml"
    toml_file.write_text(_SPEC_EXAMPLE_TOML)
    registry = LocalTomlSuiteRegistry(root=toml_file)
    apps = registry.list_installed()
    assert len(apps) == 1
    assert apps[0].app_id == "pd-ocr-labeler-spa"
    assert apps[0].version == "0.4.2"
    assert apps[0].default_port == 8001


def test_read_prunes_stale_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("PD_SUITE_DATA_DIR", str(tmp_path))
    toml_content = (
        _SPEC_EXAMPLE_TOML
        + """
[apps.pd-stale-app]
app_id = "pd-stale-app"
package = "pd_stale"
version = "0.1.0"
binary = "/nonexistent/bin/pd-stale"
default_port = 9999
icon = "stale"
display_name = "Stale App"
enabled = true
registered_at = "2026-01-01T00:00:00+00:00"
"""
    )
    toml_file = tmp_path / "installed.toml"
    toml_file.write_text(toml_content)
    registry = LocalTomlSuiteRegistry(root=toml_file)
    apps = registry.list_installed()
    # stale entry (binary doesn't exist) is dropped from result
    assert len(apps) == 1
    assert apps[0].app_id == "pd-ocr-labeler-spa"


def test_read_uses_filelock(tmp_path, monkeypatch):
    monkeypatch.setenv("PD_SUITE_DATA_DIR", str(tmp_path))
    toml_file = tmp_path / "installed.toml"
    toml_file.write_text(_SPEC_EXAMPLE_TOML)
    registry = LocalTomlSuiteRegistry(root=toml_file)

    lock_entered = []
    original_enter = filelock.FileLock.__enter__

    def spy_enter(self):
        lock_entered.append(True)
        return original_enter(self)

    with patch.object(filelock.FileLock, "__enter__", spy_enter):
        registry.list_installed()

    assert len(lock_entered) > 0
