import json
import warnings
from unittest.mock import patch

import filelock

from pd_ocr_ops.suite.prefs import LocalFilePrefs

_SPEC_EXAMPLE_JSON = json.dumps(
    {
        "common": {
            "theme": "dark",
            "density": "compact",
            "accent": "#d6925a",
            "font_size_base": 12,
            "layer_colors": {
                "word": "#4a9eff",
                "line": "#ff9f4a",
                "para": "#4aff9f",
                "block": "#ff4a9f",
            },
        },
        "apps": {
            "pd-ocr-labeler-spa": {
                "show_match_diff_default": "fuzzy-and-mismatch",
            }
        },
    }
)


def test_read_returns_defaults_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PD_SUITE_DATA_DIR", str(tmp_path))
    prefs_file = tmp_path / "ui-prefs.json"
    reader = LocalFilePrefs(root=prefs_file)
    result = reader.read()
    assert result.common.theme == "dark"
    # File should NOT be created on read
    assert not prefs_file.exists()


def test_read_parses_spec_example(tmp_path, monkeypatch):
    monkeypatch.setenv("PD_SUITE_DATA_DIR", str(tmp_path))
    prefs_file = tmp_path / "ui-prefs.json"
    prefs_file.write_text(_SPEC_EXAMPLE_JSON)
    reader = LocalFilePrefs(root=prefs_file)
    result = reader.read()
    assert result.common.theme == "dark"
    assert result.common.density == "compact"
    assert result.common.layer_colors.word == "#4a9eff"
    assert result.apps["pd-ocr-labeler-spa"]["show_match_diff_default"] == "fuzzy-and-mismatch"


def test_read_unknown_keys_in_common_section_ignored_with_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("PD_SUITE_DATA_DIR", str(tmp_path))
    prefs_file = tmp_path / "ui-prefs.json"
    prefs_file.write_text(json.dumps({"common": {"theme": "dark", "x_unknown": "val"}}))
    reader = LocalFilePrefs(root=prefs_file)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = reader.read()
    assert result.common.theme == "dark"
    warning_msgs = [str(warning.message) for warning in w]
    assert any("x_unknown" in msg for msg in warning_msgs)


def test_read_uses_filelock(tmp_path, monkeypatch):
    monkeypatch.setenv("PD_SUITE_DATA_DIR", str(tmp_path))
    prefs_file = tmp_path / "ui-prefs.json"
    prefs_file.write_text(_SPEC_EXAMPLE_JSON)
    reader = LocalFilePrefs(root=prefs_file)

    lock_entered = []
    original_enter = filelock.FileLock.__enter__

    def spy_enter(self):
        lock_entered.append(True)
        return original_enter(self)

    with patch.object(filelock.FileLock, "__enter__", spy_enter):
        reader.read()

    assert len(lock_entered) > 0
