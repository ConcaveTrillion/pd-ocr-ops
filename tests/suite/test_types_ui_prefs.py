import pytest
from pydantic import ValidationError

from pdomain_ops.suite.types import CommonUIPrefs, UIPrefs


def test_ui_prefs_defaults_match_spec():
    prefs = UIPrefs()
    assert prefs.common.theme == "dark"
    assert prefs.common.accent == "#d6925a"
    assert prefs.common.density == "normal"
    assert prefs.apps == {}


def test_ui_prefs_rejects_bad_hex_accent():
    with pytest.raises(ValidationError):
        UIPrefs(common={"theme": "dark", "accent": "orange", "layer_colors": {}})


def test_ui_prefs_rejects_bad_font_size():
    with pytest.raises(ValidationError):
        UIPrefs(common={"theme": "dark", "font_size_base": 200, "layer_colors": {}})


def test_ui_prefs_apps_freeform_dict():
    apps_data = {"pdomain-ocr-labeler-spa": {"show_match_diff_default": "fuzzy-and-mismatch"}}
    prefs = UIPrefs(apps=apps_data)
    data = prefs.model_dump(mode="json")
    roundtripped = UIPrefs.model_validate(data)
    assert roundtripped.apps == apps_data


def test_common_ui_prefs_font_scale_default():
    prefs = CommonUIPrefs()
    assert prefs.font_scale == 1.0


def test_common_ui_prefs_font_scale_roundtrip():
    prefs = CommonUIPrefs(font_scale=1.2)
    data = prefs.model_dump(mode="json")
    assert data["font_scale"] == pytest.approx(1.2)
    roundtripped = CommonUIPrefs.model_validate(data)
    assert roundtripped.font_scale == pytest.approx(1.2)


def test_common_ui_prefs_font_scale_too_low():
    with pytest.raises(ValidationError):
        CommonUIPrefs(font_scale=0.7)


def test_common_ui_prefs_font_scale_too_high():
    with pytest.raises(ValidationError):
        CommonUIPrefs(font_scale=1.5)
