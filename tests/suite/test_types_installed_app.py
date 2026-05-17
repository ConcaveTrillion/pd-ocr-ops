from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from pd_ocr_ops.suite.types import InstalledApp


def _make_app(**kwargs):
    defaults = dict(
        app_id="pd-ocr-labeler-spa",
        package="pd_ocr_labeler_spa",
        version="0.4.2",
        binary="/usr/local/bin/pd-ocr-labeler",
        default_port=8001,
        icon="labeler",
        display_name="OCR Labeler",
        registered_at=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return InstalledApp(**defaults)


def test_installed_app_constructs():
    app = _make_app()
    assert app.app_id == "pd-ocr-labeler-spa"
    assert app.enabled is True


def test_installed_app_requires_absolute_binary():
    with pytest.raises(ValidationError) as exc_info:
        _make_app(binary="relative/path/binary")
    assert "absolute" in str(exc_info.value).lower()


def test_installed_app_roundtrip():
    app = _make_app()
    data = app.model_dump(mode="json")
    roundtripped = InstalledApp.model_validate(data)
    assert roundtripped == app
