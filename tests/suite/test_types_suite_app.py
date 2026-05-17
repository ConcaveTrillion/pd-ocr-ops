import pytest
from pydantic import ValidationError

from pd_ocr_ops.suite.types import SuiteApp


def test_suite_app_constructs_minimum_fields():
    app = SuiteApp(
        app_id="pd-ocr-labeler-spa",
        display_name="OCR Labeler",
        package="pd_ocr_labeler_spa",
        default_port=8001,
        icon="labeler",
    )
    assert app.app_id == "pd-ocr-labeler-spa"
    assert app.description is None
    assert app.binary_name is None


def test_suite_app_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        SuiteApp(
            app_id="x",
            display_name="X",
            package="x",
            default_port=8000,
            icon="x",
            unknown_field="oops",
        )


def test_suite_app_roundtrip_json():
    app = SuiteApp(
        app_id="pd-ocr-labeler-spa",
        display_name="OCR Labeler",
        package="pd_ocr_labeler_spa",
        default_port=8001,
        icon="labeler",
        description="Labels OCR output",
        binary_name="pd-ocr-labeler",
    )
    roundtripped = SuiteApp.model_validate(app.model_dump())
    assert roundtripped == app
