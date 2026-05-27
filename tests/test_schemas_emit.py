import json
import subprocess
import sys
from pathlib import Path

# Locate the repo root from this test file so the schemas-emit subprocess
# runs from the correct cwd in any checkout. Previously hardcoded to
# /workspaces/ocr-container/pdomain-ocr-ops, which only existed in the local
# dev container and failed everywhere else (including CI).
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _emit() -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "pdomain_ops.schemas"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"schemas emit failed:\n{result.stderr}"
    return json.loads(result.stdout)


def test_emit_returns_top_level_dict():
    data = _emit()
    assert isinstance(data, dict)
    assert len(data) > 0


def test_emit_includes_suite_app():
    data = _emit()
    assert "SuiteApp" in data
    schema = data["SuiteApp"]
    props = schema.get("properties", {})
    assert "app_id" in props
    assert "display_name" in props
    assert "package" in props
    assert "default_port" in props
    assert "icon" in props


def test_emit_includes_installed_app():
    data = _emit()
    assert "InstalledApp" in data


def test_emit_includes_ui_prefs_and_common_ui_prefs():
    data = _emit()
    assert "UIPrefs" in data
    assert "CommonUIPrefs" in data


def test_emit_includes_layer_colors():
    data = _emit()
    assert "LayerColors" in data


def test_emit_includes_stage_result():
    data = _emit()
    assert "StageResult" in data


def test_emit_includes_job_status_job_event_job_spec():
    data = _emit()
    assert "JobStatus" in data
    assert "JobEvent" in data
    assert "JobSpec" in data


def test_emit_includes_launch_result_discriminated_union():
    data = _emit()
    assert "LaunchResult" in data
    schema = data["LaunchResult"]
    # Should use oneOf or anyOf with discriminator
    assert "oneOf" in schema or "anyOf" in schema


def test_emit_stage_result_device_enum_values():
    data = _emit()
    schema = data["StageResult"]
    # The device enum values should be in the schema
    schema_str = json.dumps(schema)
    for expected in ["local", "mps", "cpu", "modal", "shared_container"]:
        assert expected in schema_str, f"Missing device value: {expected}"
