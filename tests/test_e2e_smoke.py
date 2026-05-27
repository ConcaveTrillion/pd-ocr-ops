"""End-to-end smoke tests: real FastAPI app + suite adapters."""

import json
import subprocess
import sys
from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from pdomain_ops import mount_routes
from pdomain_ops.suite.registry import LocalTomlSuiteRegistry
from pdomain_ops.suite.types import InstalledApp

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_e2e_mount_routes_real_fastapi_app(tmp_path, monkeypatch):
    monkeypatch.setenv("PD_SUITE_DATA_DIR", str(tmp_path))

    # Create the FastAPI app with real local adapters
    app = FastAPI()
    mount_routes(app)

    # Register an app via LocalTomlSuiteRegistry
    registry = LocalTomlSuiteRegistry(root=tmp_path / "installed.toml")
    installed = InstalledApp(
        app_id="pd-test-app",
        package="pd_test_app",
        version="1.0.0",
        binary="/usr/bin/python3",
        default_port=8001,
        icon="test",
        display_name="Test App",
        registered_at=_NOW,
    )
    registry.register(installed)

    client = TestClient(app)

    # GET /api/suite/installed -> 200 with the registered app
    resp = client.get("/api/suite/installed")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["app_id"] == "pd-test-app"

    # PUT /api/suite/prefs/common -> 204
    common_payload = {
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
    resp = client.put("/api/suite/prefs/common", json=common_payload)
    assert resp.status_code == 204

    # GET /api/suite/prefs -> 200 with new common values
    resp = client.get("/api/suite/prefs")
    assert resp.status_code == 200
    prefs_data = resp.json()
    assert prefs_data["common"]["theme"] == "light"


def test_e2e_schemas_emit_dump_full(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "pdomain_ops.schemas"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Schema emit failed:\n{result.stderr}"
    schemas = json.loads(result.stdout)
    # At least 11 schemas present
    assert len(schemas) >= 11
    # All values should be dicts (valid JSON Schema objects)
    for name, schema in schemas.items():
        assert isinstance(schema, dict), f"Schema for {name} is not a dict"
