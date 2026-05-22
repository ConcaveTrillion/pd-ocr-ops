"""Tests for register_self() auto-detect helper (Task 2.5)."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import types
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import patch

import pytest

from pd_ocr_ops.suite.registry import LocalTomlSuiteRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_pkg(
    tmp_path: Path,
    pkg_name: str,
    fragment: dict,
    version: str = "1.2.3",
) -> types.ModuleType:
    """Create an in-memory fake package with a pd-suite.json fragment.

    The package has an importlib.resources-compatible file via a
    __spec__.submodule_search_locations pointing to tmp_path/pkg_name.
    """
    pkg_dir = tmp_path / pkg_name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "pd-suite.json").write_text(json.dumps(fragment))

    # Register the package in sys.modules
    mod = types.ModuleType(pkg_name)
    mod.__package__ = pkg_name
    mod.__path__ = [str(pkg_dir)]
    mod.__spec__ = importlib.util.spec_from_file_location(
        pkg_name,
        str(pkg_dir / "__init__.py"),
        submodule_search_locations=[str(pkg_dir)],
    )
    sys.modules[pkg_name] = mod

    return mod


def _patch_version(pkg_name: str, version: str):
    """Patch importlib.metadata.version to return a fixed version for pkg_name."""

    def _fake_version(name: str) -> str:
        if name == pkg_name:
            return version
        raise PackageNotFoundError(name)

    return patch("importlib.metadata.version", side_effect=_fake_version)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_register_self_reads_fragment_from_caller_package(tmp_path, monkeypatch):
    """register_self() reads pd-suite.json from the calling package."""
    pkg_name = "fake_suite_app_reads"
    fragment = {
        "app_id": "pd-fake-app",
        "display_name": "Fake App",
        "package": pkg_name,
        "default_port": 8042,
        "icon": "fake",
    }
    _make_fake_pkg(tmp_path, pkg_name, fragment)
    monkeypatch.setenv("PD_SUITE_DATA_DIR", str(tmp_path))
    toml_file = tmp_path / "installed.toml"

    with _patch_version(pkg_name, "1.2.3"):
        with patch("sys.argv", ["/usr/bin/python3"]):
            # Call register_self as if called from inside fake_suite_app_reads
            from pd_ocr_ops.suite.register_self import register_self

            register_self(_caller_package=pkg_name, _registry_root=toml_file)

    registry = LocalTomlSuiteRegistry(root=toml_file)
    apps = registry.list_installed()
    assert any(a.app_id == "pd-fake-app" for a in apps)


def test_register_self_fills_binary_from_argv(tmp_path, monkeypatch):
    """register_self() sets binary from sys.argv[0] (resolved)."""
    pkg_name = "fake_suite_app_argv"
    fragment = {
        "app_id": "pd-argv-app",
        "display_name": "Argv App",
        "package": pkg_name,
        "default_port": 8043,
        "icon": "argv",
    }
    _make_fake_pkg(tmp_path, pkg_name, fragment)
    monkeypatch.setenv("PD_SUITE_DATA_DIR", str(tmp_path))
    toml_file = tmp_path / "installed.toml"
    argv_path = "/usr/bin/python3"  # real path that exists

    with _patch_version(pkg_name, "1.0.0"):
        with patch("sys.argv", [argv_path]):
            from pd_ocr_ops.suite.register_self import register_self

            register_self(_caller_package=pkg_name, _registry_root=toml_file)

    registry = LocalTomlSuiteRegistry(root=toml_file)
    apps = registry.list_installed()
    matching = [a for a in apps if a.app_id == "pd-argv-app"]
    assert matching, "app not registered"
    # binary is stored as the resolved absolute path (follows symlinks)
    expected = str(Path(argv_path).resolve())
    assert matching[0].binary == expected


def test_register_self_fills_version_from_importlib_metadata(tmp_path, monkeypatch):
    """register_self() reads version from importlib.metadata."""
    pkg_name = "fake_suite_app_version"
    fragment = {
        "app_id": "pd-version-app",
        "display_name": "Version App",
        "package": pkg_name,
        "default_port": 8044,
        "icon": "version",
    }
    _make_fake_pkg(tmp_path, pkg_name, fragment)
    monkeypatch.setenv("PD_SUITE_DATA_DIR", str(tmp_path))
    toml_file = tmp_path / "installed.toml"

    with _patch_version(pkg_name, "3.4.5"):
        with patch("sys.argv", ["/usr/bin/python3"]):
            from pd_ocr_ops.suite.register_self import register_self

            register_self(_caller_package=pkg_name, _registry_root=toml_file)

    registry = LocalTomlSuiteRegistry(root=toml_file)
    apps = registry.list_installed()
    matching = [a for a in apps if a.app_id == "pd-version-app"]
    assert matching, "app not registered"
    assert matching[0].version == "3.4.5"


def test_register_self_kwargs_override_fragment(tmp_path, monkeypatch):
    """register_self(**overrides) merges overrides onto auto-detected fields."""
    pkg_name = "fake_suite_app_override"
    fragment = {
        "app_id": "pd-override-app",
        "display_name": "Override App",
        "package": pkg_name,
        "default_port": 8000,
        "icon": "override",
    }
    _make_fake_pkg(tmp_path, pkg_name, fragment)
    monkeypatch.setenv("PD_SUITE_DATA_DIR", str(tmp_path))
    toml_file = tmp_path / "installed.toml"

    with _patch_version(pkg_name, "1.0.0"):
        with patch("sys.argv", ["/usr/bin/python3"]):
            from pd_ocr_ops.suite.register_self import register_self

            register_self(
                _caller_package=pkg_name,
                _registry_root=toml_file,
                default_port=9999,
            )

    registry = LocalTomlSuiteRegistry(root=toml_file)
    apps = registry.list_installed()
    matching = [a for a in apps if a.app_id == "pd-override-app"]
    assert matching, "app not registered"
    assert matching[0].default_port == 9999


def test_register_self_missing_fragment_raises_clear_error(tmp_path, monkeypatch):
    """register_self() raises FileNotFoundError with package name when pd-suite.json missing."""
    pkg_name = "fake_suite_app_no_fragment"
    pkg_dir = tmp_path / pkg_name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("")
    # No pd-suite.json written

    mod = types.ModuleType(pkg_name)
    mod.__package__ = pkg_name
    mod.__path__ = [str(pkg_dir)]
    mod.__spec__ = importlib.util.spec_from_file_location(
        pkg_name,
        str(pkg_dir / "__init__.py"),
        submodule_search_locations=[str(pkg_dir)],
    )
    sys.modules[pkg_name] = mod

    monkeypatch.setenv("PD_SUITE_DATA_DIR", str(tmp_path))
    toml_file = tmp_path / "installed.toml"

    with _patch_version(pkg_name, "1.0.0"):
        with patch("sys.argv", ["/usr/bin/python3"]):
            from pd_ocr_ops.suite.register_self import register_self

            with pytest.raises(FileNotFoundError, match=pkg_name):
                register_self(_caller_package=pkg_name, _registry_root=toml_file)
