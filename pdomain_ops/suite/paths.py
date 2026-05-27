"""XDG-aware path helpers for the pdomain-suite data directory."""

from __future__ import annotations

import os
from pathlib import Path


def suite_data_dir() -> Path:
    """Return the pdomain-suite data directory, creating it if needed.

    Respects PD_SUITE_DATA_DIR env var as an override (for tests and containers).
    """
    override = os.environ.get("PD_SUITE_DATA_DIR")
    if override:
        path = Path(override)
    else:
        import platformdirs

        path = Path(platformdirs.user_data_dir("pdomain-suite", appauthor=False))
    path.mkdir(parents=True, exist_ok=True)
    return path


def installed_toml_path() -> Path:
    """Return the path to installed.toml."""
    return suite_data_dir() / "installed.toml"


def ui_prefs_json_path() -> Path:
    """Return the path to ui-prefs.json."""
    return suite_data_dir() / "ui-prefs.json"


def jobs_db_path() -> Path:
    """Return the path to jobs.db."""
    return suite_data_dir() / "jobs.db"
