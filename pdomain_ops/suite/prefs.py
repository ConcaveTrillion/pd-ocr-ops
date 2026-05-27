"""PrefsAdapter Protocol + LocalFilePrefs implementation."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, runtime_checkable

import filelock
from typing_extensions import Protocol

from pdomain_ops.suite.types import CommonUIPrefs, UIPrefs


@runtime_checkable
class PrefsAdapter(Protocol):
    """Protocol for suite UI preferences storage implementations."""

    def read(self) -> UIPrefs:
        """Read current prefs; return defaults if not yet persisted."""
        ...

    def write_common(self, common: CommonUIPrefs) -> None:
        """Persist the common section; preserve per-app sections."""
        ...

    def write_app(self, app_id: str, payload: dict[str, Any]) -> None:
        """Persist the per-app blob for app_id; preserve other sections."""
        ...


class LocalFilePrefs:
    """Local JSON file-based prefs adapter."""

    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            from pdomain_ops.suite.paths import ui_prefs_json_path

            root = ui_prefs_json_path()
        self._path = Path(root)
        self._lock_path = self._path.with_suffix(".json.lock")

    def _read_raw(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text())
        except Exception:
            return {}

    def _write_raw(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def read(self) -> UIPrefs:
        """Read prefs, returning defaults if file doesn't exist. Non-destructive."""
        with filelock.FileLock(str(self._lock_path)):
            raw = self._read_raw()

        if not raw:
            return UIPrefs()

        # Warn about unknown keys in common section
        common_raw = raw.get("common", {})
        known_common_keys = {
            "theme",
            "density",
            "accent",
            "font_size_base",
            "font_scale",
            "layer_colors",
        }
        unknown_keys = set(common_raw.keys()) - known_common_keys
        for key in unknown_keys:
            warnings.warn(f"Unknown key in UIPrefs common section: {key!r} (ignored)", stacklevel=2)

        return UIPrefs.model_validate(raw)

    def write_common(self, common: CommonUIPrefs) -> None:
        """Update only the common section, preserving apps section."""
        with filelock.FileLock(str(self._lock_path)):
            data = self._read_raw()
            data["common"] = json.loads(common.model_dump_json())
            self._write_raw(data)

    def write_app(self, app_id: str, payload: dict[str, Any]) -> None:
        """Update only the per-app section for app_id, preserving everything else."""
        with filelock.FileLock(str(self._lock_path)):
            data = self._read_raw()
            if "apps" not in data:
                data["apps"] = {}
            data["apps"][app_id] = payload
            self._write_raw(data)
