"""SuiteRegistryAdapter Protocol + LocalTomlSuiteRegistry implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, runtime_checkable

import filelock
import tomli
import tomli_w
from typing_extensions import Protocol

if TYPE_CHECKING:
    pass

from pd_ocr_ops.suite.types import InstalledApp


@runtime_checkable
class SuiteRegistryAdapter(Protocol):
    """Protocol for suite app registry implementations."""

    def list_installed(self) -> list[InstalledApp]: ...

    def register(self, app: InstalledApp) -> None: ...

    def unregister(self, app_id: str) -> None: ...


class LocalTomlSuiteRegistry:
    """Local TOML-based suite registry."""

    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            from pd_ocr_ops.suite.paths import installed_toml_path

            root = installed_toml_path()
        self._path = Path(root)
        self._lock_path = self._path.with_suffix(".toml.lock")

    def _read_raw(self) -> dict:
        """Read the raw TOML data; return empty dict if file doesn't exist."""
        if not self._path.exists():
            return {"apps": {}}
        try:
            with open(self._path, "rb") as f:
                data = tomli.load(f)
            if "apps" not in data:
                data["apps"] = {}
            return data
        except Exception:
            return {"apps": {}}

    def _write_raw(self, data: dict) -> None:
        """Write raw TOML data to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as f:
            tomli_w.dump(data, f)

    def list_installed(self) -> list[InstalledApp]:
        """Return all installed apps, pruning stale entries from the result."""
        with filelock.FileLock(str(self._lock_path)):
            data = self._read_raw()
        apps = []
        for app_id, app_data in data.get("apps", {}).items():
            try:
                app = InstalledApp.model_validate(app_data)
                # Prune stale entries: if binary doesn't exist on disk, skip
                if not Path(app.binary).exists():
                    continue
                apps.append(app)
            except Exception:
                continue
        return apps

    def register(self, app: InstalledApp) -> None:
        """Register or refresh an app entry, pruning stale entries on write."""
        with filelock.FileLock(str(self._lock_path)):
            data = self._read_raw()
            apps = data.get("apps", {})
            # Prune stale entries on disk
            stale_ids = [
                app_id
                for app_id, app_data in apps.items()
                if not Path(app_data.get("binary", "")).exists()
                and app_data.get("app_id") != app.app_id
            ]
            for stale_id in stale_ids:
                del apps[stale_id]
            # Write/refresh the new entry
            apps[app.app_id] = app.model_dump(mode="json")
            data["apps"] = apps
            self._write_raw(data)

    def unregister(self, app_id: str) -> None:
        """Remove an app entry; noop if not present."""
        with filelock.FileLock(str(self._lock_path)):
            data = self._read_raw()
            apps = data.get("apps", {})
            apps.pop(app_id, None)
            data["apps"] = apps
            self._write_raw(data)
