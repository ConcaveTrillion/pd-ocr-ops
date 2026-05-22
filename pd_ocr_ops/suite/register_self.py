"""register_self() — one-liner app registration helper.

Each pd-* app calls this at startup to register itself with the
LocalTomlSuiteRegistry.  Auto-detects app metadata from:
  - The calling package's ``pd-suite.json`` fragment (via importlib.resources)
  - ``importlib.metadata.version(package)`` for the version string
  - ``sys.argv[0]`` for the binary path

Keyword overrides merge on top of the auto-detected values, so operators
can supply an explicit binary path or port override without changing the
fragment.
"""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import json
import sys
from pathlib import Path
from typing import Any


def register_self(
    *,
    _caller_package: str | None = None,
    _registry_root: Path | None = None,
    **overrides: Any,
) -> None:
    """Register the calling package with the suite registry.

    Parameters
    ----------
    _caller_package:
        Override the auto-detected caller package name.  Use in tests or
        when calling from a helper module that wraps ``register_self``.
    _registry_root:
        Override the registry path (tests only).
    **overrides:
        Field overrides applied on top of the ``pd-suite.json`` fragment
        before constructing the ``InstalledApp``.  Useful for operators
        supplying a custom binary path or port.

    Raises:
    ------
    FileNotFoundError
        When the calling package contains no ``pd-suite.json`` fragment.
    """
    from pd_ocr_ops.suite.registry import LocalTomlSuiteRegistry
    from pd_ocr_ops.suite.types import InstalledApp

    # --- Resolve the caller package ---
    if _caller_package is None:
        import inspect

        frame = inspect.stack()[1]
        caller_globals = frame[0].f_globals
        pkg = caller_globals.get("__package__") or caller_globals.get("__name__", "")
        # Use the top-level package name
        _caller_package = pkg.split(".")[0] if pkg else ""

    # Non-optional alias — _caller_package is guaranteed str by this point
    # (assigned in the if-block above when initially None).
    caller_pkg: str = _caller_package or ""

    # --- Read the pd-suite.json fragment ---
    try:
        pkg_files = importlib.resources.files(caller_pkg)
        fragment_file = pkg_files / "pd-suite.json"
        # files() returns a Traversable; read_text() raises FileNotFoundError
        # when the resource doesn't exist, but the error message may not
        # mention the package name — we wrap it.
        raw = fragment_file.read_text(encoding="utf-8")
    except (FileNotFoundError, TypeError, ModuleNotFoundError) as exc:
        raise FileNotFoundError(
            f"pd-suite.json not found in package {caller_pkg!r}. "
            "Each pd-* app must ship a pd-suite.json resource in its wheel."
        ) from exc

    fragment: dict[str, Any] = json.loads(raw)

    # --- Auto-fill binary and version ---
    binary = str(Path(sys.argv[0]).resolve()) if sys.argv else ""
    try:
        version = importlib.metadata.version(caller_pkg)
    except importlib.metadata.PackageNotFoundError:
        version = "0.0.0"

    # --- Merge: fragment < auto-detected < overrides ---
    fields: dict[str, Any] = {
        "binary": binary,
        "version": version,
        **fragment,
        **overrides,
    }

    app = InstalledApp(**fields)

    registry = LocalTomlSuiteRegistry(root=_registry_root)
    registry.register(app)
