"""bootstrap_spa() — high-level SPA startup helper.

Every pd-* SPA can collapse its entry-point boilerplate to::

    from pdomain_ops.suite import bootstrap_spa

    def main() -> None:
        port = bootstrap_spa(
            preferred=8004,
            caller_package="pdomain_ocr_simple_gui",
        )
        uvicorn.run(app, host="127.0.0.1", port=port)

The helper:
1. Resolves the starting port (env-var override or ``preferred``).
2. Calls :func:`find_available_port` — propagates ``RuntimeError``
   if no free port is found (100 consecutive ports unavailable is a
   real infrastructure problem).
3. Calls :func:`register_self` in a best-effort manner; failure is
   logged as a warning and does **not** abort startup.
4. Prints a startup URL line to stdout.
5. Returns the bound port for the caller to pass to ``uvicorn.run``.
"""

from __future__ import annotations

import logging
import os

from pdomain_ops.suite.ports import find_available_port
from pdomain_ops.suite.register_self import register_self

logger = logging.getLogger(__name__)


def bootstrap_spa(
    *,
    preferred: int,
    caller_package: str,
    port_env: str | None = None,
    host: str = "127.0.0.1",
    url_label: str | None = None,
) -> int:
    """Pick a free port, register the app with the suite registry, print
    the URL, and return the bound port.

    Resolution order for the chosen port:

    1. If *port_env* is set **and** that env var is set to a valid int,
       probe from that value.
    2. Otherwise probe from *preferred*.

    :func:`find_available_port` raises :class:`RuntimeError` when no
    free port is found in the probe window — that error propagates
    unchanged (100 consecutive ports unavailable is a real bug, not a
    transient condition).

    :func:`register_self` failure is non-fatal; the exception is caught,
    a ``WARNING`` is emitted via the module logger, and startup continues.

    Parameters
    ----------
    preferred:
        Preferred port number; the first value probed when *port_env*
        is not set (or the env var is absent / non-integer).
    caller_package:
        Python package name (underscored) of the calling SPA, e.g.
        ``"pdomain_ocr_simple_gui"``.  Forwarded to
        :func:`register_self` and used as the default URL label.
    port_env:
        Name of an environment variable whose integer value overrides
        *preferred* as the starting probe point.  ``None`` disables
        this mechanism.
    host:
        Interface address to bind.  Forwarded to
        :func:`find_available_port` and used in the printed URL.
        Defaults to ``"127.0.0.1"``.
    url_label:
        Human-readable name to print in the startup URL line.  Defaults
        to *caller_package*.

    Returns:
    -------
    int
        The actual bound port (pass to ``uvicorn.run(..., port=port)``).

    Raises:
    ------
    RuntimeError
        When :func:`find_available_port` cannot find a free port in the
        probe window.
    """
    # --- 1. Resolve starting port ---
    start_port = preferred
    if port_env is not None:
        env_val = os.environ.get(port_env)
        if env_val is not None:
            try:
                start_port = int(env_val)
            except ValueError:
                logger.warning(
                    "Env var %s=%r is not a valid integer; falling back to preferred port %d.",
                    port_env,
                    env_val,
                    preferred,
                )

    # --- 2. Find a free port (RuntimeError propagates) ---
    port = find_available_port(preferred=start_port, host=host)

    # --- 3. Register with suite registry (best-effort) ---
    try:
        register_self(_caller_package=caller_package, actual_port=port)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Suite registry registration failed for %r (non-fatal): %s",
            caller_package,
            exc,
        )

    # --- 4. Print startup URL ---
    label = url_label if url_label is not None else caller_package
    print(f"🚀 {label} at http://{host}:{port}/")  # noqa: T201

    # --- 5. Return the bound port ---
    return port
