"""Port-availability helper for SPA bootstrap.

pd-* SPAs call ``find_available_port`` at startup to avoid ``EADDRINUSE``
crashes when the preferred port is already taken by another process.
"""

from __future__ import annotations

import socket


def find_available_port(
    preferred: int,
    host: str = "127.0.0.1",
    max_attempts: int = 100,
) -> int:
    """Return ``preferred`` if it can be bound, else the next free port.

    Walks ``preferred``, ``preferred+1``, ... up to ``max_attempts``.
    Raises :class:`RuntimeError` if nothing free is found in the window.

    The probe socket is always closed before returning.  The caller will
    bind the port for real (e.g. via uvicorn).  A TOCTOU race is
    possible but acceptable — if another process wins the race, uvicorn
    will raise ``EADDRINUSE`` immediately and the SPA can retry with a
    fresh ``find_available_port`` call.

    Parameters
    ----------
    preferred:
        The first port number to try.
    host:
        The interface address to probe.  Defaults to ``"127.0.0.1"``.
    max_attempts:
        How many consecutive ports to try before giving up.

    Returns:
    -------
    int
        The first port in ``[preferred, preferred + max_attempts)`` that
        can be bound.

    Raises:
    ------
    RuntimeError
        When no free port is found within ``max_attempts`` probes.
    """
    for offset in range(max_attempts):
        port = preferred + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
                probe.bind((host, port))
        except OSError:
            # EADDRINUSE (or similar) — try the next port
            continue
        else:
            # bind succeeded — socket is closed by the context manager
            return port

    raise RuntimeError(
        f"Could not find a free port in range [{preferred}, "
        f"{preferred + max_attempts}) on {host!r}."
    )
