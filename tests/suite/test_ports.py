"""Tests for find_available_port helper."""

from __future__ import annotations

import socket

import pytest

from pdomain_ops.suite.ports import find_available_port


def _bind_port(port: int, host: str = "127.0.0.1") -> socket.socket:
    """Bind a real socket to reserve the port; caller must close it."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    sock.bind((host, port))
    return sock


def _find_free_port(host: str = "127.0.0.1") -> int:
    """Ask the OS for an ephemeral free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


class TestFindAvailablePort:
    """Unit tests for find_available_port."""

    def test_returns_preferred_when_free(self) -> None:
        """Returns preferred port when nothing is bound there."""
        port = _find_free_port()
        result = find_available_port(port)
        assert result == port

    def test_returns_next_when_preferred_taken(self) -> None:
        """Returns preferred+1 when preferred is already bound."""
        port = _find_free_port()
        sock = _bind_port(port)
        try:
            result = find_available_port(port)
            assert result == port + 1
        finally:
            sock.close()

    def test_raises_when_max_attempts_exhausted(self) -> None:
        """Raises RuntimeError when max_attempts=1 and preferred is taken."""
        port = _find_free_port()
        sock = _bind_port(port)
        try:
            with pytest.raises(RuntimeError, match="free port"):
                find_available_port(port, max_attempts=1)
        finally:
            sock.close()

    def test_walks_multiple_hops(self) -> None:
        """Walks more than 1 hop when consecutive ports are taken."""
        base = _find_free_port()
        sock0 = _bind_port(base)
        sock1 = _bind_port(base + 1)
        try:
            result = find_available_port(base)
            assert result == base + 2
        finally:
            sock0.close()
            sock1.close()

    def test_accepts_custom_host(self) -> None:
        """Works with an explicit host argument (127.0.0.1)."""
        port = _find_free_port(host="127.0.0.1")
        result = find_available_port(port, host="127.0.0.1")
        assert result == port

    def test_returned_port_is_bindable(self) -> None:
        """The returned port can actually be bound by the caller."""
        port = _find_free_port()
        result = find_available_port(port)
        # Verify we can bind it — no OSError means the probe socket was closed
        sock = _bind_port(result)
        sock.close()
