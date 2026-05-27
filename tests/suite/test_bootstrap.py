"""Tests for pdomain_ops.suite.bootstrap.bootstrap_spa()."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from pdomain_ops.suite.bootstrap import bootstrap_spa

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PREFERRED = 8000
_ALL_INTERFACES = "0.0.0.0"  # noqa: S104


def _make_find_port_patch(return_port: int):
    return patch(
        "pdomain_ops.suite.bootstrap.find_available_port",
        return_value=return_port,
    )


def _make_register_patch():
    return patch("pdomain_ops.suite.bootstrap.register_self")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_uses_preferred_when_free(capsys):
    """bootstrap_spa returns the preferred port and calls register_self."""
    with _make_find_port_patch(_PREFERRED) as mock_find, _make_register_patch() as mock_reg:
        result = bootstrap_spa(
            preferred=_PREFERRED,
            caller_package="my_app",
        )

    assert result == _PREFERRED
    mock_find.assert_called_once_with(preferred=_PREFERRED, host="127.0.0.1")
    mock_reg.assert_called_once_with(_caller_package="my_app", actual_port=_PREFERRED)

    captured = capsys.readouterr()
    assert f"http://127.0.0.1:{_PREFERRED}/" in captured.out


def test_uses_env_override(monkeypatch, capsys):
    """When port_env is set and the env var contains an int, find_available_port
    is called with that value as preferred."""
    monkeypatch.setenv("MY_APP_PORT", "9090")
    with _make_find_port_patch(9090) as mock_find, _make_register_patch():
        result = bootstrap_spa(
            preferred=_PREFERRED,
            caller_package="my_app",
            port_env="MY_APP_PORT",
        )

    assert result == 9090
    mock_find.assert_called_once_with(preferred=9090, host="127.0.0.1")


def test_env_override_not_set_falls_back_to_preferred(monkeypatch, capsys):
    """When port_env is provided but the env var is absent, use preferred."""
    monkeypatch.delenv("ABSENT_PORT", raising=False)
    with _make_find_port_patch(_PREFERRED) as mock_find, _make_register_patch():
        bootstrap_spa(
            preferred=_PREFERRED,
            caller_package="my_app",
            port_env="ABSENT_PORT",
        )

    mock_find.assert_called_once_with(preferred=_PREFERRED, host="127.0.0.1")


def test_register_self_failure_non_fatal(caplog, capsys):
    """A register_self exception must not abort bootstrap_spa; a warning is logged."""
    with (
        _make_find_port_patch(_PREFERRED),
        patch(
            "pdomain_ops.suite.bootstrap.register_self",
            side_effect=Exception("registry unavailable"),
        ),
        caplog.at_level(logging.WARNING, logger="pdomain_ops.suite.bootstrap"),
    ):
        result = bootstrap_spa(
            preferred=_PREFERRED,
            caller_package="my_app",
        )

    assert result == _PREFERRED
    assert any("registry" in record.message.lower() for record in caplog.records)


def test_find_available_port_runtime_error_propagates():
    """RuntimeError from find_available_port must bubble up unchanged."""
    with (
        patch(
            "pdomain_ops.suite.bootstrap.find_available_port",
            side_effect=RuntimeError("no free port"),
        ),
        _make_register_patch(),
    ):
        with pytest.raises(RuntimeError, match="no free port"):
            bootstrap_spa(
                preferred=_PREFERRED,
                caller_package="my_app",
            )


def test_url_label_override(capsys):
    """When url_label is supplied, the printed line uses it instead of caller_package."""
    with _make_find_port_patch(_PREFERRED), _make_register_patch():
        bootstrap_spa(
            preferred=_PREFERRED,
            caller_package="my_app",
            url_label="My Cool App",
        )

    captured = capsys.readouterr()
    assert "My Cool App" in captured.out
    assert "my_app" not in captured.out


def test_url_label_defaults_to_caller_package(capsys):
    """When url_label is omitted, caller_package is printed in the URL line."""
    with _make_find_port_patch(_PREFERRED), _make_register_patch():
        bootstrap_spa(
            preferred=_PREFERRED,
            caller_package="my_app",
        )

    captured = capsys.readouterr()
    assert "my_app" in captured.out


def test_custom_host_passed_to_find_port_and_url(capsys):
    """The host parameter is forwarded to find_available_port and the URL."""
    with _make_find_port_patch(_PREFERRED) as mock_find, _make_register_patch():
        bootstrap_spa(
            preferred=_PREFERRED,
            caller_package="my_app",
            host=_ALL_INTERFACES,
        )

    mock_find.assert_called_once_with(preferred=_PREFERRED, host=_ALL_INTERFACES)
    captured = capsys.readouterr()
    assert f"http://{_ALL_INTERFACES}:" in captured.out
