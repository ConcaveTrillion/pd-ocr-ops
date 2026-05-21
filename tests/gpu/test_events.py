"""Tests for the @@PDEVENT@@ stdout line parser."""

import pytest

from pd_ocr_ops.gpu.events import is_event_line, parse_event_line


def test_is_event_line_recognizes_prefix():
    assert is_event_line('@@PDEVENT@@ {"kind": "log", "payload": {}}')
    assert is_event_line('   @@PDEVENT@@ {"kind": "log", "payload": {}}')


def test_is_event_line_rejects_plain_log():
    assert not is_event_line("epoch 1/3 loss=0.5")
    assert not is_event_line("")


def test_parse_event_line_returns_kind_and_payload():
    kind, payload = parse_event_line('@@PDEVENT@@ {"kind": "progress", "payload": {"pct": 0.5}}')
    assert kind == "progress"
    assert payload == {"pct": 0.5}


def test_parse_event_line_defaults_missing_payload_to_empty_dict():
    kind, payload = parse_event_line('@@PDEVENT@@ {"kind": "state"}')
    assert kind == "state"
    assert payload == {}


def test_parse_event_line_rejects_non_event_line():
    with pytest.raises(ValueError, match="not a @@PDEVENT@@ line"):
        parse_event_line("plain log output")


def test_parse_event_line_rejects_malformed_json():
    with pytest.raises(ValueError, match="malformed"):
        parse_event_line("@@PDEVENT@@ {not json")


def test_parse_event_line_rejects_invalid_kind():
    with pytest.raises(ValueError, match="invalid @@PDEVENT@@ kind"):
        parse_event_line('@@PDEVENT@@ {"kind": "epoch", "payload": {}}')


def test_parse_event_line_rejects_non_object_payload():
    with pytest.raises(ValueError, match="not an object"):
        parse_event_line('@@PDEVENT@@ ["not", "an", "object"]')


def test_parse_event_line_rejects_non_object_payload_field():
    with pytest.raises(ValueError, match="payload field is not an object"):
        parse_event_line('@@PDEVENT@@ {"kind": "log", "payload": "oops"}')
