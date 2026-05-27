"""Smoke tests for SharedContainerStageDispatcher import + Protocol shape."""

from __future__ import annotations

import asyncio

import pytest

from pdomain_ops.gpu import (
    SharedContainerBackend,
    SharedContainerStageDispatcher,
)
from pdomain_ops.gpu.types import ProcessPageRequest


def test_legacy_alias_points_at_new_class() -> None:
    assert SharedContainerBackend is SharedContainerStageDispatcher


def test_instantiates_with_base_url_and_api_key() -> None:
    d = SharedContainerStageDispatcher("https://gpu.example.com", "secret-key")
    assert d.name == "shared_container"


def test_methods_raise_not_implemented_until_wired() -> None:
    d = SharedContainerStageDispatcher("https://gpu.example.com", "k")
    req = ProcessPageRequest.model_construct(project_id="p1", idx0=0, config_overrides={})
    with pytest.raises(NotImplementedError):
        asyncio.run(d.process_page(req))
