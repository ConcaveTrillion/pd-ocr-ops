"""Tests-first for ``ModalStageDispatcher`` dispatch using a Fake Modal runtime.

The real ``modal.Function.lookup(app, fn).remote.aio(payload)`` call can't
run in this devcontainer, but the dispatch shape can: build a fake
``Function`` object whose ``remote.aio`` returns a canned response, monkeypatch
``modal.Function`` so ``_load_function`` finds it, and assert the dispatcher
serialises requests + parses responses correctly.

Locks in:
  - ``process_page`` serialises ``ProcessPageRequest``, awaits ``.remote.aio``,
    and re-validates as ``ProcessPageResponse``,
  - ``run_ocr`` serialises ``OcrPageRequest`` -> ``OcrPageResponse``,
  - ``run_batch`` sends a list of dicts, receives a list of dicts,
  - ``ModalBackend`` legacy alias resolves to ``ModalStageDispatcher``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from pd_ocr_ops.gpu.types import (
    BatchJobItem,
    OcrPageRequest,
    ProcessPageRequest,
)


class FakeFunction:
    def __init__(self, return_value: Any) -> None:
        self._rv = return_value
        self.calls: list[Any] = []
        self.remote = SimpleNamespace(aio=self._aio)

    async def _aio(self, payload: Any) -> Any:
        self.calls.append(payload)
        if callable(self._rv):
            return self._rv(payload)
        return self._rv


class FakeFunctionRegistry:
    """Mimics ``modal.Function.lookup(app, fn)``."""

    def __init__(self, fns: dict[tuple[str, str], FakeFunction]) -> None:
        self._fns = fns

    def lookup(self, app: str, fn: str) -> FakeFunction:
        return self._fns[(app, fn)]


@pytest.fixture
def modal_module(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str], FakeFunction]:
    """Inject a fake ``modal`` module so ``from modal import Function`` works."""
    fns: dict[tuple[str, str], FakeFunction] = {}
    registry = FakeFunctionRegistry(fns)
    fake = SimpleNamespace(Function=registry)
    monkeypatch.setitem(sys.modules, "modal", fake)
    return fns


@pytest.mark.asyncio
async def test_process_page_serialises_request_and_validates_response(
    modal_module: dict[tuple[str, str], FakeFunction],
) -> None:
    from pd_ocr_ops.gpu.modal_dispatcher import ModalStageDispatcher

    expected_response = {
        "processed_image_key": "projects/p/processed/x.png",
        "processed_image_url": "https://cdn.example/x.png",
        "dimensions": [1100, 800],
        "processing_time_ms": 1234,
        "backend": "modal",
        "cold_start_ms": 12000,
    }
    fn = FakeFunction(return_value=expected_response)
    modal_module[("pd-ocr-ops", "process_page")] = fn

    dispatcher = ModalStageDispatcher(token_id="x", token_secret="y")
    req = ProcessPageRequest(
        project_id="p",
        idx0=42,
        config_overrides={"threshold_level": 200},
        output_context="commit",
    )
    resp = await dispatcher.process_page(req)

    assert resp.processed_image_key == expected_response["processed_image_key"]
    assert resp.dimensions == (1100, 800)
    assert resp.cold_start_ms == 12000

    # Dispatcher got a JSON-serialisable dict, not the Pydantic model.
    assert isinstance(fn.calls[0], dict)
    assert fn.calls[0]["idx0"] == 42
    assert fn.calls[0]["config_overrides"]["threshold_level"] == 200


@pytest.mark.asyncio
async def test_run_ocr_round_trip(
    modal_module: dict[tuple[str, str], FakeFunction],
) -> None:
    from pd_ocr_ops.gpu.modal_dispatcher import ModalStageDispatcher

    fn = FakeFunction(
        return_value={
            "text": "hello world",
            "words": [],
            "text_key": "projects/p/ocr_text/x.txt",
        }
    )
    modal_module[("pd-ocr-ops", "run_ocr")] = fn

    dispatcher = ModalStageDispatcher(token_id="x", token_secret="y")
    resp = await dispatcher.run_ocr(OcrPageRequest(project_id="p", idx0=7))
    assert resp.text == "hello world"
    assert resp.text_key == "projects/p/ocr_text/x.txt"
    assert fn.calls[0]["idx0"] == 7


@pytest.mark.asyncio
async def test_run_batch_sends_list_of_dicts(
    modal_module: dict[tuple[str, str], FakeFunction],
) -> None:
    from pd_ocr_ops.gpu.modal_dispatcher import ModalStageDispatcher

    def echo(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "job_type": item["job_type"],
                "project_id": item["project_id"],
                "idx0": item["idx0"],
                "ok": True,
                "payload": {},
            }
            for item in payload
        ]

    fn = FakeFunction(return_value=echo)
    modal_module[("pd-ocr-ops", "run_batch")] = fn

    dispatcher = ModalStageDispatcher(token_id="x", token_secret="y")
    items = [
        BatchJobItem(job_type="batch_process_pages", project_id="p", idx0=0),
        BatchJobItem(job_type="batch_process_pages", project_id="p", idx0=1),
    ]
    results = await dispatcher.run_batch(items)

    assert len(results) == 2
    assert results[0].ok is True
    assert results[1].idx0 == 1

    # Confirm the wire payload was a plain list[dict].
    sent = fn.calls[0]
    assert isinstance(sent, list)
    assert all(isinstance(p, dict) for p in sent)
    assert sent[0]["idx0"] == 0


@pytest.mark.asyncio
async def test_custom_app_name_routes_to_correct_modal_app(
    modal_module: dict[tuple[str, str], FakeFunction],
) -> None:
    """app_name constructor arg lets pgdp-prep keep its 'pgdp-prep' deployment."""
    from pd_ocr_ops.gpu.modal_dispatcher import ModalStageDispatcher

    fn = FakeFunction(
        return_value={
            "text": "custom app",
            "words": [],
            "text_key": "projects/p/ocr_text/x.txt",
        }
    )
    modal_module[("pgdp-prep", "run_ocr")] = fn

    dispatcher = ModalStageDispatcher(token_id="x", token_secret="y", app_name="pgdp-prep")
    resp = await dispatcher.run_ocr(OcrPageRequest(project_id="p", idx0=0))
    assert resp.text == "custom app"


def test_legacy_alias_points_at_new_class() -> None:
    """ModalBackend alias must resolve to ModalStageDispatcher."""
    from pd_ocr_ops.gpu.modal_dispatcher import ModalBackend, ModalStageDispatcher

    assert ModalBackend is ModalStageDispatcher
