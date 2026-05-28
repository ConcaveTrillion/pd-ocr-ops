"""Step 8 — deferred remote batch stubs.

ModalStageDispatcher and SharedContainerStageDispatcher must raise
NotImplementedError for run_ocr_batch (Wave 5 deferral). These tests
document the deferral and keep the dispatchers Protocol-conformant.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from pdomain_ops.gpu.types import OcrBatchRequest


@pytest.fixture
def modal_module_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a minimal fake modal module so ModalStageDispatcher can be imported."""
    fake = SimpleNamespace(Function=SimpleNamespace(lookup=lambda *a: None))
    monkeypatch.setitem(sys.modules, "modal", fake)


def test_modal_run_ocr_batch_raises_not_implemented(
    modal_module_stub: None,
) -> None:
    """ModalStageDispatcher.run_ocr_batch must raise NotImplementedError (Wave 5 deferred)."""
    from pdomain_ops.gpu.modal_dispatcher import ModalStageDispatcher

    dispatcher = ModalStageDispatcher(token_id="x", token_secret="y")
    req = OcrBatchRequest(images=[b"fake"], source_identifiers=["p0"])
    with pytest.raises(NotImplementedError):
        asyncio.run(dispatcher.run_ocr_batch(req))


def test_shared_container_run_ocr_batch_raises_not_implemented() -> None:
    """SharedContainerStageDispatcher.run_ocr_batch must raise NotImplementedError (Wave 5 deferred)."""
    from pdomain_ops.gpu.shared_container_dispatcher import SharedContainerStageDispatcher

    dispatcher = SharedContainerStageDispatcher("https://gpu.example.com", "k")
    req = OcrBatchRequest(images=[b"fake"], source_identifiers=["p0"])
    with pytest.raises(NotImplementedError):
        asyncio.run(dispatcher.run_ocr_batch(req))
