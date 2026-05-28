"""Step 7 — LocalStageDispatcher.run_ocr_batch integration tests.

Tests that the Local dispatcher:
- accepts OcrBatchRequest with image bytes
- delegates to run_doctr_batch
- returns list of page dicts
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pdomain_ops.gpu.local_stage import LocalStageDispatcher
from pdomain_ops.gpu.types import OcrBatchRequest


@pytest.mark.asyncio
async def test_local_run_ocr_batch_returns_page_dicts(monkeypatch: pytest.MonkeyPatch) -> None:
    """LocalStageDispatcher.run_ocr_batch returns one page dict per image."""
    import cv2
    import pdomain_book_tools.hf as _hf_mod
    import pdomain_book_tools.ocr.doctr_support as _doctr_support

    import pdomain_ops.gpu.default_stages as ds

    ds._predictor_cache.clear()

    fake_det = Path("/fake/det.pt")
    fake_reco = Path("/fake/reco.pt")
    monkeypatch.setattr(_hf_mod, "resolve_ocr_models", lambda: (fake_det, fake_reco))
    sentinel_predictor = object()
    monkeypatch.setattr(
        _doctr_support,
        "get_finetuned_torch_doctr_predictor",
        lambda d, r: sentinel_predictor,
    )

    dispatcher = LocalStageDispatcher()

    # Build minimal valid image bytes
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    img_bytes = buf.tobytes()

    fake_page_a = MagicMock()
    fake_page_a.to_dict.return_value = {"words": [], "idx": 0}
    fake_page_b = MagicMock()
    fake_page_b.to_dict.return_value = {"words": [], "idx": 1}
    fake_doc = MagicMock()
    fake_doc.pages = [fake_page_a, fake_page_b]

    import pdomain_book_tools.ocr.document as _doc_mod

    with patch.object(
        _doc_mod.Document,
        "from_images_ocr_via_doctr",
        classmethod(lambda cls, images, source_identifiers=None, predictor=None, **kw: fake_doc),
    ):
        req = OcrBatchRequest(
            images=[img_bytes, img_bytes],
            source_identifiers=["pg-0", "pg-1"],
        )
        results = await dispatcher.run_ocr_batch(req)

    assert len(results) == 2
    assert results[0] == {"words": [], "idx": 0}
    assert results[1] == {"words": [], "idx": 1}


@pytest.mark.asyncio
async def test_local_run_ocr_batch_uses_book_tools_batched_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_ocr_batch delegates to from_images_ocr_via_doctr with the correct identifiers."""
    import cv2
    import pdomain_book_tools.hf as _hf_mod
    import pdomain_book_tools.ocr.doctr_support as _doctr_support

    import pdomain_ops.gpu.default_stages as ds

    ds._predictor_cache.clear()

    fake_det = Path("/fake/det.pt")
    fake_reco = Path("/fake/reco.pt")
    monkeypatch.setattr(_hf_mod, "resolve_ocr_models", lambda: (fake_det, fake_reco))
    sentinel_predictor = object()
    monkeypatch.setattr(
        _doctr_support,
        "get_finetuned_torch_doctr_predictor",
        lambda d, r: sentinel_predictor,
    )

    dispatcher = LocalStageDispatcher()

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    img_bytes = buf.tobytes()

    captured: dict[str, Any] = {}
    fake_page = MagicMock()
    fake_page.to_dict.return_value = {}
    fake_doc = MagicMock()
    fake_doc.pages = [fake_page]

    import pdomain_book_tools.ocr.document as _doc_mod

    def _capture(
        cls: Any,
        images: list[Any],
        source_identifiers: list[str] | None = None,
        predictor: Any = None,
        **kw: Any,
    ) -> MagicMock:
        captured["images"] = images
        captured["source_identifiers"] = source_identifiers
        return fake_doc

    with patch.object(
        _doc_mod.Document,
        "from_images_ocr_via_doctr",
        classmethod(_capture),
    ):
        req = OcrBatchRequest(
            images=[img_bytes],
            source_identifiers=["p0"],
            engine="doctr",
        )
        results = await dispatcher.run_ocr_batch(req)

    assert len(results) == 1
    assert captured["source_identifiers"] == ["p0"]
