"""TDD tests for run_doctr_batch — the location-independent batch worker.

Step 3: worker test — stub predictor, assert batched call + correct return.
Step 6: OOM-backoff tests — OOM once succeeds, OOM at floor falls back to CPU,
        non-OOM RuntimeError re-raises.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Step 3 — worker returns correct page dicts + calls predictor once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_doctr_batch_returns_page_per_image() -> None:
    """run_doctr_batch returns one page dict per input image, in order."""
    from pdomain_ops.gpu.doctr_batch import run_doctr_batch

    img_a = np.zeros((100, 100, 3), dtype=np.uint8)
    img_b = np.zeros((100, 100, 3), dtype=np.uint8)

    fake_page_a = MagicMock()
    fake_page_a.to_dict.return_value = {"page": "a"}
    fake_page_b = MagicMock()
    fake_page_b.to_dict.return_value = {"page": "b"}

    fake_doc = MagicMock()
    fake_doc.pages = [fake_page_a, fake_page_b]

    stub_predictor = MagicMock()

    import pdomain_book_tools.ocr.document as _doc_mod

    with patch.object(
        _doc_mod.Document,
        "from_images_ocr_via_doctr",
        classmethod(lambda cls, images, source_identifiers=None, predictor=None, **kw: fake_doc),
    ):
        result = run_doctr_batch([img_a, img_b], predictor=stub_predictor, device="cpu")

    assert result == [{"page": "a"}, {"page": "b"}]


@pytest.mark.asyncio
async def test_run_doctr_batch_calls_predictor_once_for_batch() -> None:
    """run_doctr_batch must call from_images_ocr_via_doctr exactly once (batched call)."""
    from pdomain_ops.gpu.doctr_batch import run_doctr_batch

    img_a = np.zeros((100, 100, 3), dtype=np.uint8)
    img_b = np.zeros((100, 100, 3), dtype=np.uint8)

    fake_page_a = MagicMock()
    fake_page_a.to_dict.return_value = {"p": 0}
    fake_page_b = MagicMock()
    fake_page_b.to_dict.return_value = {"p": 1}

    fake_doc = MagicMock()
    fake_doc.pages = [fake_page_a, fake_page_b]

    call_log: list[tuple[Any, ...]] = []

    def _fake_from_images(
        cls: Any,
        images: list[Any],
        source_identifiers: list[str] | None = None,
        predictor: Any = None,
        **kw: Any,
    ) -> MagicMock:
        call_log.append((images, source_identifiers, predictor))
        return fake_doc

    stub_predictor = MagicMock()

    import pdomain_book_tools.ocr.document as _doc_mod

    with patch.object(
        _doc_mod.Document,
        "from_images_ocr_via_doctr",
        classmethod(_fake_from_images),
    ):
        run_doctr_batch([img_a, img_b], predictor=stub_predictor, device="cpu")

    # The predictor should be used in a single batched call
    assert len(call_log) == 1
    images_arg, _, pred_arg = call_log[0]
    assert len(images_arg) == 2
    assert pred_arg is stub_predictor


@pytest.mark.asyncio
async def test_run_doctr_batch_accepts_bytes_input() -> None:
    """run_doctr_batch accepts bytes images (decodes to ndarray internally)."""
    import cv2

    from pdomain_ops.gpu.doctr_batch import run_doctr_batch

    # Create a real small PNG bytes buffer
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    img_bytes = buf.tobytes()

    fake_page = MagicMock()
    fake_page.to_dict.return_value = {"decoded": True}
    fake_doc = MagicMock()
    fake_doc.pages = [fake_page]

    stub_predictor = MagicMock()

    import pdomain_book_tools.ocr.document as _doc_mod

    decoded_images: list[Any] = []

    def _capture(
        cls: Any,
        images: list[Any],
        source_identifiers: list[str] | None = None,
        predictor: Any = None,
        **kw: Any,
    ) -> MagicMock:
        decoded_images.extend(images)
        return fake_doc

    with patch.object(
        _doc_mod.Document,
        "from_images_ocr_via_doctr",
        classmethod(_capture),
    ):
        result = run_doctr_batch([img_bytes], predictor=stub_predictor, device="cpu")

    assert result == [{"decoded": True}]
    # The decoded image should be a numpy ndarray, not bytes
    assert isinstance(decoded_images[0], np.ndarray)


# ---------------------------------------------------------------------------
# Step 6 — OOM-backoff tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oom_once_halves_batch_size_and_succeeds() -> None:
    """On OOM, det_bs is halved, empty_cache called, build_smaller invoked, retry succeeds."""
    from pdomain_ops.gpu.doctr_batch import run_doctr_batch

    img = np.zeros((100, 100, 3), dtype=np.uint8)

    fake_page = MagicMock()
    fake_page.to_dict.return_value = {"ok": True}
    fake_doc = MagicMock()
    fake_doc.pages = [fake_page]

    stub_predictor = MagicMock()
    stub_predictor_2 = MagicMock()

    build_smaller_calls: list[tuple[int, int]] = []

    def _build_smaller(det_bs: int, reco_bs: int) -> MagicMock:
        build_smaller_calls.append((det_bs, reco_bs))
        return stub_predictor_2

    call_count = {"n": 0}

    def _from_images(
        cls: Any,
        images: list[Any],
        source_identifiers: list[str] | None = None,
        predictor: Any = None,
        **kw: Any,
    ) -> MagicMock:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Simulate OOM on first attempt
            raise RuntimeError("CUDA out of memory")
        return fake_doc

    import pdomain_book_tools.ocr.document as _doc_mod

    with (
        patch.object(
            _doc_mod.Document,
            "from_images_ocr_via_doctr",
            classmethod(_from_images),
        ),
        patch("pdomain_ops.gpu.doctr_batch._empty_cuda_cache") as mock_cache,
    ):
        result = run_doctr_batch(
            [img],
            predictor=stub_predictor,
            device="local",
            build_smaller=_build_smaller,
        )

    assert result == [{"ok": True}]
    # build_smaller was called with halved det_bs
    assert len(build_smaller_calls) == 1
    det_bs_used, _ = build_smaller_calls[0]
    assert det_bs_used >= 1
    mock_cache.assert_called_once()


@pytest.mark.asyncio
async def test_oom_at_floor_falls_back_to_cpu(caplog: pytest.LogCaptureFixture) -> None:
    """When det_bs==1 still OOMs, fall back to per-image CPU path with a warning."""
    import logging

    from pdomain_ops.gpu.doctr_batch import run_doctr_batch

    img = np.zeros((100, 100, 3), dtype=np.uint8)

    fake_page = MagicMock()
    fake_page.to_dict.return_value = {"cpu_fallback": True}
    fake_doc = MagicMock()
    fake_doc.pages = [fake_page]

    stub_predictor = MagicMock()

    call_count = {"gpu_calls": 0}

    def _from_images(
        cls: Any,
        images: list[Any],
        source_identifiers: list[str] | None = None,
        predictor: Any = None,
        **kw: Any,
    ) -> MagicMock:
        call_count["gpu_calls"] += 1
        if predictor is stub_predictor:
            raise RuntimeError("CUDA out of memory")
        # CPU fallback predictor (or no predictor) succeeds
        return fake_doc

    import pdomain_book_tools.ocr.document as _doc_mod

    # build_smaller always returns a predictor that also OOMs
    def _build_smaller_oom(det_bs: int, reco_bs: int) -> MagicMock:
        return stub_predictor  # still OOMs

    with (
        patch.object(
            _doc_mod.Document,
            "from_images_ocr_via_doctr",
            classmethod(_from_images),
        ),
        patch("pdomain_ops.gpu.doctr_batch._empty_cuda_cache"),
        patch("pdomain_ops.gpu.doctr_batch._pick_doctr_batch_sizes_fn", return_value=(1, 128)),
        caplog.at_level(logging.WARNING, logger="pdomain_ops.gpu.doctr_batch"),
    ):
        result = run_doctr_batch(
            [img],
            predictor=stub_predictor,
            device="local",
            build_smaller=_build_smaller_oom,
        )

    assert result == [{"cpu_fallback": True}]
    assert any("OOM" in r.message or "fallback" in r.message.lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_non_oom_runtime_error_reraises() -> None:
    """A non-OOM RuntimeError is re-raised immediately without OOM backoff."""
    from pdomain_ops.gpu.doctr_batch import run_doctr_batch

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    stub_predictor = MagicMock()

    def _from_images(
        cls: Any,
        images: list[Any],
        source_identifiers: list[str] | None = None,
        predictor: Any = None,
        **kw: Any,
    ) -> MagicMock:
        raise RuntimeError("some other GPU error")

    import pdomain_book_tools.ocr.document as _doc_mod

    with patch.object(
        _doc_mod.Document,
        "from_images_ocr_via_doctr",
        classmethod(_from_images),
    ):
        with pytest.raises(RuntimeError, match="some other GPU error"):
            run_doctr_batch([img], predictor=stub_predictor, device="cpu")
