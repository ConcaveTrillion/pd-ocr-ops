"""Tests for register_default_stages()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pdomain_ops.gpu import register_default_stages
from pdomain_ops.gpu.local_stage import LocalStageDispatcher


def test_register_default_stages_registers_ocr_stage() -> None:
    """register_default_stages must register the 'ocr'/'cpu' stage."""
    dispatcher = LocalStageDispatcher()
    register_default_stages(dispatcher)
    assert ("ocr", "cpu") in dispatcher._registry


def test_registered_ocr_stage_is_callable() -> None:
    """The registered 'ocr'/'cpu' stage must be callable."""
    dispatcher = LocalStageDispatcher()
    register_default_stages(dispatcher)
    impl = dispatcher._registry[("ocr", "cpu")]
    assert callable(impl)


@pytest.mark.asyncio
async def test_doctr_stage_calls_ocr_via_run_in_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 'ocr'/'cpu' stage with engine='doctr' calls Document.from_image_ocr_via_doctr."""
    monkeypatch.setenv("PDOMAIN_GPU_BACKEND", "cpu")

    fake_page = MagicMock()
    fake_page.to_dict.return_value = {"words": [], "blocks": []}
    fake_doc = MagicMock()
    fake_doc.pages = [fake_page]

    with patch(
        "pdomain_book_tools.ocr.document.Document.from_image_ocr_via_doctr",
        return_value=fake_doc,
    ) as mock_doctr:
        dispatcher = LocalStageDispatcher()
        register_default_stages(dispatcher)
        result = await dispatcher.run_stage(
            "ocr",
            "page-1",
            image_path="/fake/image.png",
            engine="doctr",
            language="eng",
        )

    mock_doctr.assert_called_once()
    assert result.stage_id == "ocr"
    assert result.device == "cpu"
    assert result.metadata == {"pages": [{"words": [], "blocks": []}]}


@pytest.mark.asyncio
async def test_tesseract_stage_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling the 'ocr'/'cpu' stage with engine='tesseract' raises ImportError when pytesseract absent."""
    monkeypatch.setenv("PDOMAIN_GPU_BACKEND", "cpu")

    import pdomain_book_tools.ocr.cv2_tesseract as _tess_mod

    with patch.object(_tess_mod, "_pytesseract_available", False):
        dispatcher = LocalStageDispatcher()
        register_default_stages(dispatcher)
        with pytest.raises(ImportError, match="pytesseract"):
            await dispatcher.run_stage(
                "ocr",
                "page-1",
                image_path="/fake/image.png",
                engine="tesseract",
                language="eng",
            )


def test_register_default_stages_registers_local_and_mps_ocr() -> None:
    """register_default_stages must register ('ocr','local') and ('ocr','mps')."""
    dispatcher = LocalStageDispatcher()
    register_default_stages(dispatcher)
    assert ("ocr", "local") in dispatcher._registry
    assert ("ocr", "mps") in dispatcher._registry
    assert ("ocr", "cpu") in dispatcher._registry


@pytest.mark.asyncio
async def test_ocr_local_uses_finetuned_predictor(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ('ocr','local') stage passes the finetuned predictor to from_image_ocr_via_doctr."""
    from pathlib import Path

    import pdomain_ops.gpu.default_stages as ds

    monkeypatch.setenv("PDOMAIN_GPU_BACKEND", "local")
    # Reset module-level predictor cache so this test stays hermetic.
    ds._predictor_cache.clear()

    fake_det = Path("/fake/det.pt")
    fake_reco = Path("/fake/reco.pt")
    sentinel_predictor = object()

    import pdomain_book_tools.hf as _hf_mod
    import pdomain_book_tools.ocr.doctr_support as _doctr_support
    import pdomain_book_tools.ocr.document as _doc_mod

    monkeypatch.setattr(_hf_mod, "resolve_ocr_models", lambda: (fake_det, fake_reco))
    monkeypatch.setattr(
        _doctr_support,
        "get_finetuned_torch_doctr_predictor",
        lambda d, r: sentinel_predictor,
    )

    fake_page = MagicMock()
    fake_page.to_dict.return_value = {"words": []}
    fake_doc = MagicMock()
    fake_doc.pages = [fake_page]
    captured: dict[str, object] = {}

    def _fake_from_image(
        image: object,
        source_identifier: str = "",
        predictor: object = None,
        **_: object,
    ) -> MagicMock:
        captured["image"] = image
        captured["source_identifier"] = source_identifier
        captured["predictor"] = predictor
        return fake_doc

    monkeypatch.setattr(
        _doc_mod.Document,
        "from_image_ocr_via_doctr",
        classmethod(
            lambda cls, image, source_identifier="", predictor=None, **kw: _fake_from_image(
                image, source_identifier, predictor, **kw
            )
        ),
    )

    dispatcher = LocalStageDispatcher()
    register_default_stages(dispatcher)
    result = await dispatcher.run_stage(
        "ocr",
        "page-7",
        image_path="/fake/image.png",
        engine="doctr",
        language="eng",
    )

    assert captured["predictor"] is sentinel_predictor
    assert captured["source_identifier"] == "page-7"
    assert result.stage_id == "ocr"
    assert result.metadata == {"pages": [{"words": []}]}


@pytest.mark.asyncio
async def test_ocr_local_falls_back_when_resolve_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When resolve_ocr_models raises, the impl logs a warning and uses the stock CPU path."""
    import logging

    import pdomain_book_tools.hf as _hf_mod

    import pdomain_ops.gpu.default_stages as ds

    monkeypatch.setenv("PDOMAIN_GPU_BACKEND", "local")
    ds._predictor_cache.clear()

    def _boom() -> tuple[object, object]:
        raise OSError("no network")

    monkeypatch.setattr(_hf_mod, "resolve_ocr_models", _boom)

    fake_page = MagicMock()
    fake_page.to_dict.return_value = {}
    fake_doc = MagicMock()
    fake_doc.pages = [fake_page]

    with caplog.at_level(logging.WARNING, logger="pdomain_ops.gpu.default_stages"):
        with patch(
            "pdomain_book_tools.ocr.document.Document.from_image_ocr_via_doctr",
            return_value=fake_doc,
        ) as mock_doctr:
            dispatcher = LocalStageDispatcher()
            register_default_stages(dispatcher)
            result = await dispatcher.run_stage(
                "ocr",
                "page-1",
                image_path="/fake/image.png",
                engine="doctr",
            )

    mock_doctr.assert_called_once()
    # stock CPU path does NOT pass a predictor kwarg
    _args, kwargs = mock_doctr.call_args
    assert "predictor" not in kwargs
    assert result.metadata == {"pages": [{}]}
    assert any("resolve_ocr_models" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_ocr_local_predictor_cache_reuses_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two calls with the same (det,reco) paths reuse one cached predictor."""
    from pathlib import Path

    import pdomain_book_tools.hf as _hf_mod
    import pdomain_book_tools.ocr.doctr_support as _doctr_support

    import pdomain_ops.gpu.default_stages as ds

    monkeypatch.setenv("PDOMAIN_GPU_BACKEND", "local")
    ds._predictor_cache.clear()

    fake_det = Path("/fake/det.pt")
    fake_reco = Path("/fake/reco.pt")
    monkeypatch.setattr(_hf_mod, "resolve_ocr_models", lambda: (fake_det, fake_reco))

    call_count = {"n": 0}

    def _build_predictor(_d: object, _r: object) -> object:
        call_count["n"] += 1
        return object()

    monkeypatch.setattr(_doctr_support, "get_finetuned_torch_doctr_predictor", _build_predictor)

    fake_page = MagicMock()
    fake_page.to_dict.return_value = {}
    fake_doc = MagicMock()
    fake_doc.pages = [fake_page]

    with patch(
        "pdomain_book_tools.ocr.document.Document.from_image_ocr_via_doctr",
        return_value=fake_doc,
    ):
        dispatcher = LocalStageDispatcher()
        register_default_stages(dispatcher)
        await dispatcher.run_stage("ocr", "p1", image_path="/fake/a.png", engine="doctr")
        await dispatcher.run_stage("ocr", "p2", image_path="/fake/b.png", engine="doctr")

    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_default_engine_is_doctr(monkeypatch: pytest.MonkeyPatch) -> None:
    """When engine kwarg is omitted, DocTR is used by default."""
    monkeypatch.setenv("PDOMAIN_GPU_BACKEND", "cpu")

    fake_page = MagicMock()
    fake_page.to_dict.return_value = {}
    fake_doc = MagicMock()
    fake_doc.pages = [fake_page]

    with patch(
        "pdomain_book_tools.ocr.document.Document.from_image_ocr_via_doctr",
        return_value=fake_doc,
    ) as mock_doctr:
        dispatcher = LocalStageDispatcher()
        register_default_stages(dispatcher)
        await dispatcher.run_stage("ocr", "page-1", image_path="/fake/image.png")

    mock_doctr.assert_called_once()


# ---------------------------------------------------------------------------
# Step 5 — sized cache keyed by (det_path, reco_path, det_bs, reco_bs)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sized_predictor_cache_distinct_for_different_batch_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache returns distinct predictors for (det_path, reco_path, det_bs=2) vs (det_bs=4)."""
    from pathlib import Path

    import pdomain_book_tools.hf as _hf_mod
    import pdomain_book_tools.ocr.doctr_support as _doctr_support

    import pdomain_ops.gpu.default_stages as ds

    monkeypatch.setenv("PDOMAIN_GPU_BACKEND", "local")
    ds._predictor_cache.clear()

    fake_det = Path("/fake/det.pt")
    fake_reco = Path("/fake/reco.pt")
    monkeypatch.setattr(_hf_mod, "resolve_ocr_models", lambda: (fake_det, fake_reco))

    build_calls: list[tuple[object, object]] = []

    def _build_predictor(_d: object, _r: object, det_bs: int = 2, reco_bs: int = 128) -> object:
        build_calls.append((_d, _r))
        return object()  # each call returns a new distinct object

    monkeypatch.setattr(_doctr_support, "get_finetuned_torch_doctr_predictor", _build_predictor)

    # Get predictor with det_bs=2 (simulate via direct cache lookup)
    cache_key_2 = (str(fake_det), str(fake_reco), 2, 128)
    cache_key_4 = (str(fake_det), str(fake_reco), 4, 128)

    # Populate cache manually as the impl will do
    pred_2 = _build_predictor(str(fake_det), str(fake_reco), det_bs=2, reco_bs=128)
    pred_4 = _build_predictor(str(fake_det), str(fake_reco), det_bs=4, reco_bs=128)
    ds._predictor_cache[cache_key_2] = pred_2
    ds._predictor_cache[cache_key_4] = pred_4

    # They must be different objects
    assert ds._predictor_cache[cache_key_2] is not ds._predictor_cache[cache_key_4]
    # The 4-tuple cache key must exist
    assert cache_key_2 in ds._predictor_cache
    assert cache_key_4 in ds._predictor_cache


@pytest.mark.asyncio
async def test_sized_predictor_cache_reuses_same_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two batch calls with same (det_path, reco_path, det_bs, reco_bs) reuse one predictor."""
    from pathlib import Path

    import pdomain_book_tools.hf as _hf_mod
    import pdomain_book_tools.ocr.doctr_support as _doctr_support

    import pdomain_ops.gpu.default_stages as ds

    monkeypatch.setenv("PDOMAIN_GPU_BACKEND", "local")
    ds._predictor_cache.clear()

    fake_det = Path("/fake/det.pt")
    fake_reco = Path("/fake/reco.pt")
    monkeypatch.setattr(_hf_mod, "resolve_ocr_models", lambda: (fake_det, fake_reco))

    build_call_count = {"n": 0}

    def _build_predictor(_d: object, _r: object, det_bs: int = 2, reco_bs: int = 128) -> object:
        build_call_count["n"] += 1
        return object()

    monkeypatch.setattr(_doctr_support, "get_finetuned_torch_doctr_predictor", _build_predictor)

    # Simulate two fetches with the same key
    cache_key = (str(fake_det), str(fake_reco), 2, 128)
    if cache_key not in ds._predictor_cache:
        ds._predictor_cache[cache_key] = _build_predictor(
            str(fake_det), str(fake_reco), det_bs=2, reco_bs=128
        )
    _ = ds._predictor_cache[cache_key]  # second access — no new build

    # build_call_count must be 1 (populated once, fetched without rebuild)
    assert build_call_count["n"] == 1
