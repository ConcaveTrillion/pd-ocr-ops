"""Tests for register_default_stages()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pd_ocr_ops.gpu import register_default_stages
from pd_ocr_ops.gpu.local_stage import LocalStageDispatcher


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
    monkeypatch.setenv("PD_GPU_BACKEND", "cpu")

    fake_page = MagicMock()
    fake_page.to_dict.return_value = {"words": [], "blocks": []}
    fake_doc = MagicMock()
    fake_doc.pages = [fake_page]

    with patch(
        "pd_book_tools.ocr.document.Document.from_image_ocr_via_doctr",
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
    monkeypatch.setenv("PD_GPU_BACKEND", "cpu")

    import pd_book_tools.ocr.cv2_tesseract as _tess_mod  # pyright: ignore[reportMissingTypeStubs]  # pd-book-tools ships no py.typed

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


@pytest.mark.asyncio
async def test_default_engine_is_doctr(monkeypatch: pytest.MonkeyPatch) -> None:
    """When engine kwarg is omitted, DocTR is used by default."""
    monkeypatch.setenv("PD_GPU_BACKEND", "cpu")

    fake_page = MagicMock()
    fake_page.to_dict.return_value = {}
    fake_doc = MagicMock()
    fake_doc.pages = [fake_page]

    with patch(
        "pd_book_tools.ocr.document.Document.from_image_ocr_via_doctr",
        return_value=fake_doc,
    ) as mock_doctr:
        dispatcher = LocalStageDispatcher()
        register_default_stages(dispatcher)
        await dispatcher.run_stage("ocr", "page-1", image_path="/fake/image.png")

    mock_doctr.assert_called_once()
