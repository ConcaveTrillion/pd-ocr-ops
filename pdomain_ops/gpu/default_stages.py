"""Default stage registration for pdomain-ops LocalStageDispatcher.

Wires DocTR and Tesseract CPU runners from pdomain-book-tools into the
dispatcher.  Call once at app startup::

    from pdomain_ops.gpu import LocalStageDispatcher, register_default_stages
    dispatcher = LocalStageDispatcher()
    register_default_stages(dispatcher)

The stage callable signature expected by LocalStageDispatcher is::

    async def impl(page_id: str, device: str, **kwargs) -> dict

Kwargs forwarded by pdomain-ocr-simple-gui (and any other caller):

* ``image_path: str`` — path to the image file to OCR
* ``engine: str`` — which engine to use (``"doctr"`` / ``"tesseract"``).
  Defaults to ``"doctr"``.
* ``language: str`` — language hint (e.g. ``"eng"``)

Stage registry layout
---------------------
The ``(stage_id, device)`` registry key uses the *hardware* device concept,
not the OCR engine name.  Engine selection is a runtime parameter passed via
``kwargs["engine"]``:

* ``("ocr", "cpu")`` — single impl that dispatches to DocTR or Tesseract
  depending on ``kwargs.get("engine", "doctr")``.

If pdomain-book-tools OCR functions are not available at import time this
module does *not* raise; unavailable engines are handled gracefully inside
the impl (an ``ImportError`` is re-raised with a clear message).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pdomain_ops.gpu.local_stage import LocalStageDispatcher

_log = logging.getLogger(__name__)

# Predictor cache keyed by ``(str(det_path), str(reco_path), det_bs, reco_bs)``.
# Building the finetuned DocTR predictor (load .pt files + ``.to(device)``)
# costs hundreds of ms per call; the underlying HF files are cached by
# huggingface_hub but the in-memory torch module is not.  Cache lives at
# module scope so every call in the same process reuses one predictor per
# (det, reco, det_bs, reco_bs) tuple.  The batch-size dimensions are part of
# the key because DocTR bakes det_bs / reco_bs into the predictor at build
# time; a predictor built for det_bs=4 must not be reused for det_bs=2.
_predictor_cache: dict[tuple[str, str, int, int], Any] = {}


def register_default_stages(dispatcher: LocalStageDispatcher) -> None:
    """Register the built-in OCR stages onto *dispatcher*.

    Registered entries:

    * ``("ocr", "cpu")`` — unified DocTR/Tesseract CPU runner.  Engine
      is selected at call time via ``kwargs["engine"]``
      (``"doctr"`` or ``"tesseract"``; defaults to ``"doctr"``).
    * ``("ocr", "local")`` — DocTR runner using finetuned weights from
      HF (``CT2534/pd-ocr-models``) on the autoselected torch device
      (CUDA > MPS > CPU).  Falls through to the cpu impl on network /
      HF errors.
    * ``("ocr", "mps")`` — same impl as ``("ocr", "local")``; the
      finetuned predictor builder already handles MPS via torch's
      device autoselection.

    Calling this function more than once on the same dispatcher replaces
    the existing registrations with a ``UserWarning`` (standard
    ``LocalStageDispatcher`` behaviour).
    """
    dispatcher.register_stage("ocr", "cpu", _ocr_cpu_impl)
    dispatcher.register_stage("ocr", "local", _ocr_local_impl)
    dispatcher.register_stage("ocr", "mps", _ocr_local_impl)


# ---------------------------------------------------------------------------
# Stage impl
# ---------------------------------------------------------------------------


async def _ocr_cpu_impl(page_id: str, device: str, **kwargs: Any) -> dict[str, Any]:
    """Unified CPU OCR stage — dispatches to DocTR or Tesseract by engine kwarg.

    Args:
        page_id: Opaque page identifier recorded in the result.
        device: Hardware device string (always ``"cpu"`` for this impl).
        **kwargs: Forwarded call-site parameters.  Recognised keys:

            * ``image_path`` (str): Path to the image file to OCR.
            * ``engine`` (str): ``"doctr"`` (default) or ``"tesseract"``.
            * ``language`` (str): Language hint (e.g. ``"eng"``).

    Returns:
        dict with a ``"pages"`` key containing a list of page dicts.

    Raises:
        ImportError: When the requested engine's library is not installed.
        ValueError: When *image_path* cannot be loaded.
    """
    image_path: str = kwargs.get("image_path", "")
    engine: str = kwargs.get("engine", "doctr")
    language: str = kwargs.get("language", "eng")

    loop = asyncio.get_event_loop()

    if engine == "tesseract":
        fn = _make_tesseract_sync(image_path=image_path, page_id=page_id, language=language)
    else:
        fn = _make_doctr_sync(image_path=image_path, page_id=page_id)

    return await loop.run_in_executor(None, fn)


def _make_doctr_sync(*, image_path: str, page_id: str) -> Any:
    """Return a zero-arg callable that runs DocTR OCR synchronously."""

    def _run() -> dict[str, Any]:
        from pdomain_book_tools.ocr.document import (
            Document,
        )

        doc = Document.from_image_ocr_via_doctr(
            image=image_path,
            source_identifier=page_id,
        )
        pages = doc.pages
        if not pages:
            return {"pages": []}
        return {"pages": [p.to_dict() for p in pages]}

    return _run


async def _ocr_local_impl(page_id: str, device: str, **kwargs: Any) -> dict[str, Any]:
    """GPU/finetuned OCR stage — DocTR with CT2534 finetuned weights.

    Resolves the finetuned detection + recognition ``.pt`` files via
    :func:`pdomain_book_tools.hf.resolve_ocr_models` (defaults pull from
    ``CT2534/pd-ocr-models``), builds the predictor on the autoselected
    torch device, and passes it to
    :meth:`pdomain_book_tools.ocr.document.Document.from_image_ocr_via_doctr`.

    Tesseract isn't GPU-bound, so when ``engine="tesseract"`` this impl
    delegates to the same Tesseract path the CPU impl uses.

    On any HF resolution / network failure this impl logs a warning and
    falls through to :func:`_ocr_cpu_impl` so the stage still succeeds on
    stock CPU weights.
    """
    image_path: str = kwargs.get("image_path", "")
    engine: str = kwargs.get("engine", "doctr")
    language: str = kwargs.get("language", "eng")

    if engine == "tesseract":
        loop = asyncio.get_event_loop()
        fn = _make_tesseract_sync(image_path=image_path, page_id=page_id, language=language)
        return await loop.run_in_executor(None, fn)

    # DocTR + finetuned weights.  Resolve paths; on network/HF failure
    # fall through to the stock-CPU impl.
    try:
        from pdomain_book_tools.hf import resolve_ocr_models

        det_path, reco_path = resolve_ocr_models()
    except Exception as exc:  # noqa: BLE001 — intentional broad catch for graceful fallback
        _log.warning(
            "resolve_ocr_models() failed (%s: %s); falling through to stock CPU OCR.",
            type(exc).__name__,
            exc,
        )
        return await _ocr_cpu_impl(page_id, device, **kwargs)

    from pdomain_ops.gpu.device import pick_doctr_batch_sizes

    det_bs, reco_bs = pick_doctr_batch_sizes(device, chunk_pages=1)

    loop = asyncio.get_event_loop()
    fn = _make_doctr_finetuned_sync(
        image_path=image_path,
        page_id=page_id,
        det_path=str(det_path),
        reco_path=str(reco_path),
        det_bs=det_bs,
        reco_bs=reco_bs,
    )
    return await loop.run_in_executor(None, fn)


def _make_doctr_finetuned_sync(
    *,
    image_path: str,
    page_id: str,
    det_path: str,
    reco_path: str,
    det_bs: int = 2,
    reco_bs: int = 128,
) -> Any:
    """Return a zero-arg callable that runs finetuned DocTR OCR synchronously.

    The predictor is cached by ``(det_path, reco_path, det_bs, reco_bs)`` so
    predictors built for different batch sizes are not shared.
    """

    def _run() -> dict[str, Any]:
        from pdomain_book_tools.ocr.doctr_support import (
            get_finetuned_torch_doctr_predictor,
        )
        from pdomain_book_tools.ocr.document import Document

        # 4-tuple key: batch-size dims are baked into the predictor at build time.
        cache_key = (det_path, reco_path, det_bs, reco_bs)
        predictor = _predictor_cache.get(cache_key)
        if predictor is None:
            predictor = get_finetuned_torch_doctr_predictor(det_path, reco_path)
            _predictor_cache[cache_key] = predictor

        doc = Document.from_image_ocr_via_doctr(
            image=image_path,
            source_identifier=page_id,
            predictor=predictor,
        )
        pages = doc.pages
        if not pages:
            return {"pages": []}
        return {"pages": [p.to_dict() for p in pages]}

    return _run


def _make_tesseract_sync(*, image_path: str, page_id: str, language: str) -> Any:
    """Return a zero-arg callable that runs Tesseract OCR synchronously."""

    def _run() -> dict[str, Any]:
        import cv2
        from pdomain_book_tools.ocr.cv2_tesseract import (
            _pytesseract_available,
            tesseract_ocr_cv2_image,
        )

        if not _pytesseract_available:
            raise ImportError(
                "pytesseract is not installed. "
                "Install the [tesseract] extra to use the Tesseract engine."
            )

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from path: {image_path!r}")
        page = tesseract_ocr_cv2_image(
            image,
            source_path=image_path,
            lang=language,
        )
        return {"pages": [page.to_dict()]}

    return _run
