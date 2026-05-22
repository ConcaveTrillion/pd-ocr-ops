"""Default stage registration for pd-ocr-ops LocalStageDispatcher.

Wires DocTR and Tesseract CPU runners from pd-book-tools into the
dispatcher.  Call once at app startup::

    from pd_ocr_ops.gpu import LocalStageDispatcher, register_default_stages
    dispatcher = LocalStageDispatcher()
    register_default_stages(dispatcher)

The stage callable signature expected by LocalStageDispatcher is::

    async def impl(page_id: str, device: str, **kwargs) -> dict

Kwargs forwarded by pd-ocr-simple-gui (and any other caller):

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

If pd-book-tools OCR functions are not available at import time this
module does *not* raise; unavailable engines are handled gracefully inside
the impl (an ``ImportError`` is re-raised with a clear message).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pd_ocr_ops.gpu.local_stage import LocalStageDispatcher


def register_default_stages(dispatcher: LocalStageDispatcher) -> None:
    """Register the built-in OCR stages onto *dispatcher*.

    Registered entries:

    * ``("ocr", "cpu")`` — unified DocTR/Tesseract CPU runner.  Engine
      is selected at call time via ``kwargs["engine"]``
      (``"doctr"`` or ``"tesseract"``; defaults to ``"doctr"``).

    Calling this function more than once on the same dispatcher replaces
    the existing registrations with a ``UserWarning`` (standard
    ``LocalStageDispatcher`` behaviour).
    """
    dispatcher.register_stage("ocr", "cpu", _ocr_cpu_impl)


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
        from pd_book_tools.ocr.document import (  # pyright: ignore[reportMissingTypeStubs]  # pd-book-tools ships no py.typed
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


def _make_tesseract_sync(*, image_path: str, page_id: str, language: str) -> Any:
    """Return a zero-arg callable that runs Tesseract OCR synchronously."""

    def _run() -> dict[str, Any]:
        import cv2
        from pd_book_tools.ocr.cv2_tesseract import (  # pyright: ignore[reportMissingTypeStubs]  # no py.typed
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
