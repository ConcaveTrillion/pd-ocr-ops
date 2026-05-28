"""Location-independent DocTR batch worker.

``run_doctr_batch`` is the reusable GPU-side worker for batched OCR.  It:

- Receives images as ndarrays OR bytes (decodes bytes -> ndarray via OpenCV).
- Sizes det_bs / reco_bs via :func:`~pdomain_ops.gpu.device.pick_doctr_batch_sizes`.
- Delegates the actual OCR call to
  :meth:`pdomain_book_tools.ocr.document.Document.from_images_ocr_via_doctr`
  (multi-image, one predictor call).
- Returns a flat list of page dicts -- one per input image, in order.
- Wraps the call in OOM backoff:
    1. OOM detected -> ``del predictor; torch.cuda.empty_cache()``;
       halve ``det_bs``; call ``build_smaller(det_bs, reco_bs)`` for a
       rebuilt predictor; retry.
    2. ``det_bs == 1`` still OOM -> per-image CPU fallback + WARNING log.
    3. Non-OOM ``RuntimeError`` / ``Exception`` -> re-raises immediately.

The function does NOT own a predictor cache -- that is the caller's job
(e.g., ``LocalStageDispatcher`` or a future Modal entrypoint).

OOM detection
-------------
The following exception types are treated as GPU OOM:

* ``torch.cuda.OutOfMemoryError`` (torch >= 2.0)
* ``RuntimeError`` whose ``str(e)`` contains ``"out of memory"``
* ``MemoryError``
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from pdomain_ops.gpu.device import pick_doctr_batch_sizes as _pick_doctr_batch_sizes_fn

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_log = logging.getLogger(__name__)


def _is_oom(exc: BaseException) -> bool:
    """Return True when *exc* is a GPU-OOM or generic memory-exhaustion error."""
    # torch.cuda.OutOfMemoryError (torch >= 2.0) is a subclass of RuntimeError.
    # We check the class name first to avoid importing torch unconditionally.
    cls_name = type(exc).__name__
    if cls_name in {"OutOfMemoryError", "CUDAOutOfMemoryError"}:
        return True
    if isinstance(exc, MemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _empty_cuda_cache() -> None:
    """Call torch.cuda.empty_cache() if torch is available; no-op otherwise."""
    try:
        import torch  # pyright: ignore[reportMissingImports]  # optional [gpu] dep

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001,S110 — best-effort cache clear; log not useful here
        pass


def _decode_image(img: np.ndarray | bytes) -> np.ndarray:
    """Decode *img* to an ndarray if it is bytes; pass through ndarrays."""
    if isinstance(img, np.ndarray):
        return img
    import cv2  # pyright: ignore[reportMissingImports]  # optional [gpu] dep; always present with doctr

    arr = np.frombuffer(img, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("cv2.imdecode returned None -- image bytes are not a valid image format")
    return decoded


def _cpu_fallback_single(image: np.ndarray, source_id: str) -> dict[str, Any]:
    """Run single-image CPU DocTR OCR as a last-resort OOM fallback."""
    from pdomain_book_tools.ocr.document import Document

    doc = Document.from_images_ocr_via_doctr(
        images=[image],
        source_identifiers=[source_id],
        # No predictor -- stock CPU weights, no VRAM pressure.
    )
    if not doc.pages:
        return {}
    return doc.pages[0].to_dict()


def run_doctr_batch(
    images: Sequence[np.ndarray | bytes],
    *,
    predictor: Any,
    device: str,
    build_smaller: Callable[[int, int], Any] | None = None,
    source_identifiers: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Run batched DocTR OCR on *images*, returning one page dict per input.

    Parameters
    ----------
    images:
        List of images as ``numpy.ndarray`` (HxWxC, uint8) or raw image
        ``bytes`` (PNG/JPEG; decoded internally via OpenCV).
    predictor:
        A built DocTR predictor (e.g. from
        :func:`pdomain_book_tools.ocr.doctr_support.get_finetuned_torch_doctr_predictor`).
        The caller owns the predictor and its cache -- this function does
        NOT cache it.
    device:
        Hardware device string (``"local"``, ``"mps"``, or ``"cpu"``).
        Used to size ``det_bs`` / ``reco_bs`` via
        :func:`~pdomain_ops.gpu.device.pick_doctr_batch_sizes`.
    build_smaller:
        Optional callback ``(det_bs, reco_bs) -> predictor`` that returns
        a rebuilt predictor sized to the given batch sizes.  Required for
        OOM backoff on CUDA devices; ignored on CPU.
    source_identifiers:
        Per-image identifiers forwarded to book-tools for provenance
        tracking. Defaults to ``["0", "1", ...]`` when omitted.

    Returns:
    -------
    list[dict]:
        One page dict per input image, in the same order as *images*.
        Each dict is the output of ``Page.to_dict()`` from book-tools.

    Raises:
    ------
    RuntimeError:
        Any non-OOM ``RuntimeError`` is re-raised immediately.
    Exception:
        Any unexpected exception other than OOM types is re-raised.
    """
    if source_identifiers is None:
        source_identifiers = [str(i) for i in range(len(images))]

    # Decode bytes -> ndarray up front (cheap, keeps OCR path clean).
    decoded: list[np.ndarray] = [_decode_image(img) for img in images]

    det_bs, reco_bs = _pick_doctr_batch_sizes_fn(device, len(decoded))

    from pdomain_book_tools.ocr.document import Document

    current_predictor = predictor

    while True:
        try:
            doc = Document.from_images_ocr_via_doctr(
                images=decoded,
                source_identifiers=list(source_identifiers),
                predictor=current_predictor,
            )
            pages = doc.pages
            return [p.to_dict() for p in pages]

        except Exception as exc:
            if not _is_oom(exc):
                raise

            # OOM path -- free VRAM, try with smaller batch sizes.
            _log.warning(
                "OOM during DocTR batch (det_bs=%d, reco_bs=%d): %s -- attempting backoff",
                det_bs,
                reco_bs,
                exc,
            )
            del current_predictor
            _empty_cuda_cache()

            if det_bs <= 1:
                # Floor reached -- cannot shrink further; fall back to per-image CPU OCR.
                _log.warning(
                    "OOM with det_bs=1 on device=%r; falling back to per-image CPU OCR",
                    device,
                )
                return [
                    _cpu_fallback_single(img, sid)
                    for img, sid in zip(decoded, source_identifiers, strict=True)
                ]

            det_bs = max(1, det_bs // 2)
            if build_smaller is not None:
                current_predictor = build_smaller(det_bs, reco_bs)
            else:
                # No build_smaller provided -- cannot rebuild; fall back to CPU.
                _log.warning(
                    "OOM with no build_smaller callback; falling back to per-image CPU OCR",
                )
                return [
                    _cpu_fallback_single(img, sid)
                    for img, sid in zip(decoded, source_identifiers, strict=True)
                ]
