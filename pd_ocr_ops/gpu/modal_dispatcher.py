"""Modal GPUBackend — dispatches GPU work to a Modal Function.

This is the canonical home for Modal-based GPU dispatch in the pd-* suite.
Deploy the companion app with::

    modal deploy src/pd_ocr_ops/gpu/modal_app.py

Cherry-picked-from: pd-prep-for-pgdp@e36c199df466ff45b70d2a452dd7512dcc2a17c9
"""

from __future__ import annotations

import logging
from typing import Any

from pd_ocr_ops.gpu.types import (
    BatchJobItem,
    BatchJobResult,
    BatchProgressCb,
    GPUBackend,
    OcrPageRequest,
    OcrPageResponse,
    ProcessPageRequest,
    ProcessPageResponse,
)

log = logging.getLogger(__name__)


class ModalStageDispatcher(GPUBackend):
    """GPUBackend that dispatches GPU work to a Modal Function.

    Args:
        token_id: Modal API token ID.
        token_secret: Modal API token secret.
        app_name: Modal app name to look up. Defaults to ``"pd-ocr-ops"``.
            Pass ``"pgdp-prep"`` to keep an existing pgdp-prep deployment
            alive during the migration window.
    """

    name = "modal"
    PROCESS_PAGE_FN = "process_page"
    RUN_OCR_FN = "run_ocr"
    RUN_BATCH_FN = "run_batch"

    def __init__(
        self,
        token_id: str,
        token_secret: str,
        app_name: str = "pd-ocr-ops",
    ) -> None:
        self._token_id = token_id
        self._token_secret = token_secret
        self._app_name = app_name
        self._fns: dict[str, Any] = {}

    def _load_function(self, fn_name: str) -> Any:
        """Load a Modal Function by name, caching the result."""
        cached = self._fns.get(fn_name)
        if cached is not None:
            return cached
        try:
            # optional [modal] extra; basedpyright flags the module on the `from` line
            from modal import Function  # pyright: ignore[reportMissingImports]
        except ImportError as e:
            raise RuntimeError(
                "Modal backend requires the [modal] extra: install with"
                " 'pip install pd-ocr-ops[modal]'"
            ) from e
        # modal stubs omit Function.lookup (valid at runtime); only surfaces
        # when the [modal] extra is installed.
        fn = Function.lookup(self._app_name, fn_name)  # pyright: ignore[reportAttributeAccessIssue]
        self._fns[fn_name] = fn
        return fn

    async def process_page(self, req: ProcessPageRequest) -> ProcessPageResponse:
        """Process (threshold/deskew) a single page image via Modal."""
        fn = self._load_function(self.PROCESS_PAGE_FN)
        result = await fn.remote.aio(req.model_dump())
        return ProcessPageResponse.model_validate(result)

    async def run_ocr(self, req: OcrPageRequest) -> OcrPageResponse:
        """Run OCR on a single page via Modal."""
        fn = self._load_function(self.RUN_OCR_FN)
        result = await fn.remote.aio(req.model_dump())
        return OcrPageResponse.model_validate(result)

    async def run_batch(
        self,
        items: list[BatchJobItem],
        *,
        progress_cb: BatchProgressCb | None = None,
    ) -> list[BatchJobResult]:
        """Run a batch of jobs via Modal with optional progress callback."""
        fn = self._load_function(self.RUN_BATCH_FN)
        payload = [item.model_dump() for item in items]
        results_raw = await fn.remote.aio(payload)
        results = [BatchJobResult.model_validate(r) for r in results_raw]
        if progress_cb is not None:
            total = len(results)
            for i, result in enumerate(results, start=1):
                try:
                    await progress_cb(i, total, result)
                except Exception:
                    log.exception("modal run_batch progress_cb raised; continuing")
        return results


# Legacy alias — pgdp-prep's pre-migration name.
ModalBackend = ModalStageDispatcher
