"""Shared-GPU-container GPUBackend — HTTP client stub.

This is the canonical home for shared-container-based GPU dispatch in the
pd-* suite. All methods raise ``NotImplementedError`` until the HTTP client
is wired in a follow-up plan.

Cherry-picked-from: pdomain-prep-for-pgdp@e36c199df466ff45b70d2a452dd7512dcc2a17c9
"""

from __future__ import annotations

from pdomain_ops.gpu.types import (
    BatchJobItem,
    BatchJobResult,
    BatchProgressCb,
    GPUBackend,
    OcrBatchRequest,
    OcrPageRequest,
    OcrPageResponse,
    ProcessPageRequest,
    ProcessPageResponse,
)


class SharedContainerStageDispatcher(GPUBackend):
    """GPUBackend that dispatches to a shared GPU container via HTTP.

    All methods are stubs pending HTTP client wiring (Phase 4).

    Args:
        base_url: Base URL of the shared GPU container service.
        api_key: API key for authenticating with the service.
    """

    name = "shared_container"

    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    async def process_page(self, req: ProcessPageRequest) -> ProcessPageResponse:
        """Process a page image — not yet wired."""
        raise NotImplementedError("shared_container.process_page not yet wired")

    async def run_ocr(self, req: OcrPageRequest) -> OcrPageResponse:
        """Run OCR on a page — not yet wired."""
        raise NotImplementedError("shared_container.run_ocr not yet wired")

    async def run_batch(
        self,
        items: list[BatchJobItem],
        *,
        progress_cb: BatchProgressCb | None = None,
    ) -> list[BatchJobResult]:
        """Run a batch of jobs — not yet wired."""
        raise NotImplementedError("shared_container.run_batch not yet wired")

    async def run_ocr_batch(self, req: OcrBatchRequest) -> list[dict[str, object]]:
        """Batched OCR via shared container — deferred to Wave 5.

        See pdomain-ops/docs/plans/2026-05-28-batched-ocr-dispatch.md
        """
        raise NotImplementedError(
            "Wave 5: remote batch — see pdomain-ops/docs/plans/2026-05-28-batched-ocr-dispatch.md"
        )


# Legacy alias — pgdp-prep's pre-migration name.
SharedContainerBackend = SharedContainerStageDispatcher
