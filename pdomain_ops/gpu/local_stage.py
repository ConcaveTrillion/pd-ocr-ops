"""LocalStageDispatcher — in-process stage registry for local mode."""

from __future__ import annotations

import asyncio
import time
import warnings
from typing import TYPE_CHECKING, Any

from pdomain_ops.gpu.device import pick_device, pick_doctr_batch_sizes
from pdomain_ops.gpu.types import OcrBatchRequest, StageResult

if TYPE_CHECKING:
    from collections.abc import Callable

_VALID_DEVICES = frozenset({"local", "mps", "cpu", "modal", "shared_container"})


class UnknownStageError(KeyError):
    """Raised when a stage_id is not registered in the local registry."""


class LocalStageDispatcher:
    """In-process stage dispatcher.

    The registry is intentionally empty in Phase 1 — pgdp-prep's STAGE_IMPL
    migrates in plan #7.

    Registry keys are (stage_id, device) tuples.
    Impl signature: async def impl(page_id: str, device: str, **kwargs) -> dict
    """

    def __init__(
        self,
        registry: dict[tuple[str, str], Callable[..., Any]] | None = None,
    ) -> None:
        self._registry: dict[tuple[str, str], Callable[..., Any]] = dict(registry or {})

    def register_stage(self, stage_id: str, device: str, impl: Callable[..., Any]) -> None:
        """Register a stage implementation. Warns if replacing an existing entry."""
        if device not in _VALID_DEVICES:
            raise ValueError(f"device {device!r} is not valid. Allowed: {sorted(_VALID_DEVICES)}")
        key = (stage_id, device)
        if key in self._registry:
            warnings.warn(
                f"Replacing existing stage impl for ({stage_id!r}, {device!r})",
                UserWarning,
                stacklevel=2,
            )
        self._registry[key] = impl

    def unregister_stage(self, stage_id: str, device: str) -> None:
        """Remove a stage implementation."""
        self._registry.pop((stage_id, device), None)

    async def run_stage(
        self,
        stage_id: str,
        page_id: str,
        *,
        device: str | None = None,
        **kwargs: Any,
    ) -> StageResult:
        """Dispatch a stage call to the registered implementation.

        When *device* is given (e.g. a user CPU/GPU choice), it overrides
        auto-detection. Otherwise pick_device() chooses.
        Fallthrough order: requested/detected device -> "cpu" (if not in registry).
        """
        device = device or pick_device()

        # Try preferred device first, fall through to cpu
        impl = self._registry.get((stage_id, device))
        if impl is None and device != "cpu":
            impl = self._registry.get((stage_id, "cpu"))
            if impl is not None:
                device = "cpu"

        if impl is None:
            raise UnknownStageError(
                f"No implementation registered for stage {stage_id!r} "
                f"(tried device={device!r} and fallback cpu)"
            )

        start_ns = time.monotonic_ns()
        result_dict = await impl(page_id=page_id, device=device, **kwargs)
        duration_ms = (time.monotonic_ns() - start_ns) // 1_000_000

        return StageResult(
            stage_id=stage_id,
            page_id=page_id,
            device=device,  # pyright: ignore[reportArgumentType]  # narrowed by _VALID_DEVICES + fallback logic above
            duration_ms=duration_ms,
            metadata=result_dict or {},
        )

    async def run_ocr_batch(self, req: OcrBatchRequest) -> list[dict[str, object]]:
        """Run batched OCR on multiple pages.

        Accepts image bytes (not paths) so the same interface works remotely
        in Wave 5.  Delegates to :func:`~pdomain_ops.gpu.doctr_batch.run_doctr_batch`
        with a sized predictor fetched from / stored in the module-level
        predictor cache in :mod:`pdomain_ops.gpu.default_stages`.

        The ``build_smaller`` callback closes over the cache so that OOM
        backoff can rebuild a predictor at reduced batch sizes and store it
        back under its own cache key.
        """
        from pdomain_ops.gpu.default_stages import _predictor_cache
        from pdomain_ops.gpu.doctr_batch import run_doctr_batch

        device = pick_device()
        det_bs, reco_bs = pick_doctr_batch_sizes(device, len(req.images))

        def _get_or_build_predictor(d_bs: int, r_bs: int) -> Any:
            """Fetch from cache or build a predictor for the given batch sizes."""
            try:
                from pdomain_book_tools.hf import resolve_ocr_models
                from pdomain_book_tools.ocr.doctr_support import (
                    get_finetuned_torch_doctr_predictor,
                )
            except ImportError:
                return None  # No finetuned models available; run_doctr_batch will use CPU

            det_path, reco_path = resolve_ocr_models()
            cache_key = (str(det_path), str(reco_path), d_bs, r_bs)
            predictor = _predictor_cache.get(cache_key)
            if predictor is None:
                predictor = get_finetuned_torch_doctr_predictor(str(det_path), str(reco_path))
                _predictor_cache[cache_key] = predictor
            return predictor

        predictor = _get_or_build_predictor(det_bs, reco_bs)

        def build_smaller(d_bs: int, r_bs: int) -> Any:
            return _get_or_build_predictor(d_bs, r_bs)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: run_doctr_batch(
                req.images,
                predictor=predictor,
                device=device,
                build_smaller=build_smaller,
                source_identifiers=req.source_identifiers,
            ),
        )
