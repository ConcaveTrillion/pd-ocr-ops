"""LocalStageDispatcher — in-process stage registry for local mode."""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any

from pdomain_ops.gpu.device import pick_device
from pdomain_ops.gpu.types import StageResult

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

    async def run_stage(self, stage_id: str, page_id: str, **kwargs: Any) -> StageResult:
        """Dispatch a stage call to the registered implementation.

        Fallthrough order: pick_device() -> "cpu" (if device not in registry).
        """
        device = pick_device()

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
