"""GPU adapter Protocol classes: StageDispatcher + LongJobRunner."""

from __future__ import annotations

from typing import TYPE_CHECKING, runtime_checkable

from typing_extensions import Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pd_ocr_ops.gpu.types import JobEvent, JobStatus, StageResult


@runtime_checkable
class StageDispatcher(Protocol):
    """Short, sync-ish GPU stage calls (OCR, layout, char-bbox).

    Mirrors pgdp-prep's existing STAGE_IMPL registry shape.
    """

    async def run_stage(self, stage_id: str, page_id: str, **kwargs: object) -> StageResult:
        """Dispatch a short GPU stage call and return the result."""
        ...


@runtime_checkable
class LongJobRunner(Protocol):
    """Long-running job management (training runs, batch synth, etc.)."""

    async def submit(self, kind: str, spec: dict[str, object]) -> str:
        """Submit a new job; returns the job_id."""
        ...

    async def status(self, job_id: str) -> JobStatus:
        """Return current status of a job."""
        ...

    async def cancel(self, job_id: str) -> None:
        """Cancel a running or queued job."""
        ...

    async def stream_events(self, job_id: str) -> AsyncIterator[JobEvent]:
        """Stream events for a job until terminal state."""
        ...
