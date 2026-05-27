"""GPU adapter types: StageResult, JobStatus, JobEvent, JobSpec."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime  # noqa: TC003  # Pydantic requires runtime import
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StageResult(BaseModel):
    """Result of a short GPU stage call."""

    model_config = ConfigDict(extra="forbid")

    stage_id: str
    page_id: str
    device: Literal["local", "mps", "cpu", "modal", "shared_container"]
    duration_ms: int
    output_key: str | None = None
    metadata: dict[str, Any] = {}


class JobStatus(BaseModel):
    """Status of a long-running job."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    kind: str
    state: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    progress: float = 0.0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None

    @field_validator("progress")
    @classmethod
    def validate_progress(cls, v: float) -> float:
        """Validate that progress is in [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"progress must be between 0.0 and 1.0, got: {v}")
        return v


class JobEvent(BaseModel):
    """A single event in a job's event stream."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    seq: int
    at: datetime
    kind: Literal["progress", "log", "state", "metric"]
    payload: dict[str, Any]


class JobSpec(BaseModel):
    """Specification for submitting a new long-running job."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    params: dict[str, Any] = {}
    priority: Literal["interactive", "batch"] = "batch"


# ── Phase 1.7 dispatch wire-shapes (lifted from pdomain-prep-for-pgdp) ────────────
# Cherry-picked-from: pdomain-prep-for-pgdp@e36c199df466ff45b70d2a452dd7512dcc2a17c9


class DispatchModel(BaseModel):
    """Base for GPU dispatch wire-shape models.

    Sets json_schema_serialization_defaults_required=True so fields with
    defaults emit as required in OpenAPI serialization schema. Mirrors
    pdomain-prep-for-pgdp's ApiModel for the migration window.

    History: replicated from pdomain-prep-for-pgdp ApiModel in Phase 1.7.
    """

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)


class ProcessPageRequest(DispatchModel):
    """Request to process (threshold/deskew) a single page image."""

    project_id: str
    idx0: int
    config_overrides: dict[
        str, Any
    ] = {}  # generic — pdomain-prep-for-pgdp validates as PageConfigOverrides
    output_context: Literal["workbench", "commit"] = "workbench"


class ProcessPageResponse(DispatchModel):
    """Response from a process_page call."""

    processed_image_key: str
    processed_image_url: str
    dimensions: tuple[int, int]
    processing_time_ms: int
    backend: Literal["local", "cpu", "mps", "modal", "shared_container"]
    cold_start_ms: int = 0


class OcrPageRequest(DispatchModel):
    """Request to run OCR on a single page."""

    project_id: str
    idx0: int
    split_suffix: str | None = None
    engine: Literal["doctr", "tesseract"] | None = None
    model_key: str | None = None
    batch_mode: bool = False


class OcrPageResponse(DispatchModel):
    """Response from a run_ocr call."""

    text: str
    words: list[dict[str, Any]] = Field(default_factory=list)  # generic — OcrWord in pgdp-prep
    text_key: str


class BatchJobItem(DispatchModel):
    """A single item in a batch job submission."""

    job_type: str
    project_id: str
    idx0: int
    payload: dict[str, Any] = Field(default_factory=dict)


class BatchJobResult(DispatchModel):
    """Result for a single item from a batch job run."""

    job_type: str
    project_id: str
    idx0: int
    ok: bool
    error: str | None = None
    error_type: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


BatchProgressCb = Callable[[int, int, BatchJobResult], Awaitable[None]]
"""Callback type for batch job progress notifications: (done, total, result) -> None."""


@runtime_checkable
class GPUBackend(Protocol):
    """Legacy pgdp-prep-style GPU dispatch Protocol.

    Has specific process_page/run_ocr/run_batch methods as used by
    pdomain-prep-for-pgdp. Coexists with StageDispatcher (generic run_stage)
    during the Phase 1.7 migration window.

    NOT an alias for StageDispatcher — they have different method signatures.
    StageDispatcher is the forward-looking generic interface; GPUBackend is
    the pgdp-prep-specific legacy interface retained until pgdp-prep migrates.

    History: moved from pdomain-prep-for-pgdp adapters/gpu/base.py in Phase 1.7.
    """

    name: Literal["local", "cpu", "mps", "modal", "shared_container"]

    async def process_page(self, req: ProcessPageRequest) -> ProcessPageResponse:
        """Process (threshold/deskew) a single page image."""
        ...

    async def run_ocr(self, req: OcrPageRequest) -> OcrPageResponse:
        """Run OCR on a single page."""
        ...

    async def run_batch(
        self,
        items: list[BatchJobItem],
        *,
        progress_cb: BatchProgressCb | None = None,
    ) -> list[BatchJobResult]:
        """Run a batch of jobs with optional progress callback."""
        ...
