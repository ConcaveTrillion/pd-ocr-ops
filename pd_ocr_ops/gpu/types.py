"""GPU adapter types: StageResult, JobStatus, JobEvent, JobSpec."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator


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
