from datetime import datetime, timezone

from pd_ocr_ops.gpu.types import JobEvent, JobSpec, JobStatus, StageResult


def test_stage_result_shape():
    result = StageResult(
        stage_id="ocr",
        page_id="page-001",
        device="local",
        duration_ms=123,
    )
    assert result.output_key is None
    assert result.metadata == {}
    # Roundtrip JSON
    data = result.model_dump(mode="json")
    roundtripped = StageResult.model_validate(data)
    assert roundtripped == result


def test_job_status_shape():
    status = JobStatus(
        job_id="job-abc",
        kind="training_run",
        state="queued",
    )
    assert status.progress == 0.0
    assert status.started_at is None
    assert status.finished_at is None
    assert status.error is None
    # Roundtrip
    data = status.model_dump(mode="json")
    roundtripped = JobStatus.model_validate(data)
    assert roundtripped == status


def test_job_event_shape():
    now = datetime.now(timezone.utc)
    event = JobEvent(
        job_id="job-abc",
        seq=1,
        at=now,
        kind="progress",
        payload={"pct": 0.5},
    )
    data = event.model_dump(mode="json")
    roundtripped = JobEvent.model_validate(data)
    assert roundtripped == event


def test_job_spec_shape():
    spec = JobSpec(
        kind="training_run",
        params={"epochs": 100, "lr": 0.001},
    )
    assert spec.priority == "batch"
    data = spec.model_dump(mode="json")
    roundtripped = JobSpec.model_validate(data)
    assert roundtripped == spec
