from datetime import UTC, datetime

from pdomain_ops.gpu.types import (
    BatchJobItem,
    BatchJobResult,
    GPUBackend,
    JobEvent,
    JobSpec,
    JobStatus,
    OcrPageRequest,
    OcrPageResponse,
    ProcessPageRequest,
    ProcessPageResponse,
    StageResult,
)


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
    now = datetime.now(UTC)
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


# ── Phase 1.7 dispatch wire-shape tests ──────────────────────────────────────


def test_process_page_request_roundtrip():
    req = ProcessPageRequest(
        project_id="p1",
        idx0=0,
        config_overrides={"threshold_level": 128},
    )
    assert req.output_context == "workbench"
    data = req.model_dump(mode="json")
    roundtripped = ProcessPageRequest.model_validate(data)
    assert roundtripped == req


def test_process_page_response_roundtrip():
    resp = ProcessPageResponse(
        processed_image_key="out.png",
        processed_image_url="/img/out.png",
        dimensions=(800, 1200),
        processing_time_ms=42,
        backend="local",
    )
    assert resp.cold_start_ms == 0
    data = resp.model_dump(mode="json")
    roundtripped = ProcessPageResponse.model_validate(data)
    assert roundtripped == resp


def test_ocr_page_request_roundtrip():
    req = OcrPageRequest(project_id="p1", idx0=0)
    assert req.engine is None
    assert req.batch_mode is False
    data = req.model_dump(mode="json")
    roundtripped = OcrPageRequest.model_validate(data)
    assert roundtripped == req


def test_ocr_page_response_roundtrip():
    resp = OcrPageResponse(text="Hello world", text_key="out.txt")
    assert resp.words == []
    data = resp.model_dump(mode="json")
    roundtripped = OcrPageResponse.model_validate(data)
    assert roundtripped == resp


def test_batch_job_item_and_result_roundtrip():
    item = BatchJobItem(job_type="ocr", project_id="p1", idx0=2)
    assert item.payload == {}
    result = BatchJobResult(job_type="ocr", project_id="p1", idx0=2, ok=True)
    assert result.error is None
    for model in (item, result):
        data = model.model_dump(mode="json")
        roundtripped = type(model).model_validate(data)
        assert roundtripped == model


def test_gpu_backend_protocol_is_runtime_checkable():
    # Python 3.13 uses _is_runtime_protocol; older versions used __runtime_checkable__.
    # Check either flag, or verify isinstance() doesn't raise TypeError (the
    # functional proof that @runtime_checkable is applied).
    is_runtime = getattr(GPUBackend, "_is_runtime_protocol", False) or getattr(
        GPUBackend, "__runtime_checkable__", False
    )
    assert is_runtime, "GPUBackend must be @runtime_checkable"
