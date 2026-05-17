import pytest

from pd_ocr_ops.gpu.local_jobs import LocalLongJobRunner, UnknownJobError


@pytest.fixture
def runner(tmp_path):
    return LocalLongJobRunner(db_path=tmp_path / "jobs.db")


@pytest.mark.asyncio
async def test_status_returns_queued_after_submit(runner):
    job_id = await runner.submit("training_run", {})
    status = await runner.status(job_id)
    assert status.state == "queued"
    assert status.job_id == job_id


@pytest.mark.asyncio
async def test_status_unknown_job_raises(runner):
    with pytest.raises(UnknownJobError):
        await runner.status("nonexistent-job-id")


@pytest.mark.asyncio
async def test_internal_state_transition_writes_progress(runner):
    job_id = await runner.submit("training_run", {})
    runner._set_state(job_id, "running", progress=0.5)
    status = await runner.status(job_id)
    assert status.state == "running"
    assert abs(status.progress - 0.5) < 0.001
    assert status.started_at is not None


@pytest.mark.asyncio
async def test_internal_state_transition_to_succeeded_sets_finished_at(runner):
    job_id = await runner.submit("training_run", {})
    runner._set_state(job_id, "running", progress=0.0)
    runner._set_state(job_id, "succeeded", progress=1.0)
    status = await runner.status(job_id)
    assert status.state == "succeeded"
    assert status.finished_at is not None


@pytest.mark.asyncio
async def test_internal_state_transition_to_failed_records_error(runner):
    job_id = await runner.submit("training_run", {})
    runner._set_failed(job_id, "OOM")
    status = await runner.status(job_id)
    assert status.state == "failed"
    assert status.error == "OOM"
