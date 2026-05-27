from unittest.mock import MagicMock

import pytest

from pdomain_ops.gpu.local_jobs import LocalLongJobRunner, UnknownJobError


@pytest.fixture
def runner(tmp_path):
    return LocalLongJobRunner(db_path=tmp_path / "jobs.db")


@pytest.mark.asyncio
async def test_cancel_queued_job_marks_cancelled(runner):
    job_id = await runner.submit("training_run", {})
    await runner.cancel(job_id)
    status = await runner.status(job_id)
    assert status.state == "cancelled"
    assert status.finished_at is not None


@pytest.mark.asyncio
async def test_cancel_running_job_signals_subprocess(runner):
    job_id = await runner.submit("training_run", {})
    runner._set_state(job_id, "running", progress=0.1)
    # Install a fake subprocess
    mock_proc = MagicMock()
    mock_proc.wait.return_value = None
    runner._processes[job_id] = mock_proc
    await runner.cancel(job_id)
    mock_proc.terminate.assert_called_once()
    status = await runner.status(job_id)
    assert status.state == "cancelled"


@pytest.mark.asyncio
async def test_cancel_succeeded_job_is_noop(runner):
    job_id = await runner.submit("training_run", {})
    runner._set_state(job_id, "succeeded", progress=1.0)
    # Should not raise, state should stay succeeded
    await runner.cancel(job_id)
    status = await runner.status(job_id)
    assert status.state == "succeeded"


@pytest.mark.asyncio
async def test_cancel_unknown_job_raises(runner):
    with pytest.raises(UnknownJobError):
        await runner.cancel("nonexistent-job-id")
