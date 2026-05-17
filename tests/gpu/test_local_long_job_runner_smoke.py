import sys

import pytest

from pd_ocr_ops.gpu.local_jobs import LocalLongJobRunner


@pytest.fixture
def runner(tmp_path):
    return LocalLongJobRunner(db_path=tmp_path / "jobs.db", poll_interval_s=0.05)


@pytest.mark.asyncio
async def test_run_sleep_job_succeeds(runner):
    """Submit a real subprocess job and verify it transitions to succeeded."""
    job_id = await runner.submit_with_process(
        "sleep_job",
        {},
        cmd=[sys.executable, "-c", "import time; time.sleep(0.1)"],
    )
    # Wait for supervisor to complete
    import asyncio

    for _ in range(50):  # Up to 5 seconds
        status = await runner.status(job_id)
        if status.state in ("succeeded", "failed"):
            break
        await asyncio.sleep(0.1)

    status = await runner.status(job_id)
    assert status.state == "succeeded"


@pytest.mark.asyncio
async def test_run_failing_subprocess_marks_failed(runner):
    """Submit a subprocess that exits with non-zero and verify state -> failed."""
    job_id = await runner.submit_with_process(
        "failing_job",
        {},
        cmd=[sys.executable, "-c", "import sys; sys.exit(1)"],
    )
    import asyncio

    for _ in range(50):  # Up to 5 seconds
        status = await runner.status(job_id)
        if status.state in ("succeeded", "failed"):
            break
        await asyncio.sleep(0.1)

    status = await runner.status(job_id)
    assert status.state == "failed"
    assert status.error is not None
