import json
import sqlite3

import pytest

from pd_ocr_ops.gpu.local_jobs import LocalLongJobRunner


@pytest.fixture
def runner(tmp_path):
    return LocalLongJobRunner(db_path=tmp_path / "jobs.db")


@pytest.mark.asyncio
async def test_submit_returns_job_id(runner):
    job_id = await runner.submit("training_run", {"epochs": 100})
    assert isinstance(job_id, str)
    assert len(job_id) > 0


@pytest.mark.asyncio
async def test_submit_writes_row_to_db(runner):
    job_id = await runner.submit("training_run", {"epochs": 100})
    conn = sqlite3.connect(str(runner._db_path))
    row = conn.execute(
        "SELECT state, kind, spec_json FROM jobs WHERE job_id=?", (job_id,)
    ).fetchone()
    conn.close()
    assert row is not None
    state, kind, spec_json = row
    assert state == "queued"
    assert kind == "training_run"
    assert json.loads(spec_json) == {"epochs": 100}


@pytest.mark.asyncio
async def test_submit_serializes_complex_spec(runner):
    spec = {"nested": {"a": 1, "b": [1, 2, 3]}, "flag": True}
    job_id = await runner.submit("batch_synth", spec)
    conn = sqlite3.connect(str(runner._db_path))
    row = conn.execute("SELECT spec_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    conn.close()
    assert json.loads(row[0]) == spec
