"""Supervisor parses worker @@PDEVENT@@ stdout into JobEvents."""

import asyncio
import textwrap

import pytest

from pd_ocr_ops.gpu.local_jobs import LocalLongJobRunner


@pytest.fixture
def runner(tmp_path):
    return LocalLongJobRunner(db_path=tmp_path / "jobs.db", poll_interval_s=0.02)


def _worker_script(tmp_path, body: str) -> str:
    path = tmp_path / "worker.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(path)


async def _drain(runner, job_id, *, cap=200):
    events = []
    async for ev in runner.stream_events(job_id):
        events.append(ev)
        if len(events) > cap:
            break
    return events


@pytest.mark.asyncio
async def test_supervisor_parses_pdevent_lines_into_job_events(runner, tmp_path):
    """A worker emitting @@PDEVENT@@ lines surfaces them as JobEvents."""
    script = _worker_script(
        tmp_path,
        """
        import json, sys
        for i in range(3):
            print("@@PDEVENT@@ " + json.dumps(
                {"kind": "progress", "payload": {"pct": (i + 1) / 3}}
            ), flush=True)
        print("@@PDEVENT@@ " + json.dumps(
            {"kind": "metric", "payload": {"cer": 0.1}}
        ), flush=True)
        sys.exit(0)
        """,
    )
    import sys

    job_id = await runner.submit_with_process("training_run", {}, cmd=[sys.executable, script])
    events = await _drain(runner, job_id)

    progress = [e for e in events if e.kind == "progress"]
    metric = [e for e in events if e.kind == "metric"]
    assert len(progress) == 3
    assert len(metric) == 1
    assert metric[0].payload == {"cer": 0.1}
    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs)

    status = await runner.status(job_id)
    assert status.state == "succeeded"


@pytest.mark.asyncio
async def test_supervisor_captures_plain_stdout_as_log_events(runner, tmp_path):
    """Non-@@PDEVENT@@ stdout lines are captured as log events, not dropped."""
    script = _worker_script(
        tmp_path,
        """
        import json, sys
        print("starting up", flush=True)
        print("@@PDEVENT@@ " + json.dumps(
            {"kind": "progress", "payload": {"pct": 1.0}}
        ), flush=True)
        print("all done", flush=True)
        sys.exit(0)
        """,
    )
    import sys

    job_id = await runner.submit_with_process("training_run", {}, cmd=[sys.executable, script])
    events = await _drain(runner, job_id)

    logs = [e for e in events if e.kind == "log"]
    log_lines = [e.payload.get("line") for e in logs]
    assert "starting up" in log_lines
    assert "all done" in log_lines
    assert any(e.kind == "progress" for e in events)


@pytest.mark.asyncio
async def test_supervisor_skips_malformed_event_lines_without_crashing(runner, tmp_path):
    """A malformed @@PDEVENT@@ line is logged and skipped; job still succeeds."""
    script = _worker_script(
        tmp_path,
        """
        import json, sys
        print("@@PDEVENT@@ {not valid json", flush=True)
        print("@@PDEVENT@@ " + json.dumps(
            {"kind": "progress", "payload": {"pct": 0.5}}
        ), flush=True)
        sys.exit(0)
        """,
    )
    import sys

    job_id = await runner.submit_with_process("training_run", {}, cmd=[sys.executable, script])
    events = await _drain(runner, job_id)

    assert any(e.kind == "progress" for e in events)
    status = await runner.status(job_id)
    assert status.state == "succeeded"


@pytest.mark.asyncio
async def test_terminal_state_still_set_on_failure(runner, tmp_path):
    """Existing terminal-state behavior preserved when worker exits non-zero."""
    script = _worker_script(
        tmp_path,
        """
        import json, sys
        print("@@PDEVENT@@ " + json.dumps(
            {"kind": "progress", "payload": {"pct": 0.5}}
        ), flush=True)
        print("boom", file=sys.stderr, flush=True)
        sys.exit(1)
        """,
    )
    import sys

    job_id = await runner.submit_with_process("training_run", {}, cmd=[sys.executable, script])
    for _ in range(100):
        status = await runner.status(job_id)
        if status.state in ("succeeded", "failed"):
            break
        await asyncio.sleep(0.05)
    status = await runner.status(job_id)
    assert status.state == "failed"
    assert status.error is not None
