"""SQLite jobs table + LocalLongJobRunner implementation."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import filelock

from pd_ocr_ops.gpu.events import is_event_line, parse_event_line
from pd_ocr_ops.gpu.types import JobEvent, JobStatus

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})

_log = logging.getLogger(__name__)


class UnknownJobError(KeyError):
    """Raised when a job_id is not found in the database."""


def init_jobs_db(path: Path) -> None:
    """Create the jobs + job_events tables if they don't exist. Idempotent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            spec_json TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'queued',
            progress REAL NOT NULL DEFAULT 0.0,
            started_at TEXT,
            finished_at TEXT,
            error TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS job_events (
            job_id TEXT NOT NULL REFERENCES jobs(job_id),
            seq INTEGER NOT NULL,
            at TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY (job_id, seq)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs (state)
    """)
    conn.commit()
    conn.close()


class LocalLongJobRunner:
    """SQLite-backed long-running job runner for local mode."""

    def __init__(
        self,
        db_path: Path | None = None,
        poll_interval_s: float = 0.5,
    ) -> None:
        if db_path is None:
            from pd_ocr_ops.suite.paths import jobs_db_path

            db_path = jobs_db_path()
        self._db_path = Path(db_path)
        self._lock_path = self._db_path.with_suffix(".db.lock")
        self._poll_interval_s = poll_interval_s
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        init_jobs_db(self._db_path)

    def _now_iso(self) -> str:
        return datetime.now(UTC).isoformat()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    async def submit(self, kind: str, spec: dict[str, object]) -> str:
        """Submit a new job; returns the job_id. State: queued."""
        job_id = str(uuid.uuid4())
        with filelock.FileLock(str(self._lock_path)):
            conn = self._conn()
            sql = (
                "INSERT INTO jobs (job_id, kind, spec_json, state, created_at)"
                " VALUES (?, ?, ?, 'queued', ?)"
            )
            conn.execute(sql, (job_id, kind, json.dumps(spec), self._now_iso()))
            conn.commit()
            conn.close()
        return job_id

    async def submit_with_process(self, kind: str, spec: dict[str, object], cmd: list[str]) -> str:
        """Submit a job and launch a real subprocess; supervise it."""
        job_id = await self.submit(kind, spec)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._processes[job_id] = proc
        self._set_state(job_id, "running", progress=0.0)
        # Fire-and-forget supervision task
        asyncio.create_task(self._supervise(job_id, proc))
        return job_id

    def _drain_stdout(self, job_id: str, proc: subprocess.Popen[bytes]) -> None:
        """Read worker stdout line by line, routing @@PDEVENT@@ lines to events.

        Runs in an executor thread. Lines carrying the @@PDEVENT@@ protocol
        become typed JobEvents; everything else is captured as a log event.
        Malformed event lines are logged and skipped so a bad line can never
        crash the supervisor.
        """
        if proc.stdout is None:
            return
        for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip("\n")
            if not line:
                continue
            if is_event_line(line):
                try:
                    kind, payload = parse_event_line(line)
                except ValueError as exc:
                    _log.warning("skipping malformed @@PDEVENT@@ line: %s", exc)
                    continue
                self._append_event(job_id, kind, payload)
            else:
                self._append_event(job_id, "log", {"line": line})

    async def _supervise(self, job_id: str, proc: subprocess.Popen[bytes]) -> None:
        """Drain worker stdout into events, wait for exit, and set terminal state."""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._drain_stdout, job_id, proc)
            returncode = await loop.run_in_executor(None, proc.wait)
            if returncode == 0:
                self._set_state(job_id, "succeeded", progress=1.0)
            else:
                stderr_output = ""
                if proc.stderr:
                    raw = proc.stderr.read()
                    stderr_output = raw.decode(errors="replace").strip()
                error_msg = (
                    stderr_output.split("\n")[-1] if stderr_output else f"exit code {returncode}"
                )
                self._set_failed(job_id, error_msg)
        except Exception as e:
            self._set_failed(job_id, str(e))
        finally:
            self._processes.pop(job_id, None)

    async def status(self, job_id: str) -> JobStatus:
        """Return current status of a job."""
        with filelock.FileLock(str(self._lock_path)):
            conn = self._conn()
            sql = (
                "SELECT job_id, kind, state, progress, started_at, finished_at, error"
                " FROM jobs WHERE job_id=?"
            )
            row = conn.execute(sql, (job_id,)).fetchone()
            conn.close()

        if row is None:
            raise UnknownJobError(f"Job not found: {job_id!r}")

        job_id_, kind, state, progress, started_at, finished_at, error = row
        return JobStatus(
            job_id=job_id_,
            kind=kind,
            state=state,
            progress=progress,
            started_at=datetime.fromisoformat(started_at) if started_at else None,
            finished_at=datetime.fromisoformat(finished_at) if finished_at else None,
            error=error,
        )

    async def cancel(self, job_id: str) -> None:
        """Cancel a running or queued job."""
        # Check if job exists
        status = await self.status(job_id)  # raises UnknownJobError if missing

        if status.state in _TERMINAL_STATES:
            return  # noop for terminal states

        # Signal running subprocess if present
        proc = self._processes.get(job_id)
        if proc is not None:
            proc.terminate()
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, proc.wait)

        self._set_state(job_id, "cancelled", progress=status.progress)

    def _set_state(self, job_id: str, state: str, progress: float = 0.0) -> None:
        """Internal helper to update job state."""
        now = self._now_iso()
        with filelock.FileLock(str(self._lock_path)):
            conn = self._conn()
            # Set started_at if transitioning to running
            if state == "running":
                sql = (
                    "UPDATE jobs SET state=?, progress=?,"
                    " started_at=COALESCE(started_at, ?) WHERE job_id=?"
                )
                conn.execute(sql, (state, progress, now, job_id))
            elif state in ("succeeded", "failed", "cancelled"):
                conn.execute(
                    "UPDATE jobs SET state=?, progress=?, finished_at=? WHERE job_id=?",
                    (state, progress, now, job_id),
                )
            else:
                conn.execute(
                    "UPDATE jobs SET state=?, progress=? WHERE job_id=?",
                    (state, progress, job_id),
                )
            conn.commit()
            conn.close()

    def _set_failed(self, job_id: str, error: str) -> None:
        """Internal helper to mark a job as failed with error message."""
        now = self._now_iso()
        with filelock.FileLock(str(self._lock_path)):
            conn = self._conn()
            conn.execute(
                "UPDATE jobs SET state='failed', error=?, finished_at=? WHERE job_id=?",
                (error, now, job_id),
            )
            conn.commit()
            conn.close()

    def _append_event(self, job_id: str, kind: str, payload: dict[str, object]) -> None:
        """Append an event to the job_events table."""
        now = self._now_iso()
        with filelock.FileLock(str(self._lock_path)):
            conn = self._conn()
            seq = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM job_events WHERE job_id=?",
                (job_id,),
            ).fetchone()[0]
            sql = (
                "INSERT INTO job_events (job_id, seq, at, kind, payload_json)"
                " VALUES (?, ?, ?, ?, ?)"
            )
            conn.execute(sql, (job_id, seq, now, kind, json.dumps(payload)))
            conn.commit()
            conn.close()

    async def stream_events(self, job_id: str) -> AsyncIterator[JobEvent]:
        """Stream events for a job until terminal state.

        Polling-backed (local mode). Polls every poll_interval_s.
        """
        # Check job exists upfront
        _ = await self.status(job_id)  # raises UnknownJobError if missing

        last_seq = 0
        while True:
            with filelock.FileLock(str(self._lock_path)):
                conn = self._conn()
                sql = (
                    "SELECT seq, at, kind, payload_json FROM job_events"
                    " WHERE job_id=? AND seq > ? ORDER BY seq"
                )
                rows = conn.execute(sql, (job_id, last_seq)).fetchall()
                state_row = conn.execute(
                    "SELECT state FROM jobs WHERE job_id=?",
                    (job_id,),
                ).fetchone()
                conn.close()

            for seq, at, kind, payload_json in rows:
                yield JobEvent(
                    job_id=job_id,
                    seq=seq,
                    at=datetime.fromisoformat(at),
                    kind=kind,
                    payload=json.loads(payload_json),
                )
                last_seq = seq

            if state_row and state_row[0] in _TERMINAL_STATES:
                break

            await asyncio.sleep(self._poll_interval_s)
