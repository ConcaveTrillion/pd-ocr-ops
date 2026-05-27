import sqlite3

from pdomain_ops.gpu.local_jobs import init_jobs_db


def test_init_db_creates_jobs_table(tmp_path):
    db_path = tmp_path / "jobs.db"
    init_jobs_db(db_path)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("PRAGMA table_info(jobs)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}
    conn.close()
    assert "job_id" in columns
    assert "kind" in columns
    assert "spec_json" in columns
    assert "state" in columns
    assert "progress" in columns
    assert "started_at" in columns
    assert "finished_at" in columns
    assert "error" in columns
    assert "created_at" in columns


def test_init_db_creates_job_events_table(tmp_path):
    db_path = tmp_path / "jobs.db"
    init_jobs_db(db_path)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("PRAGMA table_info(job_events)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}
    conn.close()
    assert "job_id" in columns
    assert "seq" in columns
    assert "at" in columns
    assert "kind" in columns
    assert "payload_json" in columns


def test_init_db_idempotent(tmp_path):
    db_path = tmp_path / "jobs.db"
    init_jobs_db(db_path)
    # Second call should not raise
    init_jobs_db(db_path)


def test_init_db_adds_index_on_state(tmp_path):
    db_path = tmp_path / "jobs.db"
    init_jobs_db(db_path)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("PRAGMA index_list(jobs)")
    indexes = [row[1] for row in cursor.fetchall()]
    conn.close()
    assert any("state" in idx.lower() for idx in indexes)
