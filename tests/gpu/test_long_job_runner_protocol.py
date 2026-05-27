from pdomain_ops.gpu.protocols import LongJobRunner


def test_protocol_methods_present():
    assert hasattr(LongJobRunner, "submit")
    assert hasattr(LongJobRunner, "status")
    assert hasattr(LongJobRunner, "cancel")
    assert hasattr(LongJobRunner, "stream_events")


def test_protocol_is_runtime_checkable():
    class FakeRunner:
        async def submit(self, kind: str, spec: dict) -> str:
            return "job-abc"

        async def status(self, job_id: str):
            from pdomain_ops.gpu.types import JobStatus

            return JobStatus(job_id=job_id, kind="test", state="queued")

        async def cancel(self, job_id: str) -> None:
            pass

        async def stream_events(self, job_id: str):
            yield

    assert isinstance(FakeRunner(), LongJobRunner)
