from pdomain_ops.gpu.protocols import StageDispatcher


def test_protocol_methods_present():
    assert hasattr(StageDispatcher, "run_stage")


def test_protocol_is_runtime_checkable():
    class FakeDispatcher:
        async def run_stage(self, stage_id: str, page_id: str, **kwargs):
            from pdomain_ops.gpu.types import StageResult

            return StageResult(
                stage_id=stage_id,
                page_id=page_id,
                device="cpu",
                duration_ms=0,
            )

    assert isinstance(FakeDispatcher(), StageDispatcher)
