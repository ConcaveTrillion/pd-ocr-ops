import pytest

from pd_ocr_ops.gpu.local_stage import LocalStageDispatcher, UnknownStageError
from pd_ocr_ops.gpu.types import StageResult


@pytest.mark.asyncio
async def test_construct_with_empty_registry():
    dispatcher = LocalStageDispatcher(registry={})
    assert dispatcher is not None


@pytest.mark.asyncio
async def test_run_stage_unknown_id_raises():
    dispatcher = LocalStageDispatcher(registry={})
    with pytest.raises(UnknownStageError) as exc_info:
        await dispatcher.run_stage("missing", "page-1")
    assert "missing" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_stage_dispatches_to_registered_impl(monkeypatch):
    monkeypatch.setenv("PD_GPU_BACKEND", "cpu")

    async def fake(page_id, device, **kwargs):
        return {"foo": "bar"}

    registry = {("ocr", "cpu"): fake}
    dispatcher = LocalStageDispatcher(registry=registry)
    result = await dispatcher.run_stage("ocr", "page-1")
    assert isinstance(result, StageResult)
    assert result.stage_id == "ocr"
    assert result.page_id == "page-1"
    assert result.device == "cpu"
    assert result.duration_ms >= 0
    assert result.metadata == {"foo": "bar"}


@pytest.mark.asyncio
async def test_run_stage_falls_through_to_cpu_when_local_missing(monkeypatch):
    monkeypatch.setenv("PD_GPU_BACKEND", "local")

    async def cpu_impl(page_id, device, **kwargs):
        return {}

    registry = {("ocr", "cpu"): cpu_impl}
    dispatcher = LocalStageDispatcher(registry=registry)
    # Should fall through from "local" to "cpu"
    result = await dispatcher.run_stage("ocr", "page-1")
    assert result.device == "cpu"


@pytest.mark.asyncio
async def test_run_stage_propagates_kwargs(monkeypatch):
    monkeypatch.setenv("PD_GPU_BACKEND", "cpu")
    received_kwargs = {}

    async def impl(page_id, device, **kwargs):
        received_kwargs.update(kwargs)
        return {}

    registry = {("ocr", "cpu"): impl}
    dispatcher = LocalStageDispatcher(registry=registry)
    await dispatcher.run_stage("ocr", "page-1", threshold=0.9, mode="fast")
    assert received_kwargs == {"threshold": 0.9, "mode": "fast"}
