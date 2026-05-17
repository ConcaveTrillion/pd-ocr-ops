import warnings

import pytest

from pd_ocr_ops.gpu.local_stage import LocalStageDispatcher


async def _fake_impl(page_id, device, **kwargs):
    return {}


def test_register_stage_adds_entry():
    dispatcher = LocalStageDispatcher(registry={})
    dispatcher.register_stage("ocr", "cpu", _fake_impl)
    assert ("ocr", "cpu") in dispatcher._registry


def test_register_stage_replaces_existing():
    dispatcher = LocalStageDispatcher(registry={})
    dispatcher.register_stage("ocr", "cpu", _fake_impl)

    async def new_impl(page_id, device, **kwargs):
        return {"new": True}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dispatcher.register_stage("ocr", "cpu", new_impl)

    assert dispatcher._registry[("ocr", "cpu")] is new_impl
    assert any("replac" in str(warning.message).lower() for warning in w)


def test_register_stage_rejects_unknown_device():
    dispatcher = LocalStageDispatcher(registry={})
    with pytest.raises(ValueError):
        dispatcher.register_stage("ocr", "jupiter", _fake_impl)
