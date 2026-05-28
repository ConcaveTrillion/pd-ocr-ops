from pdomain_ops.gpu.protocols import StageDispatcher
from pdomain_ops.gpu.types import OcrBatchRequest


def test_protocol_methods_present():
    assert hasattr(StageDispatcher, "run_stage")


def test_protocol_has_run_ocr_batch():
    """StageDispatcher Protocol must declare run_ocr_batch for batched OCR."""
    assert hasattr(StageDispatcher, "run_ocr_batch")


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

        async def run_ocr_batch(self, req: OcrBatchRequest) -> list[dict]:
            return []

    assert isinstance(FakeDispatcher(), StageDispatcher)


def test_ocr_batch_request_carries_image_bytes():
    """OcrBatchRequest must carry images as bytes, not paths."""
    img_a = b"PNG\x00fake-image-a"
    img_b = b"PNG\x00fake-image-b"
    req = OcrBatchRequest(
        images=[img_a, img_b],
        source_identifiers=["page-0", "page-1"],
    )
    assert len(req.images) == 2
    assert req.images[0] == img_a
    assert req.images[1] == img_b
    assert req.source_identifiers == ["page-0", "page-1"]


def test_ocr_batch_request_optional_fields_have_defaults():
    """engine and language are optional in OcrBatchRequest."""
    req = OcrBatchRequest(images=[b"x"], source_identifiers=["p0"])
    assert req.engine == "doctr"
    assert req.language == "eng"
