"""GPU adapter package: protocols, device detection, local implementations."""

from pd_ocr_ops.gpu.default_stages import register_default_stages
from pd_ocr_ops.gpu.device import pick_device
from pd_ocr_ops.gpu.local_stage import LocalStageDispatcher
from pd_ocr_ops.gpu.protocols import LongJobRunner, StageDispatcher

__all__ = [
    "LocalStageDispatcher",
    "LongJobRunner",
    "StageDispatcher",
    "pick_device",
    "register_default_stages",
]
