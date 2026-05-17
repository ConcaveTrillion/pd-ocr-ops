"""GPU adapter package: protocols, device detection, local implementations."""

from pd_ocr_ops.gpu.device import pick_device
from pd_ocr_ops.gpu.protocols import LongJobRunner, StageDispatcher

__all__ = [
    "StageDispatcher",
    "LongJobRunner",
    "pick_device",
]
