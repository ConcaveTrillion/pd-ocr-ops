"""GPU adapter package: protocols, device detection, local implementations."""

from pdomain_ops.gpu.default_stages import register_default_stages
from pdomain_ops.gpu.device import pick_device
from pdomain_ops.gpu.local_stage import LocalStageDispatcher
from pdomain_ops.gpu.modal_dispatcher import ModalBackend, ModalStageDispatcher
from pdomain_ops.gpu.protocols import LongJobRunner, StageDispatcher
from pdomain_ops.gpu.shared_container_dispatcher import (
    SharedContainerBackend,
    SharedContainerStageDispatcher,
)
from pdomain_ops.gpu.types import (
    BatchJobItem,
    BatchJobResult,
    BatchProgressCb,
    DispatchModel,
    GPUBackend,
    OcrPageRequest,
    OcrPageResponse,
    ProcessPageRequest,
    ProcessPageResponse,
)

__all__ = [
    "BatchJobItem",
    "BatchJobResult",
    "BatchProgressCb",
    "DispatchModel",
    "GPUBackend",
    "LocalStageDispatcher",
    "LongJobRunner",
    "ModalBackend",
    "ModalStageDispatcher",
    "OcrPageRequest",
    "OcrPageResponse",
    "ProcessPageRequest",
    "ProcessPageResponse",
    "SharedContainerBackend",
    "SharedContainerStageDispatcher",
    "StageDispatcher",
    "pick_device",
    "register_default_stages",
]
