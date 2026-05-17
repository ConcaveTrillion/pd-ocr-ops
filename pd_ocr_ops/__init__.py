"""pd-ocr-ops: Suite plumbing and GPU dispatch adapters for the pd-* OCR suite."""

__version__ = "0.1.0"

from pd_ocr_ops.suite.routes import mount_routes
from pd_ocr_ops.suite.types import SuiteAdapters

__all__ = [
    "__version__",
    "mount_routes",
    "SuiteAdapters",
]
