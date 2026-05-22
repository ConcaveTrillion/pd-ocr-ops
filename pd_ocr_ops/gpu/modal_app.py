"""Modal app for pd-ocr-ops GPU work. Deployed via ``modal deploy src/pd_ocr_ops/gpu/modal_app.py``.

Cherry-picked-from: pd-prep-for-pgdp@e36c199df466ff45b70d2a452dd7512dcc2a17c9
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

try:
    import modal  # pyright: ignore[reportMissingImports]  # optional [modal] extra, no stubs

    _modal_available: bool = True
except ImportError:
    modal: ModuleType | None = None
    _modal_available = False

# Public flag for test introspection.
_MODAL_AVAILABLE: bool = _modal_available

if _modal_available and modal is not None:
    # modal ships no type stubs — every modal.* call below is Unknown-typed
    # and decorators are untyped. Suppressions are pyright-native and scoped.
    image = (
        modal.Image.debian_slim(python_version="3.13")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        .apt_install("libgl1", "libglib2.0-0")
        .pip_install(
            "fastapi>=0.115",
            "pydantic>=2.9",
            "huggingface_hub>=0.23",
            "transformers>=4.45",
            "torch",
            "opencv-python-headless",
            "numpy",
            "Pillow",
        )
        .pip_install_from_pyproject("pyproject.toml")
        .add_local_python_source("pd_ocr_ops")
    )
    app = modal.App("pd-ocr-ops", image=image)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    GPU_PROFILE = "T4"
    DEFAULT_TIMEOUT_S = 60 * 10

    @app.function(gpu=GPU_PROFILE, timeout=DEFAULT_TIMEOUT_S)  # pyright: ignore[reportUnknownMemberType,reportUntypedFunctionDecorator]
    def process_page(payload: dict[str, object]) -> dict[str, object]:
        """Scaffold — real S3 wiring is a separate follow-up plan."""
        del payload
        raise NotImplementedError("Modal process_page needs S3 storage wired — scaffold only")

    @app.function(gpu=GPU_PROFILE, timeout=DEFAULT_TIMEOUT_S)  # pyright: ignore[reportUnknownMemberType,reportUntypedFunctionDecorator]
    def run_ocr(payload: dict[str, object]) -> dict[str, object]:
        """Scaffold — real S3 wiring is a separate follow-up plan."""
        del payload
        raise NotImplementedError("Modal run_ocr needs S3 storage wired — scaffold only")

    @app.function(gpu=GPU_PROFILE, timeout=DEFAULT_TIMEOUT_S * 6)  # pyright: ignore[reportUnknownMemberType,reportUntypedFunctionDecorator]
    def run_batch(payloads: list[dict[str, object]]) -> list[dict[str, object]]:
        """Scaffold — real batch handler is a separate follow-up plan."""
        from pd_ocr_ops.gpu.types import BatchJobItem, BatchJobResult

        results: list[dict[str, object]] = []
        for p in payloads:
            item = BatchJobItem.model_validate(p)
            results.append(
                BatchJobResult(
                    job_type=item.job_type,
                    project_id=item.project_id,
                    idx0=item.idx0,
                    ok=False,
                    error="modal run_batch scaffold — handler not implemented",
                ).model_dump()
            )
        return results
