"""GPU device detection helper."""

from __future__ import annotations

import os
import warnings
from typing import Literal

_VALID_DEVICES = frozenset({"local", "mps", "cpu"})


def _cuda_available() -> bool:
    """Return True if a CUDA-capable GPU is accessible.

    Probes cupy first (the [gpu] extra's marker dep). Falls back to
    torch.cuda — torch is the runtime that actually executes OCR via
    DocTR, so a torch-visible CUDA device is sufficient even without
    cupy installed.
    """
    try:
        import cupy  # pyright: ignore[reportMissingImports]  # optional GPU dep; installed via [gpu] extra

        if cupy.cuda.runtime.getDeviceCount() > 0:
            return True
    except Exception:
        pass
    try:
        import torch  # pyright: ignore[reportMissingImports]  # optional GPU dep; installed via [gpu] extra

        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


def _mps_available() -> bool:
    """Return True if Apple MPS (Metal Performance Shaders) is available."""
    try:
        import torch  # pyright: ignore[reportMissingImports]  # optional GPU dep; installed via [gpu] extra

        return torch.backends.mps.is_available()
    except Exception:
        return False


def pick_device() -> Literal["local", "mps", "cpu"]:
    """Return the GPU device to use for this process.

    Resolution order:
    1. PDOMAIN_GPU_BACKEND env var (explicit override)
    2. PGDP_GPU_BACKEND env var (deprecated alias; warns)
    3. Auto-detection: CUDA -> MPS -> CPU
    """
    # Check canonical env var
    explicit = os.environ.get("PDOMAIN_GPU_BACKEND")
    if explicit:
        if explicit not in _VALID_DEVICES:
            raise ValueError(
                f"PDOMAIN_GPU_BACKEND={explicit!r} is not a valid device. "
                f"Allowed: {sorted(_VALID_DEVICES)}"
            )
        return explicit  # pyright: ignore[reportReturnType]  # narrowed by _VALID_DEVICES check above

    # Check deprecated alias
    legacy = os.environ.get("PGDP_GPU_BACKEND")
    if legacy:
        warnings.warn(
            f"PGDP_GPU_BACKEND is deprecated; use PDOMAIN_GPU_BACKEND={legacy!r} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if legacy not in _VALID_DEVICES:
            raise ValueError(
                f"PGDP_GPU_BACKEND={legacy!r} is not a valid device. "
                f"Allowed: {sorted(_VALID_DEVICES)}"
            )
        return legacy  # pyright: ignore[reportReturnType]  # narrowed by _VALID_DEVICES check above

    # Auto-detect
    if _cuda_available():
        return "local"
    if _mps_available():
        return "mps"
    return "cpu"


def _physical_cores() -> int:
    """Physical CPU core count (falls back to 1). Prefers physical over
    logical so torch's intra-op threads don't oversubscribe hyperthreads.
    """
    try:
        import psutil  # pyright: ignore[reportMissingImports]  # optional dep

        cores = psutil.cpu_count(logical=False)
        if cores:
            return int(cores)
    except Exception:
        pass
    return os.cpu_count() or 1


def _cuda_free_bytes() -> int | None:
    """Free VRAM in bytes on the active CUDA device, or None if unavailable."""
    try:
        import torch  # pyright: ignore[reportMissingImports]  # optional GPU dep

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    except Exception:
        return None


# Heuristic VRAM budget for one DocTR predictor's working set (detection +
# recognition + activations). Conservative; tune against real OOM behaviour.
_VRAM_PER_WORKER_BYTES = 2_500_000_000

# ---------------------------------------------------------------------------
# DocTR batch-size sizing constants
# ---------------------------------------------------------------------------

# Conservative estimate of peak VRAM consumed by the detection CNN while
# processing one full-resolution page tensor (backbone + activations).
# Real peak is image-size dependent; 1.2 GB covers A4 at typical scan DPI.
# Tune downward if OOM is observed in production; tune upward to allow larger
# batches on high-VRAM cards.
_VRAM_PER_PAGE_BYTES: int = 1_200_000_000

# Estimated average number of text crops produced per page by the DocTR
# detection stage.  Used to scale reco_bs relative to the chunk size so the
# recognition stage can process all crops from a chunk in a single forward
# pass when VRAM allows.
_CROPS_PER_PAGE_EST: int = 48

# Hard ceiling for reco_bs.  Recognition crops are small tensors, but GPU
# memory is still finite.  512 is safe for 8-24 GB VRAM cards.
_RECO_CEILING: int = 512


def pick_doctr_batch_sizes(
    device: str | None = None,
    chunk_pages: int = 8,
) -> tuple[int, int]:
    """Recommend (det_bs, reco_bs) for a DocTR predictor on *device*.

    Parameters
    ----------
    device:
        One of ``"local"`` (CUDA), ``"mps"``, ``"cpu"``, or ``None`` to
        auto-detect via :func:`pick_device`.
    chunk_pages:
        Number of pages in the batch being dispatched.  Used to scale
        ``reco_bs`` so the recognition stage can process all crops from the
        chunk in a single forward pass when VRAM allows.

    Returns:
    -------
    (det_bs, reco_bs):
        Both values are ints >= 1.

        * ``det_bs`` — detection batch size.  Each entry is a full-resolution
          page tensor; VRAM constrains this tightly.  DocTR default is 2.
        * ``reco_bs`` — recognition batch size.  Crops are small so this is
          bounded by crop supply (approx pages x ``_CROPS_PER_PAGE_EST``), not
          VRAM.  DocTR default is 128.

    GPU path
    --------
    Queries free VRAM via :func:`_cuda_free_bytes`.  If unavailable, returns
    the conservative fallback ``(2, 128)``.  Otherwise::

        det_bs  = clamp(1, 8, free_bytes // _VRAM_PER_PAGE_BYTES)
        reco_bs = clamp(128, _RECO_CEILING, chunk_pages * _CROPS_PER_PAGE_EST)

    CPU path
    --------
    Returns ``(1, 128)``.  torch is internally multi-threaded so a det_bs of
    1 avoids memory pressure while still benefiting from intra-op parallelism.
    """
    resolved = device or pick_device()

    if resolved == "cpu":
        return (1, 128)

    # GPU / MPS path
    free = _cuda_free_bytes()
    if free is None:
        # torch unavailable or probe failed — conservative DocTR defaults
        return (2, 128)

    det_bs = max(1, min(8, free // _VRAM_PER_PAGE_BYTES))
    reco_bs = min(_RECO_CEILING, max(128, chunk_pages * _CROPS_PER_PAGE_EST))
    return (det_bs, reco_bs)


def pick_concurrency(device: str | None = None) -> int:
    """Recommend how many OCR pages to process in parallel for *device*.

    - CPU: physical_cores // 4 (>=1). torch is already internally
      multi-threaded, so a small worker count avoids core oversubscription
      and the sustained-load thermal spikes seen on hybrid CPUs.
    - GPU: a single shared predictor serialises on one CUDA context, so
      page-level concurrency is 1. (Throughput comes from batching, not
      concurrent calls.) Returned for completeness / future batch sizing.
    """
    device = device or pick_device()
    if device == "cpu":
        return max(1, _physical_cores() // 4)
    return 1
