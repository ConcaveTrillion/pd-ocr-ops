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
    logical so torch's intra-op threads don't oversubscribe hyperthreads."""
    try:
        import psutil  # pyright: ignore[reportMissingImports]  # optional dep

        cores = psutil.cpu_count(logical=False)
        if cores:
            return int(cores)
    except Exception:  # noqa: BLE001 - best-effort; fall through to os
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
    except Exception:  # noqa: BLE001 - best-effort detection
        return None


# Heuristic VRAM budget for one DocTR predictor's working set (detection +
# recognition + activations). Conservative; tune against real OOM behaviour.
_VRAM_PER_WORKER_BYTES = 2_500_000_000


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
