"""GPU device detection helper."""

from __future__ import annotations

import os
import warnings
from typing import Literal

_VALID_DEVICES = frozenset({"local", "mps", "cpu"})


def _cuda_available() -> bool:
    """Return True if a CUDA-capable GPU is accessible."""
    try:
        import cupy  # pyright: ignore[reportMissingImports]  # optional GPU dep; installed via [gpu] extra

        return cupy.cuda.runtime.getDeviceCount() > 0
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
