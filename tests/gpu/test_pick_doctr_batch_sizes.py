"""Tests for pick_doctr_batch_sizes — VRAM/CPU detection-batch sizing helper."""

from __future__ import annotations

import pdomain_ops.gpu.device as device_mod
from pdomain_ops.gpu.device import pick_doctr_batch_sizes

# ---------------------------------------------------------------------------
# CPU fallback
# ---------------------------------------------------------------------------


def test_cpu_det_bs_in_range(monkeypatch):
    """CPU path: det_bs should be 1 or 2 (conservative, no VRAM to exploit)."""
    det_bs, _reco_bs = pick_doctr_batch_sizes("cpu", chunk_pages=8)
    assert 1 <= det_bs <= 2


def test_cpu_reco_bs_default(monkeypatch):
    """CPU path: reco_bs should be exactly 128 (DocTR default)."""
    _det_bs, reco_bs = pick_doctr_batch_sizes("cpu", chunk_pages=8)
    assert reco_bs == 128


def test_cpu_both_ints_at_least_one():
    """Both returned values are ints >= 1."""
    det_bs, reco_bs = pick_doctr_batch_sizes("cpu", chunk_pages=1)
    assert isinstance(det_bs, int)
    assert isinstance(reco_bs, int)
    assert det_bs >= 1
    assert reco_bs >= 1


# ---------------------------------------------------------------------------
# GPU — plenty of VRAM (6 GB free)
# ---------------------------------------------------------------------------

_6_GB = 6 * 1024**3


def test_gpu_ample_vram_det_bs_scales_up(monkeypatch):
    """With 6 GB free VRAM, det_bs should be >= 4 (at ~1.2 GB/page that's 5)."""
    monkeypatch.setattr(device_mod, "_cuda_free_bytes", lambda: _6_GB)
    det_bs, _reco_bs = pick_doctr_batch_sizes("local", chunk_pages=8)
    assert det_bs >= 4


def test_gpu_ample_vram_reco_bs_scales_with_chunk_pages(monkeypatch):
    """With chunk_pages=8 and ample VRAM, reco_bs should exceed 128 (default)."""
    monkeypatch.setattr(device_mod, "_cuda_free_bytes", lambda: _6_GB)
    _det_bs, reco_bs = pick_doctr_batch_sizes("local", chunk_pages=8)
    assert reco_bs > 128


def test_gpu_ample_vram_all_ints_at_least_one(monkeypatch):
    """GPU path with plenty of VRAM: both values are ints >= 1."""
    monkeypatch.setattr(device_mod, "_cuda_free_bytes", lambda: _6_GB)
    det_bs, reco_bs = pick_doctr_batch_sizes("local", chunk_pages=8)
    assert isinstance(det_bs, int)
    assert isinstance(reco_bs, int)
    assert det_bs >= 1
    assert reco_bs >= 1


# ---------------------------------------------------------------------------
# GPU — tiny VRAM (512 MB free, well below one page's working set)
# ---------------------------------------------------------------------------

_512_MB = 512 * 1024 * 1024


def test_gpu_tiny_vram_det_bs_clamped_to_one(monkeypatch):
    """With only 512 MB free, det_bs must clamp to the floor of 1 (never 0)."""
    monkeypatch.setattr(device_mod, "_cuda_free_bytes", lambda: _512_MB)
    det_bs, _reco_bs = pick_doctr_batch_sizes("local", chunk_pages=8)
    assert det_bs == 1


def test_gpu_tiny_vram_det_bs_never_zero(monkeypatch):
    """det_bs must never be 0 regardless of how little VRAM is reported."""
    monkeypatch.setattr(device_mod, "_cuda_free_bytes", lambda: 1)  # 1 byte
    det_bs, _reco_bs = pick_doctr_batch_sizes("local", chunk_pages=1)
    assert det_bs >= 1


def test_gpu_tiny_vram_reco_bs_still_at_least_one(monkeypatch):
    """reco_bs must never drop below 1 even under extreme VRAM pressure."""
    monkeypatch.setattr(device_mod, "_cuda_free_bytes", lambda: _512_MB)
    _det_bs, reco_bs = pick_doctr_batch_sizes("local", chunk_pages=1)
    assert reco_bs >= 1


# ---------------------------------------------------------------------------
# GPU — cuda_free_bytes returns None (torch unavailable / error)
# ---------------------------------------------------------------------------


def test_gpu_no_vram_info_falls_back_conservative(monkeypatch):
    """When _cuda_free_bytes returns None, should return conservative (2, 128)."""
    monkeypatch.setattr(device_mod, "_cuda_free_bytes", lambda: None)
    det_bs, reco_bs = pick_doctr_batch_sizes("local", chunk_pages=8)
    assert det_bs == 2
    assert reco_bs == 128


# ---------------------------------------------------------------------------
# Device default (picks from pick_device when device=None)
# ---------------------------------------------------------------------------


def test_device_none_uses_pick_device(monkeypatch):
    """When device=None, uses pick_device() to resolve — cpu path smoke-test."""
    monkeypatch.setattr(device_mod, "_cuda_available", lambda: False)
    monkeypatch.setattr(device_mod, "_mps_available", lambda: False)
    det_bs, reco_bs = pick_doctr_batch_sizes(None, chunk_pages=4)
    assert det_bs >= 1
    assert reco_bs >= 1


# ---------------------------------------------------------------------------
# reco_bs ceiling
# ---------------------------------------------------------------------------


def test_gpu_reco_bs_does_not_exceed_ceiling(monkeypatch):
    """reco_bs is capped by _RECO_CEILING (512) regardless of chunk_pages."""
    monkeypatch.setattr(device_mod, "_cuda_free_bytes", lambda: _6_GB)
    _det_bs, reco_bs = pick_doctr_batch_sizes("local", chunk_pages=1000)
    assert reco_bs <= 512
