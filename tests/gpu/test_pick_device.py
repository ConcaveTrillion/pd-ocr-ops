
import pytest

from pd_ocr_ops.gpu.device import pick_device


def test_picks_local_when_pd_gpu_backend_local(monkeypatch):
    monkeypatch.setenv("PD_GPU_BACKEND", "local")
    monkeypatch.delenv("PGDP_GPU_BACKEND", raising=False)
    assert pick_device() == "local"


def test_picks_mps_when_env_mps(monkeypatch):
    monkeypatch.setenv("PD_GPU_BACKEND", "mps")
    monkeypatch.delenv("PGDP_GPU_BACKEND", raising=False)
    assert pick_device() == "mps"


def test_picks_cpu_when_env_cpu(monkeypatch):
    monkeypatch.setenv("PD_GPU_BACKEND", "cpu")
    monkeypatch.delenv("PGDP_GPU_BACKEND", raising=False)
    assert pick_device() == "cpu"


def test_env_unset_falls_back_to_detection_cuda(monkeypatch):
    monkeypatch.delenv("PD_GPU_BACKEND", raising=False)
    monkeypatch.delenv("PGDP_GPU_BACKEND", raising=False)
    monkeypatch.setattr("pd_ocr_ops.gpu.device._cuda_available", lambda: True)
    monkeypatch.setattr("pd_ocr_ops.gpu.device._mps_available", lambda: False)
    assert pick_device() == "local"


def test_env_unset_falls_back_to_detection_mps(monkeypatch):
    monkeypatch.delenv("PD_GPU_BACKEND", raising=False)
    monkeypatch.delenv("PGDP_GPU_BACKEND", raising=False)
    monkeypatch.setattr("pd_ocr_ops.gpu.device._cuda_available", lambda: False)
    monkeypatch.setattr("pd_ocr_ops.gpu.device._mps_available", lambda: True)
    assert pick_device() == "mps"


def test_env_unset_falls_back_to_cpu(monkeypatch):
    monkeypatch.delenv("PD_GPU_BACKEND", raising=False)
    monkeypatch.delenv("PGDP_GPU_BACKEND", raising=False)
    monkeypatch.setattr("pd_ocr_ops.gpu.device._cuda_available", lambda: False)
    monkeypatch.setattr("pd_ocr_ops.gpu.device._mps_available", lambda: False)
    assert pick_device() == "cpu"


def test_unknown_env_value_raises(monkeypatch):
    monkeypatch.setenv("PD_GPU_BACKEND", "jupiter")
    monkeypatch.delenv("PGDP_GPU_BACKEND", raising=False)
    with pytest.raises(ValueError) as exc_info:
        pick_device()
    assert "jupiter" in str(exc_info.value)


def test_pgdp_env_var_alias_warns(monkeypatch, recwarn):
    monkeypatch.delenv("PD_GPU_BACKEND", raising=False)
    monkeypatch.setenv("PGDP_GPU_BACKEND", "mps")
    result = pick_device()
    assert result == "mps"
    warning_messages = [str(w.message) for w in recwarn.list]
    assert any("PGDP_GPU_BACKEND" in msg or "deprecated" in msg.lower() for msg in warning_messages)
