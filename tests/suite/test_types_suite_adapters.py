import pytest

from pd_ocr_ops.suite.auth import AuthAdapter
from pd_ocr_ops.suite.prefs import PrefsAdapter
from pd_ocr_ops.suite.registry import SuiteRegistryAdapter
from pd_ocr_ops.suite.sibling_spawn import SiblingLaunchAdapter
from pd_ocr_ops.suite.storage import StorageAdapter
from pd_ocr_ops.suite.types import SuiteAdapters


def test_suite_adapters_local_returns_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("PD_SUITE_DATA_DIR", str(tmp_path))
    bundle = SuiteAdapters.local()
    assert isinstance(bundle.registry, SuiteRegistryAdapter)
    assert isinstance(bundle.prefs, PrefsAdapter)
    assert isinstance(bundle.launcher, SiblingLaunchAdapter)
    assert isinstance(bundle.auth, AuthAdapter)
    assert isinstance(bundle.storage, StorageAdapter)
    assert bundle.registry.__class__.__name__ == "LocalTomlSuiteRegistry"


def test_local_bundle_uses_xdg_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("PD_SUITE_DATA_DIR", str(tmp_path))
    bundle = SuiteAdapters.local()
    # Registry reads from tmp_path/installed.toml
    # (no file yet, so list_installed returns [])
    assert bundle.registry.list_installed() == []


def test_suite_adapters_from_env_raises_not_implemented():
    with pytest.raises(NotImplementedError) as exc_info:
        SuiteAdapters.from_env()
    msg = str(exc_info.value).lower()
    assert "phase 4" in msg or "hosted" in msg
