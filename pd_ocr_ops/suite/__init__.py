"""Suite plumbing: registry, prefs, launcher, auth, storage, routes."""

from pd_ocr_ops.suite.types import (
    CommonUIPrefs,
    InstalledApp,
    LayerColors,
    SuiteAdapters,
    SuiteApp,
    UIPrefs,
)

__all__ = [
    "SuiteApp",
    "InstalledApp",
    "LayerColors",
    "CommonUIPrefs",
    "UIPrefs",
    "SuiteAdapters",
]
