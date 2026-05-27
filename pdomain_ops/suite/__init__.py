"""Suite plumbing: registry, prefs, launcher, auth, storage, routes."""

from pdomain_ops.suite.bootstrap import bootstrap_spa
from pdomain_ops.suite.ports import find_available_port
from pdomain_ops.suite.register_self import register_self
from pdomain_ops.suite.types import (
    CommonUIPrefs,
    InstalledApp,
    LayerColors,
    SuiteAdapters,
    SuiteApp,
    UIPrefs,
)

__all__ = [
    "CommonUIPrefs",
    "InstalledApp",
    "LayerColors",
    "SuiteAdapters",
    "SuiteApp",
    "UIPrefs",
    "bootstrap_spa",
    "find_available_port",
    "register_self",
]
