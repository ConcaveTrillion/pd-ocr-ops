"""pdomain-ops: Suite plumbing and GPU dispatch adapters for the pd-* suite."""

from importlib.metadata import version

__version__ = version("pdomain-ops")

from pdomain_ops.suite.routes import mount_routes
from pdomain_ops.suite.types import SuiteAdapters

__all__ = [
    "SuiteAdapters",
    "__version__",
    "mount_routes",
]
