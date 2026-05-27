"""SiblingLaunchAdapter Protocol + LocalSpawnLauncher implementation."""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, runtime_checkable

import httpx
from pydantic import BaseModel, Field
from typing_extensions import Protocol

if TYPE_CHECKING:
    from pdomain_ops.suite.types import InstalledApp

    pass


class LaunchResultOpened(BaseModel):
    """Sibling launched (or already running); url is ready to use."""

    kind: Literal["opened"] = "opened"
    url: str
    spawned: bool
    pid: int | None = None


class LaunchResultRequiresHostConfig(BaseModel):
    """Hosted-mode launch requires user to configure a remote host first."""

    kind: Literal["requires-host-config"] = "requires-host-config"
    sibling_id: str


LaunchResult = Annotated[
    LaunchResultOpened | LaunchResultRequiresHostConfig,
    Field(discriminator="kind"),
]


class LaunchTimeoutError(TimeoutError):
    """Raised when a spawned app doesn't become healthy within timeout_s."""


@runtime_checkable
class SiblingLaunchAdapter(Protocol):
    """Protocol for sibling app launcher implementations."""

    async def launch(self, app: InstalledApp) -> LaunchResult:
        """Spawn the sibling app if not running; return its URL."""
        ...


# Allowlist of env vars forwarded to spawned siblings
_ENV_ALLOWLIST_PREFIXES = ("PATH", "HOME", "USER", "PD_SUITE_", "PYTHONPATH")


class LocalSpawnLauncher:
    """Local-mode launcher: spawn sibling binary + poll /healthz."""

    def __init__(self, timeout_s: float = 30.0, poll_interval_s: float = 0.1) -> None:
        self._timeout_s = timeout_s
        self._poll_interval_s = poll_interval_s

    def _build_env(self) -> dict[str, str]:
        """Build allowlist-filtered env dict for spawned process."""
        return {
            key: val
            for key, val in os.environ.items()
            if any(key == prefix or key.startswith(prefix) for prefix in _ENV_ALLOWLIST_PREFIXES)
        }

    async def launch(self, app: InstalledApp) -> LaunchResult:
        """Spawn sibling if not already running, poll until healthy."""
        url = f"http://localhost:{app.default_port}"
        healthz_url = f"{url}/healthz"

        # Check if already running
        try:
            async with httpx.AsyncClient(timeout=0.5) as client:
                resp = await client.get(healthz_url)
                if resp.status_code == 200:
                    return LaunchResultOpened(url=url, spawned=False, pid=None)
        except Exception:
            pass

        # Spawn the process
        env = self._build_env()
        cmd = [app.binary, "--port", str(app.default_port)]
        proc = subprocess.Popen(cmd, env=env, cwd=Path.home())

        # Poll until healthy or timeout
        deadline = time.monotonic() + self._timeout_s
        while time.monotonic() < deadline:
            await asyncio.sleep(self._poll_interval_s)
            try:
                async with httpx.AsyncClient(timeout=0.5) as client:
                    resp = await client.get(healthz_url)
                    if resp.status_code == 200:
                        return LaunchResultOpened(url=url, spawned=True, pid=proc.pid)
            except Exception:
                pass

        # Timed out — do NOT terminate (user may still want to inspect)
        # See docstring: subprocess termination on timeout is a Phase 4 concern.
        raise LaunchTimeoutError(
            f"App {app.app_id!r} did not become healthy within {self._timeout_s}s "
            f"(binary: {app.binary}, port: {app.default_port})"
        )
