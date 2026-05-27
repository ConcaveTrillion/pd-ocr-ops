"""FastAPI router for suite plumbing endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response

from pdomain_ops.suite.types import CommonUIPrefs, InstalledApp, SuiteAdapters

_ALLOWED_ICON_SIZES = {1024, 512, 256, 128, 64, 32, 16}


def mount_routes(
    app: FastAPI,
    adapters: SuiteAdapters | None = None,
    *,
    suite_app: InstalledApp | None = None,
) -> None:
    """Mount suite routes under /api/suite/* and /healthz onto a FastAPI app.

    Parameters
    ----------
    app:
        The FastAPI application to mount routes onto.
    adapters:
        Suite adapter bundle.  Defaults to ``SuiteAdapters.local()``.
    suite_app:
        The ``InstalledApp`` representing this process.  Stored on
        ``app.state.suite_app`` and served from ``GET /healthz``.
        When ``None``, ``/healthz`` returns ``"unknown"`` placeholders.
    """
    if adapters is None:
        adapters = SuiteAdapters.local()

    # Record the process start time and the suite_app metadata for /healthz.
    _start_time = time.monotonic()
    app.state.suite_app = suite_app

    router = APIRouter(prefix="/api/suite", tags=["suite"])

    @router.get("/installed")
    async def get_installed() -> list[dict[str, object]]:
        apps = adapters.registry.list_installed()
        return [a.model_dump(mode="json") for a in apps]

    @router.post("/launch")
    async def launch_app(app_id: str) -> dict[str, object]:
        installed_apps = adapters.registry.list_installed()
        found = next((a for a in installed_apps if a.app_id == app_id), None)
        if found is None:
            raise HTTPException(status_code=404, detail=f"unknown app: {app_id}")
        if not found.enabled:
            raise HTTPException(status_code=409, detail=f"app is disabled: {app_id}")
        result = await adapters.launcher.launch(found)
        return result.model_dump(mode="json")

    @router.get("/prefs")
    async def get_prefs() -> dict[str, object]:
        prefs = adapters.prefs.read()
        return prefs.model_dump(mode="json")

    @router.put("/prefs/common", status_code=204)
    async def put_prefs_common(common: CommonUIPrefs) -> Response:
        adapters.prefs.write_common(common)
        return Response(status_code=204)

    @router.put("/prefs/apps/{app_id}", status_code=204)
    async def put_prefs_app(app_id: str, request: Request) -> Response:
        payload = await request.json()
        adapters.prefs.write_app(app_id, payload)
        return Response(status_code=204)

    app.include_router(router)

    # Icon serving under /api/icons/{size}
    icons_router = APIRouter(prefix="/api/icons", tags=["icons"])

    @icons_router.get("/{size}")
    async def get_icon(size: int, app_id: str) -> Response:
        if size not in _ALLOWED_ICON_SIZES:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported size: {size}. Allowed: {sorted(_ALLOWED_ICON_SIZES)}",
            )
        installed_apps = adapters.registry.list_installed()
        found = next((a for a in installed_apps if a.app_id == app_id), None)
        if found is None:
            raise HTTPException(status_code=404, detail=f"unknown app: {app_id}")
        from pathlib import Path

        binary_dir = Path(found.binary).parent.parent
        icon_path = binary_dir / "icons" / f"{size}x{size}.png"
        if not icon_path.exists():
            raise HTTPException(status_code=404, detail=f"icon not found for app: {app_id}")
        return Response(content=icon_path.read_bytes(), media_type="image/png")

    app.include_router(icons_router)

    # Centralized health endpoint — mounted at /healthz (not under /api/suite/).
    # No auth required; LocalSpawnLauncher polls this to detect readiness.
    @app.get("/healthz", tags=["health"])
    async def healthz() -> dict[str, object]:
        current: InstalledApp | None = app.state.suite_app
        return {
            "status": "ok",
            "app_id": current.app_id if current is not None else "unknown",
            "version": current.version if current is not None else "unknown",
            "uptime_s": time.monotonic() - _start_time,
        }
