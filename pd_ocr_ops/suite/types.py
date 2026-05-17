"""Core Pydantic models for suite plumbing."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


class SuiteApp(BaseModel):
    """Catalog entry for a known suite application."""

    model_config = ConfigDict(extra="forbid")

    app_id: str
    display_name: str
    package: str
    default_port: int
    icon: str
    description: str | None = None
    binary_name: str | None = None


class InstalledApp(BaseModel):
    """Registry row representing a concrete installation of a SuiteApp."""

    model_config = ConfigDict(extra="forbid")

    app_id: str
    package: str
    version: str
    binary: str
    default_port: int
    icon: str
    display_name: str
    enabled: bool = True
    registered_at: datetime = None  # type: ignore[assignment]

    def model_post_init(self, __context: Any) -> None:
        if self.registered_at is None:
            object.__setattr__(self, "registered_at", datetime.now(timezone.utc))

    @field_validator("binary")
    @classmethod
    def binary_must_be_absolute(cls, v: str) -> str:
        if not v.startswith("/"):
            raise ValueError(f"binary path must be absolute, got: {v!r}")
        return v


_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _validate_hex(v: str) -> str:
    if not _HEX_COLOR_RE.match(v):
        raise ValueError(f"Expected #RRGGBB hex color, got: {v!r}")
    return v


class LayerColors(BaseModel):
    """Hex color assignments for OCR layer overlays."""

    model_config = ConfigDict(extra="forbid")

    word: str = "#4a9eff"
    line: str = "#ff9f4a"
    para: str = "#4aff9f"
    block: str = "#ff4a9f"

    @field_validator("word", "line", "para", "block")
    @classmethod
    def validate_hex(cls, v: str) -> str:
        return _validate_hex(v)


class CommonUIPrefs(BaseModel):
    """Cross-app common UI preferences."""

    model_config = ConfigDict(extra="ignore")

    theme: str = "dark"
    density: str = "normal"
    accent: str = "#d6925a"
    font_size_base: int = 12
    layer_colors: LayerColors = None  # type: ignore[assignment]

    def model_post_init(self, __context: Any) -> None:
        if self.layer_colors is None:
            object.__setattr__(self, "layer_colors", LayerColors())

    @field_validator("theme")
    @classmethod
    def validate_theme(cls, v: str) -> str:
        allowed = {"light", "dark"}
        if v not in allowed:
            raise ValueError(f"theme must be one of {allowed}, got: {v!r}")
        return v

    @field_validator("density")
    @classmethod
    def validate_density(cls, v: str) -> str:
        allowed = {"compact", "normal", "comfortable"}
        if v not in allowed:
            raise ValueError(f"density must be one of {allowed}, got: {v!r}")
        return v

    @field_validator("accent")
    @classmethod
    def validate_accent(cls, v: str) -> str:
        return _validate_hex(v)

    @field_validator("font_size_base")
    @classmethod
    def validate_font_size(cls, v: int) -> int:
        if not (8 <= v <= 24):
            raise ValueError(f"font_size_base must be between 8 and 24, got: {v}")
        return v


class UIPrefs(BaseModel):
    """Shared cross-app UI preferences."""

    model_config = ConfigDict(extra="forbid")

    common: CommonUIPrefs = None  # type: ignore[assignment]
    apps: dict[str, dict[str, Any]] = {}

    def model_post_init(self, __context: Any) -> None:
        if self.common is None:
            object.__setattr__(self, "common", CommonUIPrefs())


# --- Adapter Protocol stubs (filled in later milestones) ---
# These are minimal stubs so SuiteAdapters can reference them.


class SuiteAdapters(BaseModel):
    """Bundle of suite adapters passed to mount_routes()."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    registry: Any = None
    prefs: Any = None
    launcher: Any = None
    auth: Any = None
    storage: Any = None

    @classmethod
    def local(cls) -> "SuiteAdapters":
        """Return a fully-wired bundle of local-mode adapters."""
        from pd_ocr_ops.suite import paths
        from pd_ocr_ops.suite.auth import NoAuthAdapter
        from pd_ocr_ops.suite.prefs import LocalFilePrefs
        from pd_ocr_ops.suite.registry import LocalTomlSuiteRegistry
        from pd_ocr_ops.suite.sibling_spawn import LocalSpawnLauncher
        from pd_ocr_ops.suite.storage import LocalFsStorage

        return cls(
            registry=LocalTomlSuiteRegistry(),
            prefs=LocalFilePrefs(),
            launcher=LocalSpawnLauncher(),
            auth=NoAuthAdapter(),
            storage=LocalFsStorage(root=paths.suite_data_dir() / "storage"),
        )

    @classmethod
    def from_env(cls) -> "SuiteAdapters":
        raise NotImplementedError(
            "hosted-mode adapters land in Phase 4; from_env() is not yet implemented"
        )
