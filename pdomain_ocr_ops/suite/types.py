"""Core Pydantic models for suite plumbing."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    description: str | None = None
    enabled: bool = True
    registered_at: datetime = None  # pyright: ignore[reportAssignmentType]  # Pydantic deferred default via model_post_init

    def model_post_init(self, __context: Any) -> None:
        """Set registered_at to now if not provided at construction time."""
        if self.registered_at is None:
            object.__setattr__(self, "registered_at", datetime.now(UTC))

    @field_validator("binary")
    @classmethod
    def binary_must_be_absolute(cls, v: str) -> str:
        """Validate that binary path is absolute."""
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
        """Validate each color value is a #RRGGBB hex string."""
        return _validate_hex(v)


class CommonUIPrefs(BaseModel):
    """Cross-app common UI preferences."""

    model_config = ConfigDict(extra="ignore")

    theme: str = "dark"
    density: str = "normal"
    accent: str = "#d6925a"
    font_size_base: int = 12
    font_scale: Annotated[float, Field(ge=0.8, le=1.4)] = 1.0
    layer_colors: LayerColors = None  # pyright: ignore[reportAssignmentType]  # Pydantic deferred default via model_post_init

    def model_post_init(self, __context: Any) -> None:
        """Set layer_colors to defaults if not provided at construction time."""
        if self.layer_colors is None:
            object.__setattr__(self, "layer_colors", LayerColors())

    @field_validator("theme")
    @classmethod
    def validate_theme(cls, v: str) -> str:
        """Validate theme is one of the allowed values."""
        allowed = {"light", "dark"}
        if v not in allowed:
            raise ValueError(f"theme must be one of {allowed}, got: {v!r}")
        return v

    @field_validator("density")
    @classmethod
    def validate_density(cls, v: str) -> str:
        """Validate density is one of the allowed values."""
        allowed = {"compact", "normal", "comfortable"}
        if v not in allowed:
            raise ValueError(f"density must be one of {allowed}, got: {v!r}")
        return v

    @field_validator("accent")
    @classmethod
    def validate_accent(cls, v: str) -> str:
        """Validate accent is a #RRGGBB hex color."""
        return _validate_hex(v)

    @field_validator("font_size_base")
    @classmethod
    def validate_font_size(cls, v: int) -> int:
        """Validate font_size_base is in the allowed range [8, 24]."""
        if not (8 <= v <= 24):
            raise ValueError(f"font_size_base must be between 8 and 24, got: {v}")
        return v


class UIPrefs(BaseModel):
    """Shared cross-app UI preferences."""

    model_config = ConfigDict(extra="forbid")

    common: CommonUIPrefs = None  # pyright: ignore[reportAssignmentType]  # Pydantic deferred default via model_post_init
    apps: dict[str, dict[str, Any]] = {}

    def model_post_init(self, __context: Any) -> None:
        """Set common to defaults if not provided at construction time."""
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
    def local(cls) -> SuiteAdapters:
        """Return a fully-wired bundle of local-mode adapters."""
        from pdomain_ocr_ops.suite import paths
        from pdomain_ocr_ops.suite.auth import NoAuthAdapter
        from pdomain_ocr_ops.suite.prefs import LocalFilePrefs
        from pdomain_ocr_ops.suite.registry import LocalTomlSuiteRegistry
        from pdomain_ocr_ops.suite.sibling_spawn import LocalSpawnLauncher
        from pdomain_ocr_ops.suite.storage import LocalFsStorage

        return cls(
            registry=LocalTomlSuiteRegistry(),
            prefs=LocalFilePrefs(),
            launcher=LocalSpawnLauncher(),
            auth=NoAuthAdapter(),
            storage=LocalFsStorage(root=paths.suite_data_dir() / "storage"),
        )

    @classmethod
    def from_env(cls) -> SuiteAdapters:
        """Return hosted-mode adapters configured from environment variables.

        Not yet implemented — see Phase 4 roadmap.
        """
        raise NotImplementedError(
            "hosted-mode adapters land in Phase 4; from_env() is not yet implemented"
        )
