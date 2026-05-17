"""Emit JSON Schema for all public Pydantic models in pd_ocr_ops."""

from __future__ import annotations

import json

from pydantic import TypeAdapter

from pd_ocr_ops.gpu.types import JobEvent, JobSpec, JobStatus, StageResult
from pd_ocr_ops.suite.sibling_spawn import (
    LaunchResult,
    LaunchResultOpened,
    LaunchResultRequiresHostConfig,
)
from pd_ocr_ops.suite.types import (
    CommonUIPrefs,
    InstalledApp,
    LayerColors,
    SuiteApp,
    UIPrefs,
)

# The ordered tuple of public models to emit schemas for.
# SuiteAdapters is NOT here — its fields are Protocol classes which
# Pydantic can't JSON-schema cleanly.
PUBLIC_MODELS = (
    SuiteApp,
    InstalledApp,
    LayerColors,
    CommonUIPrefs,
    UIPrefs,
    LaunchResultOpened,
    LaunchResultRequiresHostConfig,
    StageResult,
    JobStatus,
    JobEvent,
    JobSpec,
)


def emit_schemas() -> dict[str, dict[str, object]]:
    """Emit all schemas as a dict keyed by model class name."""
    schemas: dict[str, dict[str, object]] = {}

    for model_cls in PUBLIC_MODELS:
        adapter = TypeAdapter(model_cls)
        schemas[model_cls.__name__] = adapter.json_schema()

    # Emit the discriminated LaunchResult union separately
    # so the discriminator is preserved for pd-ui's TypeScript generator
    launch_result_adapter = TypeAdapter(LaunchResult)
    schemas["LaunchResult"] = launch_result_adapter.json_schema()

    return schemas


def main() -> None:
    """CLI entry point: emit JSON Schema to stdout."""
    schemas = emit_schemas()
    print(json.dumps(schemas, indent=2))  # noqa: T201  # CLI output is intentional


if __name__ == "__main__":
    main()
