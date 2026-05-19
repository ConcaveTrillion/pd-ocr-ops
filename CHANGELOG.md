# Changelog

## [0.2.0] - 2026-05-19

### Added

- GPU dispatch wire-shape types migrated from pd-prep-for-pgdp:
  `ProcessPageRequest`, `ProcessPageResponse`, `OcrPageRequest`, `OcrPageResponse`,
  `BatchJobItem`, `BatchJobResult`, `BatchProgressCb`, `GPUBackend` Protocol
  (`pd_ocr_ops.gpu.types`)
- `ModalStageDispatcher` (renamed from `ModalBackend`; legacy alias kept)
- `SharedContainerStageDispatcher` stub (renamed from `SharedContainerBackend`; legacy alias kept)
- `modal_app.py` — Modal deploy scaffold for `pd-ocr-ops` app name
- Optional dep group `modal = ["modal>=0.66"]`

## [0.1.0] - 2026-05-10

### Added (v0.1.0)

- Initial release: `StageDispatcher` / `LongJobRunner` Protocols
- `LocalStageDispatcher` + `LocalLongJobRunner` implementations
- `register_default_stages()` for DocTR and Tesseract OCR
- Suite plumbing: `mount_routes()`, `register_self()`, `prefs`, `sibling_spawn`, `desktop`
- `schemas.emit` for JSON Schema generation
