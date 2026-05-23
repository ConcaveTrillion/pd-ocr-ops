# Lint-rule Deviations — pd-ocr-ops

Standing suppressions and per-file rule overrides in this repo.
Each entry records: the rule, the tool, the file(s) affected, and the
justification. Update this file whenever a new suppression is added.

Reference implementation: `pd-book-tools/docs/conventions/lint-deviations.md`.
Workspace rule: `CONVENTIONS.md` → "Document every lint-rule suppression".

---

## Inline suppressions

### 1. `reportMissingImports` — basedpyright

**Files:** `pd_ocr_ops/gpu/device.py` (`import cupy`, `import torch`),
`pd_ocr_ops/gpu/modal_dispatcher.py` (`from modal import Function`),
`pd_ocr_ops/gpu/modal_app.py` (`import modal`).

**Suppression form:** `# pyright: ignore[reportMissingImports]` inline on the
guarded `import` line.

**Justification.** `cupy` and `torch` are optional `[gpu]`-extra dependencies;
`modal` is the optional `[modal]`-extra dependency. Each import sits inside a
`try`/`except ImportError` block so the module loads cleanly on installs
without the extra. The CI environment (`make setup` → `uv sync --group dev`)
installs none of these extras, so the imports genuinely fail to resolve there
— the suppression is correct, not masking a real bug. basedpyright does not
honor mypy's `# type: ignore[import-not-found]`; the pyright-native code is
required.

### 2. `reportAttributeAccessIssue` — basedpyright

**Files:**

- `pd_ocr_ops/gpu/device.py` (`torch.backends.mps.is_available()`)
- `pd_ocr_ops/gpu/default_stages.py` (`cv2.imread`)
- `pd_ocr_ops/gpu/modal_dispatcher.py` (`Function.lookup`)

**Suppression form:** `# pyright: ignore[reportAttributeAccessIssue]` inline.

**Justification.** Third-party stub gaps. `torch` and `cv2` ship incomplete
stubs that omit `torch.backends.mps` and `cv2.imread` respectively, though
both are valid at runtime. `modal`'s stubs omit `Function.lookup`; that
diagnostic only surfaces when the `[modal]` extra is installed (CI does not
install it). All three attributes are correct at runtime.

### 3. `reportArgumentType` / `reportReturnType` / `reportAssignmentType` — basedpyright

**Files:**

- `pd_ocr_ops/gpu/local_stage.py` (`device=device` argument)
- `pd_ocr_ops/gpu/device.py` (`return explicit`, `return legacy`)
- `pd_ocr_ops/suite/types.py` (`registered_at`, `layer_colors`, `common` fields)

**Suppression form:** `# pyright: ignore[reportArgumentType]`,
`# pyright: ignore[reportReturnType]`, `# pyright: ignore[reportAssignmentType]`
inline.

**Justification.** Two patterns:

- **Flow-narrowing pyright cannot trace.** In `local_stage.py` and `device.py`
  the value is validated against `_VALID_DEVICES` (with explicit fallback
  logic) before use; the literal-type narrowing does not survive pyright's
  flow analysis. The inline comment names the narrowing step in each case.
- **Pydantic deferred defaults.** In `suite/types.py` the three fields are
  declared `= None` and populated in `model_post_init`. The declared type is
  the post-init type (not `... | None`), so pyright flags the `None` default.
  This is the intentional "deferred default via `model_post_init`" Pydantic
  pattern.

### 4. `reportUnknownMemberType` / `reportUnknownVariableType` / `reportUntypedFunctionDecorator` — basedpyright

**Files:** `pd_ocr_ops/gpu/modal_app.py` (`modal.Image`, `modal.App`,
`@app.function` decorators).

**Suppression form:** `# pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]`
and `# pyright: ignore[reportUnknownMemberType,reportUntypedFunctionDecorator]`
inline.

**Justification.** `modal` ships no type stubs, so every `modal.*` member
access is `Unknown`-typed and `@app.function` is an untyped decorator. The
suppressions are scoped to the Modal-deploy entry point inside the
`if _modal_available` guard. A block comment above the guard explains the
shared rationale. These only surface when the `[modal]` extra is installed.

### 5. `TC003` — ruff (typing-only-standard-library-import)

**File:** `pd_ocr_ops/gpu/types.py` (`from datetime import datetime`).

**Suppression form:** `# noqa: TC003` inline.

**Justification.** `TC003` wants standard-library type-only imports moved into
a `TYPE_CHECKING` block. `datetime` is used as a Pydantic model field
annotation, and Pydantic resolves field annotations at runtime — moving the
import under `TYPE_CHECKING` would break model construction.

### 6. `T201` — ruff (print found)

**File:** `pd_ocr_ops/schemas/emit.py` (`print(json.dumps(...))`).

**Suppression form:** `# noqa: T201` inline.

**Justification.** `schemas.emit` is a CLI entry point; emitting the JSON
schema document to stdout via `print()` is its intended output mechanism.

### 7. `TRY004` — ruff (type-check-without-type-error)

**File:** `pd_ocr_ops/gpu/events.py` (two `raise ValueError` sites after
`isinstance` checks).

**Suppression form:** `# noqa: TRY004` inline.

**Justification.** `TRY004` suggests raising `TypeError` after a failed
`isinstance` check. The `parse_pdevent` contract is that *all* malformed
`@@PDEVENT@@` payloads — JSON decode failures, non-object payloads, invalid
kinds — raise `ValueError` uniformly. Callers catch a single exception type;
splitting some cases out to `TypeError` would break that contract.

---

## Config-level deviations — ruff

All entries below are documented inline at their definition site in
`pyproject.toml` (`[tool.ruff.lint] ignore` and `per-file-ignores`). This
section is the catalogue; the inline comments are the point-of-deviation
rationale.

### 8. Project-wide `[tool.ruff.lint] ignore`

| Rule | Reason |
|------|--------|
| `E501` | Long docstrings, error messages, URLs; 88-char wrapping adds noise. |
| `D203` / `D212` | Conflict with `D211` / `D213` respectively; one of each pair is picked. |
| `D100` / `D104` / `D107` | Missing module/package/`__init__` docstrings; added incrementally, not in one sweep. |
| `D105` | `__repr__`/`__eq__` magic methods are self-documenting. |
| `D205` | Blank-line-between-summary-and-description; enforcing across the backlog is too noisy for one commit. |
| `PLR0913` | OCR/pipeline functions legitimately take many params; config-object refactor not warranted. |
| `PLR2004` | Magic-value comparison common in threshold/port/timeout code. |
| `PLR0912` / `PLR0911` / `PLR0915` | High branch/return/statement counts in pipeline functions; refactor not warranted. |
| `TRY003` | Long messages outside exception classes; too noisy for an f-string-message library. |
| `COM812` | Conflicts with the ruff formatter's auto-style. |
| `PLC0415` | Deferred imports are a legitimate pattern (break circular deps, defer optional-heavy modules). |
| `ANN401` | Some functions legitimately accept/return `Any` (JSON deserialisers, generic dispatch). |
| `B008` | FastAPI `Depends()` and Pydantic `field()` use call-in-default legitimately in route defs. |

### 9. `per-file-ignores`

| File glob | Rules | Reason |
|-----------|-------|--------|
| `tests/**/*.py` | `S101, S105, S106, S311, T201, ANN, D, PLR2004, PT011, S108, PLR0133, PLW2901, PERF401, BLE001, PLW1510, SIM117` | Test idioms: `assert`, fixture passwords/random, no annotation/docstring requirement, broad excepts in stress tests, intentional `subprocess.run` without `check=`. |
| `scripts/*.py` | `T201, D, S607` | Scripts use `print()`; partial executable paths (`uv`, `git`) are idiomatic. (No `scripts/` dir exists yet — dormant.) |
| `**/__init__.py` | `D104, F401, TC` | Re-export modules; `F401` is the public-API-surface mechanism. |
| `**/_*.py` | `D` | Private modules; docstrings not required by internal convention. |
| `pd_ocr_ops/gpu/device.py` | `BLE001` | Optional-dep probes must catch all exceptions (`cupy`/`torch` can raise `ImportError`, `RuntimeError`, etc.). |
| `pd_ocr_ops/gpu/local_jobs.py` | `BLE001, S603, RUF006` | `subprocess.Popen` is the mechanism; fire-and-forget `create_task` is the intentional supervision pattern; supervisor catch-all. |
| `pd_ocr_ops/suite/prefs.py` | `BLE001` | JSON parse failure is treated as empty/default (resilience). |
| `pd_ocr_ops/suite/registry.py` | `BLE001, TRY300, S112` | TOML parse failure treated as empty registry; read-or-default `try`/`except` structure; skip stale/corrupt entries with `try`/`except`/`continue`. |
| `pd_ocr_ops/suite/sibling_spawn.py` | `BLE001, S603, S110` | `subprocess.Popen` for app spawning; health-poll catch-all; intentional poll-until-healthy loop. |

---

## Config-level deviations — basedpyright

### 10. `typeCheckingMode = "recommended"` with `failOnWarnings` deferred

**Config:** `pyproject.toml` `[tool.basedpyright]`.

**Justification.** "recommended" is the workspace-canonical strict mode.
`failOnWarnings` is intentionally left commented out: recommended mode
surfaces warnings from optional GPU dep stubs (`cupy`/`torch`) that lag
runtime. Enable incrementally as stub coverage improves.

### 11. `reportImportCycles = "none"`

**Config:** `pyproject.toml` `[tool.basedpyright]`.

**Justification.** Import cycles are structural and resolved via
`TYPE_CHECKING` guards. Genuine import-order problems surface as runtime
`ImportError`s, which tests catch.

### 12. CI typecheck scope: `pd_ocr_ops` only — NOT `tests/`

**Config:** `Makefile` `typecheck` target runs
`basedpyright pd_ocr_ops --level error` (the `pd_ocr_ops` package only).

**Justification.** `make ci` typechecks the shipped package at error level.
The `tests/` tree is *not* typechecked by CI: test files contain bare `dict` /
`list` annotations and other reportMissingTypeArgument-class diagnostics that
are idiomatic for fixtures and not worth a strict pass. `[tool.basedpyright]`
`include` still lists `tests` so editor/IDE checking covers them, and a
separate `executionEnvironment` gives `tests` and `scripts` more leniency.
This scoping is the reason `make ci` is green despite `basedpyright --level
error` reporting test-tree diagnostics when run repo-wide.

**Needs review.** If a future change makes the CI typecheck cover `tests/`,
the bare-`dict`/`list` annotations across the test tree must be type-argument
-completed first (one such site, `tests/suite/test_register_self.py`'s
`fragment: dict` helper param, is a known example).

---

## Resolved / removed suppressions

### 13. mypy-style `# type: ignore[...]` artifacts — REMOVED

**Files (historical):** `pd_ocr_ops/suite/register_self.py`
(`# type: ignore[attr-defined]`), `pd_ocr_ops/gpu/modal_app.py`
(`# type: ignore[assignment]`), `tests/suite/test_register_self.py`
(`# type: ignore[assignment]` ×4), `tests/gpu/test_modal_dispatcher.py`
(`# type: ignore[type-arg]`).

**Resolution.** These were mypy-style suppressions; basedpyright does not
honor mypy codes. Each was audited by stripping it and re-running
basedpyright:

- `register_self.py` and `modal_app.py`: stripping produced **0 errors** and
  **0 warnings** at the suppressed lines — they silenced nothing. Removed.
- `test_register_self.py`: the `# type: ignore[assignment]` on the
  `mod.__path__` / `mod.__spec__` lines did *not* suppress the real
  diagnostic — `importlib.util` was flagged `reportAttributeAccessIssue`
  because `import importlib.util` was missing. The underlying issue was fixed
  by adding the explicit `import importlib.util`, after which the assignments
  type-check cleanly with no suppression.
- `test_modal_dispatcher.py`: the bare `list[dict]` was completed to
  `list[dict[str, Any]]`, removing the diagnostic and the suppression.

No suppression in this group was converted to `# pyright: ignore[...]` —
none was silencing a basedpyright diagnostic that the correct fix did not
eliminate outright.
