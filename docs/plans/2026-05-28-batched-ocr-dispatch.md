---
repo: ConcaveTrillion/ocr-container-meta
plan_type: cross-cut
status: draft
synced: never
---

# Batched OCR Dispatch — VRAM-aware GPU batching with OOM backoff

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking. This plan spans three repos — implement bottom-up
> (pdomain-book-tools → pdomain-ops → consumers) in the waves defined below.

**Goal:** Replace the current per-page sequential/concurrent OCR loop with a single
**chunked, batched** dispatch path that (a) exploits DocTR's real GPU batching, (b)
auto-sizes the batch from available VRAM/RAM, (c) backs off and retries on OOM, and
(d) isolates failures so one bad chunk never loses the whole job. The dispatch
mechanics live in **pdomain-ops** because every OCR consumer (simple-gui, pdomain-ocr-cli,
labeler-spa, trainer-spa) routes through the same `StageDispatcher`, so this pattern is
shared infrastructure, not an app concern.

**Architecture:** book-tools exposes DocTR's batch-size kwargs and a list entry point;
pdomain-ops owns hardware sizing (`pick_doctr_batch_sizes`) and a batched stage with an
OOM-backoff loop + sized predictor cache; consumers chunk their pages and make one batched
stage call per chunk with per-chunk failure isolation.

**Tech Stack:** Python 3.11+, DocTR 1.0.2 (PyTorch), pdomain-ops `LocalStageDispatcher`,
psutil + `torch.cuda.mem_get_info` for sizing, pytest + pytest-asyncio.

---

## Background — how DocTR batching actually works

Investigated against DocTR 1.0.2 source (installed under the book-tools venv):

- `OCRPredictor.forward(pages)` (`doctr/models/predictor/pytorch.py:81`) hands the whole
  page list to the detection predictor, and (line 133) flattens **all** word-crops from
  **all** pages into a single recognition call.
- `DetectionPredictor.forward` (`doctr/models/detection/predictor/pytorch.py:52-59`):
  `pre_processor(pages)` → `PreProcessor.batch_inputs` does
  `torch.stack(samples[i*bs:(i+1)*bs], dim=0)` — it **stacks images into one batched
  tensor**, then `self.model(batch)` runs them through the device in **one forward pass**
  (data-parallel across the batch dim). This is genuine GPU batching, **not** a Python
  loop over images.
- Default batch sizes (`doctr/models/zoo.py:24-25`): **`det_bs = 2`**, **`reco_bs = 128`**.

**Implications:**

1. **Peak VRAM is set by `det_bs`, not by how many pages you pass.** A list of N pages
   becomes `ceil(N/det_bs)` sequential forward passes, each of size ≤ `det_bs`.
   Therefore: *splitting our page list does not reduce peak VRAM — only lowering `det_bs`
   does.*
2. **Detection is the batching win.** Today book-tools calls `predictor([rgb])` with a
   **single** image, so detection runs a batch of 1 even though `det_bs=2` could hold 2.
   Passing multiple pages lets DocTR batch detection forward passes.
3. **Recognition is already saturated per page.** A book page usually has >128 word-crops,
   so `reco_bs=128` is near-full from a single page; raising `reco_bs` only helps when
   batching **many pages' crops together**, and crops are tiny so it is not VRAM-bound —
   `reco_bs` is bounded by *crop supply*, not memory.

---

## Design

### Two distinct knobs (keep them separate)

| Knob | Layer | Bounds | Purpose |
|---|---|---|---|
| `chunk_size` (a.k.a. `batch_pages`) | consumer / dispatcher | failure-isolation, progress, retry-blast-radius | how many pages per `predictor()` call |
| `det_bs` | book-tools predictor build | **VRAM** | DocTR internal detection forward-batch; the OOM lever |
| `reco_bs` | book-tools predictor build | crop supply (high VRAM ceiling) | DocTR internal recognition forward-batch |

A chunk of `C` pages at `det_bs=B` runs `ceil(C/B)` internal detection passes. Typically
`chunk_size ≥ det_bs`.

### Option B — unified batched path, no concurrency

Both CPU and GPU use the **same** chunked-batched code path; device-specific sizing and
backoff differ:

- **GPU:** `det_bs` from free VRAM; CUDA OOM → halve `det_bs`, rebuild predictor, retry
  the chunk; floor (det_bs=1 still OOM) → CPU fallback for that chunk.
- **CPU:** `det_bs` modest (torch intra-op threads parallelize the stacked tensor);
  `MemoryError` backoff (rare at 67 GB) with the same halve-and-retry shape. **No asyncio
  worker-pool concurrency** — concurrent batched calls would each spawn torch threads and
  oversubscribe cores (the thermal-spike risk on hybrid CPUs). One batched call lets torch
  use all cores cleanly.

This **replaces** the Phase-1 per-page asyncio concurrency in simple-gui's `run_project`
(commit `4e03f8b`). The user-facing `parallel_pages` field is renamed to **`batch_pages`**
(pages per call); it no longer means concurrency.

### Chunked processing — failure isolation

```python
for chunk in chunks(pages, chunk_size):
    try:
        results = run_batched_with_oom_backoff(chunk)   # halve det_bs → retry → CPU floor
        write_results(results)                          # per-page sidecar/txt/output mirror
        mark_chunk_succeeded(chunk)
    except Exception as e:                              # non-OOM, or even CPU-floor failure
        logger.exception(...)
        mark_pages_failed(chunk, error=str(e))          # only THIS chunk's pages
        # do NOT abort — continue to the next chunk
    persist_status(); await status_callback(...)        # progress per chunk
```

Three nested resilience levels:
1. **Inside a chunk:** CUDA OOM → halve `det_bs`, rebuild, retry; floor → CPU fallback.
2. **Per chunk:** any other exception fails only that chunk's pages; the loop continues.
3. **Progress:** a status callback per chunk; a late failure never erases earlier success.

Geometric backoff (8→4→2→1) reaches a safe size in log steps. Retries are idempotent (OCR
is pure), so re-running a chunk is safe; smaller `chunk_size` bounds retry recompute waste.

### OOM detection

```python
def _is_oom(e: BaseException) -> bool:
    if isinstance(e, torch.cuda.OutOfMemoryError):   # torch >= 1.13
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()
```
On CPU, also treat `MemoryError` as the backoff trigger. **Re-raise anything that is not
OOM** so real bugs surface. Before rebuilding, `del` the old predictor reference and call
`torch.cuda.empty_cache()` so the failed allocation's reserved blocks are released —
otherwise the retry OOMs again.

---

## Parallelization — execution waves

The three layers form a hard dependency chain (book-tools → ops → simple-gui), so the plan
is **not** fully parallel. It decomposes into **waves**: tasks *within* a wave are
independent (different repos/files, no shared state) and run **concurrently, each in its own
git worktree** (`superpowers:using-git-worktrees`); waves run **sequentially** because a
later wave consumes an earlier wave's API.

```
Wave 1 (parallel):   Task 1 (book-tools batch kwargs)   ║   Task 2 (ops pick_doctr_batch_sizes)
                              │                                        │
                              └──────────────┬─────────────────────────┘
Wave 2 (single):                       Task 3 (ops batched stage + OOM backoff)   ← needs 1 & 2
                                             │
Wave 3 (single):                       Task 4 (simple-gui chunked dispatch)       ← needs 3
                                             │
Wave 4 (parallel, optional):  Task 5 (cli adopt)  ║  Task 6 (labeler/trainer adopt)  ← need 3
```

**Worktree-per-task (REQUIRED).** Each task runs in its own worktree under
`<repo>/.claude/worktrees/<slug>`. Parallel tasks in a wave touch **different repos**, so
there is no merge contention within a wave. After a task verifies green, integrate with
`superpowers:finishing-a-development-branch` (worktree → local merge to `main`, no push).

**Wave gates (critical for local-dev).** Because the repos consume each other as editable
siblings, between waves you must merge the prior wave to each repo's `main` and re-link:
run `make local-setup-py` in the consuming repo so its editable sibling has the new API
before the dependent wave starts. Wave 2 needs Wave 1 merged + re-linked; Wave 3 needs
Wave 2 merged + re-linked.

**Execution model.** Use `superpowers:subagent-driven-development`: one implementer subagent
per task with spec + quality review after each. Wave-1's two tasks are dispatched **in a
single message** (two Agent calls) for true parallelism; Waves 2 and 3 are single tasks.
Set `model: sonnet` on implementers.

---

## Wave 1 — independent foundations (Task 1 ∥ Task 2)

### Task 1: book-tools — expose `det_bs`/`reco_bs` + batch entry point

**Repo:** `pdomain-book-tools` · **Worktree:** `.claude/worktrees/batch-kwargs` · **Model:** sonnet

**Files:**
- Modify: `pdomain_book_tools/ocr/doctr_support.py` (`get_finetuned_torch_doctr_predictor`,
  `get_default_doctr_predictor`, `_assemble_doctr_predictor` at line 187)
- Modify: `pdomain_book_tools/ocr/document.py` (add `Document.from_images_ocr_via_doctr`;
  per-image preprocessing pattern lives at `from_image_ocr_via_doctr`, line 166)
- Test: `tests/ocr/test_doctr_support.py`, `tests/ocr/test_document_batch_ocr.py` (new)

- [ ] **Step 1 — failing test for batch-size pass-through.** In `test_doctr_support.py`,
      monkeypatch the `ocr_predictor` / `detection_predictor` / `recognition_predictor`
      names imported inside `_assemble_doctr_predictor` to recorders; assert
      `_assemble_doctr_predictor(det, reco, pretrained=False, det_bs=8, reco_bs=256)`
      forwards `det_bs=8` to `ocr_predictor` and `detection_predictor(batch_size=8)`, and
      `reco_bs=256` to `recognition_predictor(batch_size=256)`.
- [ ] **Step 2 — run, verify fail:** `make test-k K=assemble_batch_sizes AI=1` → FAIL
      (unexpected kwarg).
- [ ] **Step 3 — implement.** Add `det_bs: int = 2, reco_bs: int = 128` to
      `_assemble_doctr_predictor` and the two public getters; forward to the three DocTR
      public factories: `ocr_predictor(det_bs=…, reco_bs=…)`,
      `detection_predictor(batch_size=det_bs)`, `recognition_predictor(batch_size=reco_bs)`.
      No DocTR edits.
- [ ] **Step 4 — run, verify pass.**
- [ ] **Step 5 — failing test for batch OCR.** In `test_document_batch_ocr.py`, stub a
      predictor that returns a 2-page doctr `Document`; call
      `Document.from_images_ocr_via_doctr([img_a, img_b], source_identifiers=["a","b"],
      predictor=stub)`; assert the result has 2 pages in input order and the stub received a
      single call with a 2-element list.
- [ ] **Step 6 — run, verify fail** (AttributeError: no such method).
- [ ] **Step 7 — implement `from_images_ocr_via_doctr`.** Mirror `from_image_ocr_via_doctr`'s
      per-image cv2/PIL→ndarray preprocessing (document.py:166-251) over the list, build
      `[rgb1, rgb2, …]`, call `predictor([...])` **once**, map returned doctr pages back to
      our `Document` (one page per input).
- [ ] **Step 8 — run, verify pass; then `make ci AI=1`.**
- [ ] **Step 9 — commit:** `feat(ocr): det_bs/reco_bs kwargs + batch OCR entry point`.

**Integration:** `finishing-a-development-branch` → merge to `pdomain-book-tools` `main`
(no push).

### Task 2: ops — `pick_doctr_batch_sizes` hardware sizing

**Repo:** `pdomain-ops` · **Worktree:** `.claude/worktrees/batch-sizing` · **Model:** sonnet

**Files:**
- Modify: `pdomain_ops/gpu/device.py` (add `pick_doctr_batch_sizes`; reuse existing
  `_cuda_free_bytes`, `_physical_cores`), `pdomain_ops/gpu/__init__.py` (export)
- Test: `tests/gpu/test_pick_doctr_batch_sizes.py` (new)

- [ ] **Step 1 — failing tests.** Assert: `pick_doctr_batch_sizes("cpu", chunk_pages=8)`
      → conservative det_bs (1–2), reco_bs≈128; with `_cuda_free_bytes` monkeypatched to
      6 GB and device `"local"`, det_bs scales up (≥4) and reco_bs grows with `chunk_pages`
      under a ceiling; both always ≥1.
- [ ] **Step 2 — run, verify fail** (ImportError).
- [ ] **Step 3 — implement** `pick_doctr_batch_sizes(device: str | None = None,
      chunk_pages: int = 8) -> tuple[int, int]` next to `pick_concurrency`. GPU: det_bs =
      `clamp(_cuda_free_bytes() // _VRAM_PER_PAGE_BYTES, 1, 8)`; reco_bs =
      `min(_RECO_CEILING, max(128, chunk_pages * _CROPS_PER_PAGE_EST))`. CPU: det_bs from a
      small constant (default 1), reco_bs=128. Export from `pdomain_ops.gpu`.
- [ ] **Step 4 — run, verify pass; `make typecheck` then `make test AI=1`.**
- [ ] **Step 5 — commit:** `feat(gpu): pick_doctr_batch_sizes VRAM/CPU sizing`.

**Integration:** merge to `pdomain-ops` `main` (no push).

> **WAVE GATE 1→2:** merge Task 1 + Task 2; in `pdomain-ops` run `make local-setup-py` so
> its editable `pdomain-book-tools` exposes `from_images_ocr_via_doctr` before Wave 2.

---

## Wave 2 — ops batched stage + OOM backoff (Task 3, single)

### Task 3: ops — batched stage impl, OOM backoff, sized predictor cache

**Repo:** `pdomain-ops` · **Worktree:** `.claude/worktrees/batch-stage` · **Model:** sonnet
**Depends on:** Task 1 (batch entry point) + Task 2 (`pick_doctr_batch_sizes`).

**Files:**
- Modify: `pdomain_ops/gpu/default_stages.py` (`_ocr_local_impl`, module predictor cache,
  new batched impl + registration), `pdomain_ops/gpu/local_stage.py` (batch dispatch entry)
- Test: `tests/gpu/test_default_stages.py`

- [ ] **Step 1 — cache-key test.** Assert the module predictor cache returns **distinct**
      predictors for `(det_path, reco_path, det_bs=2)` vs `(…, det_bs=4)` (today keyed only
      on paths). Monkeypatch `get_finetuned_torch_doctr_predictor` to a call counter.
- [ ] **Step 2 — run, verify fail** (same predictor returned for both).
- [ ] **Step 3 — implement** cache key `(str(det_path), str(reco_path), det_bs, reco_bs)`.
- [ ] **Step 4 — OOM-backoff tests.** (a) Stub predictor raises
      `torch.cuda.OutOfMemoryError` on first call then succeeds → assert det_bs halved
      (8→4), `torch.cuda.empty_cache` called, chunk result returned. (b) det_bs=1 OOM →
      delegates to `_ocr_cpu_impl` (assert CPU path invoked, warning logged). (c) non-OOM
      `RuntimeError` re-raises unchanged.
- [ ] **Step 5 — run, verify fail.**
- [ ] **Step 6 — implement** a batched impl accepting `image_paths: list[str]`: size via
      `pick_doctr_batch_sizes`, call `from_images_ocr_via_doctr` through the cached
      predictor inside the backoff loop — `_is_oom(e)` (torch OOM **or** RuntimeError
      "out of memory" **or** `MemoryError` on CPU); on OOM `del predictor;
      torch.cuda.empty_cache()`, halve det_bs/reco_bs, rebuild via cache, retry; floor
      (det_bs==1) → per-image `_ocr_cpu_impl`. Return a list of page dicts (one per input).
      Register a batch stage key and add `run_batch_stage` to `LocalStageDispatcher`
      (keep single-image `run_stage` as a back-compat wrapper).
- [ ] **Step 7 — run, verify pass; `make typecheck`; `make test AI=1`.**
- [ ] **Step 8 — commit:** `feat(gpu): batched OCR stage with OOM backoff + sized cache`.

**Integration:** merge to `pdomain-ops` `main` (no push).

> **WAVE GATE 2→3:** in `pdomain-ocr-simple-gui` run `make local-setup-py` to re-link the
> new ops (`run_batch_stage`) before Wave 3.

---

## Wave 3 — simple-gui chunked dispatch (Task 4, single)

### Task 4: simple-gui — chunked batched `run_project` + `batch_pages`

**Repo:** `pdomain-ocr-simple-gui` · **Worktree:** `.claude/worktrees/chunked-dispatch` · **Model:** sonnet
**Depends on:** Task 3 (batch stage).

**Files:**
- Modify: `src/pdomain_ocr_simple_gui/pipeline.py` (`run_project` — replace the Phase-1
  asyncio worker pool with the chunked loop; remove `resolve_concurrency`/`Semaphore`)
- Modify: `src/pdomain_ocr_simple_gui/models.py`, `src/pdomain_ocr_simple_gui/routes/jobs.py`
  (rename `parallel_pages` → `batch_pages`)
- Modify: `frontend/src/components/JobConfigInline.tsx`, `frontend/src/lib/testids.ts`
  (field/label rename)
- Test: `tests/test_pipeline.py`, `frontend/src/components/JobConfigInline.test.tsx`

- [ ] **Step 1 — failing test: chunk failure isolation.** Mock the batch stage so chunk 1
      succeeds and chunk 2 raises a non-OOM error; assert chunk-1 pages are `succeeded` with
      sidecars written, chunk-2 pages are `failed` with `error` set, and the job reaches a
      terminal `failed` state (not an aborted run).
- [ ] **Step 2 — run, verify fail:** `make test AI=1`.
- [ ] **Step 3 — implement** the chunked loop in `run_project`: split `images` into
      `chunk_size` groups (`batch_pages` override or default 8); one `run_batch_stage` call
      per chunk inside `try/except` that marks only that chunk's pages failed and continues;
      per-chunk status callback ("Processed X/N pages"); keep status mutations serialized
      (no concurrency). Delete `resolve_concurrency` + the `asyncio.Semaphore` pool.
- [ ] **Step 4 — run, verify pass.**
- [ ] **Step 5 — rename `parallel_pages` → `batch_pages`** on `CreateJobRequest`,
      `ProjectSpec`, the POST body, and the form field/label ("Pages per batch (blank =
      auto)"); update the `JobConfigInline.test.tsx` body assertion
      (`expect(body.batch_pages)`).
- [ ] **Step 6 — verify:** `make test AI=1`, `make frontend-test AI=1`,
      `make typecheck AI=1`, `make local-frontend-build`.
- [ ] **Step 7 — commit:** `feat(pipeline): chunked batched dispatch + batch_pages`.
- [ ] **Step 8 — browser smoke (manual):** `make local-run`, drop a multi-page folder,
      confirm progress advances per chunk and a forced mid-job error fails only that chunk.

**Integration:** merge to `pdomain-ocr-simple-gui` `main` (no push).

---

## Wave 4 — optional consumer adoption (Task 5 ∥ Task 6)

Independent once Task 3 exists; parallel, different repos. Not required for the first cut.

### Task 5: pdomain-ocr-cli adopts the batch stage
**Repo:** `pdomain-ocr-cli` · **Worktree:** `.claude/worktrees/batch-adopt` · **Model:** sonnet
- [ ] Route whole-book OCR through `run_batch_stage`; chunk by `batch_pages`; same OOM
      resilience. TDD against a stub dispatcher. `make ci AI=1`. Commit.

### Task 6: labeler-spa / trainer-spa re-OCR via batch stage
**Repos:** `pdomain-ocr-labeler-spa`, `pdomain-ocr-trainer-spa` (separate worktrees) · **Model:** sonnet
- [ ] On-demand re-OCR routes through the batch stage (chunk size 1 is fine — they gain OOM
      resilience for free). TDD per repo. Commit.

---

## Implications for other tools

The dispatcher is shared, so this benefits every OCR consumer:

- **pdomain-ocr-cli** processes whole books — chunked GPU batching is the biggest win there.
- **labeler-spa / trainer-spa** re-OCR pages on demand — they get VRAM-safe batching and
  OOM resilience for free once they route through the batch stage.
- `pick_doctr_batch_sizes` + the OOM-backoff loop are device-detection siblings of
  `pick_device` / `pick_concurrency`, so they stay in `pdomain_ops.gpu` as the one place
  hardware policy lives.

## Open questions

- Default `chunk_size`: start at ~8 pages (balances overhead amortization vs
  retry-blast-radius); revisit after measuring real book runs.
- Whether to retire the single-image `run_stage` once all consumers move to the batch path,
  or keep it as a thin wrapper (`run_batch_stage([one])`).
- CPU `det_bs` default: 1 (simplest, torch threads parallelize spatial dims) vs a small
  constant (amortizes per-call overhead). Measure before fixing.
- `_VRAM_PER_PAGE_BYTES` calibration for `pick_doctr_batch_sizes` — start conservative
  (~1.2 GB/page detection working set), tune against observed OOM thresholds.
