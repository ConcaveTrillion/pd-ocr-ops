# Dynamic-port SPA bootstrap

## Problem

pd-* SPAs default to a fixed port (e.g. simple-gui = 8004). When
something else on the machine already binds that port the app crashes on
startup with `OSError: [Errno 98] Address already in use`. Hardcoding a
fallback port just shifts the collision risk to the fallback.

## Solution

`pdomain_ocr_ops.suite` exposes two helpers that work together:

- `find_available_port(preferred, host, max_attempts)` — probes the
  preferred port and walks upward until a free one is found.
- `register_self(..., actual_port=port)` — records the real bound port
  in the suite registry so cross-app linking reads the correct address.

## Canonical bootstrap snippet

```python
import uvicorn
from pdomain_ocr_ops.suite import find_available_port, register_self

PREFERRED_PORT = 8004

def main() -> None:
    port = find_available_port(PREFERRED_PORT)
    register_self(
        _caller_package="pdomain_ocr_simple_gui",
        actual_port=port,
    )
    uvicorn.run("pdomain_ocr_simple_gui.app:app", host="127.0.0.1", port=port)
```

`find_available_port` uses a transient `socket.bind` probe with
`SO_REUSEADDR=0`, so it accurately reflects whether the OS considers the
port free. The probe socket is always closed before returning — uvicorn
binds it for real. A TOCTOU race is possible (another process could
claim the port between the probe and uvicorn's bind) but is extremely
unlikely in practice; if it occurs uvicorn raises `EADDRINUSE`
immediately, and calling `find_available_port` again resolves it.

## Notes for stage-2 SPA consumers

When adopting this pattern in `pdomain-ocr-simple-gui`,
`pdomain-prep-for-pgdp`, `pdomain-ocr-labeler-spa`, and the trainer SPA:

1. Import `find_available_port` and `register_self` from
   `pdomain_ocr_ops.suite`.
2. Call `find_available_port(PREFERRED_PORT)` before `uvicorn.run`.
3. Pass the result to both `register_self(actual_port=port)` and
   `uvicorn.run(..., port=port)`.
4. The `_caller_package` argument is the top-level Python package name
   (e.g. `"pdomain_ocr_simple_gui"`), not the distribution name.
5. If the app already calls `register_self` without `actual_port`, just
   add the parameter — the default (`None`) is backward-compatible.
