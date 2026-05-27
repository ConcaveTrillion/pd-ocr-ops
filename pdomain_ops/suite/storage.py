"""StorageAdapter Protocol + LocalFsStorage implementation."""

from __future__ import annotations

from pathlib import Path
from typing import runtime_checkable

from typing_extensions import Protocol


@runtime_checkable
class StorageAdapter(Protocol):
    """Protocol for storage adapter implementations."""

    def read(self, key: str) -> bytes:
        """Read bytes at key; raises if not found."""
        ...

    def write(self, key: str, data: bytes) -> None:
        """Write bytes to key, creating parents as needed."""
        ...

    def exists(self, key: str) -> bool:
        """Return True if key exists in storage."""
        ...

    def delete(self, key: str) -> None:
        """Delete key; noop if not present."""
        ...

    def list_prefix(self, prefix: str) -> list[str]:
        """Return all keys that start with prefix, sorted."""
        ...


class LocalFsStorage:
    """Local filesystem storage adapter."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        if key.startswith("/"):
            raise ValueError(f"Storage key must be relative, got: {key!r}")
        resolved = (self._root / key).resolve()
        if not str(resolved).startswith(str(self._root.resolve())):
            raise ValueError(f"Path traversal detected in key: {key!r}")
        return resolved

    def read(self, key: str) -> bytes:
        """Read bytes at key."""
        return self._resolve(key).read_bytes()

    def write(self, key: str, data: bytes) -> None:
        """Write bytes to key, creating parent directories as needed."""
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def exists(self, key: str) -> bool:
        """Return True if key exists on disk."""
        try:
            return self._resolve(key).exists()
        except ValueError:
            return False

    def delete(self, key: str) -> None:
        """Delete key from disk; noop if not present."""
        path = self._resolve(key)
        if path.exists():
            path.unlink()

    def list_prefix(self, prefix: str) -> list[str]:
        """Return sorted list of all keys under prefix."""
        base = self._root.resolve()
        prefix_path = (self._root / prefix).resolve()
        return sorted(
            str(p.resolve().relative_to(base))
            for p in self._root.rglob("*")
            if p.is_file() and str(p.resolve()).startswith(str(prefix_path))
        )
