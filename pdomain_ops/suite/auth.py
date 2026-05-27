"""AuthAdapter Protocol + NoAuthAdapter implementation."""

from __future__ import annotations

from typing import runtime_checkable

from pydantic import BaseModel
from typing_extensions import Protocol


class Identity(BaseModel):
    """Authenticated user identity."""

    user_id: str
    display_name: str


@runtime_checkable
class AuthAdapter(Protocol):
    """Protocol for authentication adapter implementations."""

    async def authenticate(self, request: object) -> Identity:
        """Authenticate the request and return the Identity."""
        ...

    async def is_authenticated(self, request: object) -> bool:
        """Return True if the request carries a valid auth credential."""
        ...


class NoAuthAdapter:
    """Single-user local-mode auth adapter — always authenticated."""

    async def authenticate(self, request: object) -> Identity:
        """Return a fixed local identity (single-user mode)."""
        return Identity(user_id="local", display_name="Local User")

    async def is_authenticated(self, request: object) -> bool:
        """Always return True in local single-user mode."""
        return True
