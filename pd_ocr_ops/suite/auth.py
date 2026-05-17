"""AuthAdapter Protocol + NoAuthAdapter implementation."""

from __future__ import annotations

from typing import runtime_checkable

from pydantic import BaseModel
from typing_extensions import Protocol


class Identity(BaseModel):
    user_id: str
    display_name: str


@runtime_checkable
class AuthAdapter(Protocol):
    """Protocol for authentication adapter implementations."""

    async def authenticate(self, request: object) -> Identity: ...

    async def is_authenticated(self, request: object) -> bool: ...


class NoAuthAdapter:
    """Single-user local-mode auth adapter — always authenticated."""

    async def authenticate(self, request: object) -> Identity:
        return Identity(user_id="local", display_name="Local User")

    async def is_authenticated(self, request: object) -> bool:
        return True
