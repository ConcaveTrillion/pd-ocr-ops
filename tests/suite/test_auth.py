import pytest

from pdomain_ops.suite.auth import AuthAdapter, Identity, NoAuthAdapter


@pytest.mark.asyncio
async def test_no_auth_adapter_returns_single_user():
    adapter = NoAuthAdapter()
    identity = await adapter.authenticate(request=None)
    assert isinstance(identity, Identity)
    assert identity.user_id == "local"
    assert identity.display_name == "Local User"


def test_protocol_runtime_checkable():
    adapter = NoAuthAdapter()
    assert isinstance(adapter, AuthAdapter)


def test_protocol_methods_present():
    assert hasattr(AuthAdapter, "authenticate")
    assert hasattr(AuthAdapter, "is_authenticated")
