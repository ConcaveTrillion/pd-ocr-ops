from pdomain_ops.suite.registry import SuiteRegistryAdapter


def test_protocol_methods_present():
    assert hasattr(SuiteRegistryAdapter, "list_installed")
    assert hasattr(SuiteRegistryAdapter, "register")
    assert hasattr(SuiteRegistryAdapter, "unregister")


def test_protocol_is_runtime_checkable():
    # runtime_checkable lets isinstance() work against the Protocol
    class FakeRegistry:
        def list_installed(self):
            return []

        def register(self, app):
            pass

        def unregister(self, app_id):
            pass

    assert isinstance(FakeRegistry(), SuiteRegistryAdapter)
