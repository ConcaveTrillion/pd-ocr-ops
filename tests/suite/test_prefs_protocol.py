from pdomain_ops.suite.prefs import PrefsAdapter


def test_protocol_methods_present():
    assert hasattr(PrefsAdapter, "read")
    assert hasattr(PrefsAdapter, "write_common")
    assert hasattr(PrefsAdapter, "write_app")


def test_protocol_is_runtime_checkable():
    class FakePrefs:
        def read(self):
            from pdomain_ops.suite.types import UIPrefs

            return UIPrefs()

        def write_common(self, common):
            pass

        def write_app(self, app_id, payload):
            pass

    assert isinstance(FakePrefs(), PrefsAdapter)
