def test_top_level_exports_present():
    from pdomain_ops import SuiteAdapters, mount_routes
    from pdomain_ops.gpu import LongJobRunner, StageDispatcher, pick_device
    from pdomain_ops.suite import InstalledApp, SuiteApp, UIPrefs

    assert mount_routes is not None
    assert SuiteAdapters is not None
    assert SuiteApp is not None
    assert InstalledApp is not None
    assert UIPrefs is not None
    assert StageDispatcher is not None
    assert LongJobRunner is not None
    assert pick_device is not None
