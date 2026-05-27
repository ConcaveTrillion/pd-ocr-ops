from pdomain_ops.suite.sibling_spawn import (
    LaunchResultOpened,
    LaunchResultRequiresHostConfig,
    SiblingLaunchAdapter,
)


def test_protocol_present():
    assert hasattr(SiblingLaunchAdapter, "launch")


def test_launch_result_discriminated():
    opened = LaunchResultOpened(url="http://localhost:8001", spawned=True, pid=123)
    assert opened.kind == "opened"
    assert opened.spawned is True

    requires_host = LaunchResultRequiresHostConfig(sibling_id="pd-app-a")
    assert requires_host.kind == "requires-host-config"
    assert requires_host.sibling_id == "pd-app-a"
