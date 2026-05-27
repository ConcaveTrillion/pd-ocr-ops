import pytest

from pdomain_ops.suite.storage import LocalFsStorage, StorageAdapter


def test_protocol_methods_present():
    assert hasattr(StorageAdapter, "read")
    assert hasattr(StorageAdapter, "write")
    assert hasattr(StorageAdapter, "exists")
    assert hasattr(StorageAdapter, "delete")
    assert hasattr(StorageAdapter, "list_prefix")


def test_protocol_runtime_checkable():
    storage = LocalFsStorage(root="/tmp")
    assert isinstance(storage, StorageAdapter)


def test_local_fs_storage_round_trip(tmp_path):
    storage = LocalFsStorage(root=tmp_path)
    storage.write("test.json", b'{"key": "value"}')
    assert storage.exists("test.json")
    data = storage.read("test.json")
    assert data == b'{"key": "value"}'
    storage.delete("test.json")
    assert not storage.exists("test.json")
    files = storage.list_prefix("")
    assert "test.json" not in files


def test_local_fs_storage_rejects_absolute_paths(tmp_path):
    storage = LocalFsStorage(root=tmp_path)
    with pytest.raises(ValueError):
        storage.write("/etc/passwd", b"x")


def test_local_fs_storage_rejects_traversal(tmp_path):
    storage = LocalFsStorage(root=tmp_path)
    with pytest.raises(ValueError):
        storage.write("../outside", b"x")


def test_local_fs_storage_creates_intermediate_dirs(tmp_path):
    storage = LocalFsStorage(root=tmp_path)
    storage.write("a/b/c.json", b"data")
    assert (tmp_path / "a" / "b" / "c.json").exists()
