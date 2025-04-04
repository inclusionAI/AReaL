import os
import shutil
import tempfile
import time
import uuid
from unittest.mock import patch

import pytest

from realhf.base.name_resolve import (
    NameEntryExistsError,
    NameEntryNotFoundError,
    NfsNameRecordRepository,
)


@pytest.fixture
def temp_nfs_root():
    # Create a temporary directory to simulate NFS root
    temp_dir = tempfile.mkdtemp()
    original_root = NfsNameRecordRepository.RECORD_ROOT
    NfsNameRecordRepository.RECORD_ROOT = temp_dir
    yield temp_dir
    # Cleanup
    NfsNameRecordRepository.RECORD_ROOT = original_root
    shutil.rmtree(temp_dir)


@pytest.fixture
def nfs_repo(temp_nfs_root):
    repo = NfsNameRecordRepository()
    yield repo
    repo.reset()


def test_add_basic(nfs_repo):
    # Test basic add functionality
    nfs_repo.add("test_key", "test_value")
    assert nfs_repo.get("test_key") == "test_value"

    # Verify file was created
    assert os.path.isfile(
        os.path.join(NfsNameRecordRepository.RECORD_ROOT, "test_key/ENTRY")
    )

    # Non-string value
    nfs_repo.add(
        "test_key", 123, replace=True
    )  # Should fail if non-string values aren't converted
    assert nfs_repo.get("test_key") == str(123)

    with pytest.raises(ValueError):
        nfs_repo.add("", "value")


def test_add_with_replace(nfs_repo):
    # Test add with replace=False (should raise)
    nfs_repo.add("test_key", "test_value")
    with pytest.raises(NameEntryExistsError):
        nfs_repo.add("test_key", "new_value", replace=False)

    # Test add with replace=True
    nfs_repo.add("test_key", "new_value", replace=True)
    assert nfs_repo.get("test_key") == "new_value"


def test_add_delete_on_exit(nfs_repo):
    # Test delete_on_exit flag
    nfs_repo.add("test_key1", "value1", delete_on_exit=True)
    nfs_repo.add("test_key2", "value2", delete_on_exit=False)

    assert "test_key1" in nfs_repo._NfsNameRecordRepository__to_delete
    assert "test_key2" not in nfs_repo._NfsNameRecordRepository__to_delete


def test_delete(nfs_repo):
    # Test deleting existing key
    nfs_repo.add("test_key", "test_value")
    nfs_repo.delete("test_key")
    with pytest.raises(NameEntryNotFoundError):
        nfs_repo.get("test_key")

    # Test deleting non-existent key
    with pytest.raises(NameEntryNotFoundError):
        nfs_repo.delete("non_existent_key")


def test_delete_cleanup_dirs(nfs_repo):
    # Test that empty parent directories are cleaned up
    nfs_repo.add("test/path/key", "value")
    assert os.path.isdir(os.path.join(NfsNameRecordRepository.RECORD_ROOT, "test/path"))

    nfs_repo.delete("test/path/key")
    # Should clean up empty parent directories
    assert not os.path.exists(
        os.path.join(NfsNameRecordRepository.RECORD_ROOT, "test/path")
    )
    assert not os.path.exists(os.path.join(NfsNameRecordRepository.RECORD_ROOT, "test"))


def test_clear_subtree(nfs_repo):
    # Test clearing a subtree
    nfs_repo.add("test_root/key1", "value1")
    nfs_repo.add("test_root/key2", "value2")
    nfs_repo.add("test_root/sub/key3", "value3")
    nfs_repo.add("other_root/key", "value")

    nfs_repo.clear_subtree("test_root")

    # Verify subtree is gone
    assert nfs_repo.get_subtree("test_root") == []
    assert nfs_repo.find_subtree("test_root") == []

    # Verify other tree is intact
    assert nfs_repo.get("other_root/key") == "value"


def test_get(nfs_repo):
    # Test getting existing key
    nfs_repo.add("test_key", "test_value")
    assert nfs_repo.get("test_key") == "test_value"

    # Test getting non-existent key
    with pytest.raises(NameEntryNotFoundError):
        nfs_repo.get("non_existent_key")


def test_get_stale_file_handle_recovery(nfs_repo):
    # Test handling of stale file handles
    nfs_repo.add("test_key", "test_value")

    # Mock os.open to raise OSError with errno 116 (ESTALE) first few times
    original_open = open

    def mock_open(*args, **kwargs):
        mock_open.call_count += 1
        if mock_open.call_count <= 3:  # Fail first 3 times
            raise OSError(116, "Stale file handle")
        return original_open(*args, **kwargs)

    mock_open.call_count = 0

    with patch("builtins.open", mock_open):
        assert nfs_repo.get("test_key") == "test_value"
    assert mock_open.call_count == 4


def test_get_subtree(nfs_repo):
    # Test getting subtree values
    nfs_repo.add("test_root/key1", "value1")
    nfs_repo.add("test_root/key2", "value2")
    nfs_repo.add("test_root/sub/key3", "value3")

    values = nfs_repo.get_subtree("test_root")
    assert set(values) == {"value1", "value2"}


def test_find_subtree(nfs_repo):
    # Test finding subtree keys
    nfs_repo.add("test_root/key1", "value1")
    nfs_repo.add("test_root/key2", "value2")
    nfs_repo.add("test_root/sub/key3", "value3")

    keys = nfs_repo.find_subtree("test_root")
    assert set(keys) == {"test_root/key1", "test_root/key2"}
    assert keys == sorted(keys)  # Should be sorted


def test_reset(nfs_repo):
    # Test reset functionality
    nfs_repo.add("test_key1", "value1", delete_on_exit=True)
    nfs_repo.add("test_key2", "value2", delete_on_exit=False)

    nfs_repo.reset()

    # Only test_key1 should be deleted
    with pytest.raises(NameEntryNotFoundError):
        nfs_repo.get("test_key1")
    assert nfs_repo.get("test_key2") == "value2"


def test_context_manager(nfs_repo):
    # Test context manager functionality
    with NfsNameRecordRepository() as repo:
        repo.add("test_key", "test_value", delete_on_exit=True)
        assert repo.get("test_key") == "test_value"

    # Key should be deleted after context exits
    with pytest.raises(NameEntryNotFoundError):
        nfs_repo.get("test_key")


def test_destructor(nfs_repo):
    # Test destructor functionality
    repo = NfsNameRecordRepository()
    repo.add("test_key", "test_value", delete_on_exit=True)

    # Simulate object destruction
    repo.__del__()

    # Key should be deleted
    with pytest.raises(NameEntryNotFoundError):
        nfs_repo.get("test_key")


def test_add_subentry(nfs_repo):
    # Test subentry creation
    sub_name = nfs_repo.add_subentry("test_root", "sub_value")
    assert sub_name.startswith("test_root/")
    assert nfs_repo.get(sub_name) == "sub_value"


def test_wait(nfs_repo):
    # Test wait functionality
    import threading

    def delayed_add():
        time.sleep(0.1)
        nfs_repo.add("test_key", "test_value")

    job = threading.Thread(target=delayed_add, daemon=True)
    job.start()

    # Should return once key is added
    assert nfs_repo.wait("test_key", timeout=2) == "test_value"
    job.join()

    # Test timeout
    with pytest.raises(TimeoutError):
        nfs_repo.wait("non_existent_key", timeout=0.1)


def test_watch_names(nfs_repo):
    # Test watch functionality
    callback_called = False

    def callback():
        nonlocal callback_called
        callback_called = True

    nfs_repo.add("test_key", "test_value")
    nfs_repo.watch_names("test_key", callback)

    # Delete the key to trigger callback
    nfs_repo.delete("test_key")

    # Wait for callback
    time.sleep(5)  # Give watcher thread time to execute
    assert callback_called


def test_concurrent_access(nfs_repo):
    # Test concurrent access to the same key
    import threading

    nfs_repo.add("test_key", "initial_value")

    def modify_value():
        for i in range(10):
            current = nfs_repo.get("test_key")
            nfs_repo.add("test_key", f"modified_{i}", replace=True)
            time.sleep(0.01)

    threads = [threading.Thread(target=modify_value) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Final value should be one of the modified values
    final_value = nfs_repo.get("test_key")
    assert final_value.startswith("modified_")
