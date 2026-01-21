import os
import shutil
import tempfile
import threading
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

from realhf.base.name_resolve import (
    Etcd3NameRecordRepository,
    NameEntryExistsError,
    NameEntryNotFoundError,
    NfsNameRecordRepository,
)

# Define backend configurations for parameterized tests
BACKENDS = [
    ("memory", {}),
    ("nfs", {}),
    ("ray", {}),
]
if os.environ.get("TESTING_ETCD_ADDR"):
    BACKENDS.append(
        (
            "etcd3",
            {
                "host": os.getenv("TESTING_ETCD_ADDR").split(":")[0],
                "port": int(os.getenv("TESTING_ETCD_ADDR").split(":")[1]),
            },
        )
    )


@pytest.fixture(params=BACKENDS, ids=[b[0] for b in BACKENDS])
def name_resolve(request):
    """Fixture that provides a name resolve repository for each backend type."""
    backend_type, kwargs = request.param

    # Special handling for NFS backend to use temp directory
    if backend_type == "nfs":
        temp_dir = tempfile.mkdtemp()
        from realhf.base.name_resolve import NfsNameRecordRepository

        repo = NfsNameRecordRepository(temp_dir)
        yield repo
        repo.reset()
        shutil.rmtree(temp_dir)
    elif backend_type == "memory":
        from realhf.base.name_resolve import MemoryNameRecordRepository

        repo = MemoryNameRecordRepository()
        yield repo
        repo.reset()
    elif backend_type == "etcd3":
        from realhf.base.name_resolve import Etcd3NameRecordRepository

        repo = Etcd3NameRecordRepository(**kwargs)
        yield repo
        repo.reset()
    elif backend_type == "ray":
        from realhf.base.name_resolve import RayNameResolveRepository

        repo = RayNameResolveRepository(**kwargs)
        yield repo
        repo.reset()


def test_basic_add_get(name_resolve):
    """Test basic add and get functionality."""
    # Add a new entry
    name_resolve.add("test_key", "test_value")
    assert name_resolve.get("test_key") == "test_value"

    # Test with non-string value (should be converted to string)
    name_resolve.add("test_key_int", 123, replace=True)
    assert name_resolve.get("test_key_int") == "123"


def test_add_with_replace(name_resolve):
    """Test add operation with replace flag."""
    name_resolve.add("test_key", "initial_value")

    # Should fail when replace=False
    with pytest.raises(NameEntryExistsError):
        name_resolve.add("test_key", "new_value", replace=False)

    # Should succeed when replace=True
    name_resolve.add("test_key", "new_value", replace=True)
    assert name_resolve.get("test_key") == "new_value"


def test_delete(name_resolve):
    """Test delete operation."""
    name_resolve.add("test_key", "test_value")
    name_resolve.delete("test_key")

    # Verify deletion
    with pytest.raises(NameEntryNotFoundError):
        name_resolve.get("test_key")

    # Deleting non-existent key should raise
    with pytest.raises(NameEntryNotFoundError):
        name_resolve.delete("non_existent_key")


def test_clear_subtree(name_resolve):
    """Test clearing a subtree of keys."""
    # Create a subtree of keys
    name_resolve.add("test_root/key1", "value1")
    name_resolve.add("test_root/key2", "value2")
    name_resolve.add("test_root/sub/key3", "value3")
    name_resolve.add("other_root/key", "value")

    # Clear the subtree
    name_resolve.clear_subtree("test_root")

    # Verify subtree is gone
    assert name_resolve.get_subtree("test_root") == []
    assert name_resolve.find_subtree("test_root") == []

    # Verify other tree remains
    assert name_resolve.get("other_root/key") == "value"


def test_get_subtree(name_resolve):
    """Test retrieving values from a subtree."""
    name_resolve.add("test_root/key1", "value1")
    name_resolve.add("test_root/key2", "value2")
    name_resolve.add("test_root/sub/key3", "value3")

    values = name_resolve.get_subtree("test_root")
    assert set(values) == {"value1", "value2", "value3"}


def test_find_subtree(name_resolve):
    """Test finding keys in a subtree."""
    name_resolve.add("test_root/key1", "value1")
    name_resolve.add("test_root/key2", "value2")
    name_resolve.add("test_root/sub/key3", "value3")

    keys = name_resolve.find_subtree("test_root")
    assert set(keys) == {"test_root/key1", "test_root/key2", "test_root/sub/key3"}
    assert keys == sorted(keys)  # Should be sorted


def test_add_subentry(name_resolve):
    """Test adding subentries with automatic UUID generation."""
    sub_name = name_resolve.add_subentry("test_root", "sub_value")
    assert sub_name.startswith("test_root/")
    assert len(sub_name.split("/")[-1]) == 8  # UUID part should be 8 chars
    assert name_resolve.get(sub_name) == "sub_value"


def test_wait(name_resolve):
    """Test waiting for a key to appear."""

    def delayed_add():
        time.sleep(0.1)
        name_resolve.add("test_key", "test_value")

    thread = threading.Thread(target=delayed_add, daemon=True)
    thread.start()

    # Should return once key is added
    assert name_resolve.wait("test_key", timeout=2) == "test_value"
    thread.join()

    # Test timeout
    with pytest.raises(TimeoutError):
        name_resolve.wait("non_existent_key", timeout=0.1)


def test_watch_names(name_resolve):
    """Test watching keys for changes."""
    callback_called = False

    def callback():
        nonlocal callback_called
        callback_called = True

    name_resolve.add("test_key", "test_value")
    name_resolve.watch_names("test_key", callback, poll_frequency=0.1)

    # Delete the key to trigger callback
    time.sleep(0.1)  # Ensure watcher is ready
    name_resolve.delete("test_key")

    # Wait for callback
    time.sleep(2)
    assert callback_called


def test_reset(name_resolve):
    """Test reset functionality (cleanup of delete_on_exit keys)."""
    name_resolve.add("test_key1", "value1", delete_on_exit=True)
    name_resolve.add("test_key_no_delete", "value2", delete_on_exit=False)
    name_resolve.reset()

    # Only delete_on_exit=True keys should be removed
    with pytest.raises(NameEntryNotFoundError):
        name_resolve.get("test_key1")
    assert name_resolve.get("test_key_no_delete") == "value2"
    name_resolve.delete("test_key_no_delete")


def test_concurrent_access(name_resolve):
    """Test concurrent access to the same key."""
    name_resolve.add("test_key", "initial_value")

    def modify_value():
        for i in range(5):
            current = name_resolve.get("test_key")
            name_resolve.add(
                "test_key", f"modified_{threading.get_ident()}_{i}", replace=True
            )
            time.sleep(0.01)

    threads = [threading.Thread(target=modify_value) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Final value should be one of the modified values
    final_value = name_resolve.get("test_key")
    assert "modified_" in final_value


def test_path_normalization(name_resolve):
    """Test handling of different path formats."""
    # Test paths with trailing slashes
    name_resolve.add("test_path/", "value1")
    assert name_resolve.get("test_path") == "value1"
    # with pytest.raises(NameEntryNotFoundError):
    assert name_resolve.get("test_path/") == "value1"

    # Test paths with double slashes
    name_resolve.add("test//path", "value2")
    assert name_resolve.get("test//path") == "value2"

    # Test relative paths
    with pytest.raises(NameEntryExistsError):
        name_resolve.add("./test_path", "value3")
    name_resolve.add("./test_path", "value3", replace=True)
    assert name_resolve.get("./test_path") == "value3"


def test_add_with_invalid_inputs(name_resolve):
    """Test add method with invalid inputs."""
    # Test with None name
    with pytest.raises(
        Exception
    ):  # The specific exception type may vary by implementation
        name_resolve.add(None, "value")

    # Test with empty name
    with pytest.raises(Exception):
        name_resolve.add("", "value")

    # Test with None value
    name_resolve.add("test_key", None)
    assert name_resolve.get("test_key") == "None"


def test_long_paths_and_values(name_resolve):
    """Test behavior with very long path names and values."""
    long_name = "a/" * 100 + "key"
    long_value = "x" * 10000

    name_resolve.add(long_name, long_value)
    assert name_resolve.get(long_name) == long_value


def test_special_characters(name_resolve):
    """Test handling of special characters in names and values."""
    special_chars = "!@#$%^&*()_+-=[]{}|;:'\",.<>?`~ "

    # Test special characters in name
    for char in special_chars:
        try:
            name = f"test{char}key"
            name_resolve.add(name, "value")
            assert name_resolve.get(name) == "value"
            name_resolve.delete(name)
        except Exception as e:
            print(f"Failed with character '{char}': {e}")

    # Test special characters in value
    for char in special_chars:
        value = f"test{char}value"
        name_resolve.add(f"key_{char}", value)
        assert name_resolve.get(f"key_{char}") == value


def test_unicode_support(name_resolve):
    """Test support for Unicode characters in names and values."""
    unicode_name = "测试/键"
    unicode_value = "价值"

    name_resolve.add(unicode_name, unicode_value)
    assert name_resolve.get(unicode_name) == unicode_value


def test_stress_concurrent_add_get_delete(name_resolve):
    """Stress test with many concurrent operations."""
    from concurrent.futures import ThreadPoolExecutor

    num_threads = 20
    ops_per_thread = 50

    # Track success/failure counts
    results = {
        "success": 0,
        "failures": 0,
    }

    def worker(thread_id):
        try:
            for i in range(ops_per_thread):
                key = f"concurrent_key_{thread_id}_{i}"
                value = f"value_{thread_id}_{i}"

                # Add the key
                name_resolve.add(key, value)

                # Get and verify the key
                retrieved = name_resolve.get(key)
                assert retrieved == value

                # Delete the key
                name_resolve.delete(key)

                # Verify deletion
                try:
                    name_resolve.get(key)
                    results["failures"] += 1
                except NameEntryNotFoundError:
                    results["success"] += 1
        except Exception as e:
            print(f"Thread {thread_id} failed: {e}")
            results["failures"] += 1

    # Run worker threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        for future in futures:
            future.result()

    # Verify most operations succeeded
    assert (
        results["failures"] <= results["success"] * 0.1
    )  # Allow up to 10% failure rate


def test_add_subentry_uniqueness(name_resolve):
    """Test that add_subentry generates unique names."""
    # Add multiple subentries to the same root
    num_entries = 100
    entries = set()

    for _ in range(num_entries):
        sub_name = name_resolve.add_subentry("test_root", "value")
        entries.add(sub_name)

    # Verify all entries are unique
    assert len(entries) == num_entries


def test_wait_with_concurrent_delete(name_resolve):
    """Test wait behavior when a key is added and then deleted before wait completes."""

    def add_then_delete():
        time.sleep(0.1)
        name_resolve.add("test_wait_key", "test_value")
        time.sleep(1.0)
        name_resolve.delete("test_wait_key")

    thread = threading.Thread(target=add_then_delete, daemon=True)
    thread.start()

    # Wait with a timeout long enough to capture the key
    value = name_resolve.wait("test_wait_key", timeout=3.0, poll_frequency=0.05)
    assert value == "test_value"

    # Wait for the thread to complete
    thread.join()
    time.sleep(0.5)

    # Verify the key was deleted
    with pytest.raises(NameEntryNotFoundError):
        name_resolve.get("test_wait_key")


def test_wait_edge_cases(name_resolve):
    """Test edge cases for the wait method."""
    # Test with invalid timeout values
    with pytest.raises(TimeoutError):
        name_resolve.wait("nonexistent_key", timeout=0)

    # Test with negative timeout (should behave like timeout=None)
    with pytest.raises(TimeoutError):
        name_resolve.wait("nonexistent_key", timeout=-1, poll_frequency=0.01)

    # Test with very small poll frequency
    with pytest.raises(TimeoutError):
        name_resolve.wait("nonexistent_key", timeout=0.1, poll_frequency=0.001)


def test_watch_names_multiple_keys(name_resolve):
    """Test watching multiple keys."""
    callback_count = 0

    def callback():
        nonlocal callback_count
        callback_count += 1

    # Add test keys
    name_resolve.add("watch_key1", "value1")
    name_resolve.add("watch_key2", "value2")

    # Watch both keys
    name_resolve.watch_names(["watch_key1", "watch_key2"], callback, poll_frequency=0.1)

    # Delete one key
    time.sleep(0.2)  # Ensure watcher is ready
    name_resolve.delete("watch_key1")

    # Wait for callback
    time.sleep(0.5)

    # Delete second key
    name_resolve.delete("watch_key2")

    # Wait for callback
    time.sleep(1.0)

    # Callback should have been called exactly once (when the last key is deleted)
    assert callback_count == 1


def test_thread_safety_of_watch_thread_run(name_resolve):
    """Test thread safety of _watch_thread_run."""
    # Mock the get method to simulate race conditions
    original_get = name_resolve.get

    def mock_get(name):
        # First call returns normally, second call raises exception
        mock_get.counter += 1
        if mock_get.counter % 2 == 0:
            raise NameEntryNotFoundError(f"Key not found: {name}")
        return original_get(name)

    mock_get.counter = 0

    # Create a callback function that tracks calls
    callback_called = False

    def callback():
        nonlocal callback_called
        callback_called = True

    # Add a test key
    name_resolve.add("test_key", "test_value")

    # Patch the get method
    with patch.object(name_resolve, "get", side_effect=mock_get):
        # Call _watch_thread_run directly
        name_resolve._watch_thread_run("test_key", callback, 0.1, 1)

    # Verify callback was called
    assert callback_called


def test_keepalive_ttl(name_resolve):
    """Test keepalive_ttl functionality."""
    # Skip if not Etcd3NameRecordRepository, as TTL might only be fully supported there
    if "Etcd3NameRecordRepository" not in name_resolve.__class__.__name__:
        pytest.skip("keepalive_ttl test is specific to Etcd3NameRecordRepository")

    # Add a key with short TTL
    name_resolve.add("ttl_key", "ttl_value", keepalive_ttl=2)

    # Wait for less than the TTL - key should still exist
    time.sleep(1)
    assert name_resolve.get("ttl_key") == "ttl_value"

    # Mock the keep-alive mechanism to simulate failure
    with patch.object(
        name_resolve._client, "refresh_lease", side_effect=Exception("Refresh failed")
    ):
        # Wait longer than the TTL
        time.sleep(3)

        # Key should be gone if TTL is working
        with pytest.raises(NameEntryNotFoundError):
            name_resolve.get("ttl_key")


def test_subentry_with_custom_uuid(name_resolve, monkeypatch):
    """Test add_subentry with a predictable UUID for deterministic testing."""
    # Mock uuid.uuid4 to return a predictable value
    mock_uuid = MagicMock()
    mock_uuid.return_value = "12345678-1234-5678-1234-567812345678"
    monkeypatch.setattr(uuid, "uuid4", mock_uuid)

    # Add a subentry
    sub_name = name_resolve.add_subentry("test_root", "sub_value")

    # Verify the subentry has the expected name
    assert sub_name == "test_root/12345678"
    assert name_resolve.get(sub_name) == "sub_value"


def test_race_condition_in_add(name_resolve):
    """Test race condition when adding the same key concurrently."""
    if isinstance(name_resolve, NfsNameRecordRepository):
        pytest.skip("NFS repo cannot tackle race conditions")

    # Define the number of concurrent threads
    num_threads = 10
    key = "race_condition_key"
    success_count = 0
    failure_count = 0

    def add_with_same_key():
        nonlocal success_count, failure_count
        try:
            name_resolve.add(key, f"value_{threading.get_ident()}", replace=False)
            success_count += 1
        except NameEntryExistsError:
            failure_count += 1

    # Run concurrent add operations
    threads = [threading.Thread(target=add_with_same_key) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify only one thread succeeded
    assert success_count == 1
    assert failure_count == num_threads - 1

    # Verify the key exists
    assert name_resolve.get(key) is not None


def test_find_subtree_with_empty_result(name_resolve):
    """Test find_subtree behavior when no matching keys are found."""
    # Ensure no keys exist with this prefix
    prefix = "nonexistent_prefix"

    # Call find_subtree
    result = name_resolve.find_subtree(prefix)

    # Verify result is an empty list, not None
    assert result == []
    assert isinstance(result, list)


def test_get_subtree_with_empty_result(name_resolve):
    """Test get_subtree behavior when no matching keys are found."""
    # Ensure no keys exist with this prefix
    prefix = "nonexistent_prefix"

    # Call get_subtree
    result = name_resolve.get_subtree(prefix)

    # Verify result is an empty list, not None
    assert result == []
    assert isinstance(result, list)


def test_clear_subtree_with_nonexistent_prefix(name_resolve):
    """Test clear_subtree behavior with a nonexistent prefix."""
    # Ensure no keys exist with this prefix
    prefix = "nonexistent_prefix"

    # Call clear_subtree - should not raise exception
    name_resolve.clear_subtree(prefix)

    # Add a key elsewhere and verify it's not affected
    name_resolve.add("test_key", "test_value")
    assert name_resolve.get("test_key") == "test_value"


def test_nested_subtrees(name_resolve):
    """Test behavior with deeply nested subtrees."""
    # Create a deeply nested subtree
    name_resolve.add("root/level1/level2/level3/key1", "value1")
    name_resolve.add("root/level1/level2/key2", "value2")
    name_resolve.add("root/level1/key3", "value3")

    # Test get_subtree at different levels
    assert set(name_resolve.get_subtree("root")) == {"value1", "value2", "value3"}
    assert set(name_resolve.get_subtree("root/level1/level2")) == {"value1", "value2"}

    # Test find_subtree at different levels
    assert set(name_resolve.find_subtree("root/level1")) == {
        "root/level1/level2/level3/key1",
        "root/level1/level2/key2",
        "root/level1/key3",
    }

    # Clear a subtree
    name_resolve.clear_subtree("root/level1/level2")

    # Verify only the specified subtree was cleared
    with pytest.raises(NameEntryNotFoundError):
        name_resolve.get("root/level1/level2/level3/key1")
    with pytest.raises(NameEntryNotFoundError):
        name_resolve.get("root/level1/level2/key2")
    assert name_resolve.get("root/level1/key3") == "value3"


def test_corner_case_get_same_as_prefix(name_resolve):
    """Test get behavior when a key is both a prefix and a value."""
    # Add entries
    name_resolve.add("prefix", "parent_value")
    name_resolve.add("prefix/child", "child_value")

    # Verify both keys can be retrieved individually
    assert name_resolve.get("prefix") == "parent_value"
    assert name_resolve.get("prefix/child") == "child_value"

    # Verify get_subtree includes both values
    values = name_resolve.get_subtree("prefix")
    assert set(values) == {"parent_value", "child_value"}

    # Verify find_subtree includes both keys
    keys = name_resolve.find_subtree("prefix")
    assert set(keys) == {"prefix", "prefix/child"}


@pytest.mark.skipif(
    os.getenv("TESTING_ETCD_ADDR") is None, reason="ETCD3 not configured"
)
def test_etcd3_specific_features(name_resolve):
    if not isinstance(name_resolve, Etcd3NameRecordRepository):
        pytest.skip("ETCD3 specific test")
    # Test the keepalive thread
    name_resolve.add("test_key", "test_value", keepalive_ttl=2)
    time.sleep(1)  # Wait for the keepalive thread to refresh the lease
    # Ensure the key still exists
    value, _ = name_resolve._client.get("test_key")
    assert value.decode("utf-8") == "test_value"
    time.sleep(2)  # Wait for the lease to expire
    with pytest.raises(NameEntryNotFoundError):
        name_resolve.get("test_key")


@pytest.mark.skipif(
    os.getenv("TESTING_ETCD_ADDR") is not None, reason="NFS specific test"
)
def test_nfs_specific_features(name_resolve):
    """Test features specific to NFS backend."""
    from realhf.base.name_resolve import NfsNameRecordRepository

    if not isinstance(name_resolve, NfsNameRecordRepository):
        pytest.skip("NFS specific test")

    # Test handling of stale file handles
    name_resolve.add("test_key", "test_value")

    original_open = open
    call_count = 0

    def mock_open(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 3:  # Fail first 3 times
            raise OSError(116, "Stale file handle")
        return original_open(*args, **kwargs)

    with patch("builtins.open", mock_open):
        assert name_resolve.get("test_key") == "test_value"
    assert call_count == 4
