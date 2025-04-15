import threading
import time
from queue import Empty as QueueEmpty
from typing import Any

import pytest

from realhf.system.push_pull_stream import (  # Replace with your actual module name
    ZMQJsonPuller,
    ZMQJsonPusher,
)

# Constants for testing
TEST_PORT = 5557  # Different from default to avoid conflicts
TEST_HOST = "127.0.0.1"
TIMEOUT = 1000  # ms


# Fixtures for clean setup/teardown
@pytest.fixture
def puller():
    """Fixture providing a puller instance"""
    with ZMQJsonPuller(host=TEST_HOST, port=TEST_PORT, default_timeout_ms=TIMEOUT) as p:
        yield p


@pytest.fixture
def pusher():
    """Fixture providing a pusher instance"""
    with ZMQJsonPusher(host=TEST_HOST, port=TEST_PORT) as p:
        yield p


# Helper functions
def run_pusher_in_thread(data: Any, count: int = 1, delay: float = 0.1):
    """Helper to run pusher in a separate thread"""

    def pusher_thread():
        with ZMQJsonPusher(host=TEST_HOST, port=TEST_PORT) as p:
            for _ in range(count):
                p.push(data)
                time.sleep(delay)

    thread = threading.Thread(target=pusher_thread)
    thread.start()
    return thread


## Test Cases


def test_basic_push_pull(pusher, puller):
    """Test basic push-pull functionality with simple data"""
    test_data = {"key": "value", "num": 42, "flag": True}

    pusher.push(test_data)
    time.sleep(1)
    received = puller.pull()

    assert received == test_data


def test_empty_queue_raises(puller):
    """Test that pulling from empty queue raises QueueEmpty"""
    with pytest.raises(QueueEmpty):
        puller.pull()


def test_push_non_json_data(pusher):
    """Test that pushing non-JSON data raises TypeError"""

    class NonJSON:
        pass

    with pytest.raises(TypeError):
        pusher.push(NonJSON())

    with pytest.raises(TypeError):
        pusher.push(b"binary data")


def test_push_large_data(pusher, puller):
    """Test pushing and pulling large JSON data"""
    large_data = {
        "array": list(range(1000)),
        "nested": {"deep": [{"deeper": True} for _ in range(100)]},
    }

    pusher.push(large_data)
    received = puller.pull()

    assert received == large_data


def test_pull_with_custom_timeout(puller):
    """Test pull() with custom timeout parameter"""
    # Test with very short timeout (should raise immediately)
    start_time = time.time()
    with pytest.raises(QueueEmpty, match="No data available after 10ms timeout"):
        puller.pull(timeout_ms=10)
    elapsed = time.time() - start_time
    assert elapsed < 0.05  # Should be much less than 10ms

    # Test with longer timeout while data is coming
    test_data = {"test": "timeout"}
    thread = run_pusher_in_thread(test_data, delay=0.2)

    # Should get data within 500ms timeout
    start_time = time.time()
    received = puller.pull(timeout_ms=500)
    elapsed = time.time() - start_time

    assert received == test_data
    assert elapsed < 0.5  # Should complete before timeout
    thread.join()


def test_pull_timeout_none(puller):
    """Test that pull(timeout_ms=None) uses default timeout"""
    start_time = time.time()
    with pytest.raises(QueueEmpty):
        puller.pull(timeout_ms=None)  # Should use default 1000ms

    elapsed = time.time() - start_time
    assert abs(elapsed - 1.0) < 0.1  # Approximately default timeout


def test_pull_timeout_zero(puller):
    """Test that pull(timeout_ms=0) returns immediately"""
    start_time = time.time()
    with pytest.raises(QueueEmpty, match="No data available after 0ms timeout"):
        puller.pull(timeout_ms=0)
    elapsed = time.time() - start_time
    assert elapsed < 0.01  # Should be nearly instantaneous


def test_mixed_timeout_usage(pusher, puller):
    """Test mixing different timeout values in successive calls"""
    # First with default timeout (should wait)
    start_time = time.time()
    with pytest.raises(QueueEmpty):
        puller.pull()
    assert abs((time.time() - start_time) - 1.0) < 0.1

    # Then with short timeout
    start_time = time.time()
    with pytest.raises(QueueEmpty):
        puller.pull(timeout_ms=100)
    assert abs((time.time() - start_time) - 0.1) < 0.05

    # Then push some data
    pusher.push({"value": 42})

    # Should get it even with very long timeout
    start_time = time.time()
    assert puller.pull(timeout_ms=5000) == {"value": 42}
    assert (time.time() - start_time) < 0.1  # Should return immediately


def test_timeout_restoration_after_error(puller):
    """Test that default timeout is restored after error"""
    original_timeout = puller.default_timeout_ms

    # First verify default behavior
    with pytest.raises(QueueEmpty):
        puller.pull()

    # Change timeout temporarily
    with pytest.raises(QueueEmpty):
        puller.pull(timeout_ms=100)

    # Verify default timeout is restored
    start_time = time.time()
    with pytest.raises(QueueEmpty):
        puller.pull()
    elapsed = time.time() - start_time
    assert abs(elapsed - (original_timeout / 1000)) < 0.1


def test_concurrent_access(puller):
    """Test that puller can handle concurrent pushes"""
    test_data = {"message": "hello"}
    thread = run_pusher_in_thread(test_data)

    received = puller.pull()
    thread.join()

    assert received == test_data


def test_multiple_messages(pusher, puller):
    """Test sending and receiving multiple messages"""
    messages = [
        {"id": 1, "content": "first"},
        {"id": 2, "content": "second"},
        {"id": 3, "content": "third"},
    ]

    for msg in messages:
        pusher.push(msg)

    for expected in messages:
        assert puller.pull() == expected


def test_rapid_fire_messages(puller):
    """Test handling of many rapid messages"""
    test_data = {"count": 0}
    thread = run_pusher_in_thread(test_data, count=100, delay=0.01)

    received_count = 0
    while True:
        try:
            puller.pull()
            received_count += 1
            if received_count >= 100:
                break
        except QueueEmpty:
            if not thread.is_alive():
                break

    thread.join()
    assert received_count == 100
