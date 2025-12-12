"""Integration tests for RPC serialization with RTensor."""

import subprocess
import sys
import time

import pytest
import requests
import torch

from areal.scheduler.rpc.rtensor import BatchLayout, RTensor, ShardId, ShardLayout
from areal.scheduler.rpc.serialization import (
    deserialize_value,
    serialize_value,
)
from areal.utils.proc import kill_process_tree

RPC_SERVER_PORT = 8077


@pytest.fixture
def rpc_server():
    proc = subprocess.Popen(
        [
            "python3",
            "-m",
            "areal.scheduler.rpc.rpc_server",
            "--host",
            "localhost",
            "--port",
            str(RPC_SERVER_PORT),
        ],
        stdout=sys.stdout,
        stderr=sys.stdout,
    )
    while True:
        try:
            resp = requests.get(f"http://localhost:{RPC_SERVER_PORT}/health")
            if resp.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)
    yield
    kill_process_tree(proc.pid)


def test_rtensor_serialization_roundtrip():
    """Test RTensor can be serialized and deserialized."""
    # Create a simple RTensor with single shard
    tensor = torch.randn(10, 5)
    task_id = "task123"
    key = "test_key"
    node_addr = f"localhost:{RPC_SERVER_PORT}"

    rtensor = RTensor.from_tensor(tensor, task_id=task_id, key=key, node_addr=node_addr)

    # Serialize
    serialized = serialize_value(rtensor)

    # Deserialize without fetching
    deserialized = deserialize_value(serialized, fetch_remote=False)

    assert isinstance(deserialized, RTensor)
    assert len(deserialized.shards) == 1
    assert deserialized.shards[0].shard_id.task_id == task_id
    assert deserialized.shards[0].shard_id.key == key
    assert deserialized.shards[0].node_addr == node_addr
    assert deserialized.shards[0].shape == list(tensor.shape)
    assert deserialized.shards[0].dtype == str(tensor.dtype)


def test_rtensor_batched_serialization():
    """Test RTensor with batched layout serialization."""
    # Create a batched tensor
    batch_tensor = torch.randn(15, 8)

    # Create batch layout
    layout = BatchLayout(
        layout=[
            ShardLayout(shard_id=ShardId(task_id="task1", key="key1"), size=5),
            ShardLayout(shard_id=ShardId(task_id="task2", key="key1"), size=7),
            ShardLayout(shard_id=ShardId(task_id="task3", key="key1"), size=3),
        ]
    )

    rtensor = RTensor.from_batched(
        batch_tensor,
        layout=layout,
        key="output",
        node_addr=f"localhost:{RPC_SERVER_PORT}",
    )

    # Serialize and deserialize
    serialized = serialize_value(rtensor)
    deserialized = deserialize_value(serialized, fetch_remote=False)

    assert isinstance(deserialized, RTensor)
    assert len(deserialized.shards) == 3
    assert deserialized.shards[0].shard_id.task_id == "task1"
    assert deserialized.shards[1].shard_id.task_id == "task2"
    assert deserialized.shards[2].shard_id.task_id == "task3"


def test_deserialize_with_layout_extraction_dict():
    """Test layout extraction from dict structures."""
    # Create a dict with RTensors
    tensor1 = torch.randn(5, 3)
    tensor2 = torch.randn(7, 4)

    rtensor1 = RTensor.from_tensor(
        tensor1, task_id="task1", key="key1", node_addr=f"localhost:{RPC_SERVER_PORT}"
    )
    rtensor2 = RTensor.from_tensor(
        tensor2, task_id="task2", key="key2", node_addr=f"localhost:{RPC_SERVER_PORT}"
    )

    data = {"a": rtensor1, "b": rtensor2, "c": 42}

    # Serialize
    serialized = serialize_value(data)

    # Deserialize with layout extraction
    deserialized_data, layouts = deserialize_value(
        serialized, fetch_remote=False, return_layout=True
    )

    # Check deserialized data
    assert isinstance(deserialized_data, dict)
    assert isinstance(deserialized_data["a"], RTensor)
    assert isinstance(deserialized_data["b"], RTensor)
    assert deserialized_data["c"] == 42

    # Check layouts
    assert isinstance(layouts, dict)
    assert isinstance(layouts["a"], BatchLayout)
    assert isinstance(layouts["b"], BatchLayout)
    assert layouts["c"] is None

    # Verify layout content
    assert len(layouts["a"].layout) == 1
    assert layouts["a"].layout[0].shard_id.task_id == "task1"
    assert layouts["a"].layout[0].size == 5


def test_deserialize_with_layout_extraction_list():
    """Test layout extraction from list structures."""
    # Create a list with RTensors
    tensor1 = torch.randn(3, 2)
    tensor2 = torch.randn(4, 5)

    rtensor1 = RTensor.from_tensor(
        tensor1, task_id="task1", key="key1", node_addr=f"localhost:{RPC_SERVER_PORT}"
    )
    rtensor2 = RTensor.from_tensor(
        tensor2, task_id="task2", key="key2", node_addr=f"localhost:{RPC_SERVER_PORT}"
    )

    data = [rtensor1, rtensor2, "string", None]

    # Serialize
    serialized = serialize_value(data)

    # Deserialize with layout extraction
    deserialized_data, layouts = deserialize_value(
        serialized, fetch_remote=False, return_layout=True
    )

    # Check deserialized data
    assert isinstance(deserialized_data, list)
    assert len(deserialized_data) == 4
    assert isinstance(deserialized_data[0], RTensor)
    assert isinstance(deserialized_data[1], RTensor)
    assert deserialized_data[2] == "string"
    assert deserialized_data[3] is None

    # Check layouts
    assert isinstance(layouts, list)
    assert len(layouts) == 4
    assert isinstance(layouts[0], BatchLayout)
    assert isinstance(layouts[1], BatchLayout)
    assert layouts[2] is None
    assert layouts[3] is None


def test_fetch_remote_and_return_layout_independence(rpc_server):
    """Test fetch_remote and return_layout work independently."""
    tensor = torch.randn(6, 3)
    rtensor = RTensor.from_tensor(
        tensor, task_id="task1", key="key1", node_addr=f"localhost:{RPC_SERVER_PORT}"
    )

    # Put data in RPC server
    requests.put(
        f"http://localhost:{RPC_SERVER_PORT}/data/task1:key1",
        json=serialize_value(tensor),
    )

    # Ensure RTensor has cached data
    assert rtensor._data is not None

    serialized = serialize_value(rtensor)

    # Case 1: fetch_remote=False, return_layout=False
    result1 = deserialize_value(serialized, fetch_remote=False, return_layout=False)
    assert isinstance(result1, RTensor)

    # Case 2: fetch_remote=False, return_layout=True
    result2_data, result2_layout = deserialize_value(
        serialized, fetch_remote=False, return_layout=True
    )
    assert isinstance(result2_data, RTensor)
    assert isinstance(result2_layout, BatchLayout)

    # Case 3: fetch_remote=True, return_layout=False (would normally fetch via HTTP, but with cached data)
    # Since RTensor has cached _data, to_local() will use it
    result3 = deserialize_value(serialized, fetch_remote=True, return_layout=False)
    assert isinstance(result3, torch.Tensor)
    assert result3.shape == tensor.shape

    # Case 4: fetch_remote=True, return_layout=True
    result4_data, result4_layout = deserialize_value(
        serialized, fetch_remote=True, return_layout=True
    )
    assert isinstance(result4_data, torch.Tensor)
    assert isinstance(result4_layout, BatchLayout)
    assert result4_data.shape == tensor.shape


def test_batch_layout_find_in_structure():
    """Test BatchLayout.find_in_structure utility method."""
    layout1 = BatchLayout(
        layout=[ShardLayout(shard_id=ShardId(task_id="t1", key="k1"), size=5)]
    )
    layout2 = BatchLayout(
        layout=[ShardLayout(shard_id=ShardId(task_id="t2", key="k2"), size=3)]
    )

    # Test finding in dict
    data = {"a": {"b": layout1}, "c": 42}
    found = BatchLayout.find_in_structure(data)
    assert found == layout1

    # Test finding in list
    data = [None, [layout2, "text"]]
    found = BatchLayout.find_in_structure(data)
    assert found == layout2

    # Test not finding
    data = {"a": 1, "b": [2, 3]}
    found = BatchLayout.find_in_structure(data)
    assert found is None

    # Test exists_in_structure
    assert BatchLayout.exists_in_structure({"x": layout1})
    assert not BatchLayout.exists_in_structure({"x": 123})


def test_rtensor_from_engine_output_individual():
    """Test RTensor.from_engine_output with individual (non-batched) tensor."""
    from threading import Lock

    storage = {}
    lock = Lock()

    # No layout in input
    input_layouts = {"args": None, "kwargs": None}

    # Simple tensor result
    result = torch.randn(8, 4)

    converted = RTensor.from_engine_output(
        result,
        key="result",
        input_layouts=input_layouts,
        task_id="task123",
        node_addr=f"localhost:{RPC_SERVER_PORT}",
        storage_dict=storage,
        storage_lock=lock,
    )

    assert isinstance(converted, RTensor)
    assert len(converted.shards) == 1
    assert converted.shards[0].shard_id.task_id == "task123"
    assert converted.shards[0].shard_id.key == "result"

    # Check storage
    assert len(storage) == 1
    shard_id = converted.shards[0].shard_id
    assert shard_id in storage
    assert torch.equal(storage[shard_id], result.cpu())


def test_rtensor_from_engine_output_batched():
    """Test RTensor.from_engine_output with batched tensor."""
    from threading import Lock

    storage = {}
    lock = Lock()

    # Create layout in input
    layout = BatchLayout(
        layout=[
            ShardLayout(shard_id=ShardId(task_id="t1", key="k1"), size=3),
            ShardLayout(shard_id=ShardId(task_id="t2", key="k1"), size=5),
        ]
    )
    input_layouts = {"args": None, "kwargs": {"input": layout}}

    # Batched tensor result (total size = 3 + 5 = 8)
    result = torch.randn(8, 4)

    converted = RTensor.from_engine_output(
        result,
        key="result",
        input_layouts=input_layouts,
        task_id=None,
        node_addr=f"localhost:{RPC_SERVER_PORT}",
        storage_dict=storage,
        storage_lock=lock,
    )

    assert isinstance(converted, RTensor)
    assert len(converted.shards) == 2
    assert converted.shards[0].shard_id.task_id == "t1"
    assert converted.shards[1].shard_id.task_id == "t2"

    # Check storage
    assert len(storage) == 2


def test_rtensor_from_engine_output_nested():
    """Test RTensor.from_engine_output with nested structures."""
    from threading import Lock

    storage = {}
    lock = Lock()

    input_layouts = {"args": None, "kwargs": None}

    # Nested result with tensors
    result = {
        "logits": torch.randn(5, 10),
        "hidden": torch.randn(5),
        "metadata": {"count": 5, "name": "test"},
    }

    converted = RTensor.from_engine_output(
        result,
        input_layouts=input_layouts,
        key=None,
        task_id="task123",
        node_addr=f"localhost:{RPC_SERVER_PORT}",
        storage_dict=storage,
        storage_lock=lock,
    )

    # Check structure
    assert isinstance(converted, dict)
    assert isinstance(converted["logits"], RTensor)
    assert isinstance(converted["hidden"], RTensor)
    assert converted["metadata"]["count"] == 5
    assert converted["metadata"]["name"] == "test"

    # Check storage (3 tensors)
    assert len(storage) == 2
