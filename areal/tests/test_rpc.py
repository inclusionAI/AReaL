"""
Unit tests for RPC client and server
Using DistributedBatchMemory as implementation of DistributedBatch
"""

import threading
import time
import unittest
from unittest.mock import Mock, patch

import torch
from tensordict import TensorDict

from areal.controller.batch import DistributedBatchMemory
from areal.scheduler.rpc.rpc_client import RPCClient
from areal.scheduler.rpc.rpc_server import (
    EngineRPCServer,
    get_serve_port,
    process_input_to_distributed_batch,
    process_output_to_distributed_batch,
    start_rpc_server,
)


class MockEngine:
    """Mock engine for testing RPC functionality."""

    def __init__(self):
        self.initialized = False
        self.call_count = 0

    def initialize(self, config):
        self.initialized = True
        return {"status": "initialized", "config": config}

    def mock_method(self, *args, **kwargs):
        self.call_count += 1
        return {
            "method_called": True,
            "call_count": self.call_count,
            "args": args,
            "kwargs": kwargs,
        }

    def process_batch(self, batch):
        if hasattr(batch, "__len__"):
            return {"processed": True, "batch_size": len(batch)}
        else:
            return {"processed": True, "batch_size": 1}

    def return_origin_batch(self, batch):
        return batch


class TestRPCProcessing(unittest.TestCase):
    """Test RPC data processing functions"""

    def test_process_input_to_distributed_batch_with_memory_batch(self):
        """Test processing input parameters containing DistributedBatchMemory"""
        # Create DistributedBatchMemory instance
        data = {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "labels": torch.tensor([5, 6, 7, 8]),
            "metadata": ["text1", "text2", "text3", "text4"],
        }
        batch = DistributedBatchMemory.from_dict(data)

        # Test args and kwargs containing DistributedBatchMemory
        args = (batch, "other_arg")
        kwargs = {"batch_param": batch, "other_param": "value"}

        processed_args, processed_kwargs = process_input_to_distributed_batch(
            *args, **kwargs
        )

        # Verify DistributedBatchMemory is converted to dictionary
        self.assertIsInstance(processed_args[0], dict)
        self.assertEqual(processed_args[1], "other_arg")
        self.assertIsInstance(processed_kwargs["batch_param"], dict)
        self.assertEqual(processed_kwargs["other_param"], "value")

        # Verify converted dictionary contains original data
        converted_data = processed_args[0]
        torch.testing.assert_close(converted_data["input_ids"], data["input_ids"])
        torch.testing.assert_close(converted_data["labels"], data["labels"])
        self.assertEqual(converted_data["metadata"], data["metadata"])

    def test_process_input_no_distributed_batch(self):
        """Test processing input that does not contain DistributedBatch"""
        args = ("arg1", "arg2", torch.tensor([1, 2, 3]))
        kwargs = {"param1": "value1", "param2": torch.tensor([4, 5, 6])}

        processed_args, processed_kwargs = process_input_to_distributed_batch(
            *args, **kwargs
        )

        # Should remain unchanged
        self.assertEqual(processed_args, args)
        self.assertEqual(processed_kwargs, kwargs)

    def test_process_output_to_distributed_batch_dict(self):
        """Test converting dictionary output to DistributedBatch"""
        result = {
            "output_ids": torch.tensor([1, 2, 3]),
            "scores": torch.tensor([0.1, 0.2, 0.3]),
            "texts": ["result1", "result2", "result3"],
        }

        processed = process_output_to_distributed_batch(result)

        # Should be converted to DistributedBatchMemory
        self.assertIsInstance(processed, DistributedBatchMemory)

        # Verify data integrity
        processed_data = processed.get_data()
        torch.testing.assert_close(processed_data["output_ids"], result["output_ids"])
        torch.testing.assert_close(processed_data["scores"], result["scores"])
        self.assertEqual(processed_data["texts"], result["texts"])

    def test_process_output_to_distributed_batch_tensordict(self):
        """Test converting TensorDict output to DistributedBatch"""
        tensor_dict = TensorDict(
            {"tensor1": torch.tensor([1, 2, 3]), "tensor2": torch.tensor([4, 5, 6])},
            batch_size=[3],
        )

        processed = process_output_to_distributed_batch(tensor_dict)

        # Should be converted to DistributedBatchMemory
        self.assertIsInstance(processed, DistributedBatchMemory)

        # Verify data integrity
        processed_data = processed.get_data()
        torch.testing.assert_close(processed_data["tensor1"], tensor_dict["tensor1"])
        torch.testing.assert_close(processed_data["tensor2"], tensor_dict["tensor2"])

    def test_process_output_to_distributed_batch_list(self):
        """Test converting list/tuple output to DistributedBatch"""
        result_list = [
            {"id": 1, "value": torch.tensor([0.1])},
            {"id": 2, "value": torch.tensor([0.2])},
            {"id": 3, "value": torch.tensor([0.3])},
        ]

        processed_list = process_output_to_distributed_batch(result_list)
        self.assertIsInstance(processed_list, DistributedBatchMemory)

    def test_process_output_to_distributed_batch_other_types(self):
        """Test processing other types of output"""
        # String
        string_result = "success"
        processed = process_output_to_distributed_batch(string_result)
        self.assertEqual(processed, string_result)

        # Number
        number_result = 42
        processed = process_output_to_distributed_batch(number_result)
        self.assertEqual(processed, number_result)

        # Tensor
        tensor_result = torch.tensor([1, 2, 3])
        processed = process_output_to_distributed_batch(tensor_result)
        self.assertTrue(torch.equal(processed, tensor_result))


class TestGetServePort(unittest.TestCase):
    """Test port configuration functions"""

    def test_get_serve_port_from_args(self):
        """Test getting port from command line arguments"""
        mock_args = Mock()
        mock_args.port = 8080

        with patch.dict("os.environ", {}, clear=True):
            port = get_serve_port(mock_args)
            self.assertEqual(port, 8080)

    def test_get_serve_port_from_env_single_port(self):
        """Test getting single port from PORT_LIST environment variable"""
        mock_args = Mock()
        mock_args.port = 8080

        with patch.dict("os.environ", {"PORT_LIST": "9000"}):
            port = get_serve_port(mock_args)
            self.assertEqual(port, 9000)

    def test_get_serve_port_from_env_multiple_ports(self):
        """Test getting first port from multiple ports in PORT_LIST environment variable"""
        mock_args = Mock()
        mock_args.port = 8080

        with patch.dict("os.environ", {"PORT_LIST": "9000, 9001, 9002"}):
            port = get_serve_port(mock_args)
            self.assertEqual(port, 9000)

    def test_get_serve_port_invalid_env_port(self):
        """Test fallback when PORT_LIST contains invalid ports"""
        mock_args = Mock()
        mock_args.port = 8080

        with patch.dict("os.environ", {"PORT_LIST": "invalid_port, 9001"}):
            port = get_serve_port(mock_args)
            self.assertEqual(port, 8080)

    def test_get_serve_port_empty_env(self):
        """Test fallback when PORT_LIST is empty"""
        mock_args = Mock()
        mock_args.port = 8080

        with patch.dict("os.environ", {"PORT_LIST": ""}):
            port = get_serve_port(mock_args)
            self.assertEqual(port, 8080)

    def test_get_serve_port_whitespace_env(self):
        """Test fallback when PORT_LIST contains only whitespace"""
        mock_args = Mock()
        mock_args.port = 8080

        with patch.dict("os.environ", {"PORT_LIST": "   "}):
            port = get_serve_port(mock_args)
            self.assertEqual(port, 8080)


class TestRPCIntegration(unittest.TestCase):
    """RPC client and server integration tests"""

    @classmethod
    def setUpClass(cls):
        """Set up integration test environment"""
        # Find available port
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            cls.test_port = s.getsockname()[1]

    def setUp(self):
        """Set up for each test"""
        # Reset server state
        EngineRPCServer.engine = None

        # Start server in separate thread
        self.server_thread = threading.Thread(
            target=start_rpc_server, args=(self.test_port,), daemon=True
        )
        self.server_thread.start()

        # Wait for server to start
        time.sleep(0.5)

        # Create client
        self.client = RPCClient()
        self.client.register("test_worker", "127.0.0.1", self.test_port)

    def test_end_to_end_engine_creation_and_call(self):
        """Test complete RPC workflow: create engine and call methods"""
        # Create engine
        mock_engine = MockEngine()
        mock_config = {
            "model_name": "integration_test_model",
            "batch_size": 8,
            "max_length": 128,
        }

        # Create engine through RPC
        result = self.client.create_engine("test_worker", mock_engine, mock_config)
        self.assertEqual(result["status"], "initialized")
        self.assertEqual(result["config"], mock_config)

        # Call engine method through RPC
        result = self.client.call_engine(
            "test_worker",
            "mock_method",
            1,
            test_arg="integration_test",
            config=mock_config,
        )

        # Verify method call success
        self.assertTrue(result["method_called"])
        self.assertEqual(result["call_count"], 1)
        self.assertIn("test_arg", result["kwargs"])
        self.assertEqual(result["kwargs"]["test_arg"], "integration_test")

    def test_end_to_end_with_distributed_batch_memory(self):
        """Test end-to-end flow using DistributedBatchMemory"""
        # Create and register engine
        mock_engine = MockEngine()
        mock_config = {"model_name": "batch_test_model"}

        create_result = self.client.create_engine(
            "test_worker", mock_engine, mock_config
        )
        self.assertEqual(create_result["status"], "initialized")

        # Create DistributedBatchMemory data
        batch_data = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]]),
            "metadata": ["sample1", "sample2", "sample3"],
        }
        batch = DistributedBatchMemory.from_dict(batch_data)

        # Process batch through RPC
        process_result = self.client.call_engine(
            "test_worker", "process_batch", 1, batch
        )

        # Verify batch processing success
        self.assertTrue(process_result["processed"])
        self.assertEqual(process_result["batch_size"], 3)

        # Test tensor processing
        distrubuted_batch_result = self.client.call_engine(
            "test_worker", "return_origin_batch", 1, batch
        )
        self.assertIsInstance(distrubuted_batch_result, DistributedBatchMemory)
        tensor_result = distrubuted_batch_result.get_data()
        self.assertTrue(
            torch.equal(tensor_result["input_ids"], batch_data["input_ids"])
        )
        self.assertTrue(
            torch.equal(tensor_result["attention_mask"], batch_data["attention_mask"])
        )
        self.assertEqual(tensor_result["metadata"], batch_data["metadata"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
