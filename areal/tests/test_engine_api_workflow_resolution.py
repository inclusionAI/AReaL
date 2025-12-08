"""
Integration tests for Engine API workflow resolution changes.
Tests the new workflow parameter types (instance, class, string) across different engine implementations.
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.engine.fsdp_engine import FSDPEngine

MODEL_NAME = "Qwen/Qwen3-0.6B"


class MockRolloutWorkflow(RolloutWorkflow):
    """Mock workflow for testing engine API integration"""

    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 8, **kwargs):
        self.model_name = model_name
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.initialized = True

    async def arun_episode(self, request_data: dict[str, Any]) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "episode_result": "success",
        }


class MockInferenceEngine(InferenceEngine):
    """Mock inference engine for testing workflow resolution"""

    def __init__(self):
        self._workflow_executor = Mock()
        self._workflow_executor.submit = Mock()
        self._workflow_executor.rollout_batch = Mock(return_value={"result": "batch"})
        self._workflow_executor.prepare_batch = Mock(
            return_value={"result": "prepared"}
        )

    @property
    def workflow_executor(self):
        return self._workflow_executor

    def submit(
        self,
        data: dict[str, Any],
        workflow,
        workflow_kwargs=None,
        should_accept_fn=None,
    ):
        """Mock submit that uses WorkflowExecutor logic"""
        # Simulate the resolution logic from the real implementation
        resolved_workflow = self._resolve_workflow(workflow, workflow_kwargs)
        resolved_should_accept = self._resolve_should_accept(should_accept_fn)

        return self.workflow_executor.submit(
            data=data,
            workflow=resolved_workflow,
            should_accept_fn=resolved_should_accept,
        )

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow,
        workflow_kwargs=None,
    ):
        """Mock rollout_batch"""
        resolved_workflow = self._resolve_workflow(workflow, workflow_kwargs)

        return self.workflow_executor.rollout_batch(
            data=data, workflow=resolved_workflow
        )

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow,
        workflow_kwargs=None,
        should_accept_fn=None,
    ):
        """Mock prepare_batch"""
        resolved_workflow = self._resolve_workflow(workflow, workflow_kwargs)
        resolved_should_accept = self._resolve_should_accept(should_accept_fn)

        return self.workflow_executor.prepare_batch(
            dataloader=dataloader,
            workflow=resolved_workflow,
            should_accept_fn=resolved_should_accept,
        )

    def _resolve_workflow(self, workflow, workflow_kwargs):
        """Simplified workflow resolution for testing"""
        if isinstance(workflow, RolloutWorkflow):
            return workflow
        elif isinstance(workflow, type) and issubclass(workflow, RolloutWorkflow):
            if workflow_kwargs is None:
                raise ValueError("workflow_kwargs required for class type")
            return workflow(**workflow_kwargs)
        elif isinstance(workflow, str):
            if workflow_kwargs is None:
                raise ValueError("workflow_kwargs required for string type")
            # Mock the import
            if "MockRolloutWorkflow" in workflow:
                return MockRolloutWorkflow(**workflow_kwargs)
            else:
                raise ValueError(f"Unknown workflow path: {workflow}")
        else:
            raise TypeError(f"Invalid workflow type: {type(workflow)}")

    def _resolve_should_accept(self, should_accept_fn):
        """Simplified should_accept_fn resolution for testing"""
        if should_accept_fn is None or callable(should_accept_fn):
            return should_accept_fn
        elif isinstance(should_accept_fn, str):
            # Mock the import
            if "mock_filter" in should_accept_fn:
                return lambda x: True
            else:
                raise ValueError(f"Unknown filter path: {should_accept_fn}")
        else:
            raise TypeError(f"Invalid should_accept_fn type: {type(should_accept_fn)}")


def mock_trajectory_filter(trajectory: dict[str, Any]) -> bool:
    """Mock trajectory filter function"""
    return trajectory.get("accept", True)


class TestEngineAPIWorkflowResolution:
    """Test workflow resolution in Engine API methods"""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock inference engine for testing"""
        return MockInferenceEngine()

    @pytest.fixture
    def mock_dataloader(self):
        """Create a mock dataloader for testing"""
        dataloader = Mock(spec=StatefulDataLoader)
        return dataloader

    def test_submit_with_workflow_instance(self, mock_engine):
        """Test submit with workflow instance"""
        workflow_instance = MockRolloutWorkflow(model_name="instance_test")
        data = {"test": "data"}

        mock_engine.submit(data=data, workflow=workflow_instance)

        # Verify the workflow executor was called with the instance
        mock_engine.workflow_executor.submit.assert_called_once()
        call_args = mock_engine.workflow_executor.submit.call_args
        assert call_args[1]["workflow"] is workflow_instance

    def test_submit_with_workflow_class(self, mock_engine):
        """Test submit with workflow class and kwargs"""
        kwargs = {"model_name": "class_test", "batch_size": 16}
        data = {"test": "data"}

        mock_engine.submit(
            data=data, workflow=MockRolloutWorkflow, workflow_kwargs=kwargs
        )

        # Verify the workflow executor was called with instantiated workflow
        mock_engine.workflow_executor.submit.assert_called_once()
        call_args = mock_engine.workflow_executor.submit.call_args
        workflow = call_args[1]["workflow"]
        assert isinstance(workflow, MockRolloutWorkflow)
        assert workflow.model_name == "class_test"
        assert workflow.batch_size == 16

    def test_submit_with_workflow_string(self, mock_engine):
        """Test submit with workflow string path and kwargs"""
        workflow_path = "test.MockRolloutWorkflow"
        kwargs = {"model_name": "string_test", "batch_size": 32}
        data = {"test": "data"}

        mock_engine.submit(data=data, workflow=workflow_path, workflow_kwargs=kwargs)

        # Verify the workflow executor was called with instantiated workflow
        mock_engine.workflow_executor.submit.assert_called_once()
        call_args = mock_engine.workflow_executor.submit.call_args
        workflow = call_args[1]["workflow"]
        assert isinstance(workflow, MockRolloutWorkflow)
        assert workflow.model_name == "string_test"
        assert workflow.batch_size == 32

    def test_submit_with_should_accept_callable(self, mock_engine):
        """Test submit with should_accept_fn callable"""
        workflow_instance = MockRolloutWorkflow()
        data = {"test": "data"}

        mock_engine.submit(
            data=data,
            workflow=workflow_instance,
            should_accept_fn=mock_trajectory_filter,
        )

        # Verify the filter was passed through
        mock_engine.workflow_executor.submit.assert_called_once()
        call_args = mock_engine.workflow_executor.submit.call_args
        assert call_args[1]["should_accept_fn"] is mock_trajectory_filter

    def test_submit_with_should_accept_string(self, mock_engine):
        """Test submit with should_accept_fn string path"""
        workflow_instance = MockRolloutWorkflow()
        filter_path = "test.mock_filter"
        data = {"test": "data"}

        mock_engine.submit(
            data=data, workflow=workflow_instance, should_accept_fn=filter_path
        )

        # Verify the filter was resolved and passed through
        mock_engine.workflow_executor.submit.assert_called_once()
        call_args = mock_engine.workflow_executor.submit.call_args
        should_accept_fn = call_args[1]["should_accept_fn"]
        assert callable(should_accept_fn)

    def test_rollout_batch_with_workflow_types(self, mock_engine):
        """Test rollout_batch with different workflow types"""
        data = [{"test": "data1"}, {"test": "data2"}]

        # Test with instance
        workflow_instance = MockRolloutWorkflow()
        result = mock_engine.rollout_batch(data=data, workflow=workflow_instance)
        assert result == {"result": "batch"}

        # Test with class
        kwargs = {"model_name": "batch_test"}
        result = mock_engine.rollout_batch(
            data=data, workflow=MockRolloutWorkflow, workflow_kwargs=kwargs
        )
        assert result == {"result": "batch"}

        # Test with string
        workflow_path = "test.MockRolloutWorkflow"
        result = mock_engine.rollout_batch(
            data=data, workflow=workflow_path, workflow_kwargs=kwargs
        )
        assert result == {"result": "batch"}

    def test_prepare_batch_with_workflow_types(self, mock_engine, mock_dataloader):
        """Test prepare_batch with different workflow types"""
        kwargs = {"model_name": "prepare_test"}

        # Test with class
        result = mock_engine.prepare_batch(
            dataloader=mock_dataloader,
            workflow=MockRolloutWorkflow,
            workflow_kwargs=kwargs,
        )
        assert result == {"result": "prepared"}

        # Test with string
        workflow_path = "test.MockRolloutWorkflow"
        result = mock_engine.prepare_batch(
            dataloader=mock_dataloader, workflow=workflow_path, workflow_kwargs=kwargs
        )
        assert result == {"result": "prepared"}

    def test_error_handling_in_submit(self, mock_engine):
        """Test error handling in submit method"""
        data = {"test": "data"}

        # Test missing workflow_kwargs for class
        with pytest.raises(ValueError, match="workflow_kwargs required"):
            mock_engine.submit(data=data, workflow=MockRolloutWorkflow)

        # Test missing workflow_kwargs for string
        with pytest.raises(ValueError, match="workflow_kwargs required"):
            mock_engine.submit(data=data, workflow="test.MockRolloutWorkflow")

        # Test invalid workflow type
        with pytest.raises(TypeError, match="Invalid workflow type"):
            mock_engine.submit(data=data, workflow=123)

        # Test invalid should_accept_fn type
        workflow_instance = MockRolloutWorkflow()
        with pytest.raises(TypeError, match="Invalid should_accept_fn type"):
            mock_engine.submit(
                data=data, workflow=workflow_instance, should_accept_fn=123
            )

    def test_error_handling_in_rollout_batch(self, mock_engine):
        """Test error handling in rollout_batch method"""
        data = [{"test": "data"}]

        # Test that errors are properly propagated
        with pytest.raises(ValueError, match="workflow_kwargs required"):
            mock_engine.rollout_batch(data=data, workflow=MockRolloutWorkflow)

    def test_error_handling_in_prepare_batch(self, mock_engine, mock_dataloader):
        """Test error handling in prepare_batch method"""
        # Test that errors are properly propagated
        with pytest.raises(ValueError, match="workflow_kwargs required"):
            mock_engine.prepare_batch(
                dataloader=mock_dataloader, workflow=MockRolloutWorkflow
            )


class TestFSDPEngineWorkflowResolution:
    def test_fsdp_engine_method_signatures(self):
        """Test that FSDPEngine methods have the correct signatures"""
        # This test verifies the method signatures match the new API
        import inspect

        # Check rollout_batch signature
        rollout_batch_sig = inspect.signature(FSDPEngine.rollout_batch)
        params = list(rollout_batch_sig.parameters.keys())
        assert "workflow" in params
        assert "workflow_kwargs" in params
        assert "workflow_builder" not in params  # Should be removed

        # Check prepare_batch signature
        prepare_batch_sig = inspect.signature(FSDPEngine.prepare_batch)
        params = list(prepare_batch_sig.parameters.keys())
        assert "workflow" in params
        assert "workflow_kwargs" in params
        assert "should_accept_fn" in params
        assert "workflow_builder" not in params  # Should be removed

    @patch("areal.engine.fsdp_engine.FSDPEngine._check_rollout_engine_connected")
    def test_fsdp_engine_rollout_batch_parameter_passing(self, mock_check):
        """Test that FSDPEngine correctly passes parameters to rollout_coordinator"""
        # Create a mock FSDPEngine with minimal setup
        engine = Mock()
        engine._check_rollout_engine_connected = mock_check
        engine.rollout_coordinator = Mock()
        engine.rollout_coordinator.rollout_batch = Mock(
            return_value={"result": "fsdp_batch"}
        )

        # Bind the real method to our mock
        engine.rollout_batch = FSDPEngine.rollout_batch.__get__(engine, FSDPEngine)

        # Test the method call
        data = [{"test": "data"}]
        workflow = MockRolloutWorkflow()
        workflow_kwargs = {"model_name": "fsdp_test"}

        result = engine.rollout_batch(
            data=data,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
        )

        # Verify parameters were passed correctly
        engine.rollout_coordinator.rollout_batch.assert_called_once_with(
            data,
            granularity=1,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
        )
        assert result == {"result": "fsdp_batch"}

    @patch("areal.engine.fsdp_engine.FSDPEngine._check_rollout_engine_connected")
    def test_fsdp_engine_prepare_batch_parameter_passing(self, mock_check):
        """Test that FSDPEngine correctly passes parameters to rollout_coordinator"""
        # Create a mock FSDPEngine with minimal setup
        engine = Mock()
        engine._check_rollout_engine_connected = mock_check
        engine.rollout_coordinator = Mock()
        engine.rollout_coordinator.prepare_batch = Mock(
            return_value={"result": "fsdp_prepared"}
        )

        # Bind the real method to our mock
        engine.prepare_batch = FSDPEngine.prepare_batch.__get__(engine, FSDPEngine)

        # Test the method call
        dataloader = Mock()
        workflow = "test.MockRolloutWorkflow"
        workflow_kwargs = {"model_name": "fsdp_prepare_test"}
        should_accept_fn = "test.mock_filter"

        result = engine.prepare_batch(
            dataloader=dataloader,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
        )

        # Verify parameters were passed correctly
        engine.rollout_coordinator.prepare_batch.assert_called_once_with(
            dataloader,
            granularity=1,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
        )
        assert result == {"result": "fsdp_prepared"}
