"""Test for abort loop condition bug fix in RemoteInfEngine.agenerate.

This test verifies the fix for the bug where the generation loop would
prematurely exit when stop_reason='abort' because:

BUG (before fix):
- Loop condition: `len(accumulated_output_tokens) < gconfig.max_new_tokens`
- Problem: `gconfig.max_new_tokens` gets decremented in the loop
- Example: original_max=16384, after 12000 tokens -> gconfig.max_new_tokens=4384
- Condition: 12000 < 4384 = False -> loop exits prematurely with abort

FIX:
- Save original max_new_tokens before the loop
- Use `original_max_new_tokens` in the condition instead of `gconfig.max_new_tokens`
- This ensures the loop continues until we actually reach the generation limit
"""

from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from areal.api.cli_args import GenerationHyperparameters, InferenceEngineConfig
from areal.api.io_struct import HttpGenerationResult, HttpRequest, ModelRequest
from areal.core.remote_inf_engine import RemoteInfEngine
from areal.utils import logging


@dataclass
class MockBackend:
    """Mock backend that returns predefined responses."""

    responses: list[dict] = field(default_factory=list)
    call_count: int = 0

    def build_generation_request(
        self, req: ModelRequest, with_lora: bool
    ) -> HttpRequest:
        """Build a mock HTTP request."""
        return HttpRequest(endpoint="/generate", payload={}, method="POST")

    def parse_generation_response(self, response: dict) -> HttpGenerationResult:
        """Return predefined responses in sequence."""
        resp_data = self.responses[self.call_count]
        self.call_count += 1
        return HttpGenerationResult(
            output_tokens=resp_data["output_tokens"],
            output_logprobs=resp_data.get(
                "output_logprobs", [0.0] * len(resp_data["output_tokens"])
            ),
            stop_reason=resp_data["stop_reason"],
        )

    def get_health_check_request(self) -> HttpRequest:
        return HttpRequest(endpoint="/health", payload={}, method="GET")


@dataclass
class MockWorkflowExecutor:
    """Mock workflow executor that is never paused."""

    def is_paused(self) -> bool:
        return False


class TestAbortLoopConditionWithRealEngine:
    """Tests for the abort loop condition fix using real RemoteInfEngine."""

    @pytest.fixture
    def mock_backend_responses(self):
        """Factory fixture to create mock backend with predefined responses."""

        def _create(responses: list[dict]) -> MockBackend:
            return MockBackend(responses=responses)

        return _create

    @pytest.fixture
    def engine_config(self):
        """Create a minimal engine config for testing."""
        return InferenceEngineConfig(
            experiment_name="test_abort_loop",
            trial_name="test",
            request_timeout=30,
            request_retries=1,
        )

    def _create_engine(
        self, config: InferenceEngineConfig, backend: MockBackend
    ) -> RemoteInfEngine:
        """Create a RemoteInfEngine with mock backend."""
        engine = RemoteInfEngine(config=config, backend=backend)
        # Set up minimal state needed for agenerate
        engine.addresses = ["localhost:8000"]
        engine._version = 0
        engine.lora_initialized = False
        # Initialize logger (normally done in initialize())
        engine.logger = logging.getLogger("[Test Remote Inference Engine]")
        # Mock the workflow_executor
        engine._workflow_executor = MockWorkflowExecutor()
        return engine

    @pytest.mark.asyncio
    async def test_abort_should_continue_loop_with_fix(
        self, engine_config, mock_backend_responses
    ):
        """Test that abort continues the loop when using original_max_new_tokens.

        This test simulates the FIXED behavior:
        1. First generation returns 100 tokens with stop_reason='abort'
        2. Loop should continue because 100 < 200 (original max_new_tokens)
        3. Second generation returns 50 tokens with stop_reason='stop'
        4. Loop exits normally

        Total tokens: 150, which is less than original max (200).
        """
        responses = [
            {"output_tokens": list(range(100)), "stop_reason": "abort"},
            {"output_tokens": list(range(50)), "stop_reason": "stop"},
        ]
        backend = mock_backend_responses(responses)
        engine = self._create_engine(engine_config, backend)

        gconfig = GenerationHyperparameters(
            max_new_tokens=200,
            max_tokens=10000,
            n_samples=1,
        )
        req = ModelRequest(
            input_ids=[1, 2, 3],
            gconfig=gconfig,
        )

        # Mock arequest_with_retry to return raw response dict
        async def mock_arequest(*args, **kwargs):
            return {}  # Backend.parse_generation_response handles the actual data

        with patch(
            "areal.core.remote_inf_engine.arequest_with_retry",
            side_effect=mock_arequest,
        ):
            response = await engine.agenerate(req)

        # Verify the fix works: both responses were processed
        assert backend.call_count == 2, (
            f"With fix, backend should be called 2 times (abort then stop), "
            f"got {backend.call_count}"
        )
        assert len(response.output_tokens) == 150, (
            f"Should have 150 tokens total (100 + 50), got {len(response.output_tokens)}"
        )
        assert response.stop_reason == "stop", (
            f"Final stop_reason should be 'stop', got {response.stop_reason}"
        )

    @pytest.mark.asyncio
    async def test_abort_multiple_times_continues_with_fix(
        self, engine_config, mock_backend_responses
    ):
        """Test that multiple abort responses continue the loop with fix.

        Scenario:
        1. First call: 50 tokens, abort
        2. Second call: 60 tokens, abort
        3. Third call: 70 tokens, stop

        Total: 180 tokens < 300 max_new_tokens
        """
        responses = [
            {"output_tokens": list(range(50)), "stop_reason": "abort"},
            {"output_tokens": list(range(60)), "stop_reason": "abort"},
            {"output_tokens": list(range(70)), "stop_reason": "stop"},
        ]
        backend = mock_backend_responses(responses)
        engine = self._create_engine(engine_config, backend)

        gconfig = GenerationHyperparameters(
            max_new_tokens=300,
            max_tokens=10000,
            n_samples=1,
        )
        req = ModelRequest(input_ids=[1, 2, 3], gconfig=gconfig)

        async def mock_arequest(*args, **kwargs):
            return {}

        with patch(
            "areal.core.remote_inf_engine.arequest_with_retry",
            side_effect=mock_arequest,
        ):
            response = await engine.agenerate(req)

        assert backend.call_count == 3, (
            f"Should have 3 backend calls, got {backend.call_count}"
        )
        assert len(response.output_tokens) == 180, (
            f"Should have 180 tokens, got {len(response.output_tokens)}"
        )
        assert response.stop_reason == "stop"

    @pytest.mark.asyncio
    async def test_abort_reaches_length_limit_with_fix(
        self, engine_config, mock_backend_responses
    ):
        """Test that abort loop exits when reaching length limit.

        Scenario with fix:
        - max_new_tokens = 250
        - Each call returns 80 tokens with abort (staying within remaining limit)
        - After 3 calls: 240 tokens, remaining = 10
        - 4th call: returns 80 tokens with abort, but 240 < 250, so we enter loop
        - After 4th call: 320 tokens >= 250, exit loop

        Note: The implementation asserts req.gconfig.max_new_tokens >= 0 after each
        generation, so each generation must not exceed the remaining max_new_tokens.
        In practice, the server respects max_new_tokens, but we need to be careful
        in tests.
        """
        responses = [
            {"output_tokens": list(range(80)), "stop_reason": "abort"},
            {"output_tokens": list(range(80)), "stop_reason": "abort"},
            {"output_tokens": list(range(80)), "stop_reason": "abort"},
            {
                "output_tokens": list(range(10)),
                "stop_reason": "abort",
            },  # This will hit remaining limit
            # This 5th response won't be reached
            {"output_tokens": list(range(10)), "stop_reason": "stop"},
        ]
        backend = mock_backend_responses(responses)
        engine = self._create_engine(engine_config, backend)

        gconfig = GenerationHyperparameters(
            max_new_tokens=250,
            max_tokens=10000,
            n_samples=1,
        )
        req = ModelRequest(input_ids=[1, 2, 3], gconfig=gconfig)

        async def mock_arequest(*args, **kwargs):
            return {}

        with patch(
            "areal.core.remote_inf_engine.arequest_with_retry",
            side_effect=mock_arequest,
        ):
            response = await engine.agenerate(req)

        # 80 + 80 + 80 + 10 = 250 tokens
        assert backend.call_count == 4, (
            f"Should have 4 backend calls, got {backend.call_count}"
        )
        assert len(response.output_tokens) == 250, (
            f"Should have 250 tokens, got {len(response.output_tokens)}"
        )
        # When exiting due to length limit with abort, stop_reason is converted to "length"
        assert response.stop_reason == "length", (
            f"Final stop_reason should be 'length' (converted from abort), got {response.stop_reason}"
        )

    @pytest.mark.asyncio
    async def test_tool_calls_exits_immediately(
        self, engine_config, mock_backend_responses
    ):
        """Test that tool_calls stop_reason exits the loop immediately."""
        responses = [
            {"output_tokens": list(range(50)), "stop_reason": "tool_calls"},
            # This won't be reached
            {"output_tokens": list(range(50)), "stop_reason": "stop"},
        ]
        backend = mock_backend_responses(responses)
        engine = self._create_engine(engine_config, backend)

        gconfig = GenerationHyperparameters(
            max_new_tokens=200,
            max_tokens=10000,
            n_samples=1,
        )
        req = ModelRequest(input_ids=[1, 2, 3], gconfig=gconfig)

        async def mock_arequest(*args, **kwargs):
            return {}

        with patch(
            "areal.core.remote_inf_engine.arequest_with_retry",
            side_effect=mock_arequest,
        ):
            response = await engine.agenerate(req)

        assert backend.call_count == 1
        assert response.stop_reason == "tool_calls"
        assert len(response.output_tokens) == 50

    @pytest.mark.asyncio
    async def test_length_exits_immediately(
        self, engine_config, mock_backend_responses
    ):
        """Test that length stop_reason exits the loop immediately."""
        responses = [
            {"output_tokens": list(range(50)), "stop_reason": "length"},
            # This won't be reached
            {"output_tokens": list(range(50)), "stop_reason": "stop"},
        ]
        backend = mock_backend_responses(responses)
        engine = self._create_engine(engine_config, backend)

        gconfig = GenerationHyperparameters(
            max_new_tokens=200,
            max_tokens=10000,
            n_samples=1,
        )
        req = ModelRequest(input_ids=[1, 2, 3], gconfig=gconfig)

        async def mock_arequest(*args, **kwargs):
            return {}

        with patch(
            "areal.core.remote_inf_engine.arequest_with_retry",
            side_effect=mock_arequest,
        ):
            response = await engine.agenerate(req)

        assert backend.call_count == 1
        assert response.stop_reason == "length"
        assert len(response.output_tokens) == 50

    @pytest.mark.asyncio
    async def test_stop_exits_immediately(self, engine_config, mock_backend_responses):
        """Test that stop stop_reason exits the loop immediately."""
        responses = [
            {"output_tokens": list(range(30)), "stop_reason": "stop"},
            # This won't be reached
            {"output_tokens": list(range(50)), "stop_reason": "stop"},
        ]
        backend = mock_backend_responses(responses)
        engine = self._create_engine(engine_config, backend)

        gconfig = GenerationHyperparameters(
            max_new_tokens=200,
            max_tokens=10000,
            n_samples=1,
        )
        req = ModelRequest(input_ids=[1, 2, 3], gconfig=gconfig)

        async def mock_arequest(*args, **kwargs):
            return {}

        with patch(
            "areal.core.remote_inf_engine.arequest_with_retry",
            side_effect=mock_arequest,
        ):
            response = await engine.agenerate(req)

        assert backend.call_count == 1
        assert response.stop_reason == "stop"
        assert len(response.output_tokens) == 30


class TestBugReproduction:
    """Tests that reproduce the original bug behavior for documentation.

    These tests simulate what WOULD happen with the buggy code to demonstrate
    the issue that was fixed. They don't use the actual RemoteInfEngine since
    the bug is already fixed, but they document the expected buggy behavior.
    """

    def test_abort_prematurely_exits_without_fix_simulation(self):
        """Simulate the BUGGY behavior (before fix) for documentation.

        This test simulates what would happen WITHOUT the fix:
        1. First generation returns 100 tokens with stop_reason='abort'
        2. gconfig.max_new_tokens is decremented: 200 - 100 = 100
        3. Loop condition: 100 < 100 = False -> loop exits!
        4. Log shows: "Finish generation loop with stop_reason=abort"

        This documents the bug that was fixed.
        """
        # Simulate the BUGGY loop logic
        gconfig = GenerationHyperparameters(
            max_new_tokens=200,
            n_samples=1,
        )

        accumulated_output_tokens = []
        stop_reason = None
        # BUG: NOT saving original_max_new_tokens

        # Simulate first iteration
        gen_result_tokens = list(range(100))
        stop_reason = "abort"
        accumulated_output_tokens.extend(gen_result_tokens)
        gconfig.max_new_tokens -= len(gen_result_tokens)  # Now 100

        # Check if loop would continue (BUGGY condition)
        buggy_condition = (
            stop_reason not in ["stop", "tool_calls", "length"]
            and len(accumulated_output_tokens)
            < gconfig.max_new_tokens  # 100 < 100 = False!
        )

        # Verify BUGGY behavior - loop would exit prematurely
        assert buggy_condition is False, (
            "With bug, the loop condition should be False (100 < 100 = False)"
        )
        assert stop_reason == "abort", (
            "With bug, stop_reason remains 'abort' - this is the bug symptom"
        )
        assert len(accumulated_output_tokens) == 100, (
            "With bug, only first batch of tokens is accumulated"
        )

    def test_abort_continues_with_fix_simulation(self):
        """Simulate the FIXED behavior for documentation.

        This test simulates what happens WITH the fix:
        1. Save original_max_new_tokens = 200
        2. First generation returns 100 tokens with stop_reason='abort'
        3. gconfig.max_new_tokens is decremented: 200 - 100 = 100
        4. Loop condition: 100 < 200 (original) = True -> continue!
        """
        gconfig = GenerationHyperparameters(
            max_new_tokens=200,
            n_samples=1,
        )

        accumulated_output_tokens = []
        stop_reason = None
        original_max_new_tokens = gconfig.max_new_tokens  # FIX: save original

        # Simulate first iteration
        gen_result_tokens = list(range(100))
        stop_reason = "abort"
        accumulated_output_tokens.extend(gen_result_tokens)
        gconfig.max_new_tokens -= len(gen_result_tokens)  # Now 100

        # Check if loop would continue (FIXED condition)
        fixed_condition = (
            stop_reason not in ["stop", "tool_calls", "length"]
            and len(accumulated_output_tokens)
            < original_max_new_tokens  # 100 < 200 = True!
        )

        # Verify FIXED behavior - loop continues
        assert fixed_condition is True, (
            "With fix, the loop condition should be True (100 < 200 = True)"
        )


class TestGconfigModification:
    """Tests to verify gconfig.max_new_tokens is modified during the loop."""

    def test_gconfig_max_new_tokens_decremented(self):
        """Verify that gconfig.max_new_tokens is decremented in each iteration."""
        gconfig = GenerationHyperparameters(max_new_tokens=200, n_samples=1)
        original_value = gconfig.max_new_tokens

        # Simulate what happens in the loop
        tokens_generated = 100
        gconfig.max_new_tokens -= tokens_generated

        assert gconfig.max_new_tokens == 100, (
            "gconfig.max_new_tokens should be decremented"
        )
        assert original_value == 200, (
            "Original value should be unchanged if saved separately"
        )

    def test_model_request_copy_creates_new_gconfig(self):
        """Verify that ModelRequest.copy() creates a new gconfig instance."""
        gconfig = GenerationHyperparameters(max_new_tokens=200, n_samples=1)
        req = ModelRequest(input_ids=[1, 2, 3], gconfig=gconfig)

        req_copy = req.copy()

        # Modify the copy
        req_copy.gconfig.max_new_tokens -= 100

        # Original should be unchanged
        assert req.gconfig.max_new_tokens == 200, "Original gconfig should be unchanged"
        assert req_copy.gconfig.max_new_tokens == 100, (
            "Copy's gconfig should be modified"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
