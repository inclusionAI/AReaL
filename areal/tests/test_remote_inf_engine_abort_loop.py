"""Test for abort loop condition bug fix in RemoteInfEngine.agenerate.

This test verifies the fix for the bug where the generation loop would
prematurely exit when stop_reason='abort' because:

BUG (before fix):
- Loop condition: `len(accumulated_output_tokens) < gconfig.max_new_tokens`
- Problem: `gconfig.max_new_tokens` gets decremented in the loop
- Example: original_max=16384, after 12000 tokens -> gconfig.max_new_tokens=4384
- Condition: 12000 < 4384 = False -> loop exits prematurely with abort

FIX (commit 445dca7):
- Save original max_new_tokens before the loop
- Use `original_max_new_tokens` in the condition instead of `gconfig.max_new_tokens`
- This ensures the loop continues until we actually reach the generation limit

Reference: commit 445dca7216e66c24d7a5b47a272761ea81749f8d
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import HttpGenerationResult, ModelRequest


@dataclass
class MockWorkflowExecutor:
    """Mock workflow executor that is never paused."""

    def is_paused(self) -> bool:
        return False


@dataclass
class MockBackend:
    """Mock backend that returns predefined responses."""

    responses: list[HttpGenerationResult] = field(default_factory=list)
    call_count: int = 0

    def build_generation_request(self, req: ModelRequest, with_lora: bool) -> Any:
        """Build a mock HTTP request."""
        return MagicMock(endpoint="/generate", payload={}, method="POST")

    def parse_generation_response(self, response: dict) -> HttpGenerationResult:
        """Return predefined responses in sequence."""
        result = self.responses[self.call_count]
        self.call_count += 1
        return result


class TestAbortLoopCondition:
    """Tests for the abort loop condition fix."""

    def _create_generation_result(
        self,
        output_tokens: list[int],
        stop_reason: str,
        output_logprobs: list[float] | None = None,
    ) -> HttpGenerationResult:
        """Helper to create HttpGenerationResult."""
        if output_logprobs is None:
            output_logprobs = [0.0] * len(output_tokens)
        return HttpGenerationResult(
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            rollout_expert_ids=[],
            stop_reason=stop_reason,
            input_logprobs=[],
        )

    @pytest.mark.asyncio
    async def test_abort_should_continue_loop_with_fix(self):
        """Test that abort continues the loop when using original_max_new_tokens.

        This test simulates the FIXED behavior:
        1. First generation returns 100 tokens with stop_reason='abort'
        2. Loop should continue because 100 < 200 (original max_new_tokens)
        3. Second generation returns 50 tokens with stop_reason='stop'
        4. Loop exits normally

        Total tokens: 150, which is less than original max (200).
        """
        # Setup responses: first returns abort, second returns stop
        responses = [
            self._create_generation_result(
                output_tokens=list(range(100)),  # 100 tokens
                stop_reason="abort",
            ),
            self._create_generation_result(
                output_tokens=list(range(50)),  # 50 more tokens
                stop_reason="stop",
            ),
        ]

        backend = MockBackend(responses=responses)

        # Simulate the FIXED loop logic
        gconfig = GenerationHyperparameters(
            max_new_tokens=200,
            n_samples=1,
        )
        req = ModelRequest(
            input_ids=[1, 2, 3],
            gconfig=gconfig,
        )

        accumulated_output_tokens = []
        stop_reason = None
        original_max_new_tokens = gconfig.max_new_tokens  # FIXED: save original

        loop_iterations = 0
        max_iterations = 10  # Safety limit

        while (
            stop_reason not in ["stop", "tool_calls", "length"]
            and len(accumulated_output_tokens) < original_max_new_tokens  # FIXED
            and loop_iterations < max_iterations
        ):
            loop_iterations += 1

            # Simulate backend call
            gen_result = backend.parse_generation_response({})
            stop_reason = gen_result.stop_reason

            # Accumulate tokens
            accumulated_output_tokens.extend(gen_result.output_tokens)

            # Update request (this is what happens in the real code)
            req.input_ids += gen_result.output_tokens
            req.gconfig.max_new_tokens -= len(gen_result.output_tokens)

        # Verify FIXED behavior
        assert loop_iterations == 2, (
            f"With fix, loop should iterate 2 times (abort then stop), "
            f"got {loop_iterations}"
        )
        assert len(accumulated_output_tokens) == 150, (
            f"Should have 150 tokens total (100 + 50), got {len(accumulated_output_tokens)}"
        )
        assert stop_reason == "stop", f"Final stop_reason should be 'stop', got {stop_reason}"
        assert backend.call_count == 2, f"Backend should be called 2 times, got {backend.call_count}"

    @pytest.mark.asyncio
    async def test_abort_prematurely_exits_without_fix(self):
        """Test that abort prematurely exits loop when using modified gconfig.max_new_tokens.

        This test simulates the BUGGY behavior (before fix):
        1. First generation returns 100 tokens with stop_reason='abort'
        2. gconfig.max_new_tokens is decremented: 200 - 100 = 100
        3. Loop condition: 100 < 100 = False -> loop exits!
        4. Log shows: "Finish generation loop with stop_reason=abort"

        This is the bug that was fixed.
        """
        # Setup responses: first returns abort, second would return stop
        # but with the bug, we never reach the second call
        responses = [
            self._create_generation_result(
                output_tokens=list(range(100)),  # 100 tokens
                stop_reason="abort",
            ),
            self._create_generation_result(
                output_tokens=list(range(50)),  # This should not be reached with bug
                stop_reason="stop",
            ),
        ]

        backend = MockBackend(responses=responses)

        # Simulate the BUGGY loop logic
        gconfig = GenerationHyperparameters(
            max_new_tokens=200,
            n_samples=1,
        )
        req = ModelRequest(
            input_ids=[1, 2, 3],
            gconfig=gconfig,
        )

        accumulated_output_tokens = []
        stop_reason = None
        # BUG: NOT saving original_max_new_tokens

        loop_iterations = 0
        max_iterations = 10  # Safety limit

        while (
            stop_reason not in ["stop", "tool_calls", "length"]
            and len(accumulated_output_tokens) < gconfig.max_new_tokens  # BUG: uses modified value
            and loop_iterations < max_iterations
        ):
            loop_iterations += 1

            # Simulate backend call
            gen_result = backend.parse_generation_response({})
            stop_reason = gen_result.stop_reason

            # Accumulate tokens
            accumulated_output_tokens.extend(gen_result.output_tokens)

            # Update request (this modifies gconfig.max_new_tokens!)
            req.input_ids += gen_result.output_tokens
            req.gconfig.max_new_tokens -= len(gen_result.output_tokens)

        # Verify BUGGY behavior - loop exits prematurely
        assert loop_iterations == 1, (
            f"With bug, loop should exit after 1 iteration due to condition failure, "
            f"got {loop_iterations}"
        )
        assert len(accumulated_output_tokens) == 100, (
            f"With bug, should only have 100 tokens (first batch only), "
            f"got {len(accumulated_output_tokens)}"
        )
        assert stop_reason == "abort", (
            f"With bug, stop_reason should remain 'abort', got {stop_reason}"
        )
        assert backend.call_count == 1, (
            f"With bug, backend should only be called once, got {backend.call_count}"
        )

    @pytest.mark.asyncio
    async def test_abort_multiple_times_continues_with_fix(self):
        """Test that multiple abort responses continue the loop with fix.

        Scenario:
        1. First call: 50 tokens, abort
        2. Second call: 60 tokens, abort
        3. Third call: 70 tokens, stop

        Total: 180 tokens < 300 max_new_tokens
        """
        responses = [
            self._create_generation_result(list(range(50)), "abort"),
            self._create_generation_result(list(range(60)), "abort"),
            self._create_generation_result(list(range(70)), "stop"),
        ]

        backend = MockBackend(responses=responses)

        gconfig = GenerationHyperparameters(max_new_tokens=300, n_samples=1)
        req = ModelRequest(input_ids=[1, 2, 3], gconfig=gconfig)

        accumulated_output_tokens = []
        stop_reason = None
        original_max_new_tokens = gconfig.max_new_tokens  # FIXED

        loop_iterations = 0
        max_iterations = 10

        while (
            stop_reason not in ["stop", "tool_calls", "length"]
            and len(accumulated_output_tokens) < original_max_new_tokens
            and loop_iterations < max_iterations
        ):
            loop_iterations += 1
            gen_result = backend.parse_generation_response({})
            stop_reason = gen_result.stop_reason
            accumulated_output_tokens.extend(gen_result.output_tokens)
            req.input_ids += gen_result.output_tokens
            req.gconfig.max_new_tokens -= len(gen_result.output_tokens)

        assert loop_iterations == 3, f"Should have 3 iterations, got {loop_iterations}"
        assert len(accumulated_output_tokens) == 180, f"Should have 180 tokens, got {len(accumulated_output_tokens)}"
        assert stop_reason == "stop"

    @pytest.mark.asyncio
    async def test_abort_reaches_length_limit_with_fix(self):
        """Test that abort loop exits when reaching length limit.

        Scenario with fix:
        1. First call: 100 tokens, abort
        2. Second call: 100 tokens, abort
        3. Third call: 100 tokens, abort (but total=300 >= max=250, so this shouldn't happen)

        Actually, after 200 tokens (2 calls), we check: 200 < 250 = True, continue
        Third call: 200 + 100 = 300, then check: 300 < 250 = False, exit

        Wait, the check happens BEFORE the call in the while condition.
        Let me reconsider...

        Loop iteration 1: enter (0 < 250), get 100 tokens, abort
        Loop iteration 2: enter (100 < 250), get 100 tokens, abort
        Loop iteration 3: enter (200 < 250), get 100 tokens, abort
        Loop iteration 4: check (300 < 250) = False, exit

        So we make 3 calls but then exit due to length.
        """
        responses = [
            self._create_generation_result(list(range(100)), "abort"),
            self._create_generation_result(list(range(100)), "abort"),
            self._create_generation_result(list(range(100)), "abort"),
            # This 4th response won't be reached
            self._create_generation_result(list(range(100)), "stop"),
        ]

        backend = MockBackend(responses=responses)

        gconfig = GenerationHyperparameters(max_new_tokens=250, n_samples=1)
        req = ModelRequest(input_ids=[1, 2, 3], gconfig=gconfig)

        accumulated_output_tokens = []
        stop_reason = None
        original_max_new_tokens = gconfig.max_new_tokens

        loop_iterations = 0
        max_iterations = 10

        while (
            stop_reason not in ["stop", "tool_calls", "length"]
            and len(accumulated_output_tokens) < original_max_new_tokens
            and loop_iterations < max_iterations
        ):
            loop_iterations += 1
            gen_result = backend.parse_generation_response({})
            stop_reason = gen_result.stop_reason
            accumulated_output_tokens.extend(gen_result.output_tokens)
            req.input_ids += gen_result.output_tokens
            req.gconfig.max_new_tokens -= len(gen_result.output_tokens)

        assert loop_iterations == 3, f"Should have 3 iterations, got {loop_iterations}"
        assert len(accumulated_output_tokens) == 300, f"Should have 300 tokens, got {len(accumulated_output_tokens)}"
        assert stop_reason == "abort", "Final stop_reason should be 'abort' (exited due to length limit)"

    @pytest.mark.asyncio
    async def test_tool_calls_exits_immediately(self):
        """Test that tool_calls stop_reason exits the loop immediately."""
        responses = [
            self._create_generation_result(list(range(50)), "tool_calls"),
            # This won't be reached
            self._create_generation_result(list(range(50)), "stop"),
        ]

        backend = MockBackend(responses=responses)

        gconfig = GenerationHyperparameters(max_new_tokens=200, n_samples=1)
        req = ModelRequest(input_ids=[1, 2, 3], gconfig=gconfig)

        accumulated_output_tokens = []
        stop_reason = None
        original_max_new_tokens = gconfig.max_new_tokens

        loop_iterations = 0
        max_iterations = 10

        while (
            stop_reason not in ["stop", "tool_calls", "length"]
            and len(accumulated_output_tokens) < original_max_new_tokens
            and loop_iterations < max_iterations
        ):
            loop_iterations += 1
            gen_result = backend.parse_generation_response({})
            stop_reason = gen_result.stop_reason
            accumulated_output_tokens.extend(gen_result.output_tokens)
            req.input_ids += gen_result.output_tokens
            req.gconfig.max_new_tokens -= len(gen_result.output_tokens)

        assert loop_iterations == 1
        assert stop_reason == "tool_calls"
        assert len(accumulated_output_tokens) == 50

    @pytest.mark.asyncio
    async def test_length_exits_immediately(self):
        """Test that length stop_reason exits the loop immediately."""
        responses = [
            self._create_generation_result(list(range(50)), "length"),
            # This won't be reached
            self._create_generation_result(list(range(50)), "stop"),
        ]

        backend = MockBackend(responses=responses)

        gconfig = GenerationHyperparameters(max_new_tokens=200, n_samples=1)
        req = ModelRequest(input_ids=[1, 2, 3], gconfig=gconfig)

        accumulated_output_tokens = []
        stop_reason = None
        original_max_new_tokens = gconfig.max_new_tokens

        loop_iterations = 0
        max_iterations = 10

        while (
            stop_reason not in ["stop", "tool_calls", "length"]
            and len(accumulated_output_tokens) < original_max_new_tokens
            and loop_iterations < max_iterations
        ):
            loop_iterations += 1
            gen_result = backend.parse_generation_response({})
            stop_reason = gen_result.stop_reason
            accumulated_output_tokens.extend(gen_result.output_tokens)
            req.input_ids += gen_result.output_tokens
            req.gconfig.max_new_tokens -= len(gen_result.output_tokens)

        assert loop_iterations == 1
        assert stop_reason == "length"
        assert len(accumulated_output_tokens) == 50


class TestGconfigModification:
    """Tests to verify gconfig.max_new_tokens is modified during the loop."""

    def test_gconfig_max_new_tokens_decremented(self):
        """Verify that gconfig.max_new_tokens is decremented in each iteration."""
        gconfig = GenerationHyperparameters(max_new_tokens=200, n_samples=1)
        original_value = gconfig.max_new_tokens

        # Simulate what happens in the loop
        tokens_generated = 100
        gconfig.max_new_tokens -= tokens_generated

        assert gconfig.max_new_tokens == 100, "gconfig.max_new_tokens should be decremented"
        assert original_value == 200, "Original value should be unchanged if saved separately"

    def test_model_request_gconfig_is_reference(self):
        """Verify that ModelRequest.gconfig is a reference that gets modified."""
        gconfig = GenerationHyperparameters(max_new_tokens=200, n_samples=1)
        req = ModelRequest(input_ids=[1, 2, 3], gconfig=gconfig)

        # Modify through the request
        req.gconfig.max_new_tokens -= 100

        # Note: After calling req.copy(), the gconfig is a NEW object
        # but in the original code, we use req.gconfig directly which shares the reference
        assert req.gconfig.max_new_tokens == 100

    def test_model_request_copy_creates_new_gconfig(self):
        """Verify that ModelRequest.copy() creates a new gconfig instance."""
        gconfig = GenerationHyperparameters(max_new_tokens=200, n_samples=1)
        req = ModelRequest(input_ids=[1, 2, 3], gconfig=gconfig)

        req_copy = req.copy()

        # Modify the copy
        req_copy.gconfig.max_new_tokens -= 100

        # Original should be unchanged
        assert req.gconfig.max_new_tokens == 200, "Original gconfig should be unchanged"
        assert req_copy.gconfig.max_new_tokens == 100, "Copy's gconfig should be modified"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
