from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from areal.experimental.gateway.data_proxy.pause import PauseState

logger = logging.getLogger("DataProxyBackend")


@dataclass
class GenerationResult:
    """Result from a single SGLang /generate call."""

    output_tokens: list[int]
    output_logprobs: list[float]
    stop_reason: str  # 'stop' | 'abort' | 'length' | 'tool_calls'


class SGLangBackend:
    """Thin HTTP wrapper around SGLang's native /generate endpoint.

    This class is the ONLY place that knows about SGLang's response format.
    Plan 3c will wrap generate() calls in a pause/resubmit loop — keep this class stateless.
    """

    def __init__(self, backend_addr: str, request_timeout: float = 120.0):
        self.backend_addr = backend_addr.rstrip("/")
        self.request_timeout = request_timeout

    async def generate(
        self,
        input_ids: list[int],
        sampling_params: dict[str, Any],
    ) -> GenerationResult:
        """Call SGLang /generate (non-streaming) and parse response.

        Args:
            input_ids: Pre-tokenized input token IDs.
            sampling_params: SGLang sampling parameters (max_new_tokens, temperature, etc.).

        Returns:
            GenerationResult with output tokens, logprobs, and stop reason.

        Raises:
            httpx.HTTPStatusError: On non-2xx response from SGLang.
        """
        payload = {
            "input_ids": input_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            resp = await client.post(
                f"{self.backend_addr}/generate",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        meta_info = data["meta_info"]
        finish_reason = meta_info["finish_reason"]
        stop_reason = finish_reason["type"]  # 'stop' | 'abort' | 'length'

        # Handle abort-before-prefill: no output tokens
        output_token_logprobs = meta_info.get("output_token_logprobs", [])
        output_tokens = [x[1] for x in output_token_logprobs]
        output_logprobs = [x[0] for x in output_token_logprobs]

        return GenerationResult(
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            stop_reason=stop_reason,
        )


class SGLangBackendWithResubmit:
    """Wraps SGLangBackend with transparent pause/abort/resubmit loop.

    When SGLang returns stop_reason='abort' (weight update in progress),
    this wrapper:
      1. Accumulates partial output tokens
      2. Waits for PauseState to clear (resume)
      3. Resubmits with input_ids + accumulated_output_tokens
      4. Decrements max_new_tokens by len(accumulated)

    Ported from areal/infra/remote_inf_engine.py:771-843.
    """

    def __init__(
        self,
        base: SGLangBackend,
        pause_state: PauseState,
        max_resubmit_retries: int = 20,
        resubmit_wait: float = 0.5,
    ) -> None:
        self.base = base
        self.pause_state = pause_state
        self.max_resubmit_retries = max_resubmit_retries
        self.resubmit_wait = resubmit_wait

    async def generate(
        self,
        input_ids: list[int],
        sampling_params: dict[str, Any],
    ) -> GenerationResult:
        """Generate with transparent pause/abort/resubmit.

        Mirrors remote_inf_engine.py:771-843 exactly:
          - while stop_reason not in ('stop','tool_calls','length')
            and accumulated < ori_max_new_tokens:
              wait while paused, call backend, accumulate tokens,
              extend input_ids, decrement max_new_tokens
          - final abort at max_new_tokens → treat as 'length'
        """
        ori_max_new_tokens: int = sampling_params.get("max_new_tokens", 512)
        accumulated_tokens: list[int] = []
        accumulated_logprobs: list[float] = []
        stop_reason: str | None = None

        for _attempt in range(self.max_resubmit_retries):
            # Wait while paused (weight update in progress)
            while await self.pause_state.is_paused():
                await asyncio.sleep(self.resubmit_wait)

            # Update max_new_tokens to account for already-generated tokens
            updated_params = dict(sampling_params)
            updated_params["max_new_tokens"] = ori_max_new_tokens - len(
                accumulated_tokens
            )
            if updated_params["max_new_tokens"] <= 0:
                stop_reason = "length"
                break

            # Build current input: original + accumulated output
            current_input_ids = list(input_ids) + accumulated_tokens

            result = await self.base.generate(current_input_ids, updated_params)
            stop_reason = result.stop_reason

            # Always accumulate output tokens
            accumulated_tokens.extend(result.output_tokens)
            accumulated_logprobs.extend(result.output_logprobs)

            # Check loop exit conditions (matches remote_inf_engine.py:771-773)
            if stop_reason in ("stop", "tool_calls", "length"):
                break

            if len(accumulated_tokens) >= ori_max_new_tokens:
                stop_reason = "length"
                break

            # stop_reason == 'abort' → continue loop (resubmit)
            logger.debug(
                "Abort detected, resubmit attempt %d, accumulated %d tokens",
                _attempt + 1,
                len(accumulated_tokens),
            )

        # Final abort at max retries → treat as length
        # (matches remote_inf_engine.py:839-843)
        if stop_reason == "abort" or stop_reason is None:
            stop_reason = "length"

        return GenerationResult(
            output_tokens=accumulated_tokens,
            output_logprobs=accumulated_logprobs,
            stop_reason=stop_reason,
        )
