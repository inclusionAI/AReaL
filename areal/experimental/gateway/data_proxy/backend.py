from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


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
