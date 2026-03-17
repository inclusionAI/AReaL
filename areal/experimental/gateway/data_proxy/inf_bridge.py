"""InfBridge -- HTTP client implementing _AsyncGenerateEngine protocol for SGLang."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Literal

_StopReason = Literal["length", "stop", "tool_calls", "abort"]

import httpx
import numpy as np
import pybase64

if TYPE_CHECKING:
    from areal.api.io_struct import ModelRequest, ModelResponse
    from areal.experimental.gateway.data_proxy.pause import PauseState

logger = logging.getLogger("InfBridge")


class InfBridge:
    """SGLang HTTP client implementing _AsyncGenerateEngine protocol.

    Translates ModelRequest into SGLang /generate HTTP calls and ModelResponse.
    Handles pause/abort/resubmit loop transparently.
    """

    def __init__(
        self,
        backend_addr: str,
        pause_state: PauseState,
        request_timeout: float = 120.0,
        max_resubmit_retries: int = 20,
        resubmit_wait: float = 0.5,
        stream: bool = False,
        version: int = 0,
    ) -> None:
        self.backend_addr = backend_addr.rstrip("/")
        self.pause_state = pause_state
        self.request_timeout = request_timeout
        self.max_resubmit_retries = max_resubmit_retries
        self.resubmit_wait = resubmit_wait
        self.stream = stream
        self._version = version

    # -- version tracking ---------------------------------------------------

    def set_version(self, version: int) -> None:
        self._version = version

    def get_version(self) -> int:
        return self._version

    # -- pause / resume -----------------------------------------------------

    async def pause(self) -> None:
        """Pause generation by setting pause_state and calling SGLang."""
        await self.pause_state.set_paused(True)
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{self.backend_addr}/pause_generation", json={})
            resp.raise_for_status()
        logger.info("SGLang pause_generation called on %s", self.backend_addr)

    async def resume(self) -> None:
        """Resume generation by calling SGLang and clearing pause_state."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self.backend_addr}/continue_generation", json={}
            )
            resp.raise_for_status()
        await self.pause_state.set_paused(False)
        logger.info("SGLang continue_generation called on %s", self.backend_addr)

    # -- internal SGLang HTTP call ------------------------------------------

    async def _call_sglang(
        self,
        input_ids: list[int],
        sampling_params: dict[str, Any],
        *,
        return_routed_experts: bool = False,
    ) -> dict[str, Any]:
        """Call SGLang /generate and return the raw JSON response.

        Args:
            input_ids: Pre-tokenized input token IDs.
            sampling_params: SGLang sampling parameters.
            return_routed_experts: Whether to request routed expert info.

        Returns:
            Raw JSON dict from SGLang /generate endpoint.

        Raises:
            httpx.HTTPStatusError: On non-2xx response from SGLang.
        """
        payload: dict[str, Any] = {
            "input_ids": input_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
            "stream": self.stream,
        }
        if return_routed_experts:
            payload["return_routed_experts"] = True

        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            resp = await client.post(
                f"{self.backend_addr}/generate",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    # -- response parsing ---------------------------------------------------

    @staticmethod
    def _parse_response(
        data: dict[str, Any],
        *,
        return_routed_experts: bool = False,
    ) -> tuple[list[int], list[float], _StopReason, np.ndarray | None]:
        """Parse SGLang /generate response.

        Returns:
            Tuple of (output_tokens, output_logprobs, stop_reason, routed_experts).
        """
        meta_info = data["meta_info"]
        finish_reason = meta_info["finish_reason"]
        stop_reason: _StopReason = finish_reason["type"]

        # Extract routed_experts if requested and present
        routed_experts: np.ndarray | None = None
        if return_routed_experts:
            raw_experts = meta_info.get("routed_experts", None)
            if raw_experts is not None:
                num_sgl_token = (
                    meta_info["prompt_tokens"] + meta_info["completion_tokens"] - 1
                )
                routed_experts = np.frombuffer(
                    pybase64.b64decode(raw_experts.encode("utf-8")),
                    dtype=np.int32,
                ).reshape(num_sgl_token, -1)

        # Handle abort-before-prefill: no output tokens
        output_token_logprobs = meta_info.get("output_token_logprobs", [])
        output_tokens = [x[1] for x in output_token_logprobs]
        output_logprobs = [x[0] for x in output_token_logprobs]

        return output_tokens, output_logprobs, stop_reason, routed_experts

    # -- main generation with pause/abort/resubmit --------------------------

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Generate response for ModelRequest via SGLang HTTP.

        Implements _AsyncGenerateEngine protocol.
        Handles pause/abort/resubmit loop transparently.
        """
        from areal.api.io_struct import ModelResponse

        # Validate n_samples
        if req.gconfig.n_samples != 1:
            raise ValueError(
                f"InfBridge only supports n_samples=1, got {req.gconfig.n_samples}"
            )

        # Compute effective max_new_tokens
        max_new_tokens = min(
            req.gconfig.max_tokens - len(req.input_ids),
            req.gconfig.max_new_tokens,
        )
        if max_new_tokens <= 0:
            raise ValueError(
                f"max_new_tokens must be > 0, got {max_new_tokens} "
                f"(max_tokens={req.gconfig.max_tokens}, "
                f"input_len={len(req.input_ids)}, "
                f"max_new_tokens={req.gconfig.max_new_tokens})"
            )

        gconfig = req.gconfig
        return_routed_experts = req.metadata.get("return_routed_experts", False)

        # Build sampling params (matches backend.py:69-74 + sglang_remote.py)
        sampling_params: dict[str, Any] = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": gconfig.stop_token_ids,
            "ignore_eos": gconfig.ignore_eos,
            "skip_special_tokens": gconfig.skip_special_tokens,
            "frequency_penalty": gconfig.frequency_penalty,
        }
        if gconfig.stop:
            sampling_params["stop"] = gconfig.stop

        ori_max_new_tokens = max_new_tokens
        accumulated_tokens: list[int] = []
        accumulated_logprobs: list[float] = []
        stop_reason: _StopReason | None = None
        # Only keep the last routed_experts result (from final segment)
        final_routed_experts: np.ndarray | None = None

        t0 = time.monotonic()

        # Pause/abort/resubmit loop (ported from backend.py:98-165)
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
            current_input_ids = list(req.input_ids) + accumulated_tokens

            data = await self._call_sglang(
                current_input_ids,
                updated_params,
                return_routed_experts=return_routed_experts,
            )
            tokens, logprobs, stop_reason, routed_experts = self._parse_response(
                data,
                return_routed_experts=return_routed_experts,
            )

            # Always accumulate output tokens
            accumulated_tokens.extend(tokens)
            accumulated_logprobs.extend(logprobs)
            if routed_experts is not None:
                if final_routed_experts is None:
                    final_routed_experts = routed_experts
                else:
                    final_routed_experts = np.concatenate(
                        [final_routed_experts, routed_experts], axis=0
                    )

            # Check loop exit conditions (matches remote_inf_engine.py:771-773)
            if stop_reason in ("stop", "tool_calls", "length"):
                break

            if len(accumulated_tokens) >= ori_max_new_tokens:
                stop_reason = "length"
                break

            # stop_reason == 'abort' -> continue loop (resubmit)
            logger.debug(
                "Abort detected, resubmit attempt %d, accumulated %d tokens",
                _attempt + 1,
                len(accumulated_tokens),
            )

        # Final abort at max retries -> treat as length
        # (matches remote_inf_engine.py:839-843)
        if stop_reason == "abort" or stop_reason is None:
            stop_reason = "length"

        latency = time.monotonic() - t0

        return ModelResponse(
            input_tokens=list(req.input_ids),
            output_tokens=accumulated_tokens,
            output_logprobs=accumulated_logprobs,
            output_versions=[self._version] * len(accumulated_tokens),
            stop_reason=stop_reason,
            tokenizer=req.tokenizer,
            latency=latency,
            routed_experts=final_routed_experts,
        )
