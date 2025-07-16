import time

from arealite.api.cli_args import InferenceEngineConfig
from arealite.engine.sglang_remote import RemoteSGLangEngine
from arealite.api.io_struct import (
    VLMRequest,
    VLMResponse
)
from arealite.utils.http import arequest_with_retry
from realhf.base import logging, pkg_version

logger = logging.getLogger(__name__)

if pkg_version.is_available("sglang"):
    if pkg_version.is_version_greater_or_equal("sglang", "0.4.4"):
        SGLANG_TOKEN_OUTPUT_IDENTIFIER = "output_ids"
    else:
        SGLANG_TOKEN_OUTPUT_IDENTIFIER = "token_ids"

ROLLOUT_POLL_WAIT_TIME = 0.1
RID_CACHE_SIZE = 128


class VL_RemoteSGLangEngine(RemoteSGLangEngine):

    def __init__(self, config: InferenceEngineConfig):
        super().__init__(config)

    async def agenerate(self, req: VLMRequest) -> VLMResponse:
        """Async version of generate using aiohttp."""
        # Prepare request payload
        gconfig = req.gconfig
        stop_token_ids = gconfig.stop_token_ids

        if gconfig.n_samples != 1:
            raise ValueError(
                "RemoteSGLangEngine does not support n_samples > 1. "
                "Please call generate for multiple times with n_samples = 1."
            )
        sample_params = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_new_tokens": gconfig.max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": stop_token_ids,
        }

        # NOTE: rid should NOT be passed in payload
        payload = {
            "input_ids": req.input_ids.copy(),
            "image_data": req.image_data,  # ImageObject or str
            "sampling_params": sample_params,
            "return_logprob": True,
            "stream": False,
        }

        # Make request
        start_time = time.perf_counter()
        accumulated_output_tokens = []
        accumulated_output_logprobs = []
        accumulated_versions = []

        # Deal with rollout interruption
        completions = ""
        stop_reason = "length"

        if req.rid in self.rid_to_address:
            server_addr = self.rid_to_address[req.rid]
        else:
            server_addr = self.choose_server()
            if len(self.rid_queue) >= RID_CACHE_SIZE:
                # Remove the oldest entry if cache is full
                oldest_rid = self.rid_queue.pop(0)
                self.rid_to_address.pop(oldest_rid, None)
            self.rid_to_address[req.rid] = server_addr
            self.rid_queue.append(req.rid)

        while (
            stop_reason != "stop"
            and len(accumulated_output_tokens) < gconfig.max_new_tokens
        ):
            # loop until the generation is complete
            result = await arequest_with_retry(
                addr=self.choose_server(),
                endpoint="/generate",
                payload=payload,
                method="POST",
                max_retries=3,
                timeout=self.config.request_timeout,
            )

            # Parse response
            meta_info = result["meta_info"]
            output_tokens = [x[1] for x in meta_info["output_token_logprobs"]]
            output_logprobs = [x[0] for x in meta_info["output_token_logprobs"]]

            # Update accumulated outputs
            accumulated_output_tokens.extend(output_tokens)
            accumulated_output_logprobs.extend(output_logprobs)
            # FIXME: Update with actual server versions
            accumulated_versions.extend([-1] * len(output_tokens))

            # Check if generation is complete
            finish_reason = meta_info["finish_reason"]
            stop_reason = finish_reason["type"]

            payload["input_ids"] += result[SGLANG_TOKEN_OUTPUT_IDENTIFIER]
            sample_params["max_new_tokens"] = min(
                sample_params["max_new_tokens"],
                gconfig.max_new_tokens - len(output_tokens),
            )

        latency = time.perf_counter() - start_time

        return VLMResponse(
            input_tokens=req.input_ids,
            input_images=req.image_data,
            output_tokens=accumulated_output_tokens,
            output_logprobs=accumulated_output_logprobs,
            output_versions=accumulated_versions,
            stop_reason=stop_reason,
            latency=latency,
            ttft=latency,  # Simplified for non-streaming
        )

