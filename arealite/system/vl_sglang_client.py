# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import time

from arealite.api.io_struct import LLMServerInfo,VLMRequest,VLMResponse
from realhf.base import logging, pkg_version
from arealite.api.vlm_client_api import VLMClient
from arealite.api.cli_args import LLMClientConfig, TrainingArgs
from arealite.system.sglang_client import SGLangClient
logger = logging.getLogger(__name__)

if pkg_version.is_available("sglang"):
    if pkg_version.is_version_greater_or_equal("sglang", "0.4.4"):
        SGLANG_TOKEN_OUTPUT_IDENTIFIER = "output_ids"
    else:
        SGLANG_TOKEN_OUTPUT_IDENTIFIER = "token_ids"



class VL_SGLangClient(VLMClient, SGLangClient):
    """A client for interacting with both VLM and SGLang servers."""

    def __init__(self, args: TrainingArgs, client_config: LLMClientConfig):
        # Initialize the parent classes
        VLMClient.__init__(self, args, client_config)
        SGLangClient.__init__(self, args, client_config)

    async def agenerate(self, req: VLMRequest) -> VLMResponse:
        """Override the agenerate method to support both VLM and SGLang generation."""
        if not req.images:
            # If no images are provided, use SGLang generation
            return await SGLangClient.agenerate(self, req)
        if not req.text:
            assert req.input_ids is not None
            req.text = self.tokenizer.decode(req.input_ids)
            
         # Prepare request payload
        gconfig = req.gconfig
        stop_token_ids = gconfig.stop_token_ids
        if self.tokenizer.eos_token_id not in stop_token_ids:
            stop_token_ids.append(self.tokenizer.eos_token_id)
        if self.tokenizer.pad_token_id not in stop_token_ids:
            stop_token_ids.append(self.tokenizer.pad_token_id)

        assert gconfig.n_samples == 1
        sample_params = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_new_tokens": gconfig.max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": stop_token_ids,
        }

        payload = {
            "rid": req.rid,
            "text": req.text,
            "images": req.images, #  ImageObject or str
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
        completion = ""
        stop_reason = "length"

        while (
            stop_reason != "stop"
            and len(accumulated_output_tokens) < gconfig.max_new_tokens
        ):
            # loop until the generation is complete
            response, server_info = await self.arequest_with_retry(
                endpoint="/generate",
                payload=payload,
                method="POST",
                max_retries=3,
                timeout=self.client_config.request_timeout,
            )
            result = await response.json()

            # Parse response
            completion += result["text"]
            meta_info = result["meta_info"]
            output_tokens = [x[1] for x in meta_info["output_token_logprobs"]]
            output_logprobs = [x[0] for x in meta_info["output_token_logprobs"]]

            # Update accumulated outputs
            accumulated_output_tokens.extend(output_tokens)
            accumulated_output_logprobs.extend(output_logprobs)
            accumulated_versions.extend([server_info.version] * len(output_tokens))

            # Check if generation is complete
            finish_reason = meta_info["finish_reason"]
            stop_reason = finish_reason["type"]

            payload["text"] += completion

        latency = time.perf_counter() - start_time

        return VLMResponse(
            completion=completion,
            input_tokens=req.input_ids,
            input_images=req.images,
            output_tokens=accumulated_output_tokens,
            output_logprobs=accumulated_output_logprobs,
            output_versions=accumulated_versions,
            stop_reason=stop_reason,
            latency=latency,
            ttft=latency,  # Simplified for non-streaming
        )

    async def aupdate_weights_from_disk(self, server_info: LLMServerInfo, path: str):
        
        await SGLangClient.aupdate_weights_from_disk(self, server_info, path)

    async def ainit_weight_update_group(self, server_info, group_meta):

        await SGLangClient.ainit_weight_update_group(self, server_info, group_meta)

    async def aupdate_weights_from_distributed(self, server_info, weight_meta):

        await SGLangClient.aupdate_weights_from_distributed(self, server_info, weight_meta)
