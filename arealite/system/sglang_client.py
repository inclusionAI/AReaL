# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import time

from arealite.api.io_struct import LLMRequest, LLMResponse, LLMServerInfo
from arealite.api.llm_client_api import LLMClient
from realhf.base import logging, pkg_version

logger = logging.getLogger(__name__)

if pkg_version.is_available("sglang"):
    if pkg_version.is_version_greater_or_equal("sglang", "0.4.4"):
        SGLANG_TOKEN_OUTPUT_IDENTIFIER = "output_ids"
    else:
        SGLANG_TOKEN_OUTPUT_IDENTIFIER = "token_ids"


class SGLangClient(LLMClient):
    """SGLang implementation of LLMClient."""

    async def agenerate(self, req: LLMRequest) -> LLMResponse:
        """Async version of generate using aiohttp."""

        # Convert messages to prompt
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

        return LLMResponse(
            completion=completion,
            input_tokens=req.input_ids,
            output_tokens=accumulated_output_tokens,
            output_logprobs=accumulated_output_logprobs,
            output_versions=accumulated_versions,
            stop_reason=stop_reason,
            latency=latency,
            ttft=latency,  # Simplified for non-streaming
        )

    async def aupdate_weights_from_disk(self, server_info: LLMServerInfo, path: str):
        server_url = f"http://{server_info.host}:{server_info.port}"
        response, _ = await self.arequest_with_retry(
            endpoint="/update_weights_from_disk",
            payload=dict(model_path=path, allow_interrupt=True),
            method="POST",
            max_retries=3,
            timeout=self.client_config.request_timeout,
            target_server=server_info,
        )
        res = await response.json()
        assert res["success"]
        if "num_paused_requests" in res:
            logger.info(
                f"{res['num_paused_requests']} requests are interrupted "
                f"during updating weights for server {server_url}"
            )
        self.registry.update_heartbeat(
            server_info.server_id, "healthy", version=server_info.version + 1
        )

    async def ainit_weight_update_group(self, server_info, group_meta):
        payload = dict(
            master_address=group_meta.master_address,
            master_port=group_meta.master_port,
            rank_offset=group_meta.rank_offset,
            world_size=group_meta.world_size,
            group_name=group_meta.group_name,
            backend=group_meta.backend,
        )
        response, _ = await self.arequest_with_retry(
            endpoint="/init_weights_update_group",
            payload=payload,
            method="POST",
            max_retries=3,
            timeout=self.client_config.request_timeout,
            target_server=server_info,
        )
        res = await response.json()
        assert res["success"], res["message"]

    async def aupdate_weights_from_distributed(self, server_info, weight_meta):
        payload = dict(
            name=weight_meta.param_name,
            dtype=weight_meta.dtype,
            shape=weight_meta.shape,
        )
        response, _ = await self.arequest_with_retry(
            endpoint="/update_weights_from_distributed",
            payload=payload,
            method="POST",
            max_retries=3,
            timeout=self.client_config.request_timeout,
            target_server=server_info,
        )
        res = await response.json()
        assert res["success"], res["message"]
