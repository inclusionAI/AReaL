import time

import requests
import torch.distributed as dist
import transformers

from arealite.api.cli_args import LLMClientConfig, TrainingArgs
from arealite.api.engine_api import SPMDWrapper
from arealite.api.io_struct import LLMRequest, LLMResponse, LLMServerInfo
from arealite.api.llm_client_api import LLMClient
from arealite.api.llm_server_api import LLMServiceRegistry
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import constants, logging, pkg_version

logger = logging.getLogger(__name__)

if pkg_version.is_available("sglang"):
    if pkg_version.is_version_greater_or_equal("sglang", "0.4.4"):
        SGLANG_TOKEN_OUTPUT_IDENTIFIER = "output_ids"
    else:
        SGLANG_TOKEN_OUTPUT_IDENTIFIER = "token_ids"


class SGLangClient(LLMClient):
    """SGLang implementation of LLMClient."""

    def generate(self, req: LLMRequest) -> LLMResponse:
        """Generate response using SGLang server."""
        server_info = self.select_server()
        base_url = f"http://{server_info.host}:{server_info.port}"

        # Convert messages to prompt
        if not req.text:
            assert req.input_ids is not None
            req.text = self.tokenizer.decode(req.text)

        # Prepare request payload
        gconfig = req.gconfig
        stop_token_ids = gconfig.stop_token_ids
        if self.tokenizer.eos_token_id not in stop_token_ids:
            stop_token_ids.append(self.tokenizer.eos_token_id)
        if self.tokenizer.pad_token_id not in stop_token_ids:
            stop_token_ids.append(self.tokenizer.pad_token_id)

        assert gconfig.n == 1
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
        # TODO: implement interruptable rollout
        # TODO: server OOM request will not return, should retry
        start_time = time.perf_counter()
        response = requests.post(
            f"{base_url}/generate",
            json=payload,
            timeout=self.client_config.gen_timeout,
        )
        response.raise_for_status()

        # Parse response
        result = response.json()
        latency = time.perf_counter() - start_time

        # Extract completion and tokens
        completion = result["text"]
        meta_info = result["meta_info"]

        output_tokens = [x[1] for x in meta_info["output_token_logprobs"]]
        output_logprobs = [x[0] for x in meta_info["output_token_logprobs"]]

        # Determine stop reason
        finish_reason = meta_info["finish_reason"]
        stop_reason = finish_reason["type"]
        assert stop_reason in ["length", "stop"], stop_reason

        return LLMResponse(
            completion=completion,
            input_tokens=req.input_ids,
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            output_versions=[server_info.version] * len(output_tokens),
            stop_reason=stop_reason,
            latency=latency,
            ttft=latency,  # Simplified for non-streaming
        )

    async def request_update_weight(
        self, server_info: LLMServerInfo, new_param_path: str, version: int
    ):
        import aiohttp

        server_url = f"http://{server_info.host}:{server_info.port}"
        success = False
        for _ in range(self.client_config.update_weights_retries):
            async with aiohttp.ClientSession(
                server_url,
                timeout=aiohttp.ClientTimeout(
                    total=self.client_config.update_weights_timeout,
                    sock_connect=self.client_config.update_weights_timeout,
                ),
            ) as session:
                async with session.post(
                    f"/update_weights_from_disk",
                    json=dict(model_path=new_param_path, allow_interrupt=True),
                ) as resp:
                    if resp.status == 200:
                        res = await resp.json()
                        success = res["success"]
                        if success:
                            if "num_paused_requests" in res:
                                logger.info(
                                    f"{res['num_paused_requests']} requests are interrupted "
                                    f"during updating weights for server {server_url}"
                                )
                            self.registry.update_heartbeat(
                                server_info.server_id, "healthy", version=version + 1
                            )
                            return
                        logger.warning(
                            f"Update weights failed: {res['message']}. Retrying."
                        )
                    logger.warning(f"Update weights failed: {resp.reason}. Retrying.")
                time.sleep(0.1)
        raise RuntimeError("Update weights failed.")
