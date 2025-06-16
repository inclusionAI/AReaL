import time
from typing import List

import requests
import torch.distributed as dist
import transformers

from arealite.api.cli_args import LLMClientConfig, TrainingArgs
from arealite.api.engine_api import SPMDWrapper
from arealite.api.io_struct import LLMRequest, LLMResponse, Message
from arealite.api.llm_client_api import LLMClient
from arealite.api.llm_server_api import LLMServiceRegistry
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import constants, logging

logger = logging.getLogger(__name__)


class SGLangClient(LLMClient):
    """SGLang implementation of LLMClient."""

    def __init__(self, args: TrainingArgs, client_config: LLMClientConfig):
        super().__init__(args, client_config)
        self.registry = LLMServiceRegistry(args.experiment_name, args.trial_name)
        self.tokenizer: transformers.PreTrainedTokenizerFast = load_hf_tokenizer(
            client_config.tokenizer_path
        )

        self._server_idx = 0

    def _choose_server(self):
        """Get an available healthy server."""
        servers = self.registry.get_healthy_servers()
        if not servers:
            raise RuntimeError("No healthy SGLang servers available")

        # Simple round-robin selection (could be improved with load balancing)
        server_info = servers[self._server_idx % len(servers)]
        self._server_idx += 1
        return server_info

    def _convert_messages_to_prompt(self, messages: List[Message]) -> str:
        """Convert messages to a prompt string."""
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        return "\n".join(prompt_parts)

    def _tokenize_prompt(self, prompt: str) -> List[int]:
        """Tokenize the prompt."""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not initialized")
        return self.tokenizer.encode(prompt)

    def generate(self, req: LLMRequest) -> LLMResponse:
        """Generate response using SGLang server."""
        server_info = self._choose_server()
        base_url = f"http://{server_info.host}:{server_info.port}"

        # Convert messages to prompt
        prompt = self._convert_messages_to_prompt(req.messages)
        input_tokens = self.tokenizer.encode(prompt)

        # Prepare request payload
        gconfig = req.gconfig
        sample_params = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_new_tokens": gconfig.max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
        }

        payload = {
            "text": prompt,
            "sampling_params": sample_params,
            "return_logprob": True,
            "stream": False,
        }

        # Make request
        # TODO: implement interruptable rollout
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
        completion = result.get("text", "")
        output_tokens = []
        output_logprobs = []

        if "meta_info" in result:
            meta_info = result["meta_info"]
            if "output_token_logprobs" in meta_info:
                output_logprobs = [x[0] for x in meta_info["output_token_logprobs"]]

            # Determine stop reason
            finish_reason = meta_info.get("finish_reason", {})
            stop_reason = finish_reason.get("type", "stop")
            if stop_reason not in ["length", "stop", "interrupt"]:
                stop_reason = "stop"
        else:
            stop_reason = "stop"

        return LLMResponse(
            completion=completion,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            stop_reason=stop_reason,
            latency=latency,
            ttft=latency,  # Simplified for non-streaming
        )

    async def request_update_weight(self, server_info, new_param_path):
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
                            return
                        logger.warning(
                            f"Update weights failed: {res['message']}. Retrying."
                        )
                    logger.warning(f"Update weights failed: {resp.reason}. Retrying.")
                time.sleep(0.1)
        raise RuntimeError("Update weights failed.")

    def update_weights_from(self, engine: SPMDWrapper) -> None:
        """Update weights from the engine after an RL training step."""
        server_infos = self.registry.get_healthy_servers()
        n_total_servers = len(server_infos)
        m = (n_total_servers + dist.get_world_size() - 1) // dist.get_world_size()
        infos = server_infos[dist.get_rank() * m : (dist.get_rank() + 1) * m]

        path = constants.get_param_realloc_path(self.args)
        engine.save_model_to_hf(path=path)
        tik = time.perf_counter()

        if len(infos) > 0:

            def _run_in_thread():
                import asyncio

                # Create a new event loop for this thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                tasks = [self.request_update_weight(info, path) for info in infos]
                try:
                    return new_loop.run_until_complete(asyncio.gather(*tasks))
                finally:
                    new_loop.close()

            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor() as executor:
                future = executor.submit(_run_in_thread)
                _ = future.result()
        dist.barrier()

        logger.info(
            f"Updating weights for SGLang server done: {time.perf_counter() - tik:.4f}s"
        )
