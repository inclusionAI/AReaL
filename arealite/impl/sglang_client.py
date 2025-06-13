import time
from typing import List

import requests
import transformers

from arealite.api.cli_args import LLMClientConfig, TrainingArgs
from arealite.api.engine_api import SPMDWrapper
from arealite.api.io_struct import LLMRequest, LLMResponse, Message
from arealite.api.llm_client_api import LLMClient
from arealite.api.llm_server_api import LLMServiceRegistry
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import logging

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

    def _get_available_server(self):
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
        server_info = self._get_available_server()
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

    def update_weights_from(self, engine: SPMDWrapper) -> None:
        """Update weights from the engine after an RL training step."""
        server_infos = self.registry.get_healthy_servers()
        base_url = f"http://{server_info.host}:{server_info.port}"

        # This would typically save weights to disk and update the server
        # For now, we implement a placeholder that logs the action
        logger.info(f"Updating weights for server {server_info.server_id}")

        # TODO: Implement actual weight update logic
        # This would involve:
        # 1. Saving engine weights to a temporary path
        # 2. Calling the server's update_weights_from_disk endpoint
        # 3. Handling the response
