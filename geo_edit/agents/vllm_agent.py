from typing import Any, Dict, Tuple

from openai import OpenAI

from geo_edit.agents.base import BaseAgent
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)


class VLLMBasedAgent(BaseAgent):
    """Agent that interacts with a local vLLM OpenAI-compatible server."""

    def __init__(self, config):
        super().__init__(config)
        # Cumulative token counters to convert per-step token counts to cumulative
        # This makes VLLM token stats consistent with Google/OpenAI APIs
        self._cumulative_tokens_input = 0
        self._cumulative_tokens_output = 0

    def reset(self):
        super().reset()
        self._cumulative_tokens_input = 0
        self._cumulative_tokens_output = 0

    def load_model(self):
        if self._model_loaded:
            return

        self.client = OpenAI(base_url=self.config.api_base.rstrip("/") + "/v1", api_key="none")
        self.model = self.config.model_name
        self._model_loaded = True
        logger.info("Loaded vLLM model %s at %s", self.model, self.config.api_base)

    def _generate_response(self, model_input: Any) -> Tuple[Any, Dict[str, Any]]:
        gen_kwargs = dict(self.config.generate_config or {})
        if isinstance(model_input, dict):
            messages = model_input.get("messages", [])
        else:
            messages = model_input

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **gen_kwargs,
        )

        extra_info = {"original_response": str(response)}
        if response.usage is not None:
            usage = response.usage
            if isinstance(usage, dict):
                tokens_input = usage.get("prompt_tokens")
                tokens_output = usage.get("completion_tokens")
                tokens_total = usage.get("total_tokens")
            else:
                tokens_input = getattr(usage, "prompt_tokens", None)
                tokens_output = getattr(usage, "completion_tokens", None)
                tokens_total = getattr(usage, "total_tokens", None)

            if tokens_input is not None:
                self._cumulative_tokens_input += tokens_input
                extra_info["tokens_input"] = self._cumulative_tokens_input
            if tokens_output is not None:
                self._cumulative_tokens_output += tokens_output
                extra_info["tokens_output"] = self._cumulative_tokens_output
            if tokens_total is not None:
                extra_info["tokens_used"] = self._cumulative_tokens_input + self._cumulative_tokens_output
            elif tokens_input is not None and tokens_output is not None:
                extra_info["tokens_used"] = self._cumulative_tokens_input + self._cumulative_tokens_output
        return response, extra_info

    def _validate_response(self, response: Any) -> None:
        if not response.choices:
            raise ValueError("Generated response has no choices.")
