from typing import Any, Dict, Tuple

from openai import OpenAI

from geo_edit.agents.base import BaseAgent
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)


class SGLangBasedAgent(BaseAgent):
    """Agent that interacts with a local SGLang OpenAI-compatible server."""

    def load_model(self):
        if self._model_loaded:
            return

        self.client = OpenAI(base_url=self.config.api_base.rstrip("/") + "/v1", api_key="none")
        self.model = self.config.model_name
        self._model_loaded = True
        logger.info("Loaded SGLang model %s at %s", self.model, self.config.api_base)

    def _generate_response(self, model_input: Any) -> Tuple[Any, Dict[str, int | float | str | None]]:
        gen_kwargs = dict(self.config.generate_config)
        messages = model_input["messages"] if isinstance(model_input, dict) else model_input
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **gen_kwargs,
        )

        extra_info = {"original_response": str(response)}
        usage = response.usage
        if usage is not None:
            extra_info["tokens_used"] = usage.total_tokens
            extra_info["tokens_input"] = usage.prompt_tokens
            extra_info["tokens_output"] = usage.completion_tokens
        return response, extra_info

    def _validate_response(self, response: Any) -> None:
        if not response.choices:
            raise ValueError("Generated response has no choices.")
