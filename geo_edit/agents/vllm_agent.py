import time
from typing import Any, Dict, Tuple

from openai import OpenAI

from .base import AgentConfig, BaseAgent
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class VLLMBasedAgent(BaseAgent):
    """Agent that interacts with a local vLLM OpenAI-compatible server."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.client = None
        self._model_loaded = False

    def _resolve_base_url(self) -> str:
        if self.config.api_base:
            base_url = self.config.api_base.rstrip("/")
        elif self.config.port is not None:
            base_url = f"http://localhost:{self.config.port}"
        else:
            base_url = "http://localhost:8000"

        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        return base_url

    def load_model(self):
        if self._model_loaded:
            return

        base_url = self._resolve_base_url()
        api_key = self.config.api_key or "EMPTY"
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = self.config.model_name
        self._model_loaded = True
        logger.info("Loaded vLLM model %s at %s", self.model, base_url)

    def _prepare_input(self, observation: Dict[str, Any]) -> Any:
        return observation

    def _parse_response(self, raw_response: str, observation: Dict[str, Any]) -> str:
        return raw_response

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
            extra_info["tokens_used"] = response.usage.total_tokens
        return response, extra_info

    def act(self, observation: Any) -> Tuple[Any, Dict[str, Any]]:
        if self.client is None:
            self.load_model()

        for attempt in range(self.config.n_retry):
            try:
                content, extra_info = self._generate_response(observation)
                if not content.choices:
                    raise ValueError("Generated response has no choices.")

                extra_info.update(
                    {
                        "model_name": self.config.model_name,
                        "attempt": attempt + 1,
                        "step_count": self.step_count,
                    }
                )

                self.step_count += 1
                if "tokens_used" in extra_info:
                    self.total_tokens_used += extra_info["tokens_used"]

                logger.info(
                    "Step %s: Generated response in attempt %s",
                    self.step_count,
                    attempt + 1,
                )
                return content, extra_info

            except Exception as exc:
                logger.warning("Attempt %s failed: %s", attempt + 1, exc)
                if attempt < self.config.n_retry - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    error_msg = (
                        f"Failed after {self.config.n_retry} attempts: {exc}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

    def reset(self):
        self.step_count = 0
        self.total_tokens_used = 0
        self.client = None
        self._model_loaded = False
        logger.info("Agent state reset")
