import logging
from typing import Any, Dict, Tuple

from geo_edit.agents.base import AgentConfig, BaseAgent
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)


class APIBasedAgent(BaseAgent):
    """Agent that interacts with an external API for generation."""

    def load_model(self):
        if self._model_loaded:
            return

        logger.info("Loading API-based model: %s", self.config.model_name)

        if self.config.api_key is None:
            raise ValueError("API key must be provided for API-based agents.")

        if self.config.model_type == "Google":
            from google import genai
            self.client = genai.Client(api_key=self.config.api_key)
            self.model = self.config.model_name
            self._model_loaded = True
            logger.info("API-based model %s loaded successfully.", self.config.model_name)
        elif self.config.model_type == "OpenAI":
            from openai import OpenAI
            self.client = OpenAI(api_key=self.config.api_key)
            self.model = self.config.model_name
            self._model_loaded = True
            logger.info("API-based model %s loaded successfully.", self.config.model_name)
        else:
            raise NotImplementedError(f"Model type {self.config.model_type} not supported yet.")

    def _generate_response(self, model_input: Any) -> Tuple[Any, Dict[str, Any]]:
        gen_kwargs = self.config.generate_config
        extra_info = {}
        contents = model_input

        if self.config.model_type == "Google":
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=gen_kwargs,
            )
            extra_info["original_response"] = str(response)
            if response.usage_metadata is not None:
                usage = response.usage_metadata
                extra_info["tokens_used"] = usage.total_token_count
                extra_info["tokens_input"] = usage.prompt_token_count
                extra_info["tokens_output"] = usage.candidates_token_count
                extra_info["tokens_thoughts"] = getattr(usage, "thoughts_token_count", None)
            content = response.candidates[0].content
            return content, extra_info

        if self.config.model_type == "OpenAI":
            input_payload = contents["input"]
            previous_response_id = contents["previous_response_id"]
            response = self.client.responses.create(
                model=self.model,
                input=input_payload,
                previous_response_id=previous_response_id,
                **gen_kwargs,
            )
            extra_info["original_response"] = str(response)
            extra_info["response_id"] = response.id
            if response.usage is not None:
                extra_info["tokens_used"] = response.usage.total_tokens
                extra_info["tokens_input"] = response.usage.input_tokens
                extra_info["tokens_output"] = response.usage.output_tokens

            return response, extra_info

        raise NotImplementedError(f"Model type {self.config.model_type} not supported yet.")

    def _validate_response(self, response: Any) -> None:
        if self.config.model_type == "Google":
            if response.parts is None:
                logging.warning("Generated content parts is None: %s", response)
                raise ValueError("Generated content is None.")
        elif self.config.model_type == "OpenAI":
            if not response.output:
                logging.warning("Generated content output is empty: %s", response)
                raise ValueError("Generated content is empty.")
