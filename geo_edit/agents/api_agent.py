import logging
from typing import Any, Dict, Tuple
from geo_edit.agents.base import AgentConfig, BaseAgent
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

class APIBasedAgent(BaseAgent):
    """Unified agent for all API-based backends.

    Supports:
    - Google (Gemini API) - uses native google.genai client
    - OpenAI - uses OpenAI client with responses or chat_completions API
    - vLLM - uses OpenAI-compatible client (local server, supports both API modes)
    - SGLang - uses OpenAI-compatible client (local server, only supports chat_completions)

    For OpenAI/vLLM, you can choose between two API modes via config.api_mode:
    - "responses": Uses client.responses.create() (OpenAI Responses API)
    - "chat_completions": Uses client.chat.completions.create() (Chat Completions API)

    Google API always uses its native client and does not support api_mode switching.
    SGLang only supports chat_completions mode.
    """

    def load_model(self):
        if self._model_loaded:
            return
        
        logger.info("Loading API-based model: %s (type=%s)", self.config.model_name, self.config.model_type)

        # Validate api_mode for SGLang (only supports chat_completions)
        if self.config.model_type == "SGLang" and self.config.api_mode != "chat_completions":
            raise ValueError("SGLang only supports 'chat_completions' mode. Please set api_mode='chat_completions'.")

        if self.config.model_type == "Google":
            self._load_google_client()
        elif self.config.model_type == "OpenAI":
            self._load_openai_client()
        elif self.config.model_type in ("vLLM", "SGLang"):
            self._load_local_server_client()
        else:
            raise NotImplementedError(f"Model type {self.config.model_type} not supported yet.")

        self.model = self.config.model_name
        self._model_loaded = True
        logger.info("API-based model %s loaded successfully.", self.config.model_name)

    def _load_google_client(self):
        """Load Google Gemini client."""
        if self.config.api_key is None:
            raise ValueError("API key must be provided for Google API.")
        
        from google import genai
        self.client = genai.Client(api_key=self.config.api_key)

    def _load_openai_client(self):
        """Load OpenAI client."""

        from openai import OpenAI
        if self.config.api_key is None:
            raise ValueError("API key must be provided for OpenAI API.")

        client_kwargs = {"api_key": self.config.api_key}
        if self.config.api_base is not None:
            client_kwargs["base_url"] = self.config.api_base

        self.client = OpenAI(**client_kwargs)

    def _load_local_server_client(self):
        """Load OpenAI-compatible client for local servers (vLLM, SGLang).

        These servers run locally and do not require an API key.
        Only api_base (server URL) is required.
        """
        from openai import OpenAI
        if self.config.api_base is None:
            raise ValueError(f"api_base must be provided for {self.config.model_type}.")

        self.client = OpenAI(
            base_url=self.config.api_base.rstrip("/") + "/v1",
            api_key="none",  # Local servers don't require API key
        )
        logger.info("Connected to local %s server at %s", self.config.model_type, self.config.api_base)

    def _generate_response(self, model_input: Any) -> Tuple[Any, Dict[str, int | float | str | None]]:
        if self.config.model_type == "Google":
            return self._generate_google_response(model_input)
        elif self.config.model_type in ("OpenAI", "vLLM", "SGLang"):
            # OpenAI, vLLM, SGLang all use OpenAI-compatible API
            if self.config.api_mode == "responses":
                return self._generate_responses_api(model_input)
            elif self.config.api_mode == "chat_completions":
                return self._generate_chat_completions_api(model_input)

    def _generate_google_response(self, model_input: Any) -> Tuple[Any, Dict[str, int | float | str | None]]:
        """Generate response using Google Gemini API."""
        gen_kwargs = self.config.generate_config
        extra_info = {}

        response = self.client.models.generate_content(
            model=self.model,
            contents=model_input,
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

    def _generate_responses_api(self, model_input: Any) -> Tuple[Any, Dict[str, int | float | str | None]]:
        """Generate response using OpenAI Responses API (client.responses.create)."""
        gen_kwargs = dict(self.config.generate_config)
        extra_info = {}

        # Extract input payload
        if isinstance(model_input, dict) and "input" in model_input:
            input_payload = model_input["input"]
            previous_response_id = model_input.get("previous_response_id")
        else:
            input_payload = model_input
            previous_response_id = None

        # Build API call kwargs
        api_kwargs = {"model": self.model, "input": input_payload, **gen_kwargs}
        if previous_response_id is not None:
            api_kwargs["previous_response_id"] = previous_response_id

        response = self.client.responses.create(**api_kwargs)

        extra_info["original_response"] = str(response)
        extra_info["response_id"] = getattr(response, "id", None)

        if response.usage is not None:
            usage = response.usage
            extra_info["tokens_used"] = getattr(usage, "total_tokens", None)
            extra_info["tokens_input"] = getattr(usage, "input_tokens", None)
            extra_info["tokens_output"] = getattr(usage, "output_tokens", None)

        return response, extra_info

    def _generate_chat_completions_api(self, model_input: Any) -> Tuple[Any, Dict[str, int | float | str | None]]:
        """Generate response using OpenAI Chat Completions API (client.chat.completions.create)."""
        gen_kwargs = dict(self.config.generate_config)
        extra_info = {}

        # Extract messages
        if isinstance(model_input, dict) and "messages" in model_input:
            messages = model_input["messages"]
        elif isinstance(model_input, list):
            messages = model_input
        else:
            raise ValueError(f"Invalid model_input format for chat_completions API: {type(model_input)}")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **gen_kwargs,
        )

        extra_info["original_response"] = str(response)

        if response.usage is not None:
            usage = response.usage
            extra_info["tokens_used"] = usage.total_tokens
            extra_info["tokens_input"] = usage.prompt_tokens
            extra_info["tokens_output"] = usage.completion_tokens

        return response, extra_info

    def _validate_response(self, response: Any) -> None:
        if self.config.model_type == "Google":
            if response.parts is None:
                logging.warning("Generated content parts is None: %s", response)
                raise ValueError("Generated content is None.")
        elif self.config.api_mode == "responses":
            if not response.output:
                logging.warning("Generated content output is empty: %s", response)
                raise ValueError("Generated content is empty.")
        elif self.config.api_mode == "chat_completions":
            if not response.choices:
                logging.warning("Generated response has no choices: %s", response)
                raise ValueError("Generated response has no choices.")
