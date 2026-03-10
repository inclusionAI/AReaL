import logging
from typing import Any, Dict, Tuple
from geo_edit.agents.base import AgentConfig, BaseAgent
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

class APIBasedAgent(BaseAgent):
    """Unified agent for all API-based backends.

    Supports three api_mode values:
    - "google": Google Gemini native API (google.genai client)
    - "responses": OpenAI Responses API (client.responses.create)
    - "chat_completions": OpenAI Chat Completions API (client.chat.completions.create)

    Model type to api_mode mapping:
    - Google -> api_mode="google" (required)
    - OpenAI -> api_mode="responses" or "chat_completions"
    - vLLM -> api_mode="responses" or "chat_completions"
    - SGLang -> api_mode="chat_completions" (required)
    """

    def load_model(self):
        if self._model_loaded:
            return

        logger.info("Loading API-based model: %s (api_mode=%s)", self.config.model_name, self.config.api_mode)

        if self.config.api_mode == "google":
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
        if self.config.api_mode == "google":
            return self._generate_google_response(model_input)
        elif self.config.api_mode == "responses":
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
        # print(api_kwargs)
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
        gen_kwargs: Dict[str, Any] = dict(self.config.generate_config)
        extra_info = {}

        # Extract system_prompt from config (not a valid API parameter)
        system_prompt = gen_kwargs.pop("_system_prompt", None)
        # Extract reasoning_level for dynamic handling based on model type
        reasoning_level = gen_kwargs.pop("_reasoning_level", None)

        # Extract messages
        if isinstance(model_input, dict) and "messages" in model_input:
            messages = list(model_input["messages"])
        elif isinstance(model_input, list):
            messages = list(model_input)
        else:
            raise ValueError(f"Invalid model_input format for chat_completions API: {type(model_input)}")

        # Inject system message at the beginning if provided
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        if self.config.api_base and "matrixllm" in self.config.api_base:
            gen_kwargs.pop("temperature", None)  # matrixllm API does not support temperature parameter
            # matrixllm uses max_completion_tokens instead of max_tokens
            if "max_tokens" in gen_kwargs:
                gen_kwargs["max_completion_tokens"] = gen_kwargs.pop("max_tokens")

            # Handle reasoning based on model type
            if reasoning_level is not None:
                model_lower = self.model.lower()
                if "gemini" in model_lower:
                    # Gemini 2.5 uses thinking_budget, Gemini 3 uses thinking_level
                    if "gemini-2.5" in model_lower or "gemini-2" in model_lower:
                        # Gemini 2.5: use thinking_budget (0=off, -1=auto, positive=tokens)
                        # Map reasoning_level to budget: low->4096, medium->8192, high->16384
                        budget_map = {"low": 4096, "medium": 8192, "high": 16384}
                        thinking_budget = budget_map.get(reasoning_level, -1)  # -1 = auto
                        gen_kwargs["extra_body"] = {
                            "google": {
                                "thinking_config": {
                                    "include_thoughts": True,
                                    "thinking_budget": thinking_budget,
                                },
                                "thought_tag_marker": "think",
                            }
                        }
                    else:
                        # Gemini 3: use thinking_level
                        gen_kwargs["extra_body"] = {
                            "google": {
                                "thinking_config": {
                                    "include_thoughts": True,
                                    "thinking_level": reasoning_level,
                                },
                                "thought_tag_marker": "think",
                            }
                        }
                else:
                    # For GPT models: use reasoning_effort
                    gen_kwargs["reasoning_effort"] = reasoning_level
        
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
        if self.config.api_mode == "google":
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
