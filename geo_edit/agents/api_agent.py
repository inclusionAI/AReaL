import logging
import time
from typing import Any, Dict, Tuple

from geo_edit.agents.base import AgentConfig, BaseAgent
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

class APIBasedAgent(BaseAgent):
    """Agent that interacts with an external API for generation."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.client = None
        self._model_loaded = False

    def load_model(self):
        # create client for external API
        if self._model_loaded:
            return

        logger.info(f"Loading API-based model: {self.config.model_name}")
        
        if self.config.api_key is None:
            raise ValueError("API key must be provided for API-based agents.")
        
        if self.config.model_type == "Google":
            from google import genai
            self.client = genai.Client(api_key=self.config.api_key)
            self.model= self.config.model_name
            self._model_loaded = True
            logger.info(f"API-based model {self.config.model_name} loaded successfully.")
        elif self.config.model_type == "OpenAI":
            from openai import OpenAI
            self.client = OpenAI(api_key=self.config.api_key)
            self.model = self.config.model_name
            self._model_loaded = True
            logger.info(f"API-based model {self.config.model_name} loaded successfully.")
        else:
            raise NotImplementedError(f"Model type {self.config.model_type} not supported yet.")
    
    def _prepare_input(self, observation: Dict[str, Any]) -> Any:
        # API agent does not need to prepare input as the input is directly a list of dict.
        return observation
    
    def _parse_response(self, raw_response: str, observation: Dict[str, Any]) -> str:
        # API agent does not need to parse response as the response is directly returned.
        return raw_response

    def _generate_response(self, model_input: Any) -> Tuple[Any, Dict[str, Any]]:
        """Generate response using the model
        
        Args:
            model_input: list of dict containing the input for the API call
            
        Returns:
            response: Generated text
            extra_info: Additional information
        """
        
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
                extra_info["tokens_used"] = response.usage_metadata.total_token_count
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
            extra_info["tokens_used"] = response.usage.total_tokens
            return response, extra_info

        raise NotImplementedError(f"Model type {self.config.model_type} not supported yet.")

    def act(self, observation: Any) -> Tuple[Any, Dict[str, Any]]:
        if self.client is None:
            self.load_model()

        for attempt in range(self.config.n_retry):
            try:
                model_input=observation
                
                content, extra_info = self._generate_response(model_input)
                if self.config.model_type == "Google":
                    if content.parts is None:
                        logging.warning(f"Generated content parts is None: {content}")
                        raise ValueError("Generated content is None.")
                elif self.config.model_type == "OpenAI":
                    if not content.output:
                        logging.warning(f"Generated content output is empty: {content}")
                        raise ValueError("Generated content is empty.")
            
                
                extra_info.update({
                    "model_name": self.config.model_name,
                    "attempt": attempt + 1,
                    "step_count": self.step_count ,
                })
                
                self.step_count += 1
                if 'tokens_used' in extra_info:
                    self.total_tokens_used += extra_info['tokens_used']
                    
                logger.info(f"Step {self.step_count}: Generated response in attempt {attempt + 1}")

                return content, extra_info
            
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.config.n_retry - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    error_msg = f"Failed after {self.config.n_retry} attempts: {str(e)}"
                    logger.error(error_msg)
                    
                    raise RuntimeError(error_msg)

    def reset(self):
        """Reset agent state"""
        self.step_count = 0
        self.total_tokens_used = 0
        self.client=None
        self._model_loaded=False
        logger.info("Agent state reset")
        
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state
        
        Returns:
            Dictionary containing agent state
        """
        return {
            'step_count': self.step_count,
            'total_tokens_used': self.total_tokens_used,
            'total_cost': sum(self.cost),
            'config': self.config.__dict__,
        }
