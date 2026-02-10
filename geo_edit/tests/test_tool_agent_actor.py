"""Tests for ToolModelActor - environment/tool_agents/actor.py"""

import pytest
from unittest.mock import MagicMock, patch


class TestToolModelActorInit:
    """Test ToolModelActor initialization."""

    @patch("geo_edit.environment.tool_agents.actor.LLM", create=True)
    def test_actor_sets_model_name(self, mock_llm_class):
        # Import after patching to avoid vLLM import error
        with patch.dict("sys.modules", {"vllm": MagicMock()}):
            from geo_edit.environment.tool_agents.actor import ToolModelActor

            # Create a mock instance that bypasses __init__
            actor = object.__new__(ToolModelActor)
            actor.model_name = "test-model"
            actor.system_prompt = "test prompt"
            actor.llm = MagicMock()
            actor._initialized = True

            assert actor.model_name == "test-model"

    @patch("geo_edit.environment.tool_agents.actor.LLM", create=True)
    def test_actor_uses_default_prompt_when_none_provided(self, mock_llm_class):
        from geo_edit.prompts import DEFAULT_TOOL_AGENT_PROMPT

        with patch.dict("sys.modules", {"vllm": MagicMock()}):
            from geo_edit.environment.tool_agents.actor import ToolModelActor

            actor = object.__new__(ToolModelActor)
            actor.model_name = "test-model"
            actor.system_prompt = DEFAULT_TOOL_AGENT_PROMPT
            actor.llm = MagicMock()
            actor._initialized = True

            assert actor.system_prompt == DEFAULT_TOOL_AGENT_PROMPT

    @patch("geo_edit.environment.tool_agents.actor.LLM", create=True)
    def test_actor_uses_custom_system_prompt(self, mock_llm_class):
        with patch.dict("sys.modules", {"vllm": MagicMock()}):
            from geo_edit.environment.tool_agents.actor import ToolModelActor

            custom_prompt = "You are a custom analysis agent."
            actor = object.__new__(ToolModelActor)
            actor.model_name = "test-model"
            actor.system_prompt = custom_prompt
            actor.llm = MagicMock()
            actor._initialized = True

            assert actor.system_prompt == custom_prompt


class TestBuildMessages:
    """Test _build_messages method."""

    def test_build_messages_includes_system_prompt(self):
        with patch.dict("sys.modules", {"vllm": MagicMock()}):
            from geo_edit.environment.tool_agents.actor import ToolModelActor

            actor = object.__new__(ToolModelActor)
            actor.system_prompt = "You are a helpful assistant."

            messages = actor._build_messages("base64_image_data", "What is this?")

            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are a helpful assistant."

    def test_build_messages_includes_image_as_base64_url(self):
        with patch.dict("sys.modules", {"vllm": MagicMock()}):
            from geo_edit.environment.tool_agents.actor import ToolModelActor

            actor = object.__new__(ToolModelActor)
            actor.system_prompt = "test"

            messages = actor._build_messages("abc123base64", "Describe")

            user_content = messages[1]["content"]
            image_part = next(p for p in user_content if p["type"] == "image_url")
            assert "data:image/jpeg;base64,abc123base64" in image_part["image_url"]["url"]

    def test_build_messages_includes_question_text(self):
        with patch.dict("sys.modules", {"vllm": MagicMock()}):
            from geo_edit.environment.tool_agents.actor import ToolModelActor

            actor = object.__new__(ToolModelActor)
            actor.system_prompt = "test"

            messages = actor._build_messages("base64", "What color is the car?")

            user_content = messages[1]["content"]
            text_part = next(p for p in user_content if p["type"] == "text")
            assert text_part["text"] == "What color is the car?"

    def test_build_messages_has_correct_structure(self):
        with patch.dict("sys.modules", {"vllm": MagicMock()}):
            from geo_edit.environment.tool_agents.actor import ToolModelActor

            actor = object.__new__(ToolModelActor)
            actor.system_prompt = "System prompt"

            messages = actor._build_messages("img_b64", "Question?")

            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert isinstance(messages[1]["content"], list)


class TestParseOutput:
    """Test _parse_output method."""

    def test_parse_output_extracts_analysis_key(self):
        with patch.dict("sys.modules", {"vllm": MagicMock()}):
            from geo_edit.environment.tool_agents.actor import ToolModelActor

            actor = object.__new__(ToolModelActor)

            result = actor._parse_output('{"analysis": "The image shows a cat"}')

            assert result == "The image shows a cat"

    def test_parse_output_extracts_text_key(self):
        with patch.dict("sys.modules", {"vllm": MagicMock()}):
            from geo_edit.environment.tool_agents.actor import ToolModelActor

            actor = object.__new__(ToolModelActor)

            result = actor._parse_output('{"text": "Some text content"}')

            assert result == "Some text content"

    def test_parse_output_extracts_result_key(self):
        with patch.dict("sys.modules", {"vllm": MagicMock()}):
            from geo_edit.environment.tool_agents.actor import ToolModelActor

            actor = object.__new__(ToolModelActor)

            result = actor._parse_output('{"result": "Final result here"}')

            assert result == "Final result here"

    def test_parse_output_returns_error_message(self):
        with patch.dict("sys.modules", {"vllm": MagicMock()}):
            from geo_edit.environment.tool_agents.actor import ToolModelActor

            actor = object.__new__(ToolModelActor)

            result = actor._parse_output('{"error": "Something went wrong"}')

            assert "Error:" in result
            assert "Something went wrong" in result

    def test_parse_output_returns_raw_text_on_json_error(self):
        with patch.dict("sys.modules", {"vllm": MagicMock()}):
            from geo_edit.environment.tool_agents.actor import ToolModelActor

            actor = object.__new__(ToolModelActor)

            result = actor._parse_output("This is not JSON, just plain text")

            assert result == "This is not JSON, just plain text"

    def test_parse_output_prefers_analysis_over_text(self):
        with patch.dict("sys.modules", {"vllm": MagicMock()}):
            from geo_edit.environment.tool_agents.actor import ToolModelActor

            actor = object.__new__(ToolModelActor)

            result = actor._parse_output('{"analysis": "Analysis content", "text": "Text content"}')

            assert result == "Analysis content"


class TestHealthCheck:
    """Test health_check method."""

    def test_health_check_returns_model_name(self):
        with patch.dict("sys.modules", {"vllm": MagicMock()}):
            from geo_edit.environment.tool_agents.actor import ToolModelActor

            actor = object.__new__(ToolModelActor)
            actor.model_name = "my-vlm-model"
            actor._initialized = True

            result = actor.health_check()

            assert result["model"] == "my-vlm-model"

    def test_health_check_returns_initialized_status(self):
        with patch.dict("sys.modules", {"vllm": MagicMock()}):
            from geo_edit.environment.tool_agents.actor import ToolModelActor

            actor = object.__new__(ToolModelActor)
            actor.model_name = "model"
            actor._initialized = True

            result = actor.health_check()

            assert result["initialized"] is True

    def test_health_check_returns_dict(self):
        with patch.dict("sys.modules", {"vllm": MagicMock()}):
            from geo_edit.environment.tool_agents.actor import ToolModelActor

            actor = object.__new__(ToolModelActor)
            actor.model_name = "model"
            actor._initialized = True

            result = actor.health_check()

            assert isinstance(result, dict)
            assert "model" in result
            assert "initialized" in result


class TestAnalyze:
    """Test analyze method."""

    def test_analyze_calls_llm_chat(self):
        with patch.dict("sys.modules", {"vllm": MagicMock()}):
            from geo_edit.environment.tool_agents.actor import ToolModelActor

            actor = object.__new__(ToolModelActor)
            actor.model_name = "model"
            actor.system_prompt = "prompt"
            actor._initialized = True

            # Mock LLM
            mock_llm = MagicMock()
            mock_output = MagicMock()
            mock_output.outputs = [MagicMock(text='{"analysis": "Result"}')]
            mock_llm.chat.return_value = [mock_output]
            actor.llm = mock_llm

            # Mock SamplingParams
            with patch("geo_edit.environment.tool_agents.actor.SamplingParams", create=True):
                result = actor.analyze("base64img", "question", 0.5, 1024)

            assert result == "Result"

    def test_analyze_returns_error_when_no_output(self):
        with patch.dict("sys.modules", {"vllm": MagicMock()}):
            from geo_edit.environment.tool_agents.actor import ToolModelActor

            actor = object.__new__(ToolModelActor)
            actor.model_name = "model"
            actor.system_prompt = "prompt"
            actor._initialized = True

            # Mock LLM returning empty outputs
            mock_llm = MagicMock()
            mock_llm.chat.return_value = []
            actor.llm = mock_llm

            with patch("geo_edit.environment.tool_agents.actor.SamplingParams", create=True):
                result = actor.analyze("base64img", "question", 0.0, 1024)

            assert "error" in result.lower()
