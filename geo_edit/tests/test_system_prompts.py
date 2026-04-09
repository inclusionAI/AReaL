"""Tests for prompts/system_prompts.py - get_system_prompt function."""

import pytest


class TestGetSystemPrompt:
    """Test get_system_prompt function behavior."""

    def test_google_auto_mode_returns_tool_call_prompt(self):
        from geo_edit.prompts.system_prompts import (
            TOOL_CALL_SYSTEM_PROMPT,
            get_system_prompt,
        )

        result = get_system_prompt("Google", "auto")
        assert result == TOOL_CALL_SYSTEM_PROMPT

    def test_google_force_mode_returns_tool_call_prompt(self):
        from geo_edit.prompts.system_prompts import (
            TOOL_CALL_SYSTEM_PROMPT,
            get_system_prompt,
        )

        result = get_system_prompt("Google", "force")
        assert result == TOOL_CALL_SYSTEM_PROMPT

    def test_google_direct_mode_returns_api_no_tool_prompt(self):
        from geo_edit.prompts import API_NO_TOOL_SYSTEM_PROMPT, get_system_prompt

        result = get_system_prompt("Google", "direct")
        assert result == API_NO_TOOL_SYSTEM_PROMPT

    def test_openai_auto_mode_returns_tool_call_prompt(self):
        from geo_edit.prompts.system_prompts import (
            TOOL_CALL_SYSTEM_PROMPT,
            get_system_prompt,
        )

        result = get_system_prompt("OpenAI", "auto")
        assert result == TOOL_CALL_SYSTEM_PROMPT

    def test_openai_direct_mode_returns_api_no_tool_prompt(self):
        from geo_edit.prompts import API_NO_TOOL_SYSTEM_PROMPT, get_system_prompt

        result = get_system_prompt("OpenAI", "direct")
        assert result == API_NO_TOOL_SYSTEM_PROMPT

    def test_vllm_force_mode_includes_force_tool_call_prompt(self):
        from geo_edit.prompts import VLLM_FORCE_TOOL_CALL_PROMPT, get_system_prompt
        from geo_edit.prompts.system_prompts import TOOL_CALL_SYSTEM_PROMPT

        result = get_system_prompt("vLLM", "force")
        assert TOOL_CALL_SYSTEM_PROMPT in result
        assert VLLM_FORCE_TOOL_CALL_PROMPT in result

    def test_vllm_direct_mode_returns_vllm_no_tool_prompt(self):
        from geo_edit.prompts import VLLM_NO_TOOL_SYSTEM_PROMPT, get_system_prompt

        result = get_system_prompt("vLLM", "direct")
        assert result == VLLM_NO_TOOL_SYSTEM_PROMPT

    def test_vllm_auto_mode_returns_none_with_current_logic(self):
        from geo_edit.prompts import get_system_prompt

        result = get_system_prompt("vLLM", "auto")
        assert result is None

    def test_sglang_force_mode_includes_force_tool_call_prompt(self):
        from geo_edit.prompts import VLLM_FORCE_TOOL_CALL_PROMPT, get_system_prompt
        from geo_edit.prompts.system_prompts import TOOL_CALL_SYSTEM_PROMPT

        result = get_system_prompt("SGLang", "force")
        assert TOOL_CALL_SYSTEM_PROMPT in result
        assert VLLM_FORCE_TOOL_CALL_PROMPT in result

    def test_sglang_direct_mode_returns_vllm_no_tool_prompt(self):
        from geo_edit.prompts import VLLM_NO_TOOL_SYSTEM_PROMPT, get_system_prompt

        result = get_system_prompt("SGLang", "direct")
        assert result == VLLM_NO_TOOL_SYSTEM_PROMPT

    def test_sglang_auto_mode_returns_none_with_current_logic(self):
        from geo_edit.prompts import get_system_prompt

        result = get_system_prompt("SGLang", "auto")
        assert result is None

    def test_case_insensitive_and_whitespace_handling(self):
        from geo_edit.prompts.system_prompts import (
            TOOL_CALL_SYSTEM_PROMPT,
            get_system_prompt,
        )

        result = get_system_prompt(" openai ", " auto ")
        assert result == TOOL_CALL_SYSTEM_PROMPT

    def test_tool_mode_none_for_openai_returns_tool_call_prompt(self):
        from geo_edit.prompts.system_prompts import (
            TOOL_CALL_SYSTEM_PROMPT,
            get_system_prompt,
        )

        result = get_system_prompt("OpenAI")
        assert result == TOOL_CALL_SYSTEM_PROMPT


class TestSystemPromptConstants:
    """Test system prompt constant values."""

    def test_tool_call_system_prompt_has_required_tags(self):
        from geo_edit.prompts.system_prompts import TOOL_CALL_SYSTEM_PROMPT

        assert "<think>" in TOOL_CALL_SYSTEM_PROMPT
        assert "</think>" in TOOL_CALL_SYSTEM_PROMPT
        assert "<answer>" in TOOL_CALL_SYSTEM_PROMPT
        assert "</answer>" in TOOL_CALL_SYSTEM_PROMPT

    def test_api_no_tool_system_prompt_has_answer_tags(self):
        from geo_edit.prompts import API_NO_TOOL_SYSTEM_PROMPT

        assert "<answer>" in API_NO_TOOL_SYSTEM_PROMPT
        assert "</answer>" in API_NO_TOOL_SYSTEM_PROMPT

    def test_vllm_no_tool_system_prompt_has_reasoning_and_answer_tags(self):
        from geo_edit.prompts import VLLM_NO_TOOL_SYSTEM_PROMPT

        assert "<think>" in VLLM_NO_TOOL_SYSTEM_PROMPT
        assert "</think>" in VLLM_NO_TOOL_SYSTEM_PROMPT
        assert "<answer>" in VLLM_NO_TOOL_SYSTEM_PROMPT
        assert "</answer>" in VLLM_NO_TOOL_SYSTEM_PROMPT

    def test_vllm_force_tool_call_prompt_content(self):
        from geo_edit.prompts import VLLM_FORCE_TOOL_CALL_PROMPT

        assert "ALWAYS call the appropriate tool first" in VLLM_FORCE_TOOL_CALL_PROMPT
        assert "NEVER provide answers without tool results" in VLLM_FORCE_TOOL_CALL_PROMPT

    def test_tool_execution_prompts_exist(self):
        from geo_edit.prompts import (
            TOOL_EXECUTION_FAILURE_PROMPT,
            TOOL_EXECUTION_SUCCESS_PROMPT,
        )

        success_lower = TOOL_EXECUTION_SUCCESS_PROMPT.lower()
        failure_lower = TOOL_EXECUTION_FAILURE_PROMPT.lower()

        assert "success" in success_lower
        assert ("observation" in success_lower or "image" in success_lower)
        assert "text" in success_lower

        assert "fail" in failure_lower
        assert "text" in failure_lower
        assert ("observation" in failure_lower or "image" in failure_lower)

    def test_user_prompt_has_expected_placeholders(self):
        from geo_edit.prompts.system_prompts import USER_PROMPT

        assert "{task_type}" in USER_PROMPT
        assert "{Question}" in USER_PROMPT
        assert "{output_format}" in USER_PROMPT


class TestBuildUserMessage:
    """Test build_user_message helper function."""

    def test_single_image_basic(self):
        from geo_edit.prompts.system_prompts import build_user_message

        result = build_user_message(
            question="What is shown?",
            num_images=1,
            task_type="chart comprehension",
            output_format="Answer in <answer> tags.",
        )
        assert result.startswith("Observation 0:\n<image>\n")
        assert "What is shown?" in result
        assert "chart comprehension" in result
        assert "Answer in <answer> tags." in result

    def test_multi_image(self):
        from geo_edit.prompts.system_prompts import build_user_message

        result = build_user_message(
            question="Compare these.",
            num_images=3,
            task_type="visual question answering",
        )
        assert "Observation 0:\n<image>\n" in result
        assert "Observation 1:\n<image>\n" in result
        assert "Observation 2:\n<image>\n" in result

    def test_zero_images(self):
        from geo_edit.prompts.system_prompts import build_user_message

        result = build_user_message(question="Text only", num_images=0)
        assert not result.startswith("Observation")
        assert "Text only" in result

    def test_question_prefix_dedup(self):
        from geo_edit.prompts.system_prompts import build_user_message

        result = build_user_message(question="Question: What color?", num_images=0)
        assert "Question: Question:" not in result
        assert "Question: What color?" in result

    def test_default_output_format(self):
        from geo_edit.prompts.system_prompts import build_user_message, DEFAULT_OUTPUT_FORMAT

        result = build_user_message(question="Test", num_images=0)
        assert DEFAULT_OUTPUT_FORMAT in result


class TestEvalPrompts:
    """Test evaluation prompts."""

    def test_eval_prompts_exist(self):
        from geo_edit.prompts import EVAL_QUERY_PROMPT, EVAL_SYSTEM_PROMPT

        assert EVAL_SYSTEM_PROMPT is not None
        assert EVAL_QUERY_PROMPT is not None
        assert len(EVAL_SYSTEM_PROMPT) > 0
        assert len(EVAL_QUERY_PROMPT) > 0


class TestToolAgentPrompts:
    """Test tool agent prompts."""

    def test_get_tool_agent_prompt_returns_string(self):
        from geo_edit.prompts import get_tool_agent_prompt

        result = get_tool_agent_prompt("multimath")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_tool_agent_prompt_returns_empty_for_unknown(self):
        from geo_edit.prompts import get_tool_agent_prompt

        result = get_tool_agent_prompt("unknown_agent")
        assert result == ""

    def test_list_tool_agents_returns_list(self):
        from geo_edit.prompts import list_tool_agents

        result = list_tool_agents()
        assert isinstance(result, list)

    def test_list_tool_agents_contains_known_agents(self):
        from geo_edit.prompts import list_tool_agents

        result = list_tool_agents()
        assert "multimath" in result
        assert "gllava" in result
        assert "chartmoe" in result
