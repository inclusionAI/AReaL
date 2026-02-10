"""Tests for prompts/system_prompts.py - get_system_prompt function."""

import pytest


class TestGetSystemPrompt:
    """Test get_system_prompt function."""

    def test_google_auto_mode_returns_api_call_prompt(self):
        from geo_edit.prompts import get_system_prompt, API_CALL_SYSTEM_PROMPT

        result = get_system_prompt("Google", "auto")
        assert result == API_CALL_SYSTEM_PROMPT

    def test_google_force_mode_returns_api_call_prompt(self):
        from geo_edit.prompts import get_system_prompt, API_CALL_SYSTEM_PROMPT

        result = get_system_prompt("Google", "force")
        assert result == API_CALL_SYSTEM_PROMPT

    def test_google_direct_mode_returns_no_tool_prompt(self):
        from geo_edit.prompts import get_system_prompt, API_NO_TOOL_SYSTEM_PROMPT

        result = get_system_prompt("Google", "direct")
        assert result == API_NO_TOOL_SYSTEM_PROMPT

    def test_openai_auto_mode_returns_api_call_prompt(self):
        from geo_edit.prompts import get_system_prompt, API_CALL_SYSTEM_PROMPT

        result = get_system_prompt("OpenAI", "auto")
        assert result == API_CALL_SYSTEM_PROMPT

    def test_openai_direct_mode_returns_no_tool_prompt(self):
        from geo_edit.prompts import get_system_prompt, API_NO_TOOL_SYSTEM_PROMPT

        result = get_system_prompt("OpenAI", "direct")
        assert result == API_NO_TOOL_SYSTEM_PROMPT

    def test_vllm_auto_mode_returns_vllm_prompt(self):
        from geo_edit.prompts import get_system_prompt, VLLM_SYSTEM_PROMPT

        result = get_system_prompt("vLLM", "auto")
        assert result == VLLM_SYSTEM_PROMPT

    def test_vllm_force_mode_includes_force_tool_call_prompt(self):
        from geo_edit.prompts import get_system_prompt, VLLM_SYSTEM_PROMPT, VLLM_FORCE_TOOL_CALL_PROMPT

        result = get_system_prompt("vLLM", "force")
        assert VLLM_SYSTEM_PROMPT in result
        assert VLLM_FORCE_TOOL_CALL_PROMPT in result

    def test_vllm_direct_mode_returns_no_tool_prompt(self):
        from geo_edit.prompts import get_system_prompt, VLLM_NO_TOOL_SYSTEM_PROMPT

        result = get_system_prompt("vLLM", "direct")
        assert result == VLLM_NO_TOOL_SYSTEM_PROMPT

    def test_sglang_auto_mode_returns_vllm_prompt(self):
        from geo_edit.prompts import get_system_prompt, VLLM_SYSTEM_PROMPT

        result = get_system_prompt("SGLang", "auto")
        assert result == VLLM_SYSTEM_PROMPT

    def test_sglang_force_mode_includes_force_tool_call_prompt(self):
        from geo_edit.prompts import get_system_prompt, VLLM_SYSTEM_PROMPT, VLLM_FORCE_TOOL_CALL_PROMPT

        result = get_system_prompt("SGLang", "force")
        assert VLLM_SYSTEM_PROMPT in result
        assert VLLM_FORCE_TOOL_CALL_PROMPT in result

    def test_sglang_direct_mode_returns_no_tool_prompt(self):
        from geo_edit.prompts import get_system_prompt, VLLM_NO_TOOL_SYSTEM_PROMPT

        result = get_system_prompt("SGLang", "direct")
        assert result == VLLM_NO_TOOL_SYSTEM_PROMPT

    def test_case_insensitive_model_type(self):
        from geo_edit.prompts import get_system_prompt, VLLM_SYSTEM_PROMPT

        result = get_system_prompt("VLLM", "auto")
        assert result == VLLM_SYSTEM_PROMPT

        result = get_system_prompt("sglang", "auto")
        assert result == VLLM_SYSTEM_PROMPT


class TestSystemPromptConstants:
    """Test system prompt constant values."""

    def test_api_call_system_prompt_has_tool_instructions(self):
        from geo_edit.prompts import API_CALL_SYSTEM_PROMPT

        assert "<think>" in API_CALL_SYSTEM_PROMPT
        assert "</think>" in API_CALL_SYSTEM_PROMPT
        assert "<answer>" in API_CALL_SYSTEM_PROMPT
        assert "</answer>" in API_CALL_SYSTEM_PROMPT

    def test_api_no_tool_system_prompt_has_answer_tags(self):
        from geo_edit.prompts import API_NO_TOOL_SYSTEM_PROMPT

        assert "<answer>" in API_NO_TOOL_SYSTEM_PROMPT
        assert "</answer>" in API_NO_TOOL_SYSTEM_PROMPT

    def test_vllm_system_prompt_has_tool_instructions(self):
        from geo_edit.prompts import VLLM_SYSTEM_PROMPT

        assert "<think>" in VLLM_SYSTEM_PROMPT
        assert "</think>" in VLLM_SYSTEM_PROMPT
        assert "<answer>" in VLLM_SYSTEM_PROMPT
        assert "</answer>" in VLLM_SYSTEM_PROMPT

    def test_vllm_no_tool_system_prompt_has_reasoning_instructions(self):
        from geo_edit.prompts import VLLM_NO_TOOL_SYSTEM_PROMPT

        assert "<think>" in VLLM_NO_TOOL_SYSTEM_PROMPT
        assert "</think>" in VLLM_NO_TOOL_SYSTEM_PROMPT
        assert "<answer>" in VLLM_NO_TOOL_SYSTEM_PROMPT
        assert "</answer>" in VLLM_NO_TOOL_SYSTEM_PROMPT

    def test_vllm_force_tool_call_prompt_content(self):
        from geo_edit.prompts import VLLM_FORCE_TOOL_CALL_PROMPT

        assert "MUST call a tool" in VLLM_FORCE_TOOL_CALL_PROMPT

    def test_tool_execution_prompts_exist(self):
        from geo_edit.prompts import TOOL_EXECUTION_SUCCESS_PROMPT, TOOL_EXECUTION_FAILURE_PROMPT

        assert "success" in TOOL_EXECUTION_SUCCESS_PROMPT.lower() or "new image" in TOOL_EXECUTION_SUCCESS_PROMPT.lower()
        assert "fail" in TOOL_EXECUTION_FAILURE_PROMPT.lower()


class TestEvalPrompts:
    """Test evaluation prompts."""

    def test_eval_prompts_exist(self):
        from geo_edit.prompts import EVAL_SYSTEM_PROMPT, EVAL_QUERY_PROMPT

        assert EVAL_SYSTEM_PROMPT is not None
        assert EVAL_QUERY_PROMPT is not None
        assert len(EVAL_SYSTEM_PROMPT) > 0
        assert len(EVAL_QUERY_PROMPT) > 0


class TestToolAgentPrompts:
    """Test tool agent prompts."""

    def test_default_tool_agent_prompt_exists(self):
        from geo_edit.prompts import DEFAULT_TOOL_AGENT_PROMPT

        assert DEFAULT_TOOL_AGENT_PROMPT is not None
        assert len(DEFAULT_TOOL_AGENT_PROMPT) > 0

    def test_get_tool_agent_prompt_returns_string(self):
        from geo_edit.prompts import get_tool_agent_prompt

        result = get_tool_agent_prompt("multimath")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_list_tool_agents_returns_list(self):
        from geo_edit.prompts import list_tool_agents

        result = list_tool_agents()
        assert isinstance(result, list)
