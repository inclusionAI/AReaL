"""Unit tests for MiniMax workflow agents."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMiniMaxMathAgent:
    """Tests for MiniMax MathAgent workflow."""

    def test_import(self):
        """MiniMax workflow module can be imported."""
        from areal.workflow.minimax.math_agent import MathAgent, MultiTurnMathAgent

        assert MathAgent is not None
        assert MultiTurnMathAgent is not None

    def test_init_from_package(self):
        """MiniMax agents can be imported from package __init__."""
        from areal.workflow.minimax import MathAgent, MultiTurnMathAgent

        assert MathAgent is not None
        assert MultiTurnMathAgent is not None

    def test_math_agent_init_default_kwargs(self):
        """MathAgent sets default temperature to 1.0."""
        from areal.workflow.minimax.math_agent import MathAgent

        agent = MathAgent()
        assert agent.kwargs.get("temperature") == 1.0

    def test_math_agent_init_strips_max_tokens(self):
        """MathAgent strips max_tokens from kwargs."""
        from areal.workflow.minimax.math_agent import MathAgent

        agent = MathAgent(max_tokens=1024, max_turns=5, temperature=0.8)
        assert "max_tokens" not in agent.kwargs
        assert "max_turns" not in agent.kwargs
        assert agent.kwargs["temperature"] == 0.8

    def test_multi_turn_agent_init_default(self):
        """MultiTurnMathAgent initializes with correct defaults."""
        from areal.workflow.minimax.math_agent import MultiTurnMathAgent

        agent = MultiTurnMathAgent()
        assert agent.max_turns == 8
        assert agent.kwargs.get("temperature") == 1.0

    def test_multi_turn_agent_custom_turns(self):
        """MultiTurnMathAgent respects custom max_turns."""
        from areal.workflow.minimax.math_agent import MultiTurnMathAgent

        agent = MultiTurnMathAgent(max_turns=3)
        assert agent.max_turns == 3

    def test_default_base_url(self):
        """Default base URL points to MiniMax overseas API."""
        from areal.workflow.minimax.math_agent import MINIMAX_DEFAULT_BASE_URL

        assert MINIMAX_DEFAULT_BASE_URL == "https://api.minimax.io/v1"

    @pytest.mark.asyncio
    async def test_math_agent_uses_minimax_env_vars(self):
        """MathAgent reads MINIMAX_API_KEY and MINIMAX_BASE_URL from env."""
        from areal.workflow.minimax.math_agent import MathAgent

        agent = MathAgent()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "42"

        with (
            patch("areal.workflow.minimax.math_agent.AsyncOpenAI") as mock_openai_cls,
            patch.dict(
                "os.environ",
                {
                    "MINIMAX_API_KEY": "test-minimax-key",
                    "MINIMAX_BASE_URL": "https://custom.minimax.io/v1",
                },
            ),
            patch(
                "areal.workflow.minimax.math_agent.math_reward_fn", return_value=1.0
            ),
        ):
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai_cls.return_value = mock_client

            data = {
                "messages": [{"role": "user", "content": "What is 6*7?"}],
                "answer": "42",
            }
            await agent.run(data)

            mock_openai_cls.assert_called_once_with(
                base_url="https://custom.minimax.io/v1",
                api_key="test-minimax-key",
                http_client=None,
                max_retries=0,
            )

    @pytest.mark.asyncio
    async def test_math_agent_uses_default_base_url(self):
        """MathAgent falls back to default base URL when env var is not set."""
        from areal.workflow.minimax.math_agent import MathAgent

        agent = MathAgent()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "42"

        with (
            patch("areal.workflow.minimax.math_agent.AsyncOpenAI") as mock_openai_cls,
            patch.dict(
                "os.environ",
                {"MINIMAX_API_KEY": "test-key"},
                clear=False,
            ),
            patch(
                "areal.workflow.minimax.math_agent.math_reward_fn", return_value=1.0
            ),
        ):
            # Ensure MINIMAX_BASE_URL is not set
            import os

            os.environ.pop("MINIMAX_BASE_URL", None)

            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai_cls.return_value = mock_client

            data = {
                "messages": [{"role": "user", "content": "What is 6*7?"}],
                "answer": "42",
            }
            await agent.run(data)

            mock_openai_cls.assert_called_once_with(
                base_url="https://api.minimax.io/v1",
                api_key="test-key",
                http_client=None,
                max_retries=0,
            )

    @pytest.mark.asyncio
    async def test_math_agent_extra_kwargs_override_env(self):
        """extra_kwargs take priority over environment variables."""
        from areal.workflow.minimax.math_agent import MathAgent

        agent = MathAgent()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "42"

        with (
            patch("areal.workflow.minimax.math_agent.AsyncOpenAI") as mock_openai_cls,
            patch.dict(
                "os.environ",
                {
                    "MINIMAX_API_KEY": "env-key",
                    "MINIMAX_BASE_URL": "https://env.minimax.io/v1",
                },
            ),
            patch(
                "areal.workflow.minimax.math_agent.math_reward_fn", return_value=1.0
            ),
        ):
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai_cls.return_value = mock_client

            mock_http_client = MagicMock()
            data = {
                "messages": [{"role": "user", "content": "What is 6*7?"}],
                "answer": "42",
            }
            await agent.run(
                data,
                base_url="https://proxy.example.com/v1",
                api_key="proxy-key",
                http_client=mock_http_client,
            )

            mock_openai_cls.assert_called_once_with(
                base_url="https://proxy.example.com/v1",
                api_key="proxy-key",
                http_client=mock_http_client,
                max_retries=0,
            )
