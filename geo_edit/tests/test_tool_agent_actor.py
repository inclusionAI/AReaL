"""Tests for BaseToolModelActor - environment/tool_agents/actor.py

These tests verify that BaseToolModelActor is a proper abstract base class
and that concrete implementations (MultiMathActor, GLLaVAActor, ChartMoEActor)
follow the expected interface.
"""

import pytest
from abc import ABC
from unittest.mock import MagicMock, patch


class TestBaseToolModelActor:
    """Test BaseToolModelActor abstract base class."""

    def test_base_actor_is_abstract(self):
        from geo_edit.environment.tool_agents.actor import BaseToolModelActor

        assert issubclass(BaseToolModelActor, ABC)

    def test_base_actor_cannot_be_instantiated(self):
        from geo_edit.environment.tool_agents.actor import BaseToolModelActor

        with pytest.raises(TypeError):
            BaseToolModelActor(
                model_name="test",
                max_model_len=8192,
                gpu_memory_utilization=0.8,
                system_prompt="test",
            )

    def test_base_actor_has_required_abstract_methods(self):
        from geo_edit.environment.tool_agents.actor import BaseToolModelActor

        # Check abstract methods exist
        assert hasattr(BaseToolModelActor, "__init__")
        assert hasattr(BaseToolModelActor, "analyze")
        assert hasattr(BaseToolModelActor, "health_check")


class TestConcreteActorClasses:
    """Test that concrete Actor classes properly inherit from BaseToolModelActor."""

    def test_multimath_actor_inherits_from_base(self):
        from geo_edit.environment.tool_agents.actor import BaseToolModelActor
        from geo_edit.tool_definitions.agents.multimath import MultiMathActor

        assert issubclass(MultiMathActor, BaseToolModelActor)

    def test_gllava_actor_inherits_from_base(self):
        from geo_edit.environment.tool_agents.actor import BaseToolModelActor
        from geo_edit.tool_definitions.agents.gllava import GLLaVAActor

        assert issubclass(GLLaVAActor, BaseToolModelActor)

    def test_chartmoe_actor_inherits_from_base(self):
        from geo_edit.environment.tool_agents.actor import BaseToolModelActor
        from geo_edit.tool_definitions.agents.chartmoe import ChartMoEActor

        assert issubclass(ChartMoEActor, BaseToolModelActor)


class TestActorClassRegistry:
    """Test ACTOR_CLASS exports from agent modules."""

    def test_multimath_exports_actor_class(self):
        from geo_edit.tool_definitions.agents import multimath

        assert hasattr(multimath, "ACTOR_CLASS")
        assert multimath.ACTOR_CLASS.__name__ == "MultiMathActor"

    def test_gllava_exports_actor_class(self):
        from geo_edit.tool_definitions.agents import gllava

        assert hasattr(gllava, "ACTOR_CLASS")
        assert gllava.ACTOR_CLASS.__name__ == "GLLaVAActor"

    def test_chartmoe_exports_actor_class(self):
        from geo_edit.tool_definitions.agents import chartmoe

        assert hasattr(chartmoe, "ACTOR_CLASS")
        assert chartmoe.ACTOR_CLASS.__name__ == "ChartMoEActor"


class TestGetActorClass:
    """Test get_actor_class function."""

    def test_get_actor_class_returns_multimath_actor(self):
        from geo_edit.tool_definitions.agents import get_actor_class
        from geo_edit.tool_definitions.agents.multimath import MultiMathActor

        assert get_actor_class("multimath") is MultiMathActor

    def test_get_actor_class_returns_gllava_actor(self):
        from geo_edit.tool_definitions.agents import get_actor_class
        from geo_edit.tool_definitions.agents.gllava import GLLaVAActor

        assert get_actor_class("gllava") is GLLaVAActor

    def test_get_actor_class_returns_chartmoe_actor(self):
        from geo_edit.tool_definitions.agents import get_actor_class
        from geo_edit.tool_definitions.agents.chartmoe import ChartMoEActor

        assert get_actor_class("chartmoe") is ChartMoEActor

    def test_get_actor_class_raises_for_unknown_agent(self):
        from geo_edit.tool_definitions.agents import get_actor_class

        with pytest.raises(KeyError):
            get_actor_class("unknown_agent")


class TestActorParseOutput:
    """Test _parse_output method in concrete actors."""

    def _create_mock_actor(self, ActorClass):
        """Create a mock actor without actually loading the model."""
        actor = object.__new__(ActorClass)
        actor.model_name = "test-model"
        actor.system_prompt = "test"
        actor.max_model_len = 8192
        actor._initialized = True
        return actor

    def test_parse_output_extracts_analysis_key(self):
        from geo_edit.tool_definitions.agents.multimath import MultiMathActor

        actor = self._create_mock_actor(MultiMathActor)
        result = actor._parse_output('{"analysis": "The image shows a cat"}')

        assert result == "The image shows a cat"

    def test_parse_output_extracts_text_key(self):
        from geo_edit.tool_definitions.agents.gllava import GLLaVAActor

        actor = self._create_mock_actor(GLLaVAActor)
        result = actor._parse_output('{"text": "Some text content"}')

        assert result == "Some text content"

    def test_parse_output_extracts_result_key(self):
        from geo_edit.tool_definitions.agents.chartmoe import ChartMoEActor

        actor = self._create_mock_actor(ChartMoEActor)
        result = actor._parse_output('{"result": "Final result here"}')

        assert result == "Final result here"

    def test_parse_output_returns_error_message(self):
        from geo_edit.tool_definitions.agents.multimath import MultiMathActor

        actor = self._create_mock_actor(MultiMathActor)
        result = actor._parse_output('{"error": "Something went wrong"}')

        assert "Error:" in result
        assert "Something went wrong" in result

    def test_parse_output_returns_raw_text_on_json_error(self):
        from geo_edit.tool_definitions.agents.gllava import GLLaVAActor

        actor = self._create_mock_actor(GLLaVAActor)
        result = actor._parse_output("This is not JSON, just plain text")

        assert result == "This is not JSON, just plain text"


class TestActorHealthCheck:
    """Test health_check method in concrete actors."""

    def _create_mock_actor(self, ActorClass):
        """Create a mock actor without actually loading the model."""
        actor = object.__new__(ActorClass)
        actor.model_name = "test-model"
        actor.system_prompt = "test"
        actor.max_model_len = 8192
        actor._initialized = True
        return actor

    def test_health_check_returns_model_name(self):
        from geo_edit.tool_definitions.agents.multimath import MultiMathActor

        actor = self._create_mock_actor(MultiMathActor)
        result = actor.health_check()

        assert result["model"] == "test-model"

    def test_health_check_returns_initialized_status(self):
        from geo_edit.tool_definitions.agents.gllava import GLLaVAActor

        actor = self._create_mock_actor(GLLaVAActor)
        result = actor.health_check()

        assert result["initialized"] is True

    def test_health_check_returns_dict(self):
        from geo_edit.tool_definitions.agents.chartmoe import ChartMoEActor

        actor = self._create_mock_actor(ChartMoEActor)
        result = actor.health_check()

        assert isinstance(result, dict)
        assert "model" in result
        assert "initialized" in result
