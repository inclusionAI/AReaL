import sys
import types


if "ray" not in sys.modules:
    ray_module = types.ModuleType("ray")
    ray_util_module = types.ModuleType("ray.util")
    ray_sched_module = types.ModuleType("ray.util.scheduling_strategies")
    ray_sched_module.NodeAffinitySchedulingStrategy = None
    sys.modules["ray"] = ray_module
    sys.modules["ray.util"] = ray_util_module
    sys.modules["ray.util.scheduling_strategies"] = ray_sched_module

from geo_edit.agents.api_agent import APIBasedAgent
from geo_edit.agents.base import AgentConfig


def test_openai_load_model_passes_base_url_when_provided(monkeypatch):
    import openai

    captured = {}

    def _fake_openai_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(openai, "OpenAI", _fake_openai_client)

    agent = APIBasedAgent(
        AgentConfig(
            model_type="OpenAI",
            model_name="gpt-5-mini-20250807",
            api_key="test-key",
            api_base="https://llm-proxy.perflab.nvidia.com/openai/v1",
        )
    )
    agent.load_model()

    assert captured == {
        "api_key": "test-key",
        "base_url": "https://llm-proxy.perflab.nvidia.com/openai/v1",
    }


def test_openai_load_model_keeps_original_behavior_without_base_url(monkeypatch):
    import openai

    captured = {}

    def _fake_openai_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(openai, "OpenAI", _fake_openai_client)

    agent = APIBasedAgent(
        AgentConfig(
            model_type="OpenAI",
            model_name="gpt-5-mini-20250807",
            api_key="test-key",
        )
    )
    agent.load_model()

    assert captured == {"api_key": "test-key"}
