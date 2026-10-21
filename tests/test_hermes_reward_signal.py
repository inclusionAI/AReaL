import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import torch
import yaml

from areal.api.cli_args import PPOActorConfig

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "examples/hermes/config.yaml"


def _load_config() -> dict:
    return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))


def test_hermes_singleton_online_config_preserves_each_outcome_signal():
    config = _load_config()
    actor = config["actor"]

    assert config["gconfig"]["n_samples"] == 1
    assert config["train_dataset"]["batch_size"] == 1
    assert actor.get("reward_norm") is None
    assert actor.get("adv_norm") is None

    counterfactual_outcomes = torch.tensor([0.0, 1.0])
    actor_defaults = PPOActorConfig()
    reward_bias = actor.get("reward_bias", actor_defaults.reward_bias)
    reward_scaling = actor.get("reward_scaling", actor_defaults.reward_scaling)
    signed_rewards = (counterfactual_outcomes + reward_bias) * reward_scaling

    for signed_reward in signed_rewards:
        single_trajectory_token_advantages = signed_reward.expand(3)
        assert torch.count_nonzero(single_trajectory_token_advantages).item() == 3

    assert signed_rewards[0] < 0 < signed_rewards[1]
    assert not torch.equal(signed_rewards[0], signed_rewards[1])


@pytest.mark.parametrize("on_demand", [False, True])
def test_hermes_carries_episode_identity_into_every_llm_request(monkeypatch, on_demand):
    from examples.hermes.hermes import HermesAgent, _Upstream

    module = ModuleType("run_agent")
    module.AIAgent = MagicMock()
    monkeypatch.setitem(sys.modules, "run_agent", module)
    metadata = {
        "base_url": "http://inference",
        "api_key": "session-key",
        "model": "Qwen",
    }
    if on_demand:
        metadata["episode"] = {"incarnation_id": "source", "trajectory_id": 3}
    upstream = _Upstream.from_inference(metadata)
    HermesAgent()._build_agent(upstream)
    kwargs = module.AIAgent.call_args.kwargs
    if on_demand:
        assert kwargs["request_overrides"] == {
            "extra_body": {"areal_episode": metadata["episode"]}
        }
        assert (
            _Upstream.from_inference(
                {
                    **metadata,
                    "episode": {"incarnation_id": "source", "trajectory_id": 4},
                }
            )
            != upstream
        )
    else:
        assert "request_overrides" not in kwargs


@pytest.mark.asyncio
async def test_hermes_replaced_episode_cannot_run_a_closed_instance():
    from types import SimpleNamespace

    from examples.hermes.hermes import HermesAgent

    agent = HermesAgent()
    old = SimpleNamespace(session_key="session", agent=MagicMock())
    agent._sessions["session"] = SimpleNamespace(
        session_key="session", agent=MagicMock()
    )
    with pytest.raises(RuntimeError, match="replaced"):
        await agent._run_conversation(old, "late", [])
    old.agent.run_conversation.assert_not_called()


@pytest.mark.asyncio
async def test_hermes_direct_turn_cannot_inherit_cached_training_identity():
    from types import SimpleNamespace

    from examples.hermes.hermes import HermesAgent, _Upstream

    from areal.v2.agent_service.types import AgentRequest

    agent = HermesAgent()
    agent._sessions["session"] = SimpleNamespace(
        upstream=_Upstream(
            "http://inference",
            "key",
            "Qwen",
            {"incarnation_id": "source", "trajectory_id": 1},
        )
    )
    with pytest.raises(RuntimeError, match="requires an episode identity"):
        await agent._resolve_state(AgentRequest("delayed", "session", "old"))


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["group", "history", "missing_budget"])
async def test_gsm8k_agent_rejects_changed_episode_semantics_before_inference(
    monkeypatch, invalid
):
    from examples.hermes.hermes import GSM8KAgent

    from areal.v2.agent_service.types import AgentRequest
    from areal.v2.agent_service.worker.app import _CollectingEmitter

    client = MagicMock()
    monkeypatch.setattr("openai.AsyncOpenAI", client)
    metadata = {
        "areal_inference": {"base_url": "http://inference", "api_key": "key"},
        "generation": {"max_tokens": 128},
    }
    history = []
    if invalid == "group":
        metadata["generation"]["n"] = 2
    elif invalid == "history":
        history = [{"role": "user", "content": "previous problem"}]
    else:
        metadata["generation"] = {}
    with pytest.raises(ValueError):
        await GSM8KAgent().run(
            AgentRequest(
                "question", "session", "run", history=history, metadata=metadata
            ),
            emitter=_CollectingEmitter(),
        )
    client.assert_not_called()
