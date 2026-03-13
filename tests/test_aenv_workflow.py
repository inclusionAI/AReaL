"""Unit tests for AEnvironment workflow integration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from areal.infra.aenv import AenvConfig, AenvToolCallResult
from areal.workflow.aenv.workflow import AenvWorkflow


def _sync_reward_fn(**kwargs):
    return 0.75


def _make_gconfig(**openai_args):
    class _FakePreparedGConfig:
        def __init__(self, args: dict[str, Any]):
            self._args = dict(args)

        def to_openai_args_dict(self, exclude_args=None):
            result = dict(self._args)
            if exclude_args and "n_samples" in exclude_args:
                result.pop("n", None)
            return result

    class _FakeInputGConfig:
        def new_with_stop_and_pad_token_ids(self, _):
            return _FakePreparedGConfig(openai_args)

    return _FakeInputGConfig()


class _FakeAssistantMessage:
    def __init__(self, content: str, tool_calls: list[Any] | None = None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls is not None:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ]
        return payload


class _FakeCompletion:
    def __init__(self, cid: str, message: _FakeAssistantMessage):
        self.id = cid
        self.choices = [SimpleNamespace(message=message)]


@pytest.fixture
def patched_workflow_runtime(monkeypatch):
    """Patch runtime dependencies used inside AenvWorkflow."""
    import areal.workflow.aenv.workflow as workflow_module

    class _FakeStats:
        def __init__(self):
            self.logged: list[dict[str, Any]] = []

        def scalar(self, **kwargs):
            self.logged.append(kwargs)

    fake_stats = _FakeStats()

    class _FakeStatsTracker:
        @staticmethod
        def get(_):
            return fake_stats

    monkeypatch.setattr(workflow_module, "stats_tracker", _FakeStatsTracker())
    monkeypatch.setattr(workflow_module.workflow_context, "stat_scope", lambda: "test")

    class _FakeArealOpenAI:
        planned_responses: list[_FakeCompletion] = []
        last_instance: _FakeArealOpenAI | None = None

        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.create_calls: list[dict[str, Any]] = []
            self.last_reward: float | None = None
            self.discount: float | None = None
            self.export_style: str | None = None
            self.chat = SimpleNamespace(completions=self)
            self._queue = list(type(self).planned_responses)
            type(self).last_instance = self

        async def create(self, **kwargs):
            self.create_calls.append(kwargs)
            if not self._queue:
                raise RuntimeError("No planned completion response")
            return self._queue.pop(0)

        def set_last_reward(self, reward: float) -> None:
            self.last_reward = reward

        def apply_reward_discount(self, turn_discount: float = 1.0) -> None:
            self.discount = turn_discount

        def export_interactions(self, style: str):
            self.export_style = style
            return {"fake": "interactions"}

    monkeypatch.setattr(workflow_module, "ArealOpenAI", _FakeArealOpenAI)

    class _FakeAdapter:
        listed_tools = [
            {
                "name": "math/add",
                "description": "Add two numbers",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            }
        ]
        call_error: Exception | None = None
        list_tools_error: Exception | None = None
        calls: list[dict[str, Any]] = []

        def __init__(self, config):
            self.config = config

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        async def list_tools(self):
            if type(self).list_tools_error is not None:
                raise type(self).list_tools_error
            return type(self).listed_tools

        async def call_tool(
            self, tool_name: str, arguments: dict[str, Any], timeout=None
        ):
            type(self).calls.append(
                {"tool_name": tool_name, "arguments": arguments, "timeout": timeout}
            )
            if type(self).call_error is not None:
                raise type(self).call_error
            return AenvToolCallResult(
                content=[{"type": "text", "text": "2"}], is_error=False
            )

    monkeypatch.setattr(workflow_module, "AenvEnvironmentAdapter", _FakeAdapter)

    _FakeArealOpenAI.planned_responses = []
    _FakeArealOpenAI.last_instance = None
    _FakeAdapter.call_error = None
    _FakeAdapter.list_tools_error = None
    _FakeAdapter.calls = []

    return {
        "module": workflow_module,
        "stats": fake_stats,
        "client_cls": _FakeArealOpenAI,
        "adapter_cls": _FakeAdapter,
    }


@pytest.mark.asyncio
async def test_aenv_workflow_executes_tool_call_and_exports_interactions(
    patched_workflow_runtime,
):
    """Test end-to-end multi-turn flow with one successful tool call."""

    async def reward_fn(**kwargs):
        assert "messages" in kwargs
        return 1.0

    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="math/add", arguments='{"a": 1, "b": 1}'),
    )
    patched_workflow_runtime["client_cls"].planned_responses = [
        _FakeCompletion("cmp1", _FakeAssistantMessage("", tool_calls=[tool_call])),
        _FakeCompletion("cmp2", _FakeAssistantMessage("final answer", tool_calls=[])),
    ]
    patched_workflow_runtime["adapter_cls"].call_error = None
    patched_workflow_runtime["adapter_cls"].calls = []

    workflow = AenvWorkflow(
        gconfig=_make_gconfig(
            temperature=0.7,
            max_completion_tokens=64,
            top_p=0.9,
        ),
        tokenizer=SimpleNamespace(),
        aenv_config=AenvConfig(tool_call_timeout=12.0, turn_discount=0.85),
        reward_fn=reward_fn,
        max_turns=4,
        export_style="individual",
    )

    result = await workflow.arun_episode(
        engine=SimpleNamespace(),
        data={"messages": [{"role": "user", "content": "1+1=?"}]},
    )

    client = patched_workflow_runtime["client_cls"].last_instance
    assert result == {"fake": "interactions"}
    assert client is not None
    assert len(client.create_calls) == 2
    assert "tools" in client.create_calls[0]
    assert client.create_calls[0]["tool_choice"] == "auto"
    assert client.last_reward == 1.0
    assert client.discount == 0.85
    assert client.export_style == "individual"

    adapter_calls = patched_workflow_runtime["adapter_cls"].calls
    assert adapter_calls == [
        {
            "tool_name": "math/add",
            "arguments": {"a": 1, "b": 1},
            "timeout": 12.0,
        }
    ]

    second_round_messages = client.create_calls[1]["messages"]
    assert any(
        msg.get("role") == "tool"
        and msg.get("tool_call_id") == "call_1"
        and msg.get("content") == "2"
        for msg in second_round_messages
    )


@pytest.mark.asyncio
async def test_aenv_workflow_appends_tool_error_when_argument_parsing_fails(
    patched_workflow_runtime,
):
    """Test append_error policy keeps rollout alive on tool argument parse errors."""

    async def reward_fn(**kwargs):
        return 0.5

    bad_call = SimpleNamespace(
        id="call_bad",
        function=SimpleNamespace(name="math/add", arguments="{bad-json}"),
    )
    patched_workflow_runtime["client_cls"].planned_responses = [
        _FakeCompletion("cmp1", _FakeAssistantMessage("", tool_calls=[bad_call])),
        _FakeCompletion("cmp2", _FakeAssistantMessage("done", tool_calls=[])),
    ]
    patched_workflow_runtime["adapter_cls"].call_error = None
    patched_workflow_runtime["adapter_cls"].calls = []

    workflow = AenvWorkflow(
        gconfig=_make_gconfig(temperature=0.0, max_completion_tokens=64),
        tokenizer=SimpleNamespace(),
        aenv_config=AenvConfig(tool_error_policy="append_error"),
        reward_fn=reward_fn,
        max_turns=2,
    )

    await workflow.arun_episode(
        engine=SimpleNamespace(),
        data={"messages": [{"role": "user", "content": "test"}]},
    )

    client = patched_workflow_runtime["client_cls"].last_instance
    assert client is not None
    assert patched_workflow_runtime["adapter_cls"].calls == []

    second_round_messages = client.create_calls[1]["messages"]
    assert any(
        msg.get("role") == "tool"
        and msg.get("tool_call_id") == "call_bad"
        and str(msg.get("content", "")).startswith("Error:")
        for msg in second_round_messages
    )


@pytest.mark.asyncio
async def test_aenv_workflow_raises_on_tool_failure_when_policy_is_raise(
    patched_workflow_runtime,
):
    """Test raise policy surfaces tool execution failures."""

    async def reward_fn(**kwargs):
        return 1.0

    tool_call = SimpleNamespace(
        id="call_2",
        function=SimpleNamespace(name="math/add", arguments='{"a": 2, "b": 2}'),
    )
    patched_workflow_runtime["client_cls"].planned_responses = [
        _FakeCompletion("cmp1", _FakeAssistantMessage("", tool_calls=[tool_call]))
    ]
    patched_workflow_runtime["adapter_cls"].call_error = RuntimeError(
        "tool backend failure"
    )
    patched_workflow_runtime["adapter_cls"].calls = []

    workflow = AenvWorkflow(
        gconfig=_make_gconfig(temperature=0.0, max_completion_tokens=64),
        tokenizer=SimpleNamespace(),
        aenv_config=AenvConfig(tool_error_policy="raise"),
        reward_fn=reward_fn,
        max_turns=2,
    )

    with pytest.raises(RuntimeError, match="tool backend failure"):
        await workflow.arun_episode(
            engine=SimpleNamespace(),
            data={"messages": [{"role": "user", "content": "test"}]},
        )


@pytest.mark.asyncio
async def test_aenv_workflow_falls_back_to_no_tools_when_list_tools_fails(
    patched_workflow_runtime,
):
    """Test append_error policy continues rollout when tool discovery fails."""

    async def reward_fn(**kwargs):
        return 0.2

    patched_workflow_runtime["client_cls"].planned_responses = [
        _FakeCompletion("cmp1", _FakeAssistantMessage("plain answer", tool_calls=[]))
    ]
    patched_workflow_runtime["adapter_cls"].call_error = None
    patched_workflow_runtime["adapter_cls"].list_tools_error = RuntimeError(
        "list tools failed"
    )

    workflow = AenvWorkflow(
        gconfig=_make_gconfig(temperature=0.0, max_completion_tokens=64),
        tokenizer=SimpleNamespace(),
        aenv_config=AenvConfig(tool_error_policy="append_error"),
        reward_fn=reward_fn,
    )

    await workflow.arun_episode(
        engine=SimpleNamespace(),
        data={"messages": [{"role": "user", "content": "test"}]},
    )

    client = patched_workflow_runtime["client_cls"].last_instance
    assert client is not None
    assert "tools" not in client.create_calls[0]
    assert "tool_choice" not in client.create_calls[0]


@pytest.mark.asyncio
async def test_aenv_workflow_raises_when_list_tools_fails_and_policy_is_raise(
    patched_workflow_runtime,
):
    """Test raise policy propagates list_tools failures."""

    patched_workflow_runtime["client_cls"].planned_responses = [
        _FakeCompletion("cmp1", _FakeAssistantMessage("unused", tool_calls=[]))
    ]
    patched_workflow_runtime["adapter_cls"].list_tools_error = RuntimeError(
        "list tools hard failure"
    )

    workflow = AenvWorkflow(
        gconfig=_make_gconfig(temperature=0.0, max_completion_tokens=64),
        tokenizer=SimpleNamespace(),
        aenv_config=AenvConfig(tool_error_policy="raise"),
        reward_fn=None,
    )

    with pytest.raises(RuntimeError, match="list tools hard failure"):
        await workflow.arun_episode(
            engine=SimpleNamespace(),
            data={"messages": [{"role": "user", "content": "test"}]},
        )


@pytest.mark.asyncio
async def test_aenv_workflow_allows_post_tool_answer_after_last_tool_round(
    patched_workflow_runtime,
):
    """Test max_turns counts executable tool rounds and still allows post-tool answer."""

    async def reward_fn(**kwargs):
        return 1.0

    tool_call = SimpleNamespace(
        id="call_last",
        function=SimpleNamespace(name="math/add", arguments='{"a": 1, "b": 2}'),
    )
    patched_workflow_runtime["client_cls"].planned_responses = [
        _FakeCompletion("cmp1", _FakeAssistantMessage("", tool_calls=[tool_call])),
        _FakeCompletion(
            "cmp2", _FakeAssistantMessage("assistant after tool", tool_calls=[])
        ),
    ]
    patched_workflow_runtime["adapter_cls"].list_tools_error = None
    patched_workflow_runtime["adapter_cls"].call_error = None
    patched_workflow_runtime["adapter_cls"].calls = []

    workflow = AenvWorkflow(
        gconfig=_make_gconfig(temperature=0.0, max_completion_tokens=64),
        tokenizer=SimpleNamespace(),
        aenv_config=AenvConfig(),
        reward_fn=reward_fn,
        max_turns=1,
    )

    await workflow.arun_episode(
        engine=SimpleNamespace(),
        data={"messages": [{"role": "user", "content": "test"}]},
    )

    client = patched_workflow_runtime["client_cls"].last_instance
    assert client is not None
    assert len(client.create_calls) == 2
    assert len(patched_workflow_runtime["adapter_cls"].calls) == 1


@pytest.mark.asyncio
async def test_aenv_workflow_supports_sync_reward_function_path(
    patched_workflow_runtime,
):
    """Test sync reward functions work via AsyncRewardWrapper branch."""

    patched_workflow_runtime["client_cls"].planned_responses = [
        _FakeCompletion(
            "cmp1", _FakeAssistantMessage("sync reward answer", tool_calls=[])
        )
    ]
    patched_workflow_runtime["adapter_cls"].list_tools_error = None
    patched_workflow_runtime["adapter_cls"].call_error = None

    workflow = AenvWorkflow(
        gconfig=_make_gconfig(temperature=0.0, max_completion_tokens=64),
        tokenizer=SimpleNamespace(),
        aenv_config=AenvConfig(),
        reward_fn=_sync_reward_fn,
    )

    await workflow.arun_episode(
        engine=SimpleNamespace(),
        data={"messages": [{"role": "user", "content": "test"}]},
    )

    client = patched_workflow_runtime["client_cls"].last_instance
    assert client is not None
    assert client.last_reward == 0.75


@pytest.mark.asyncio
async def test_aenv_workflow_forces_single_sample_per_openai_request(
    patched_workflow_runtime,
):
    """Test workflow strips grouped rollout n-sampling before OpenAI-style generation."""
    patched_workflow_runtime["client_cls"].planned_responses = [
        _FakeCompletion(
            "cmp1", _FakeAssistantMessage("single sample answer", tool_calls=[])
        )
    ]

    workflow = AenvWorkflow(
        gconfig=_make_gconfig(
            n=4,
            temperature=0.3,
            max_completion_tokens=32,
        ),
        tokenizer=SimpleNamespace(),
        aenv_config=AenvConfig(),
        reward_fn=None,
    )

    await workflow.arun_episode(
        engine=SimpleNamespace(),
        data={"messages": [{"role": "user", "content": "test"}]},
    )

    client = patched_workflow_runtime["client_cls"].last_instance
    assert client is not None
    assert client.create_calls[0]["n"] == 1
    assert client.create_calls[0]["temperature"] == 0.3
    assert client.create_calls[0]["max_completion_tokens"] == 32


def test_example_aenv_config_defaults_to_raise_on_tool_discovery_failure():
    """Test example config fails fast instead of silently disabling tools."""
    config_path = (
        Path(__file__).resolve().parents[1] / "examples" / "aenv" / "config.yaml"
    )
    config = yaml.safe_load(config_path.read_text())

    assert config["aenv"]["tool_error_policy"] == "raise"
