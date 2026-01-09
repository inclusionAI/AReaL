import asyncio
import os

import aiohttp
import pytest

from areal.api.cli_args import InferenceEngineConfig, OpenAIConfig, SGLangConfig
from areal.api.workflow_api import AgentWorkflow
from areal.core import workflow_context
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.experimental.openai import (
    ArealOpenAI,
    OpenAIProxyServer,
    OpenAIProxyWorkflow,
)
from areal.tests.utils import get_model_path
from areal.utils import network
from areal.utils.hf_utils import load_hf_tokenizer

MODEL_PATH = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
)
tokenizer = load_hf_tokenizer(MODEL_PATH)


@pytest.fixture(scope="module")
def inf_engine():
    dist_port = network.find_free_ports(1)[0]
    host = network.gethostip()
    sglang_config = SGLangConfig(
        skip_tokenizer_init=True,
        model_path=MODEL_PATH,
        mem_fraction_static=0.2,
        context_length=128,
    )
    sglang_args = SGLangConfig.build_args(
        sglang_config=sglang_config,
        tp_size=1,
        base_gpu_id=0,
        dist_init_addr=f"{host}:{dist_port}",
    )
    config = InferenceEngineConfig(
        experiment_name="test_openai_proxy",
        trial_name="trial0",
        openai=OpenAIConfig(),
        max_head_offpolicyness=100,
    )
    engine = RemoteSGLangEngine(config)
    server_info = engine.launch_server(sglang_args)
    os.environ["AREAL_LLM_SERVER_ADDRS"] = f"{server_info.host}:{server_info.port}"
    engine.initialize()
    yield engine
    engine.destroy()


@pytest.fixture(scope="module")
def proxy_server(inf_engine):
    openai_client = ArealOpenAI(
        engine=inf_engine,
        tokenizer=tokenizer,
    )
    # Start proxy server
    openai_proxy_server = OpenAIProxyServer(model=openai_client)
    openai_proxy_server.start(wait_until_ready=True)
    assert openai_proxy_server.engine is inf_engine
    yield openai_proxy_server
    openai_proxy_server.close()


@pytest.mark.asyncio
async def test_session_lifecycle(proxy_server):
    # 1. directly start session will return 429 due to no capacity
    async with aiohttp.ClientSession() as http_session:
        async with http_session.post(
            f"{proxy_server.public_addr}/rl/start_session",
            json={"task_id": "test-task"},
        ) as resp:
            assert resp.status == 429

    # 2. after grant capacity, start/end session success
    await proxy_server.grant_capacity()

    async with aiohttp.ClientSession() as http_session:
        async with http_session.post(
            f"{proxy_server.public_addr}/rl/start_session",
            json={"task_id": "test-task"},
        ) as resp:
            assert resp.status == 200
            data = await resp.json()
            session_id = data["session_id"]
            assert session_id == "test-task-0"

    # Make a completion request
    from openai import AsyncOpenAI

    async with AsyncOpenAI(
        base_url=f"{proxy_server.public_addr}/{session_id}", max_retries=0
    ) as client:
        await client.chat.completions.create(
            **{
                "messages": [{"role": "user", "content": "Hi"}],
                "model": "default",
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": 16,
            }
        )

    # Set reward
    async with aiohttp.ClientSession() as http_session:
        async with http_session.post(
            f"{proxy_server.public_addr}/{session_id}/rl/set_reward",
            json={"reward": 1.0},
        ) as resp:
            assert resp.status == 200

    # End session
    async with aiohttp.ClientSession() as http_session:
        async with http_session.post(
            f"{proxy_server.public_addr}/{session_id}/rl/end_session",
        ) as resp:
            assert resp.status == 200

    # 3. after end session, can fetch results with `wait_for_session`
    session_data = await proxy_server.wait_for_session(session_id)
    interactions = session_data.completions.export_interactions
    assert len(interactions) >= 1


class NullAgent(AgentWorkflow):
    async def run(self, base_url: str, data: dict):
        # Verify env vars are set correctly in subprocess
        await asyncio.sleep(0)
        return {}


@pytest.mark.asyncio
async def test_offline_null_agent(proxy_server):
    workflow = OpenAIProxyWorkflow(
        mode="offline",
        agent=NullAgent(),
        proxy_server=proxy_server,
    )

    # Set workflow context for task_id
    workflow_context.set(workflow_context.WorkflowContext(task_id=0))

    result = await workflow.arun_episode(proxy_server.engine, None)
    assert result == {}


class SimpleAgent(AgentWorkflow):
    async def run(self, base_url: str, data: dict):
        from openai import AsyncOpenAI

        async with AsyncOpenAI(base_url=base_url, max_retries=0) as client:
            await client.chat.completions.create(
                model="default",
                messages=data["messages"],
                temperature=1.0,
                top_p=1.0,
                max_tokens=16,
            )
        return 1.0


@pytest.mark.asyncio
async def test_offline_simple_agent(proxy_server):
    workflow = OpenAIProxyWorkflow(
        mode="offline",
        agent=SimpleAgent(),
        proxy_server=proxy_server,
    )

    workflow_context.set(workflow_context.WorkflowContext(task_id=1))

    data = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
    result = await workflow.arun_episode(proxy_server.engine, data)

    # Should have at least one interaction
    assert len(result) == 1
    interaction = next(iter(result.values()))
    assert interaction.model_response is not None
    assert interaction.reward == 1.0


class MultiTurnAgent(AgentWorkflow):
    async def run(self, base_url: str, data: dict):
        from openai import AsyncOpenAI

        async with AsyncOpenAI(base_url=base_url, max_retries=0) as client:
            comp1 = await client.chat.completions.create(
                model="default",
                messages=data["messages"],
                temperature=1.0,
                top_p=1.0,
                max_tokens=16,
            )
            data["messages"] += [
                {"role": "assistant", "content": comp1.choices[0].message.content},
                {"role": "user", "content": "How's the weather today?"},
            ]
            comp2 = await client.chat.completions.create(
                model="default",
                messages=data["messages"],
                temperature=1.0,
                top_p=1.0,
                max_tokens=32,
            )
        rewards = {}
        rewards[comp1.id] = 0.5
        rewards[comp2.id] = 1.0
        return rewards


@pytest.mark.asyncio
async def test_offline_multiturn_agent(proxy_server):
    workflow = OpenAIProxyWorkflow(
        mode="offline",
        agent=MultiTurnAgent(),
        proxy_server=proxy_server,
    )

    workflow_context.set(workflow_context.WorkflowContext(task_id=1))

    data = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
    result = await workflow.arun_episode(proxy_server.engine, data)

    assert len(result) == 2
    interaction1, interaction2 = result.values()
    assert interaction1.reward == 1.5  # discount=1.0, 0.5 + 1.0 * 1.0 = 1.5
    assert interaction2.reward == 1.0

    assert interaction1.model_response is not None
    assert interaction2.model_response is not None

    seq1 = (
        interaction1.model_response.input_tokens
        + interaction1.model_response.output_tokens
    )
    input_ids2 = interaction2.model_response.input_tokens
    # At least the first several token ids are identical ("what is 2+2?")
    assert input_ids2[:10] == seq1[:10]


class ResponseAgent(AgentWorkflow):
    async def run(self, base_url: str, data: dict):
        from openai import AsyncOpenAI

        global tokenizer
        async with AsyncOpenAI(base_url=base_url, max_retries=0) as client:
            resp = await client.responses.create(
                model="default",
                input="test",
                temperature=1.0,
                top_p=1.0,
                max_output_tokens=16,
                tools=[],
            )
        return float("4" in resp.output_text)


@pytest.mark.asyncio
async def test_offline_response_agent(proxy_server):
    workflow = OpenAIProxyWorkflow(
        mode="offline",
        agent=ResponseAgent(),
        proxy_server=proxy_server,
    )

    workflow_context.set(workflow_context.WorkflowContext(task_id=1))

    data = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
    result = await workflow.arun_episode(proxy_server.engine, data)

    assert len(result) == 1
    interaction = next(iter(result.values()))
    assert interaction.model_response is not None
    resp = interaction.model_response
    output_text = resp.tokenizer.decode(resp.output_tokens)
    reward = interaction.reward
    assert reward == float("4" in output_text)


def test_agent_workflow_integration(proxy_server, inf_engine):
    test_data = [
        {"messages": [{"role": "user", "content": "What is 2+2?"}]},
        {"messages": [{"role": "user", "content": "What is 3+1?"}]},
    ]

    workflow = OpenAIProxyWorkflow(
        mode="offline",
        agent=SimpleAgent(),
        proxy_server=proxy_server,
    )

    result = inf_engine.rollout_batch(data=test_data, workflow=workflow)

    # Result should be a concatenated tensor dict
    assert "input_ids" in result
    assert "rewards" in result
    assert result["input_ids"].shape[0] == 2  # 2 samples
