import asyncio
import os
from concurrent.futures import ProcessPoolExecutor

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
    os.environ["OPENAI_BASE_URL"] = f"{proxy_server.public_addr}/{session_id}"
    os.environ["OPENAI_API_KEY"] = "none"
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    await client.chat.completions.create(
        **{
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "default",
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 16,
        }
    )
    os.environ.pop("OPENAI_BASE_URL")
    os.environ.pop("OPENAI_API_KEY")

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
    interactions = await proxy_server.wait_for_session(session_id)
    assert len(interactions) >= 1


class NullAgent(AgentWorkflow):
    async def run(self, base_url: str, data: dict):
        # Verify env vars are set correctly in subprocess
        assert os.environ.get("OPENAI_API_KEY") == "dummy"
        assert os.environ.get("OPENAI_BASE_URL") is not None
        await asyncio.sleep(0)
        return {}


@pytest.mark.asyncio
async def test_offline_null_agent(proxy_server):
    original_base_url = os.environ.get("OPENAI_BASE_URL")
    original_api_key = os.environ.get("OPENAI_API_KEY")

    with ProcessPoolExecutor(max_workers=1) as pool:
        workflow = OpenAIProxyWorkflow(
            mode="offline",
            agent=NullAgent(),
            proxy_server=proxy_server,
            process_pool=pool,
        )

        # Set workflow context for task_id
        workflow_context.set(workflow_context.WorkflowContext(task_id=0))

        result = await workflow.arun_episode(proxy_server.engine, None)
        assert result == {}

    # Verify local env not polluted
    assert os.environ.get("OPENAI_BASE_URL") == original_base_url
    assert os.environ.get("OPENAI_API_KEY") == original_api_key


class SimpleAgent(AgentWorkflow):
    async def run(self, base_url: str, data: dict):
        from openai import AsyncOpenAI

        async with AsyncOpenAI(base_url=base_url) as client:
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
    with ProcessPoolExecutor(max_workers=1) as pool:
        workflow = OpenAIProxyWorkflow(
            mode="offline",
            agent=SimpleAgent(),
            proxy_server=proxy_server,
            process_pool=pool,
        )

        workflow_context.set(workflow_context.WorkflowContext(task_id=1))

        data = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
        result = await workflow.arun_episode(proxy_server.engine, data)

        # Should have at least one interaction
        assert len(result) >= 1

        # Verify interaction structure
        for interaction in result.values():
            assert interaction.model_response is not None
            assert interaction.reward == 1.0


def test_agent_workflow_integration(proxy_server, inf_engine):
    test_data = [
        {"messages": [{"role": "user", "content": "What is 2+2?"}]},
        {"messages": [{"role": "user", "content": "What is 3+1?"}]},
    ]

    with ProcessPoolExecutor(max_workers=2) as pool:
        workflow = OpenAIProxyWorkflow(
            mode="offline",
            agent=SimpleAgent(),
            proxy_server=proxy_server,
            process_pool=pool,
        )

        result = inf_engine.rollout_batch(data=test_data, workflow=workflow)

    # Result should be a concatenated tensor dict
    assert "input_ids" in result
    assert "rewards" in result
    assert result["input_ids"].shape[0] == 2  # 2 samples
