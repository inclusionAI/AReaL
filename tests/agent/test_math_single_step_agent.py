import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch

from realhf.api.core.agent_api import Agent
from realhf.api.core.model_api import BundledGenerationOutputs
from realhf.base import constants, name_resolve, testing


@pytest.fixture
def mock_env():
    env = AsyncMock()
    env.reset = AsyncMock()
    env.step = AsyncMock(return_value=(None, [0.5, 0.7], None))
    return env


@pytest.fixture
def agent_config(tmp_path):
    return {
        "gconfig": MagicMock(n=2),
        "tokenizer_path": "/storage/openpsi/models/Qwen__Qwen2.5-0.5B-Instruct/",
        "success_rate_lb": 0.1,
        "success_rate_ub": 1.0,
        "reward_scaling": 2.0,
        "reward_bias": 0.1,
        "answer_save_path": tmp_path,
    }


@pytest.fixture
def agent(agent_config):
    from realhf.impl.agent.math_single_step_agent import MathSingleStepAgent

    testing.clear_name_resolve()
    constants.set_experiment_trial_names(
        testing._DEFAULT_EXPR_NAME, testing._DEFAULT_TRIAL_NAME
    )

    agent = MathSingleStepAgent(**agent_config)
    yield agent


@pytest.fixture
def mock_prompt():
    from realhf.api.core import data_api

    return data_api.SequenceSample(
        ids=[str(123)],
        data={"packed_prompts": torch.tensor([1, 2, 3])},
        keys=set(["packed_prompts"]),
        seqlens=dict(packed_prompts=[[3]]),
        dtypes=dict(packed_prompts=torch.long),
        trailing_shapes=dict(packed_prompts=()),
    )


@pytest.fixture
def mock_act():
    return BundledGenerationOutputs(
        qid=str(123),
        seqs=[[1, 2, 3, 4, 5, 6], [1, 2, 3, 7, 8, 9]],
        output_ids=[[4, 5, 6], [7, 8, 9]],
        prompt_ids=[1, 2, 3],
        logprobs=[[0, 0, -0.1, -0.2, -0.3], [0, 0, -0.3, -0.2, -0.3]],
        no_eos=[True, False],
        version_start=[1, 1],
        version_end=[2, 2],
    )


@pytest.mark.asyncio
async def test_collect_trajectory_happy_path(agent, mock_env, mock_prompt, mock_act):
    obs_queue = asyncio.Queue()
    act_queue = asyncio.Queue()
    await act_queue.put(mock_act)

    result = await agent.collect_trajectory(mock_prompt, mock_env, obs_queue, act_queue)

    assert len(result) == 1
    sample = result[0]
    assert sample.ids == [str(123)]
    assert torch.equal(sample.data["packed_prompts"], torch.tensor([1, 2, 3]))
    # r = [0.5, 0.7]
    # ((r - 0.5) * 2 - bias) * scaling, bias=0.1, scaling=2.0
    assert torch.equal(sample.data["rewards"], torch.tensor([-0.2, 0.6]))


@pytest.mark.asyncio
async def test_collect_trajectory_low_reward(
    agent_config, mock_env, mock_prompt, mock_act
):
    # Set reward lower bound higher than what env will return
    agent_config["success_rate_lb"] = 1.0
    from realhf.impl.agent.math_single_step_agent import MathSingleStepAgent

    agent = MathSingleStepAgent(**agent_config)

    obs_queue = asyncio.Queue()
    act_queue = asyncio.Queue()
    await act_queue.put(mock_act)

    result = await agent.collect_trajectory(mock_prompt, mock_env, obs_queue, act_queue)
    assert len(result) == 0


@pytest.mark.asyncio
async def test_collect_trajectory_high_reward(
    agent_config, mock_env, mock_prompt, mock_act
):
    # Set reward upper bound lower than what env will return
    agent_config["success_rate_ub"] = 0.0
    from realhf.impl.agent.math_single_step_agent import MathSingleStepAgent

    agent = MathSingleStepAgent(**agent_config)

    obs_queue = asyncio.Queue()
    act_queue = asyncio.Queue()
    await act_queue.put(mock_act)

    result = await agent.collect_trajectory(mock_prompt, mock_env, obs_queue, act_queue)
    assert len(result) == 0


@pytest.mark.asyncio
async def test_collect_trajectory_empty_act_queue(agent, mock_env, mock_prompt):
    obs_queue = asyncio.Queue()
    act_queue = asyncio.Queue()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            agent.collect_trajectory(mock_prompt, mock_env, obs_queue, act_queue),
            timeout=1,
        )


def test_log_rewards_to_file(agent, tmp_path):
    # Setup test directories
    agent.log_rewards_to_file(
        qid="123",
        prompt="test_prompt",
        prompt_len=3,
        answers=["answer1", "answer2"],
        seqlens=[5, 6],
        rewards=[0.5, 0.7],
        success=[True, False],
        version_starts=[1, 2],
        version_ends=[2, 3],
    )

    # Check generated file
    gen_file_path = Path(agent.answer_save_path) / "1" / "123.txt"
    assert gen_file_path.exists()
    with open(gen_file_path) as f:
        content = f.read()
        assert "idx: 1 / 2" in content
        assert "seqlen: 5" in content
        assert "test_prompt" in content

    # Check monitor file
    monitor_file_path = Path(agent.answer_save_path) / "1" / "123.jsonl"
    assert monitor_file_path.exists()
    with open(monitor_file_path) as f:
        data = json.loads(f.readline())
        assert data["version_start"] == 1
        assert data["prompt_len"] == 3


def test_reward_calculation(agent):
    # Test reward scaling and biasing
    raw_rewards = [0.2, 0.4]
    expected = [(0.2 - 0.1) * 2.0, (0.4 - 0.1) * 2.0]
    processed = [
        (float(r) - agent.reward_bias) * agent.reward_scaling for r in raw_rewards
    ]
    assert processed == expected
