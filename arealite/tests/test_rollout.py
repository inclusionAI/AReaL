# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import json
from datetime import datetime
from pathlib import Path

import pytest
from datasets import load_dataset

from arealite.api.cli_args import (
    DatasetPreprocessor,
    GenerationHyperparameters,
    GSM8KPreprocessor,
    MathCodeSingleStepConfig,
    RLVRConfig,
    RolloutCollectorConfig,
    SGLangConfig,
    TrainingArgs,
)
from arealite.api.io_struct import Trajectory
from arealite.api.llm_client_api import LLMClientFactory
from arealite.api.llm_server_api import LLMServerFactory
from arealite.api.rollout_api import RolloutCollectorFactory
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import name_resolve, seeding

EXPR_NAME = "test_rollout"
TRIAL_NAME = "test_rollout"
MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen3-1.7B/"


@pytest.fixture(scope="module")
def tokenizer():
    yield load_hf_tokenizer(MODEL_PATH)


@pytest.fixture(scope="module")
def args():
    args = TrainingArgs(experiment_name=EXPR_NAME, trial_name=TRIAL_NAME)
    args.rollout.model_path = MODEL_PATH
    seeding.set_random_seed(args.seed, EXPR_NAME)
    name_resolve.reconfigure(args.cluster.name_resolve)
    yield args
    name_resolve.reset()


@pytest.fixture(scope="module")
def sglang_server(args):
    args.rollout.sglang = SGLangConfig()
    server = LLMServerFactory(args).make_server(args.rollout.llm_service)
    server._startup()
    yield
    server._graceful_exit(0)


@pytest.mark.parametrize("task", ["math", "code"])
@pytest.mark.asyncio
async def test_rlvr_rollout(args, sglang_server, tokenizer, task):
    jsonl_file = Path(__file__).parent / "data" / f"rlvr_{task}_dataset.jsonl"
    args.rollout.server_backend = "sglang"
    args.rollout.gconfig = gconfig = GenerationHyperparameters(max_new_tokens=16)
    args.rollout.collector = RolloutCollectorConfig(
        type="rlvr",
        rlvr=RLVRConfig(reward_type=f"areal-{task}", solution_path=jsonl_file),
    )
    llm_client = LLMClientFactory(args).make_client(args.rollout.llm_client)
    collector = RolloutCollectorFactory(args).make_collector(args.rollout.collector)

    # Test the rollout collector with the provided JSONL data
    with open(jsonl_file, "r") as f:
        for i, l in enumerate(f.readlines()):
            data = json.loads(l)
            env_option = dict(
                query_id=data["query_id"],
                input_ids=tokenizer.encode(data["prompt"]),
                prompt=data["prompt"],
            )
            res = await collector.arun_episode(
                llm_client=llm_client,
                gconfig=gconfig,
                env_option=env_option,
            )
            assert isinstance(res, Trajectory)
            assert isinstance(res.data, dict)
            assert res.prompt == env_option
            shape = res.data["input_ids"].shape
            for k in ["prompt_mask", "logprobs", "versions"]:
                assert res.data[k].shape == shape
            assert res.stats.episode_length == 1
            assert res.stats.total_reward in [0, 1], res.stats.total_reward
            assert res.stats.start_time < datetime.now().timestamp()


@pytest.mark.asyncio
async def test_gsm8k_rollout(args, sglang_server, tokenizer):
    args.rollout.server_backend = "sglang"
    args.rollout.gconfig = gconfig = GenerationHyperparameters(max_new_tokens=16)
    args.rollout.collector = RolloutCollectorConfig(
        type="rlvr", rlvr=RLVRConfig(reward_type="gsm8k")
    )
    collector = RolloutCollectorFactory(args).make_collector(args.rollout.collector)

    args.train_dataset.path = "openai/gsm8k"
    args.train_dataset.name = "main"
    args.train_dataset.split = "train"
    args.train_dataset.preprocessor = DatasetPreprocessor(
        "gsm8k_rl", gsm8k=GSM8KPreprocessor("strict")
    )

    from arealite.api.dataset_api import DatasetFactory

    llm_client = LLMClientFactory(args).make_client(args.rollout.llm_client)
    dataset = (
        DatasetFactory(args)
        .make_dataset(args.train_dataset, rank=0, world_size=1)
        .select(range(10))
    )
    for i in range(len(dataset)):
        env_option = dataset[i]
        res = await collector.arun_episode(
            llm_client=llm_client,
            gconfig=gconfig,
            env_option=env_option,
        )
        assert isinstance(res, Trajectory)
        assert isinstance(res.data, dict)
        assert res.prompt == env_option
        shape = res.data["input_ids"].shape
        for k in ["prompt_mask", "logprobs", "versions"]:
            assert res.data[k].shape == shape
        assert res.stats.episode_length == 1
        assert res.stats.total_reward in [0, 1], res.stats.total_reward
        assert res.stats.start_time < datetime.now().timestamp()


@pytest.mark.parametrize("task", ["math", "code"])
@pytest.mark.asyncio
async def test_math_code_agentic_rollout(args, task, sglang_server, tokenizer):
    jsonl_file = Path(__file__).parent / "data" / f"rlvr_{task}_dataset.jsonl"
    args.rollout.server_backend = "sglang"
    args.rollout.gconfig = gconfig = GenerationHyperparameters(max_new_tokens=16)
    args.rollout.collector = RolloutCollectorConfig(
        type="math_code_single_step",
        math_code_single_step=MathCodeSingleStepConfig(solution_path=jsonl_file),
    )

    collector = RolloutCollectorFactory(args).make_collector(args.rollout.collector)
    llm_client = LLMClientFactory(args).make_client(args.rollout.llm_client)

    # Test the rollout collector with the provided JSONL data
    with open(jsonl_file, "r") as f:
        for i, l in enumerate(f.readlines()):
            data = json.loads(l)
            env_option = dict(
                query_id=data["query_id"],
                input_ids=tokenizer.encode(data["prompt"]),
            )
            res = await collector.arun_episode(
                llm_client=llm_client,
                gconfig=gconfig,
                env_option=env_option,
            )
            assert isinstance(res, Trajectory)
            assert isinstance(res.data, dict)
            assert res.prompt == env_option
            shape = res.data["input_ids"].shape
            for k in ["prompt_mask", "logprobs", "versions"]:
                assert res.data[k].shape == shape
            assert res.stats.episode_length == 1
            assert res.stats.total_reward in [0, 1], res.stats.total_reward
            assert res.stats.start_time < datetime.now().timestamp()
