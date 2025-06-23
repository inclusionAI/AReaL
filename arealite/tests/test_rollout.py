import json
from datetime import datetime
from pathlib import Path

import pytest

from arealite.api.cli_args import (
    GenerationHyperparameters,
    LLMClientConfig,
    LLMServiceConfig,
    MathCodeSingleStepConfig,
    RLVRConfig,
    RolloutWorkflowConfig,
    SGLangConfig,
    TrainingArgs,
)
from arealite.api.io_struct import Trajectory
from arealite.api.llm_server_api import LLMServerFactory
from arealite.api.rollout_api import RolloutWorkflowFactory
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import constants, name_resolve, seeding

EXPR_NAME = "test_rollout"
TRIAL_NAME = "test_rollout"
MODEL_PATH = "/storage/testing/models/Qwen__Qwen3-1.7B/"


@pytest.fixture(scope="module")
def tokenizer():
    yield load_hf_tokenizer(MODEL_PATH)


@pytest.fixture(scope="module")
def args():
    args = TrainingArgs(experiment_name=EXPR_NAME, trial_name=TRIAL_NAME)
    constants.set_experiment_trial_names(args.experiment_name, args.trial_name)
    seeding.set_random_seed(args.seed, EXPR_NAME)
    name_resolve.reconfigure(args.cluster.name_resolve)
    yield args
    name_resolve.reset()


@pytest.fixture(scope="module")
def sglang_server(args):
    server_args = LLMServiceConfig(EXPR_NAME, TRIAL_NAME, model_path=MODEL_PATH)
    server_args.sglang = SGLangConfig()
    server = LLMServerFactory.make_server(server_args)
    server._startup()
    yield
    server._graceful_exit(0)


@pytest.mark.parametrize("task", ["math", "code"])
def test_rlvr_rollout(args, sglang_server, tokenizer, task):
    jsonl_file = Path(__file__).parent / "data" / f"rlvr_{task}_dataset.jsonl"
    args.rollout.llm_client = LLMClientConfig(
        server_backend="sglang",
        tokenizer_path=MODEL_PATH,
        request_timeout=10,
    )
    args.rollout.gconfig = gconfig = GenerationHyperparameters(max_new_tokens=16)
    args.rollout.workflow = RolloutWorkflowConfig(
        type="rlvr", rlvr=RLVRConfig(reward_type=task, solution_path=jsonl_file)
    )

    collector = RolloutWorkflowFactory(args).make_workflow(args.rollout.workflow)

    # Test the rollout workflow with the provided JSONL data
    with open(jsonl_file, "r") as f:
        for i, l in enumerate(f.readlines()):
            data = json.loads(l)
            env_option = dict(
                query_id=data["query_id"],
                input_ids=tokenizer.encode(data["prompt"]),
            )
            res = collector.run_episode(
                gconfig,
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
def test_math_code_agentic_rollout(args, task, sglang_server, tokenizer):
    jsonl_file = Path(__file__).parent / "data" / f"rlvr_{task}_dataset.jsonl"
    args.rollout.llm_client = LLMClientConfig(
        server_backend="sglang",
        tokenizer_path=MODEL_PATH,
        request_timeout=10,
    )
    args.rollout.gconfig = gconfig = GenerationHyperparameters(max_new_tokens=16)
    args.rollout.workflow = RolloutWorkflowConfig(
        type="math_code_single_step",
        math_code_single_step=MathCodeSingleStepConfig(solution_path=jsonl_file),
    )

    collector = RolloutWorkflowFactory(args).make_workflow(args.rollout.workflow)

    # Test the rollout workflow with the provided JSONL data
    with open(jsonl_file, "r") as f:
        for i, l in enumerate(f.readlines()):
            data = json.loads(l)
            env_option = dict(
                query_id=data["query_id"],
                input_ids=tokenizer.encode(data["prompt"]),
            )
            res = collector.run_episode(
                gconfig,
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
