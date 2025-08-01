# ================================================================
# NOTE: This test file is dedicated to LocalSGLangEngine testing.
#
# Unlike remote engine setup which is managed via a pytest fixture,
# the LocalSGLangEngine requires explicit `initialize()` to construct
# the engine instance at runtime. Therefore, each test must manually
# create and destroy the engine.
#
# Because of this lifecycle difference, tests for local and remote
# engines cannot be placed in the same test file.
# ================================================================
import os
import time
import uuid

import pytest
import torch
from tensordict import TensorDict

from areal.api.cli_args import (
    GenerationHyperparameters,
    InferenceEngineConfig,
    SGLangConfig,
)
from areal.api.io_struct import LLMRequest, LLMResponse
from areal.experimental.sglang_engine import SGLangEngine
from areal.workflow.rlvr import RLVRWorkflow
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import seeding

EXPR_NAME = "test_sglang_local_engine"
TRIAL_NAME = "trial_0"
MODEL_PATH = "/storage/testing/models/Qwen__Qwen3-1.7B/"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "Qwen/Qwen2-0.5B"


def build_engine_config(**kwargs):
    return InferenceEngineConfig(
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
        **kwargs,
    )


def build_engine_args():
    return SGLangConfig.build_args(
        sglang_config=SGLangConfig(mem_fraction_static=0.3, enable_metrics=False),
        model_path=MODEL_PATH,
        tp_size=1,
        base_gpu_id=0,
        served_model_name=MODEL_PATH,
        skip_tokenizer_init=False,
    )


@pytest.mark.asyncio
async def test_local_sglang_generate():
    seeding.set_random_seed(1, EXPR_NAME)
    config = build_engine_config()
    engine = SGLangEngine(config, engine_args=build_engine_args())
    engine.initialize(None, None)

    req = LLMRequest(
        rid=str(uuid.uuid4()),
        text="hello! how are you today",
        gconfig=GenerationHyperparameters(max_new_tokens=16),
    )
    resp = await engine.agenerate(req)

    assert isinstance(resp, LLMResponse)
    assert resp.input_tokens == req.input_ids
    assert (
        len(resp.output_logprobs)
        == len(resp.output_tokens)
        == len(resp.output_versions)
    )
    engine.destroy()


@pytest.mark.parametrize("n_samples", [1, 2, 4])
def test_local_sglang_rollout(n_samples):
    seeding.set_random_seed(1, EXPR_NAME)
    config = build_engine_config(max_concurrent_rollouts=2, consumer_batch_size=2)
    engine = SGLangEngine(config, engine_args=build_engine_args())
    engine.initialize(None, None)

    gconfig = GenerationHyperparameters(
        max_new_tokens=16, greedy=False, n_samples=n_samples
    )
    tokenizer = load_hf_tokenizer(MODEL_PATH)

    workflow = RLVRWorkflow(
        reward_fn=lambda **kwargs: 1.0,
        gconfig=gconfig,
        tokenizer=tokenizer,
    )

    data = {"messages": [{"role": "user", "content": "Hello, how are you?"}]}
    result = engine.rollout([data] * 2, workflow=workflow)

    print("Here is the result ", result)
    assert isinstance(result, TensorDict)
    assert result.batch_size == torch.Size([2 * n_samples])
    engine.destroy()


@pytest.mark.parametrize("ofp", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("bs", [2, 4])
@pytest.mark.parametrize("n_samples", [2, 1])
def test_local_sglang_staleness_control(bs, ofp, n_samples):
    seeding.set_random_seed(1, EXPR_NAME)
    config = build_engine_config(consumer_batch_size=bs, max_head_offpolicyness=ofp)
    engine = SGLangEngine(config, engine_args=build_engine_args())
    engine.initialize(None, None)

    gconfig = GenerationHyperparameters(
        max_new_tokens=16, greedy=False, n_samples=n_samples
    )
    tokenizer = load_hf_tokenizer(MODEL_PATH)

    workflow = RLVRWorkflow(
        reward_fn=lambda **kwargs: 1.0,
        gconfig=gconfig,
        tokenizer=tokenizer,
    )

    data = {"messages": [{"role": "user", "content": "Hello, how are you?"}]}
    for _ in range(bs * 2):
        engine.submit(data, workflow=workflow)
    time.sleep(5)
    assert engine.output_queue.qsize() == min(bs * 2, bs * (ofp + 1))

    engine.set_version(1)
    for _ in range(bs * 2):
        engine.submit(data, workflow=workflow)
    time.sleep(5)
    assert engine.output_queue.qsize() == min(bs * 4, bs * (ofp + 2))

    engine.destroy()
