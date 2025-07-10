import os
import subprocess
import sys
import time
import uuid

import pytest
import requests
import torch
from tensordict import TensorDict

from arealite.api.cli_args import (
    GenerationHyperparameters,
    InferenceEngineConfig,
    SGLangConfig,
)
from arealite.api.io_struct import FinetuneSpec, LLMRequest, LLMResponse
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import name_resolve, network, seeding

EXPR_NAME = "test_sglang_engine"
TRIAL_NAME = "trial_0"
MODEL_PATH = "/storage/testing/models/Qwen__Qwen3-1.7B/"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "Qwen/Qwen2-0.5B"
PORT = 13887
DIST_PORT = 15887
HOST = network.gethostip()


# @pytest.fixture(scope="function")
# def sglang_local_engine():
#     from realhf.base import seeding
#     seeding.set_random_seed(1, EXPR_NAME)




# @pytest.mark.asyncio
# async def test_local_sglang_generate():
#     from arealite.engine.sglang_local import LocalSGLangEngine
#     from realhf.base import seeding
#     seeding.set_random_seed(1, EXPR_NAME)

#     config = InferenceEngineConfig(experiment_name=EXPR_NAME, trial_name=TRIAL_NAME)
#     engine_args = SGLangConfig.build_args(
#         sglang_config=SGLangConfig(mem_fraction_static=0.3),
#         model_path=MODEL_PATH,
#         tp_size=1,
#         base_gpu_id=0,
#         served_model_name=MODEL_PATH,
#         skip_tokenizer_init=False,
#     )

#     engine = LocalSGLangEngine(config, engine_args=engine_args)
#     engine.initialize(None, None)

#     print("SGLang Local Engine initialized", flush=True)
#     req = LLMRequest(
#         rid=str(uuid.uuid4()),
#         text="hello! how are you today",
#         gconfig=GenerationHyperparameters(max_new_tokens=16),
#     )
#     resp = await engine.agenerate(req)
#     print(resp.completions)
#     assert isinstance(resp, LLMResponse)
#     assert resp.input_tokens == req.input_ids
#     assert (
#         len(resp.output_logprobs)
#         == len(resp.output_tokens)
#         == len(resp.output_versions)
#     )
#     assert isinstance(resp.completions, str)
#     time.sleep(5)
#     engine.destroy()
#     time.sleep(30)  # Allow time for the engine to clean up


@pytest.mark.parametrize("n_samples", [1, 2])
def test_local_sglang_rollout(n_samples):
    from arealite.engine.sglang_local import LocalSGLangEngine
    from arealite.workflow.rlvr import RLVRWorkflow
    from realhf.base import seeding
    seeding.set_random_seed(1, EXPR_NAME)

    config = InferenceEngineConfig(
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
        max_concurrent_rollouts=2,
        consumer_batch_size=2,
    )
    engine_args = SGLangConfig.build_args(
        sglang_config=SGLangConfig(mem_fraction_static=0.3),
        model_path=MODEL_PATH,
        tp_size=1,
        base_gpu_id=0,
        served_model_name=MODEL_PATH,
        skip_tokenizer_init=False,
    )

    engine = LocalSGLangEngine(config, engine_args=engine_args)
    engine.initialize(None, None)

    gconfig = GenerationHyperparameters(
        max_new_tokens=16, greedy=False, n_samples=n_samples
    )
    tokenizer = load_hf_tokenizer(MODEL_PATH)

    workflow = RLVRWorkflow(
        reward_fn=lambda **kwargs: 1.0,  # Dummy reward function
        gconfig=gconfig,
        tokenizer=tokenizer,
    )

    data = {
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    }
    result = engine.rollout([data] * 2, workflow=workflow)

    print("Here is the result ", result)
    assert isinstance(result, TensorDict)
    bs = result.batch_size
    assert bs == torch.Size([2 * n_samples])
    engine.destroy()

    time.sleep(30)  # Allow time for the engine to clean up


# @pytest.mark.parametrize("ofp", [2])
# @pytest.mark.parametrize("bs", [2])
# @pytest.mark.parametrize("n_samples", [1])
# def test_local_sglang_staleness_control(bs, ofp, n_samples):
#     from arealite.engine.sglang_local import LocalSGLangEngine
#     from arealite.workflow.rlvr import RLVRWorkflow
#     from realhf.base import seeding
#     seeding.set_random_seed(1, EXPR_NAME)

#     config = InferenceEngineConfig(
#         experiment_name=EXPR_NAME,
#         trial_name=TRIAL_NAME,
#         consumer_batch_size=bs,
#         max_head_offpolicyness=ofp,
#     )
#     engine_args = SGLangConfig.build_args(
#         sglang_config=SGLangConfig(mem_fraction_static=0.3),
#         model_path=MODEL_PATH,
#         tp_size=1,
#         base_gpu_id=0,
#         served_model_name=MODEL_PATH,
#         skip_tokenizer_init=False,
#     )

#     engine = LocalSGLangEngine(config, engine_args=engine_args)
#     engine.initialize(None, None)

#     gconfig = GenerationHyperparameters(
#         max_new_tokens=16, greedy=False, n_samples=n_samples
#     )
#     tokenizer = load_hf_tokenizer(MODEL_PATH)

#     workflow = RLVRWorkflow(
#         reward_fn=lambda **kwargs: 1.0,  # Dummy reward function
#         gconfig=gconfig,
#         tokenizer=tokenizer,
#     )
#     data = {
#         "messages": [{"role": "user", "content": "Hello, how are you?"}],
#     }
#     for _ in range(bs * 2):
#         engine.submit(data, workflow=workflow)

#     time.sleep(5)
#     assert engine.output_queue.qsize() == min(bs * 2, bs * (ofp + 1))

#     engine.set_version(1)
#     for _ in range(bs * 2):
#         engine.submit(data, workflow=workflow)
#     time.sleep(5)
#     assert engine.output_queue.qsize() == min(bs * 4, bs * (ofp + 2))

#     engine.destroy()

#     time.sleep(10)  # Allow time for the engine to clean up
