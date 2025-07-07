import os
import uuid

import pytest

from arealite.api.cli_args import (
    GenerationHyperparameters,
    SGLangConfig,
    InferenceEngineConfig,
)
from tensordict import TensorDict
from arealite.api.io_struct import FinetuneSpec, LLMRequest, LLMResponse
from realhf.base import name_resolve, seeding
import subprocess
import sys
import requests
from realhf.api.core.data_api import load_hf_tokenizer
import time

EXPR_NAME = "test_sglang_engine"
TRIAL_NAME = "trial_0"
MODEL_PATH = "Qwen/Qwen2-0.5B"


def check_server_health(base_url):
    # Check server endpoint
    response = requests.get(
        f"{base_url}/metrics",
        timeout=30,
    )
    return response.status_code == 200


@pytest.fixture(scope="module")
def sglang_server():
    cmd = SGLangConfig.build_cmd(
        sglang_config=SGLangConfig(mem_fraction_static=0.3),
        model_path=MODEL_PATH,
        tp_size=1,
        base_gpu_id=0,
        dist_init_addr=f"localhost:8887",
        served_model_name=MODEL_PATH,
        skip_tokenizer_init=False,
    )
    # Launch process
    port = 8889
    full_command = f"{cmd} --port {port}"
    full_command = full_command.replace("\\\n", " ").replace("\\", " ")
    process = subprocess.Popen(
        full_command.split(),
        text=True,
        stdout=sys.stdout,
        stderr=sys.stdout,
    )
    base_url = f"http://localhost:{port}"
    while not check_server_health(base_url):
        time.sleep(2)
    yield
    process.terminate()


@pytest.mark.asyncio
async def test_remote_sglang_generate(sglang_server):
    from arealite.engine.sglang_remote import RemoteSGLangEngine

    config = InferenceEngineConfig(experiment_name=EXPR_NAME, trial_name=TRIAL_NAME)
    config.server_addrs = ["http://localhost:8889"]
    engine = RemoteSGLangEngine(config)
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
    assert isinstance(resp.completions, str)


@pytest.mark.parametrize("n_samples", [1, 2, 4])
def test_remote_sglang_rollout(sglang_server, n_samples):
    from arealite.engine.sglang_remote import RemoteSGLangEngine
    from arealite.workflow.rlvr import RLVRWorkflow

    config = InferenceEngineConfig(
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
        max_concurrent_rollouts=2,
        consumer_batch_size=2,
    )
    config.server_addrs = ["http://localhost:8889"]
    engine = RemoteSGLangEngine(config)

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
    assert isinstance(result, TensorDict)
    bs = result.batch_size
    assert bs == [2 * n_samples]  # Batch size should be 2
