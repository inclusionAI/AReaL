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


def check_server_health(base_url):
    # Check server endpoint
    try:
        response = requests.get(
            f"{base_url}/metrics",
            timeout=30,
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@pytest.fixture(scope="module")
def sglang_server():
    from realhf.base import seeding

    seeding.set_random_seed(1, EXPR_NAME)
    cmd = SGLangConfig.build_cmd(
        sglang_config=SGLangConfig(mem_fraction_static=0.3),
        model_path=MODEL_PATH,
        tp_size=1,
        base_gpu_id=0,
        dist_init_addr=f"{HOST}:{DIST_PORT}",
        served_model_name=MODEL_PATH,
        skip_tokenizer_init=False,
    )
    # Launch process
    full_command = f"{cmd} --port {PORT}"
    full_command = full_command.replace("\\\n", " ").replace("\\", " ")
    process = subprocess.Popen(
        full_command.split(),
        text=True,
        stdout=sys.stdout,
        stderr=sys.stdout,
    )
    base_url = f"http://{HOST}:{PORT}"
    tik = time.time()
    while time.time() - tik < 90:
        if check_server_health(base_url):
            break
        time.sleep(1)
    if time.time() - tik > 90:
        raise RuntimeError("server launch failed")
    yield
    process.terminate()


@pytest.mark.skip("")
@pytest.mark.asyncio
async def test_remote_sglang_generate(sglang_server):
    from arealite.engine.sglang_remote import RemoteSGLangEngine

    config = InferenceEngineConfig(experiment_name=EXPR_NAME, trial_name=TRIAL_NAME)
    config.server_addrs = [f"{HOST}:{PORT}"]
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


@pytest.mark.skip("")
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
    config.server_addrs = [f"{HOST}:{PORT}"]
    engine = RemoteSGLangEngine(config)
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
    assert isinstance(result, TensorDict)
    bs = result.batch_size
    assert bs == torch.Size([2 * n_samples])
    engine.destroy()


@pytest.mark.parametrize("ofp", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("bs", [2, 4])
@pytest.mark.parametrize("n_samples", [2, 1])
def test_remote_sglang_staleness_control(sglang_server, bs, ofp, n_samples):
    from arealite.engine.sglang_remote import RemoteSGLangEngine
    from arealite.workflow.rlvr import RLVRWorkflow

    config = InferenceEngineConfig(
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
        consumer_batch_size=bs,
        max_head_offpolicyness=ofp,
    )
    config.server_addrs = [f"{HOST}:{PORT}"]
    engine = RemoteSGLangEngine(config)
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
    for _ in range(bs * 2):
        engine.submit(data, workflow=workflow)

    # wait for some time
    time.sleep(15)
    assert engine.output_queue.qsize() == min(bs * 2, bs * (ofp + 1))

    # Update model version
    engine.set_version(1)
    print("Updated model version", flush=True)

    # submit again
    for _ in range(bs * 2):
        engine.submit(data, workflow=workflow)
    # wait for some time
    time.sleep(15)
    assert engine.output_queue.qsize() == min(bs * 4, bs * (ofp + 2))

    # exit
    engine.destroy()
