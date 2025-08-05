import os
import subprocess
import sys
import time

import pytest
import requests
import torch
from tensordict import TensorDict

from areal.api.cli_args import (
    GenerationHyperparameters,
    InferenceEngineConfig,
    SGLangConfig,
)
from areal.api.io_struct import WeightUpdateMeta
from areal.utils import network
from realhf.api.core.data_api import load_hf_tokenizer

EXPR_NAME = "test_sglang_engine"
TRIAL_NAME = "trial_0"
MODEL_PATH = "/storage/testing/models/Qwen__Qwen3-1.7B/"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "Qwen/Qwen2-0.5B"
PORT, DIST_PORT = network.find_free_ports(2)
HOST = network.gethostip()
# set a large timeout since we may need to download the model from hub
RUN_SERVER_TIMEOUT = 180


def check_server_health(base_url):
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        return False


@pytest.fixture(scope="module")
def sglang_server():
    from realhf.base import seeding

    seeding.set_random_seed(1, EXPR_NAME)
    cmd = SGLangConfig.build_cmd(
        sglang_config=SGLangConfig(
            skip_tokenizer_init=True,
            model_path=MODEL_PATH,
            mem_fraction_static=0.3,
        ),
        host=HOST,
        port=PORT,
        tp_size=1,
        base_gpu_id=0,
        dist_init_addr=f"{HOST}:{DIST_PORT}",
    )
    # Launch process
    cmd = cmd.replace("\\\n", " ").replace("\\", " ")
    process = subprocess.Popen(
        cmd.split(),
        text=True,
        stdout=sys.stdout,
        stderr=sys.stdout,
    )
    base_url = f"http://{HOST}:{PORT}"
    tik = time.time()
    while time.time() - tik < RUN_SERVER_TIMEOUT:
        if check_server_health(base_url):
            break
        time.sleep(1)
    if time.time() - tik > RUN_SERVER_TIMEOUT:
        raise RuntimeError("server launch failed")
    yield
    process.terminate()


@pytest.mark.parametrize("n_samples", [1, 2, 4])
def test_remote_sglang_rollout(sglang_server, n_samples):
    from areal.engine.sglang_remote import RemoteSGLangEngine
    from areal.workflow.rlvr import RLVRWorkflow

    config = InferenceEngineConfig(
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
        max_concurrent_rollouts=2,
        consumer_batch_size=2,
    )
    os.environ["AREAL_LLM_SERVER_ADDRS"] = f"{HOST}:{PORT}"
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
        enable_thinking=False,
    )

    data = {
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    }
    result = engine.rollout_batch([data] * 2, workflow=workflow)
    assert isinstance(result, TensorDict)
    bs = result.batch_size
    assert bs == torch.Size([2 * n_samples])
    engine.destroy()


@pytest.mark.parametrize("ofp", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("bs", [2, 4])
@pytest.mark.parametrize("n_samples", [2, 1])
def test_remote_sglang_staleness_control(sglang_server, bs, ofp, n_samples):
    from areal.engine.sglang_remote import RemoteSGLangEngine
    from areal.workflow.rlvr import RLVRWorkflow

    config = InferenceEngineConfig(
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
        consumer_batch_size=bs,
        max_head_offpolicyness=ofp,
    )
    os.environ["AREAL_LLM_SERVER_ADDRS"] = f"{HOST}:{PORT}"
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
        enable_thinking=False,
    )
    data = {
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    }
    for _ in range(bs * 2):
        engine.submit(data, workflow=workflow)

    # wait for some time
    time.sleep(5)
    assert engine.output_queue.qsize() == min(bs * 2, bs * (ofp + 1))

    # Update model version
    engine.set_version(1)
    print("Updated model version", flush=True)

    # submit again
    for _ in range(bs * 2):
        engine.submit(data, workflow=workflow)
    # wait for some time
    time.sleep(5)
    assert engine.output_queue.qsize() == min(bs * 4, bs * (ofp + 2))

    # exit
    engine.destroy()


def test_disk_update_weights_from_fsdp_engine(tmp_path_factory, sglang_server):
    # setup FSDP engine
    from areal.api.cli_args import OptimizerConfig, TrainEngineConfig
    from areal.api.io_struct import FinetuneSpec
    from areal.engine.fsdp_engine import FSDPEngine

    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "7777"

    engine_config = TrainEngineConfig(
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
        path=MODEL_PATH,
        optimizer=OptimizerConfig(),
    )
    engine = FSDPEngine(engine_config)
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=100, train_batch_size=2)
    engine.initialize(None, ft_spec)
    engine.model_version = 100

    # setup name resolve
    import realhf.base.name_resolve as name_resolve
    from realhf.api.cli_args import NameResolveConfig

    nfs_record_root = tmp_path_factory.mktemp("nfs_record_path")
    name_resolve_config = NameResolveConfig(type="nfs", nfs_record_root=nfs_record_root)
    name_resolve.reconfigure(name_resolve_config)
    # initialize SGLang remote engine
    from areal.api.cli_args import InferenceEngineConfig
    from areal.engine.sglang_remote import RemoteSGLangEngine

    config = InferenceEngineConfig(experiment_name=EXPR_NAME, trial_name=TRIAL_NAME)
    os.environ["AREAL_LLM_SERVER_ADDRS"] = f"{HOST}:{PORT}"
    inf_engine = RemoteSGLangEngine(config)
    inf_engine.initialize(None, None)
    inf_engine.set_version(100)
    # test update weights
    path = tmp_path_factory.mktemp("upload_weights_from_disk")
    update_weight_meta = WeightUpdateMeta(type="disk", path=str(path))
    future = inf_engine.update_weights(update_weight_meta)
    engine.upload_weights(update_weight_meta)
    future.result()
    inf_engine.destroy()
