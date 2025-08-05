import os
import subprocess
import sys
import time

import pytest
import requests

from areal.api.cli_args import (
    InferenceEngineConfig,
    OptimizerConfig,
    SGLangConfig,
    TrainEngineConfig,
)
from areal.api.io_struct import AllocationMode, FinetuneSpec, WeightUpdateMeta
from areal.engine.fsdp_engine import FSDPEngine
from areal.engine.sglang_remote import RemoteSGLangEngine
from realhf.base import network

EXPR_NAME = "test_fsdp_engine_nccl"
TRIAL_NAME = "trial_nccl"
MODEL_PATH = "/storage/testing/models/Qwen__Qwen3-1.7B/"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "Qwen/Qwen2-0.5B"
PORT = 13998
DIST_PORT = 15998
GROUP_NAME = "test_nccl_group"
MASTER_PORT = DIST_PORT + 1
HOST = network.gethostip()
RUN_SERVER_TIMEOUT = 180


def check_server_health(base_url):
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@pytest.fixture(scope="module")
def sglang_server_nccl():
    from realhf.base import seeding

    seeding.set_random_seed(1, EXPR_NAME)
    cmd = SGLangConfig.build_cmd(
        sglang_config=SGLangConfig(
            mem_fraction_static=0.2,
            model_path=MODEL_PATH,
            skip_tokenizer_init=False,
            log_level="info",
        ),
        tp_size=1,
        base_gpu_id=1,
        host=HOST,
        port=PORT,
        dist_init_addr=f"{HOST}:{DIST_PORT}",
    )
    full_command = f"{cmd} --port {PORT}"
    full_command = full_command.replace("\\\n", " ").replace("\\", " ")
    os.environ["AREAL_LLM_SERVER_ADDRS"] = f"{HOST}:{PORT}"

    print(f"full_command to start sglang server: {full_command}", flush=True)
    process = subprocess.Popen(
        full_command.split(),
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
        process.terminate()
        raise RuntimeError("server launch failed")
    yield
    process.terminate()


def test_fsdpengine_nccl_weight_update_to_remote(tmp_path_factory, sglang_server_nccl):
    # Set environment variables for torch distributed
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = HOST
    os.environ["MASTER_PORT"] = str(MASTER_PORT)

    # Initialize FSDPEngine
    engine_config = TrainEngineConfig(
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
        path=MODEL_PATH,
        optimizer=OptimizerConfig(),
    )
    engine = FSDPEngine(engine_config)
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=100, train_batch_size=2)
    engine.initialize(None, ft_spec)

    # Initialize RemoteSGLangEngine
    config = InferenceEngineConfig(experiment_name=EXPR_NAME, trial_name=TRIAL_NAME)
    config.server_addrs = [f"{HOST}:{PORT}"]
    remote_engine = RemoteSGLangEngine(config)
    remote_engine.initialize(None, None)

    # Get WeightUpdateMeta
    meta = WeightUpdateMeta.from_fsdp_nccl(
        AllocationMode.from_str("sglang.d1p1t1+d1p1t1"),
        engine,
        nccl_group_name=GROUP_NAME,
    )

    # Broadcast weights
    future = remote_engine.update_weights(meta)
    print("got future", flush=True)
    engine.upload_weights(meta)
    print("uploaded wexights to remote engine", flush=True)
    # Wait for remote engine to finish
    future.result(timeout=120)
    print("got result", flush=True)
    remote_engine.destroy()
    engine.destroy()
