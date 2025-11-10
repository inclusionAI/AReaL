import os

import pytest
import torch.distributed as dist

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    InferenceEngineConfig,
    OptimizerConfig,
    SGLangConfig,
    TrainEngineConfig,
)
from areal.api.io_struct import FinetuneSpec, WeightUpdateMeta
from areal.engine.fsdp_engine import FSDPEngine
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils import network

EXPR_NAME = "test_fsdp_engine_nccl"
TRIAL_NAME = "trial_nccl"
MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "Qwen/Qwen3-0.6B"
PORT = 13998
DIST_PORT = 15998
GROUP_NAME = "test_nccl_group"
MASTER_PORT = DIST_PORT + 1
HOST = network.gethostip()


@pytest.fixture(scope="module")
def sglang_server_nccl():
    from areal.utils import seeding

    seeding.set_random_seed(1, EXPR_NAME)
    sglang_args = SGLangConfig.build_args(
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
    os.environ["AREAL_LLM_SERVER_ADDRS"] = f"{HOST}:{PORT}"

    # Create engine instance for server management
    temp_config = InferenceEngineConfig(
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
    )
    server_manager = RemoteSGLangEngine(temp_config)

    try:
        # Launch server via engine API
        server_manager.launch_server(sglang_args)
        yield
    finally:
        # Cleanup using engine API
        server_manager.teardown_server()
        server_manager.destroy()


def test_fsdpengine_nccl_weight_update_to_remote(tmp_path_factory, sglang_server_nccl):
    # Set environment variables for torch distributed
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = HOST
    os.environ["MASTER_PORT"] = str(MASTER_PORT)
    # required by sglang
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    # Initialize FSDPEngine
    engine_config = TrainEngineConfig(
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
        path=MODEL_PATH,
        optimizer=OptimizerConfig(),
    )
    engine = FSDPEngine(engine_config)
    remote_engine = None
    try:
        engine.create_process_group()
        ft_spec = FinetuneSpec(
            total_train_epochs=1, dataset_size=100, train_batch_size=2
        )
        engine.initialize(None, ft_spec)

        # Initialize RemoteSGLangEngine
        config = InferenceEngineConfig(experiment_name=EXPR_NAME, trial_name=TRIAL_NAME)
        config.server_addrs = [f"{HOST}:{PORT}"]
        remote_engine = RemoteSGLangEngine(config)
        remote_engine.initialize()

        # Get WeightUpdateMeta
        meta = WeightUpdateMeta.from_fsdp_xccl(
            AllocationMode.from_str("sglang.d1p1t1+d1p1t1"),
            nccl_group_name=GROUP_NAME,
        )

        engine.connect_engine(remote_engine, meta)

        # Broadcast weights
        engine.update_weights(meta)
        print("uploaded weights to remote engine", flush=True)
    finally:
        # Cleanup in reverse order
        if remote_engine is not None:
            remote_engine.destroy()
        engine.destroy()
        assert not dist.is_initialized()
