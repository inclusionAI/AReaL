"""Unified test suite for inference engines (vLLM and SGLang, both local and remote)."""

import os
import subprocess
import sys
import time

import pytest
import requests

from areal.api.cli_args import (
    GenerationHyperparameters,
    InferenceEngineConfig,
    SGLangConfig,
    vLLMConfig,
)
from areal.api.io_struct import WeightUpdateMeta
from areal.utils import network
from areal.utils.data import get_batch_size
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.pkg_version import is_available

MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "Qwen/Qwen3-0.6B"

# set a large timeout since we may need to download the model from hub
RUN_SERVER_TIMEOUT = 180

IS_VLLM_INSTALLED = is_available("vllm")


def check_server_health(base_url):
    """Check if the server is healthy and ready to accept requests."""
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _dummy_reward_fn(*args, **kwargs):
    """Dummy reward function for testing."""
    return 1.0


@pytest.fixture(
    params=[
        ("vllm", "remote"),
        ("vllm", "local"),
        ("sglang", "remote"),
        ("sglang", "local"),
    ],
    ids=["vllm-remote", "vllm-local", "sglang-remote", "sglang-local"],
)
def inference_engine(request):
    """Unified fixture that provides any inference engine (vLLM/SGLang, local/remote).

    This fixture:
    1. Launches the appropriate server (for remote) or prepares engine args (for local)
    2. Yields engine metadata for test initialization
    3. Cleans up resources after all tests complete
    """
    backend, mode = request.param

    # Skip if vLLM is not installed
    if backend == "vllm" and not IS_VLLM_INSTALLED:
        pytest.skip("vLLM is not installed")

    from areal.utils import seeding

    expr_name = f"test_{mode}_{backend}_engine"
    trial_name = "trial_0"

    seeding.set_random_seed(1, expr_name)

    # Initialize engine based on backend and mode
    if mode == "remote":
        # Launch server
        port, dist_port = network.find_free_ports(2)
        host = network.gethostip()

        if backend == "vllm":
            from areal.engine.vllm_remote import RemotevLLMEngine

            cmd = vLLMConfig.build_cmd(
                vllm_config=vLLMConfig(
                    skip_tokenizer_init=False,
                    model=MODEL_PATH,
                    gpu_memory_utilization=0.1,
                ),
                host=host,
                port=port,
                tp_size=1,
                dist_init_addr=f"{host}:{dist_port}",
            )
            engine_class = RemotevLLMEngine
        else:  # sglang
            from areal.engine.sglang_remote import RemoteSGLangEngine

            cmd = SGLangConfig.build_cmd(
                sglang_config=SGLangConfig(
                    skip_tokenizer_init=True,
                    model_path=MODEL_PATH,
                    mem_fraction_static=0.3,
                ),
                host=host,
                port=port,
                tp_size=1,
                base_gpu_id=0,
                dist_init_addr=f"{host}:{dist_port}",
            )
            engine_class = RemoteSGLangEngine

        # Launch process
        cmd = cmd.replace("\\\n", " ").replace("\\", " ")
        process = subprocess.Popen(
            cmd.split(),
            text=True,
            stdout=sys.stdout,
            stderr=sys.stdout,
        )
        base_url = f"http://{host}:{port}"
        tik = time.time()
        while time.time() - tik < RUN_SERVER_TIMEOUT:
            if check_server_health(base_url):
                break
            time.sleep(1)
        if time.time() - tik > RUN_SERVER_TIMEOUT:
            process.terminate()
            raise RuntimeError(f"{backend.upper()} server launch failed")

        # Set environment for remote engine
        os.environ["AREAL_LLM_SERVER_ADDRS"] = f"{host}:{port}"

        yield {
            "engine_class": engine_class,
            "backend": backend,
            "mode": mode,
            "expr_name": expr_name,
            "trial_name": trial_name,
            "host": host,
            "port": port,
        }

        # Cleanup
        process.terminate()

    else:  # local
        if backend == "vllm":
            from vllm import EngineArgs

            from areal.engine.vllm_local import LocalvLLMEngine

            engine_args = EngineArgs(
                model=MODEL_PATH,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.3,
                trust_remote_code=True,
            )
            engine_class = LocalvLLMEngine
        else:  # sglang
            from areal.engine.sglang_local import LocalSGLangEngine

            engine_args = {
                "model_path": MODEL_PATH,
                "tp_size": 1,
                "mem_fraction_static": 0.3,
                "skip_tokenizer_init": True,
            }
            engine_class = LocalSGLangEngine

        yield {
            "engine_class": engine_class,
            "backend": backend,
            "mode": mode,
            "expr_name": expr_name,
            "trial_name": trial_name,
            "engine_args": engine_args,
        }


# ============================================================================
# Unified Tests
# ============================================================================


@pytest.mark.parametrize("n_samples", [1, 2, 4])
def test_rollout(inference_engine, n_samples):
    """Test engine rollout with different sample sizes."""
    from areal.workflow.rlvr import RLVRWorkflow

    config = InferenceEngineConfig(
        experiment_name=inference_engine["expr_name"],
        trial_name=inference_engine["trial_name"],
        max_concurrent_rollouts=2,
        consumer_batch_size=2,
    )

    engine = inference_engine["engine_class"](config)

    # Initialize based on mode
    if inference_engine["mode"] == "remote":
        engine.initialize()
    else:  # local
        engine.initialize(engine_args=inference_engine["engine_args"])

    gconfig = GenerationHyperparameters(
        max_new_tokens=16, greedy=False, n_samples=n_samples
    )
    tokenizer = load_hf_tokenizer(MODEL_PATH)

    workflow = RLVRWorkflow(
        reward_fn=_dummy_reward_fn,
        gconfig=gconfig,
        tokenizer=tokenizer,
        enable_thinking=False,
    )

    data = {
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    }
    result = engine.rollout_batch([data] * 2, workflow=workflow)
    assert isinstance(result, dict)
    bs = get_batch_size(result)
    assert bs == 2 * n_samples
    engine.destroy()


@pytest.mark.parametrize("ofp", [0, 1, 4, 16])
@pytest.mark.parametrize("bs", [2, 4])
@pytest.mark.parametrize("n_samples", [2, 1])
def test_staleness_control(inference_engine, bs, ofp, n_samples):
    """Test engine staleness control mechanism."""
    from areal.workflow.rlvr import RLVRWorkflow

    config = InferenceEngineConfig(
        experiment_name=inference_engine["expr_name"],
        trial_name=inference_engine["trial_name"],
        consumer_batch_size=bs,
        max_head_offpolicyness=ofp,
        enable_rollout_tracing=(
            inference_engine["backend"] == "sglang"
            and inference_engine["mode"] == "remote"
        ),
    )

    engine = inference_engine["engine_class"](config)

    # Initialize based on mode
    if inference_engine["mode"] == "remote":
        engine.initialize()
    else:  # local
        engine.initialize(engine_args=inference_engine["engine_args"])

    gconfig = GenerationHyperparameters(
        max_new_tokens=2, greedy=False, n_samples=n_samples
    )
    tokenizer = load_hf_tokenizer(MODEL_PATH)

    workflow = RLVRWorkflow(
        reward_fn=_dummy_reward_fn,
        gconfig=gconfig,
        tokenizer=tokenizer,
        enable_thinking=False,
    )
    data = {
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    }
    for _ in range(bs * 2):
        engine.submit(data, workflow=workflow)

    if ofp < 1:
        # Due to controlled offpolicyness, not all requests are committed
        with pytest.raises(TimeoutError):
            engine.wait(count=bs * 2, timeout=10)
    else:
        result = engine.wait(count=bs * 2, timeout=10)
        assert result["attention_mask"].shape[0] == bs * 2 * n_samples

    # Update model version
    engine.set_version(1)
    print("Updated model version", flush=True)

    # submit again
    for _ in range(bs * 2):
        engine.submit(data, workflow=workflow)

    if ofp < 2:
        # Due to controlled offpolicyness, not all requests are committed
        with pytest.raises(TimeoutError):
            engine.wait(count=bs * 4, timeout=5)
    else:
        # 2 * bs samples haved been retrived above
        results = engine.wait(count=bs * 2, timeout=5)
        assert results["attention_mask"].shape[0] == bs * 2 * n_samples

    # exit
    engine.destroy()


def test_disk_update_weights_from_fsdp_engine(tmp_path_factory, inference_engine):
    """Test disk-based weight updates from FSDP engine to inference engine."""

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
        experiment_name=inference_engine["expr_name"],
        trial_name=inference_engine["trial_name"],
        path=MODEL_PATH,
        optimizer=OptimizerConfig(),
    )
    train_engine = FSDPEngine(engine_config)
    train_engine.create_process_group()
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=100, train_batch_size=2)
    train_engine.initialize(None, ft_spec)
    train_engine.model_version = 100

    # setup name resolve
    import areal.utils.name_resolve as name_resolve
    from areal.api.cli_args import NameResolveConfig

    nfs_record_root = tmp_path_factory.mktemp("nfs_record_path")
    name_resolve_config = NameResolveConfig(type="nfs", nfs_record_root=nfs_record_root)
    name_resolve.reconfigure(name_resolve_config)

    # initialize inference engine
    config = InferenceEngineConfig(
        experiment_name=inference_engine["expr_name"],
        trial_name=inference_engine["trial_name"],
    )
    inf_engine = inference_engine["engine_class"](config)

    # Initialize based on mode
    if inference_engine["mode"] == "remote":
        inf_engine.initialize()
    else:  # local
        inf_engine.initialize(engine_args=inference_engine["engine_args"])

    inf_engine.set_version(100)

    # test update weights
    path = tmp_path_factory.mktemp("update_weights_from_disk")
    update_weight_meta = WeightUpdateMeta(type="disk", path=str(path))
    train_engine.connect_engine(inf_engine, update_weight_meta)
    train_engine.set_version(100)
    train_engine.update_weights(update_weight_meta)
    inf_engine.destroy()
