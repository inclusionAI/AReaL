"""Test suite for remote inference engines (vLLM and SGLang)."""

import os

import pytest
import torch.distributed as dist

from areal.api.cli_args import (
    GenerationHyperparameters,
    InferenceEngineConfig,
    SGLangConfig,
    vLLMConfig,
)
from areal.api.io_struct import WeightUpdateMeta
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import network
from areal.utils.data import concat_padded_tensors, get_batch_size
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.pkg_version import is_available

MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "Qwen/Qwen3-0.6B"

IS_VLLM_INSTALLED = is_available("vllm")


def _dummy_reward_fn(*args, **kwargs):
    """Dummy reward function for testing."""
    return 1.0


@pytest.fixture(params=["vllm", "sglang"], scope="module")
def inference_engine(request):
    """Fixture for remote inference engines only (vLLM and SGLang)."""
    backend = request.param

    # Skip if vLLM is not installed
    if backend == "vllm" and not IS_VLLM_INSTALLED:
        pytest.skip("vLLM is not installed")

    from areal.utils import seeding

    expr_name = f"test_remote_{backend}_engine"
    trial_name = "trial_0"

    seeding.set_random_seed(1, expr_name)

    dist_port = network.find_free_ports(1)[0]
    host = network.gethostip()

    # Configure SGLang
    sglang_config = SGLangConfig(
        skip_tokenizer_init=True,
        model_path=MODEL_PATH,
        mem_fraction_static=0.2,
        context_length=128,
    )
    sglang_args = SGLangConfig.build_args(
        sglang_config=sglang_config,
        tp_size=1,
        base_gpu_id=0,
        dist_init_addr=f"{host}:{dist_port}",
    )

    # Configure vLLM
    vllm_config = vLLMConfig(
        skip_tokenizer_init=False,
        model=MODEL_PATH,
        gpu_memory_utilization=0.2,
        max_model_len=128,
        enforce_eager=True,  # reduce launch overhead
    )
    vllm_args = vLLMConfig.build_args(
        vllm_config=vllm_config,
        tp_size=1,
        pp_size=1,
    )

    # Launch remote server and initialize engine
    if backend == "vllm":
        from areal.engine.vllm_remote import RemotevLLMEngine

        engine_class = RemotevLLMEngine
        server_args = vllm_args
    else:  # sglang
        from areal.engine.sglang_remote import RemoteSGLangEngine

        engine_class = RemoteSGLangEngine
        server_args = sglang_args

    # Create engine instance for server management
    temp_config = InferenceEngineConfig(
        experiment_name=expr_name,
        trial_name=trial_name,
        setup_timeout=360,
    )
    server_manager = engine_class(temp_config)

    try:
        # Launch server via engine API
        server_info = server_manager.launch_server(server_args)

        # Set environment for remote engine
        os.environ["AREAL_LLM_SERVER_ADDRS"] = f"{server_info.host}:{server_info.port}"

        yield {
            "engine_class": engine_class,
            "expr_name": expr_name,
            "trial_name": trial_name,
            "host": host,
            "port": server_info.port,
        }
    finally:
        # Cleanup using engine API
        server_manager.teardown_server()
        server_manager.destroy()


# ============================================================================
# Unified Tests
# ============================================================================


@pytest.mark.parametrize("n_samples", [1, 2, 4])
@pytest.mark.slow
@pytest.mark.ci
def test_rollout(inference_engine, n_samples):
    """Test engine rollout with different sample sizes."""
    from areal.workflow.rlvr import RLVRWorkflow

    config = InferenceEngineConfig(
        experiment_name=inference_engine["expr_name"],
        trial_name=inference_engine["trial_name"],
        max_concurrent_rollouts=2,
        consumer_batch_size=2,
        enable_rollout_tracing=True,
        setup_timeout=360,
        max_head_offpolicyness=int(1e10),
    )

    engine = inference_engine["engine_class"](config)
    engine.initialize()

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

    class NullWorkflow(RolloutWorkflow):
        async def arun_episode(self, engine, data):
            return None

    # Test workflow returning None
    result = engine.rollout_batch(
        [data] * 2,
        workflow=NullWorkflow(),
    )
    assert result == {}

    engine.destroy()
    assert not dist.is_initialized()


@pytest.mark.parametrize("ofp", [0, 1, 4, 16])
@pytest.mark.parametrize("bs", [2, 4])
@pytest.mark.parametrize("n_samples", [2, 1])
@pytest.mark.slow
@pytest.mark.ci
def test_staleness_control(inference_engine, bs, ofp, n_samples):
    """Test engine staleness control mechanism."""
    from areal.workflow.rlvr import RLVRWorkflow

    config = InferenceEngineConfig(
        experiment_name=inference_engine["expr_name"],
        trial_name=inference_engine["trial_name"],
        consumer_batch_size=bs,
        max_head_offpolicyness=ofp,
        enable_rollout_tracing=True,
        setup_timeout=360,
    )

    engine = inference_engine["engine_class"](config)
    engine.initialize()

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
        results = engine.wait(count=bs * 2, timeout=10)
        result = concat_padded_tensors([r for r in results if r is not None])
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
        results_list = engine.wait(count=bs * 2, timeout=5)
        results = concat_padded_tensors([r for r in results_list if r is not None])
        assert results["attention_mask"].shape[0] == bs * 2 * n_samples

    engine.destroy()
    assert not dist.is_initialized()


@pytest.mark.slow
@pytest.mark.ci
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
    inf_engine = None
    try:
        ft_spec = FinetuneSpec(
            total_train_epochs=1, dataset_size=100, train_batch_size=2
        )
        train_engine.initialize(None, ft_spec)
        train_engine.model_version = 100

        # setup name resolve
        import areal.utils.name_resolve as name_resolve
        from areal.api.cli_args import NameResolveConfig

        nfs_record_root = tmp_path_factory.mktemp("nfs_record_path")
        name_resolve_config = NameResolveConfig(
            type="nfs", nfs_record_root=nfs_record_root
        )
        name_resolve.reconfigure(name_resolve_config)

        config = InferenceEngineConfig(
            experiment_name=inference_engine["expr_name"],
            trial_name=inference_engine["trial_name"],
        )
        # initialize inference engine
        inf_engine = inference_engine["engine_class"](config)
        inf_engine.initialize()
        inf_engine.set_version(100)

        # test update weights
        path = tmp_path_factory.mktemp("update_weights_from_disk")
        update_weight_meta = WeightUpdateMeta(type="disk", path=str(path))
        train_engine.connect_engine(inf_engine, update_weight_meta)
        train_engine.set_version(100)
        train_engine.update_weights(update_weight_meta)
    finally:
        train_engine.destroy()
        if inf_engine is not None:
            inf_engine.destroy()
        assert not dist.is_initialized()
