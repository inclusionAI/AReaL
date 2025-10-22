from __future__ import annotations

import os
import time

import pytest

from areal.api.cli_args import GenerationHyperparameters, InferenceEngineConfig
from areal.api.io_struct import WeightUpdateMeta
from areal.utils.data import get_batch_size
from areal.utils.hf_utils import load_hf_tokenizer


EXPR_NAME = "test_local_sglang_engine"
TRIAL_NAME = "trial_0"
MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "Qwen/Qwen3-0.6B"


def _dummy_reward_fn(*args, **kwargs):
    return 1.0


@pytest.fixture(scope="module")
def engine_args():
    """Provide SGLang engine args for local inference."""
    return {
        "model_path": MODEL_PATH,
        "tp_size": 1,
        "mem_fraction_static": 0.3,
        "skip_tokenizer_init": True,
    }


@pytest.mark.parametrize("n_samples", [1, 2, 4])
def test_local_sglang_rollout(engine_args, n_samples):
    from areal.engine.sglang_local import LocalSGLangEngine
    from areal.workflow.rlvr import RLVRWorkflow

    config = InferenceEngineConfig(
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
        max_concurrent_rollouts=2,
        consumer_batch_size=2,
    )
    engine = LocalSGLangEngine(config)
    engine.initialize(engine_args=engine_args)

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


@pytest.mark.parametrize("ofp", [1, 4, 16])
@pytest.mark.parametrize("bs", [2, 4])
@pytest.mark.parametrize("n_samples", [2, 1])
def test_local_sglang_staleness_control(engine_args, bs, ofp, n_samples):
    from areal.engine.sglang_local import LocalSGLangEngine
    from areal.workflow.rlvr import RLVRWorkflow

    config = InferenceEngineConfig(
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
        consumer_batch_size=bs,
        max_head_offpolicyness=ofp,
    )
    engine = LocalSGLangEngine(config)
    engine.initialize(engine_args=engine_args)

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

    # wait for some time
    time.sleep(10)
    assert engine._engine.workflow_executor.output_queue.qsize() == min(
        bs * 2, bs * (ofp + 1)
    )

    # Update model version
    engine.set_version(1)
    print("Updated model version", flush=True)

    # submit again
    for _ in range(bs * 2):
        engine.submit(data, workflow=workflow)
    # wait for some time
    time.sleep(5)
    assert engine._engine.workflow_executor.output_queue.qsize() == min(
        bs * 4, bs * (ofp + 2)
    )

    # exit
    engine.destroy()


def test_disk_update_weights_from_fsdp_engine(tmp_path_factory, engine_args):
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

    # initialize SGLang local engine
    from areal.engine.sglang_local import LocalSGLangEngine

    config = InferenceEngineConfig(experiment_name=EXPR_NAME, trial_name=TRIAL_NAME)
    inf_engine = LocalSGLangEngine(config)
    inf_engine.initialize(engine_args=engine_args)
    inf_engine.set_version(100)

    # test update weights
    path = tmp_path_factory.mktemp("upload_weights_from_disk")
    update_weight_meta = WeightUpdateMeta(type="disk", path=str(path))
    train_engine.connect_engine(inf_engine, update_weight_meta)
    train_engine.set_version(100)
    train_engine.update_weights(update_weight_meta)
    inf_engine.destroy()
