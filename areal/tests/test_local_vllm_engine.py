from __future__ import annotations

import os

import pytest

from areal.api.cli_args import GenerationHyperparameters, InferenceEngineConfig
from areal.utils.data import get_batch_size
from areal.utils.hf_utils import load_hf_tokenizer


EXPR_NAME = "test_local_vllm_engine"
TRIAL_NAME = "trial_0"
MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "Qwen/Qwen3-0.6B"


def _dummy_reward_fn(*args, **kwargs):
    return 1.0


@pytest.fixture(scope="module")
def engine_args():
    """Provide vLLM engine args for local inference."""
    from vllm import EngineArgs

    return EngineArgs(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        trust_remote_code=True,
    )


@pytest.mark.parametrize("n_samples", [1, 2, 4])
def test_local_vllm_rollout(engine_args, n_samples):
    from areal.engine.vllm_local import LocalvLLMEngine
    from areal.workflow.rlvr import RLVRWorkflow

    config = InferenceEngineConfig(
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
        max_concurrent_rollouts=2,
        consumer_batch_size=2,
    )
    engine = LocalvLLMEngine(config)
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


def test_local_vllm_weight_update_not_supported(engine_args):
    """Test that weight updates correctly raise NotImplementedError for vLLM."""
    from areal.api.io_struct import WeightUpdateMeta
    from areal.engine.vllm_local import LocalvLLMEngine

    config = InferenceEngineConfig(experiment_name=EXPR_NAME, trial_name=TRIAL_NAME)
    engine = LocalvLLMEngine(config)
    engine.initialize(engine_args=engine_args)

    # Test that disk weight update is not supported
    update_weight_meta = WeightUpdateMeta(type="disk", path="/tmp/fake_path")

    with pytest.raises(NotImplementedError, match="vLLM does not support"):
        fut = engine.update_weights_from_disk(update_weight_meta)
        fut.result()  # Wait for the future to complete and raise the exception

    engine.destroy()
