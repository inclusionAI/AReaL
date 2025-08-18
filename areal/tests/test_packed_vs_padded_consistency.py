import os

import pytest
import torch
from tensordict import TensorDict

from areal.api.cli_args import MicroBatchSpec, TrainEngineConfig
from areal.engine.base_hf_engine import BaseHFEngine
from areal.utils.data import concat_padded_tensors
from areal.utils.network import find_free_ports

BS = 4
MAX_ANSWER_LEN = 16
MAX_PROMPT_LEN = 8
VOCAB_SIZE = 100


@pytest.fixture
def mock_engine_config():
    """Create a mock engine configuration."""
    config = TrainEngineConfig(
        path="mock_model",
        dtype="bfloat16",
        mb_spec=MicroBatchSpec(n_mbs=2, max_tokens_per_mb=48),
        pad_to_maximum=False,
        attn_impl="flash_attention_2",
        gradient_checkpointing=False,
        disable_dropout=True,
        init_from_scratch=True,
        optimizer=None,
    )
    return config


@pytest.fixture
def mock_padded_data():
    """Generate mock padded input data."""
    prompt_lens = torch.randint(1, MAX_PROMPT_LEN, size=(BS,))
    answer_lens = torch.randint(1, MAX_ANSWER_LEN, size=(BS,))
    all_data = []

    for prompt_len, ans_len in zip(prompt_lens, answer_lens):
        prompt_len = int(prompt_len)
        ans_len = int(ans_len)
        seq = dict(
            input_ids=torch.randint(
                0, VOCAB_SIZE, size=(prompt_len + ans_len,)
            ).unsqueeze(0),
            loss_mask=torch.tensor([0] * prompt_len + [1] * ans_len).unsqueeze(0),
            attention_mask=torch.tensor([1] * (prompt_len + ans_len)).unsqueeze(0),
        )
        all_data.append(TensorDict(seq, batch_size=[1]))

    return concat_padded_tensors(all_data)


QWEN3_PATH = "/storage/testing/models/Qwen__Qwen3-1.7B/"
if not os.path.exists(QWEN3_PATH):
    QWEN3_PATH = "Qwen/Qwen3-1.7B"
QWEN25_PATH = "/storage/testing/models/Qwen__Qwen2.5-1.5B/"
if not os.path.exists(QWEN25_PATH):
    QWEN25_PATH = "Qwen/Qwen2.5-1.5B"


@pytest.mark.parametrize(
    "model_path",
    [QWEN3_PATH, QWEN25_PATH],
)
def test_padded_vs_packed_forward_consistency(model_path, mock_padded_data):
    os.environ["RANK"] = str(0)
    os.environ["WORLD_SIZE"] = str(1)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_ports(1))
    os.environ["LOCAL_RANK"] = str(0)

    config = TrainEngineConfig(
        path=model_path,
        dtype="bfloat16",
        attn_impl="flash_attention_2",
        gradient_checkpointing=False,
        disable_dropout=True,
        init_from_scratch=True,
        optimizer=None,
    )
    engine = BaseHFEngine(config)
    engine.create_process_group()
    engine.create_device_model()
    engine.initialized = True

    # Prepare padded input
    padded_input = mock_padded_data.clone()

    # Get packed input using prepare_mb_list
    mb_list = engine.prepare_mb_list(padded_input)
    assert len(mb_list.mbs) == 1

    with torch.no_grad():
        padded_logits = engine.model(
            input_ids=padded_input["input_ids"],
            attention_mask=padded_input["attention_mask"],
            position_ids=padded_input["position_ids"],
        ).logits
        seqlens = padded_input["attention_mask"].sum(1)
        x1 = []
        for i, s in enumerate(seqlens):
            x1.append(padded_logits[i, :s])
        x1 = torch.cat(x1)

        x2 = engine.model(**mb_list.mbs[0]).logits.squeeze(0)

        assert x1.shape == x2.shape, (x1.shape, x2.shape)

        # Now compare - they should be very close (allowing for small numerical differences)
        assert torch.allclose(
            x1,
            x2,
            rtol=1e-3,
            atol=1e-2,
        ), (
            (x1 - x2).abs().meax()
        )

    engine.destroy()
