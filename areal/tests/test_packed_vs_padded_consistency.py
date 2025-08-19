import os

import pytest
import torch
from tensordict import TensorDict
from torch.testing import assert_close

from areal.api.cli_args import TrainEngineConfig
from areal.engine.base_hf_engine import BaseHFEngine
from areal.utils.data import concat_padded_tensors
from areal.utils.network import find_free_ports
from realhf.api.core.data_api import load_hf_processor_and_tokenizer

BS = 4
MAX_ANSWER_LEN = 16
MAX_PROMPT_LEN = 8
VOCAB_SIZE = 100


@pytest.fixture
def mock_padded_llm_data():
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

    return concat_padded_tensors(all_data).cuda()


QWEN3_PATH = "/storage/testing/models/Qwen__Qwen3-1.7B/"
if not os.path.exists(QWEN3_PATH):
    QWEN3_PATH = "Qwen/Qwen3-1.7B"
QWEN25_PATH = "/storage/openpsi/models/Qwen__Qwen2.5-1.5B/"
if not os.path.exists(QWEN25_PATH):
    QWEN25_PATH = "Qwen/Qwen2.5-1.5B"


@pytest.mark.parametrize(
    "model_path",
    [QWEN3_PATH, QWEN25_PATH],
)
def test_llm_consistency(model_path, mock_padded_llm_data):
    os.environ["RANK"] = str(0)
    os.environ["WORLD_SIZE"] = str(1)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_ports(1)[0])
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
    padded_input = mock_padded_llm_data.clone()

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

        mb = mb_list.padded_mbs[0]
        pad_len = mb_list.padding_lengths[0]
        x2 = engine.model(**mb).logits.squeeze(0)[:-pad_len]

        assert x1.shape == x2.shape, (x1.shape, x2.shape)

        assert_close(x1, x2, atol=2e-1, rtol=2e-1)

QWEN25_VL_PATH = "/storage/openpsi/models/Qwen2.5-VL-3B-Instruct"
if not os.path.exists(QWEN25_VL_PATH):
    QWEN25_VL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"


@pytest.fixture(params=[QWEN25_VL_PATH])
def mock_padded_vlm_data(request):
    model_path = request.param
    # TODO: create mock vlm image data
    prompt_lens = torch.randint(1, MAX_PROMPT_LEN, size=(BS,))
    answer_lens = torch.randint(1, MAX_ANSWER_LEN, size=(BS,))

    all_data = []

    processor, tokenizer = load_hf_processor_and_tokenizer(model_path)

    for prompt_len, ans_len in zip(prompt_lens, answer_lens):
        num_images = random.randint(1, 4)
        images = []
        image_tokens = []
        
        for _ in range(num_images):

            VISION_H = random.randint(4, 32) * 28  
            VISION_W = random.randint(4, 32) * 28 
            
            image = torch.randint(0, 255, size=(3, VISION_H, VISION_W)).float() / 255.0
            images.append(image)
            image_tokens.append("<|vision_start|><|image_pad|><|vision_end|>")
        
        combined_image_token = "".join(image_tokens)
        
        processed_input = processor(text=combined_image_token, images=images, return_tensors="pt")

        image_input_id = processed_input["input_ids"].squeeze(0) if processed_input["input_ids"].dim() > 1 else processed_input["input_ids"]

        prompt_len = int(prompt_len) + int(image_input_id.shape[0])
        ans_len = int(ans_len)
        total_len = prompt_len + ans_len
        
        input_ids = torch.cat(
            [
                torch.randint(
                    0, VOCAB_SIZE, size=(prompt_len + ans_len - len(image_input_id),)
                ),
                image_input_id,
            ]
        )

        pixel_values = processed_input["pixel_values"]
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)  
            
        seq = dict(
            input_ids=input_ids.unsqueeze(0),
            pixel_values=pixel_values,
            image_grid_thw=processed_input["image_grid_thw"],
            loss_mask=torch.tensor([0] * prompt_len + [1] * ans_len).unsqueeze(0),
            attention_mask=torch.tensor([1] * total_len).unsqueeze(0),
        )
        all_data.append(TensorDict(seq, batch_size=[1]))

    return concat_padded_tensors(all_data).cuda()

@pytest.mark.parametrize(
    "model_path",
    [QWEN25_VL_PATH],
)
def test_vlm_consistency(model_path, mock_padded_vlm_data):
    os.environ["RANK"] = str(0)
    os.environ["WORLD_SIZE"] = str(1)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_ports(1)[0])
    os.environ["LOCAL_RANK"] = str(0)

    config = TrainEngineConfig(
        path=model_path,
        dtype="bfloat16",
        attn_impl="flash_attention_2",
        gradient_checkpointing=False,
        disable_dropout=True,
        init_from_scratch=False,
        optimizer=None,
    )

    engine = BaseHFEngine(config)
    engine.create_process_group()
    engine.create_device_model()
    engine.initialized = True

    padded_input = mock_padded_vlm_data.clone()

    # Get packed input
    mb_list = engine.prepare_mb_list(padded_input)
    assert len(mb_list.mbs) == 1

    with torch.no_grad():
        # Padded logits
        padded_logits = engine.model(
            input_ids=padded_input["input_ids"],
            pixel_values=padded_input["pixel_values"],
            image_grid_thw=padded_input["image_grid_thw"],
            attention_mask=padded_input["attention_mask"],
        ).logits

        # Extract valid sequence logits
        seqlens = padded_input["attention_mask"].sum(1)
        x1 = []
        for i, s in enumerate(seqlens):
            x1.append(padded_logits[i, :s])
        x1 = torch.cat(x1)

        # Packed logits
        mb = mb_list.padded_mbs[0]
        pad_len = mb_list.padding_lengths[0]
        x2 = engine.model(**mb).logits.squeeze(0)[:-pad_len]

        assert x1.shape == x2.shape, f"Shape mismatch: {x1.shape} vs {x2.shape}"
        assert_close(x1, x2, atol=2e-1, rtol=2e-1)
