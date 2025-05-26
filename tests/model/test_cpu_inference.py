# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
from typing import *

import pytest
import torch
import transformers

from realhf.base import constants, logging, testing
from tests.fixtures import *

logger = logging.getLogger("tests.test_cpu")


# NOTE: To run test for a new model class, please implement and register `real_config_maker`
# in realhf.api.from_hf.<your_model_class_name> and add the model class name to the
# `model_class` fixture in this file.
@pytest.fixture(params=["llama", "gpt2", "qwen2", "mistral", "mixtral", "qwen3"])
def model_class(request):
    return request.param


@pytest.fixture
def cpu_hf_model_loaded(save_path):
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        save_path,
        trust_remote_code=True,
        force_download=True,
    ).to(torch.float32)
    hf_model.eval()
    return hf_model


@torch.no_grad()
def test_inference_cpu_consistency(
    cpu_real_model,
    cpu_hf_model_loaded,
    model_class,
    mconfig,
    save_path,
):
    from realhf.impl.model.nn.real_llm_api import ReaLModel, add_helper_functions

    max_prompt_len = mconfig.n_positions
    with constants.model_scope(testing.MODEL_NAME):
        bs = 10
        torch.manual_seed(1)
        input_ids = torch.randint(
            0, mconfig.vocab_size, (bs, max_prompt_len), dtype=torch.long
        )
        input_lens = torch.full((bs,), max_prompt_len, dtype=torch.int32)
        attention_mask = torch.arange(max_prompt_len)[None, :] < input_lens[:, None]

        # Consistency between the created ReaLModel and HuggingFace model
        logits1 = cpu_hf_model_loaded(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits * attention_mask.unsqueeze(-1)
        logits2 = cpu_real_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits * attention_mask.unsqueeze(-1)

        assert torch.allclose(logits1, logits2, atol=1e-4), (
            model_class,
            (logits1 - logits2).abs().max(),
        )

        # Consistency between the created and loaded ReaLModel
        cpu_real_model_loaded = ReaLModel(mconfig, dtype=torch.float32, device="cpu")
        cpu_real_model_loaded._instantiation_hooks.append(
            lambda: getattr(cpu_real_model_loaded, f"from_{model_class}")(
                load_dir=str(save_path), init_critic_from_actor=False
            )
        )
        cpu_real_model_loaded.instantiate()
        add_helper_functions(cpu_real_model_loaded)
        cpu_real_model_loaded.eval()

        logits3 = cpu_real_model_loaded(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits * attention_mask.unsqueeze(-1)

        assert torch.allclose(logits3, logits2, atol=1e-4), (
            model_class,
            (logits3 - logits2).abs().max(),
        )
