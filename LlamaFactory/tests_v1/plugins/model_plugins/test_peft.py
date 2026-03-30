# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llamafactory.v1.plugins.model_plugins import peft as peft_module
from llamafactory.v1.plugins.model_plugins.peft import merge_and_export_model


TINY_MODEL = "llamafactory/tiny-random-qwen3"


@pytest.fixture(scope="module")
def model_path():
    return TINY_MODEL


@pytest.fixture(scope="function")
def model(model_path):
    return AutoModelForCausalLM.from_pretrained(model_path)


@pytest.fixture(scope="function")
def tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path)


@pytest.fixture(scope="function")
def adapter_path(tmp_path):
    # Create a dummy adapter
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    base_model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    peft_model = get_peft_model(base_model, lora_config)
    save_path = tmp_path / "test_adapter"
    peft_model.save_pretrained(save_path)
    return str(save_path)


def test_find_all_linear_modules(model):
    """Verify linear modules are discoverable and include q_proj / v_proj for tiny-random-qwen3."""
    modules = peft_module._find_all_linear_modules(model)
    expected_subset = {"q_proj", "v_proj"}
    assert expected_subset.issubset(set(modules))


def test_get_lora_model(model):
    """Verify a PeftModel is returned and LoRA config takes effect."""
    config = {"name": "lora", "r": 8, "target_modules": "all", "lora_alpha": 16}
    model = peft_module.get_lora_model(model, config, is_train=True)
    assert isinstance(model, PeftModel)
    assert model.peft_config["default"].r == 8
    assert "q_proj" in model.peft_config["default"].target_modules


def test_get_freeze_model_layers(model):
    """Verify layer-wise freezing: only the last layer stays trainable."""
    # Freeze all but last layer
    config = {"name": "freeze", "freeze_trainable_layers": 1, "freeze_trainable_modules": "all"}

    # Ensure we start with something known
    model = peft_module.get_freeze_model(model, config, is_train=True)

    num_layers = model.config.num_hidden_layers
    assert num_layers > 0

    for name, param in model.named_parameters():
        if f"layers.{num_layers - 1}" in name:
            assert param.requires_grad, f"{name} should be trainable"
        elif "layers.0" in name and num_layers > 1:
            assert not param.requires_grad, f"{name} should be frozen"


def test_get_freeze_model_modules(model):
    """Verify module-wise freezing: only last-layer self_attn is trainable."""
    # Freeze specific modules (e.g. only self_attn)
    config = {"name": "freeze", "freeze_trainable_layers": 1, "freeze_trainable_modules": "self_attn"}
    model = peft_module.get_freeze_model(model, config, is_train=True)

    num_layers = model.config.num_hidden_layers

    for name, param in model.named_parameters():
        if f"layers.{num_layers - 1}" in name and "self_attn" in name:
            assert param.requires_grad, f"{name} should be trainable"
        else:
            assert not param.requires_grad, f"{name} should be frozen"


def test_load_adapter_single_for_inference(model, adapter_path):
    """Verify single adapter is merged+unloaded in inference mode."""
    # Test loading single adapter for inference (merge and unload)
    model_result = peft_module.load_adapter(model, adapter_path, is_train=False)
    assert not isinstance(model_result, PeftModel)


def test_load_adapter_resume_train(model, adapter_path):
    """Verify training mode returns a trainable PeftModel."""
    # Test loading for training
    model_result = peft_module.load_adapter(model, adapter_path, is_train=True)
    assert isinstance(model_result, PeftModel)


def test_load_adapter_train_multiple_disallowed(model, adapter_path):
    """Verify multiple adapters are rejected in training mode."""
    with pytest.raises(ValueError, match="only a single LoRA adapter"):
        peft_module.load_adapter(model, [adapter_path, adapter_path], is_train=True)


def test_load_adapter_infer_multiple_merges(model, adapter_path):
    """Verify multiple adapters are merged in inference mode."""
    # Test merging multiple adapters
    model_result = peft_module.load_adapter(model, [adapter_path, adapter_path], is_train=False)
    assert not isinstance(model_result, PeftModel)


def test_merge_and_export_model(tmp_path, adapter_path):
    """Verify merge_and_export_model produces export artifacts."""
    export_dir = tmp_path / "export"

    args_dict = {
        "model": TINY_MODEL,
        "peft_config": {
            "name": "lora",
            "adapter_name_or_path": adapter_path,
            "export_dir": str(export_dir),
            "export_size": 1,
            "infer_dtype": "float16",
        },
    }

    merge_and_export_model(args_dict)

    assert export_dir.exists()
    assert (export_dir / "config.json").exists()
    assert (export_dir / "model.safetensors").exists()
    assert (export_dir / "tokenizer_config.json").exists()
