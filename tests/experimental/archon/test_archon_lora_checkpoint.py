"""Tests for LoRA adapter checkpoint I/O and PEFT format conversion."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from areal.experimental.models.archon.lora.adapter import get_adapter_params
from areal.experimental.models.archon.lora.lora_linear import LoRALinear
from areal.experimental.models.archon.qwen2.model.state_dict_adapter import (
    Qwen2StateDictAdapter,
)

# Try to import PEFT for compatibility tests
try:
    import peft  # noqa: F401

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class TestStateDictAdapterLoRAKeys:
    """Test LoRA key conversion in Qwen2StateDictAdapter."""

    def setup_method(self):
        mock_config = Mock()
        mock_config.tie_word_embeddings = False
        self.adapter = Qwen2StateDictAdapter(mock_config)

    def test_qwen2_lora_key_conversion_attention(self):
        """Test attention LoRA key conversion (wq, wk, wv, wo)."""
        hf_key = "model.layers.0.self_attn.q_proj.lora_A.weight"
        archon_key = self.adapter._convert_key_from_hf(hf_key)
        assert archon_key == "layers.0.attention.wq.lora_a.weight"

        hf_key_back = self.adapter._convert_key_to_hf(archon_key)
        assert hf_key_back == hf_key

    def test_qwen2_lora_key_conversion_mlp(self):
        """Test MLP LoRA key conversion (w1, w2, w3)."""
        hf_key = "model.layers.5.mlp.gate_proj.lora_A.weight"
        archon_key = self.adapter._convert_key_from_hf(hf_key)
        assert archon_key == "layers.5.feed_forward.w1.lora_a.weight"

        hf_key_back = self.adapter._convert_key_to_hf(archon_key)
        assert hf_key_back == hf_key

    def test_qwen2_lora_key_conversion_lm_head(self):
        """Test LM head LoRA key conversion."""
        hf_key = "lm_head.lora_A.weight"
        archon_key = self.adapter._convert_key_from_hf(hf_key)
        assert archon_key == "output.lora_a.weight"

        hf_key_back = self.adapter._convert_key_to_hf(archon_key)
        assert hf_key_back == hf_key

    def test_qwen2_lora_case_conversion(self):
        """Test lora_A/lora_B (HF) <-> lora_a/lora_b (Archon) case handling."""
        # lora_A -> lora_a
        hf_key_a = "model.layers.0.self_attn.v_proj.lora_A.weight"
        archon_key_a = self.adapter._convert_key_from_hf(hf_key_a)
        assert "lora_a" in archon_key_a

        # lora_B -> lora_b
        hf_key_b = "model.layers.0.self_attn.v_proj.lora_B.weight"
        archon_key_b = self.adapter._convert_key_from_hf(hf_key_b)
        assert "lora_b" in archon_key_b

    def test_qwen2_all_16_lora_mappings(self):
        """Verify all 16 LoRA key patterns convert correctly."""
        lora_mappings = [
            ("model.layers.{}.self_attn.q_proj.lora_A.weight", "layers.{}.attention.wq.lora_a.weight"),
            ("model.layers.{}.self_attn.q_proj.lora_B.weight", "layers.{}.attention.wq.lora_b.weight"),
            ("model.layers.{}.self_attn.k_proj.lora_A.weight", "layers.{}.attention.wk.lora_a.weight"),
            ("model.layers.{}.self_attn.k_proj.lora_B.weight", "layers.{}.attention.wk.lora_b.weight"),
            ("model.layers.{}.self_attn.v_proj.lora_A.weight", "layers.{}.attention.wv.lora_a.weight"),
            ("model.layers.{}.self_attn.v_proj.lora_B.weight", "layers.{}.attention.wv.lora_b.weight"),
            ("model.layers.{}.self_attn.o_proj.lora_A.weight", "layers.{}.attention.wo.lora_a.weight"),
            ("model.layers.{}.self_attn.o_proj.lora_B.weight", "layers.{}.attention.wo.lora_b.weight"),
            ("model.layers.{}.mlp.gate_proj.lora_A.weight", "layers.{}.feed_forward.w1.lora_a.weight"),
            ("model.layers.{}.mlp.gate_proj.lora_B.weight", "layers.{}.feed_forward.w1.lora_b.weight"),
            ("model.layers.{}.mlp.up_proj.lora_A.weight", "layers.{}.feed_forward.w3.lora_a.weight"),
            ("model.layers.{}.mlp.up_proj.lora_B.weight", "layers.{}.feed_forward.w3.lora_b.weight"),
            ("model.layers.{}.mlp.down_proj.lora_A.weight", "layers.{}.feed_forward.w2.lora_a.weight"),
            ("model.layers.{}.mlp.down_proj.lora_B.weight", "layers.{}.feed_forward.w2.lora_b.weight"),
            ("lm_head.lora_A.weight", "output.lora_a.weight"),
            ("lm_head.lora_B.weight", "output.lora_b.weight"),
        ]

        for hf_pattern, archon_pattern in lora_mappings:
            # Substitute layer index
            hf_key = hf_pattern.replace("{}", "3")
            archon_key = archon_pattern.replace("{}", "3")

            converted = self.adapter._convert_key_from_hf(hf_key)
            assert converted == archon_key, f"from_hf failed: {hf_key} -> {converted}"

            back = self.adapter._convert_key_to_hf(archon_key)
            assert back == hf_key, f"to_hf failed: {archon_key} -> {back}"

    def test_to_peft_module_map(self):
        """Test Archon -> PEFT module name mapping."""
        assert self.adapter.to_peft_module_map["wq"] == "q_proj"
        assert self.adapter.to_peft_module_map["wk"] == "k_proj"
        assert self.adapter.to_peft_module_map["wv"] == "v_proj"
        assert self.adapter.to_peft_module_map["wo"] == "o_proj"
        assert self.adapter.to_peft_module_map["w1"] == "gate_proj"
        assert self.adapter.to_peft_module_map["w2"] == "down_proj"
        assert self.adapter.to_peft_module_map["w3"] == "up_proj"
        assert self.adapter.to_peft_module_map["output"] == "lm_head"


class TestPEFTAdapterConfig:
    """Test PEFT adapter config generation."""

    def setup_method(self):
        mock_config = Mock()
        mock_config.tie_word_embeddings = False
        self.adapter = Qwen2StateDictAdapter(mock_config)

    def test_create_peft_adapter_config(self):
        """Test adapter_config.json generation."""
        from dataclasses import dataclass

        @dataclass
        class LoRAConfig:
            rank: int
            alpha: float
            target_modules: list[str]

        lora_cfg = LoRAConfig(rank=8, alpha=16.0, target_modules=["wq", "wv"])
        config = self.adapter.create_peft_adapter_config(lora_cfg)

        assert config["peft_type"] == "LORA"
        assert config["r"] == 8
        assert config["lora_alpha"] == 16
        assert config["task_type"] == "CAUSAL_LM"
        assert "q_proj" in config["target_modules"]
        assert "v_proj" in config["target_modules"]
        assert config["bias"] == "none"

    def test_create_peft_adapter_config_with_base_model(self):
        """Test adapter config with base model path."""
        from dataclasses import dataclass

        @dataclass
        class LoRAConfig:
            rank: int
            alpha: float
            target_modules: list[str]

        lora_cfg = LoRAConfig(rank=16, alpha=32.0, target_modules=["wq"])
        config = self.adapter.create_peft_adapter_config(
            lora_cfg, base_model_path="Qwen/Qwen2-0.5B"
        )

        assert config["base_model_name_or_path"] == "Qwen/Qwen2-0.5B"
        assert config["r"] == 16
        assert config["lora_alpha"] == 32


class TestLoRAAdapterCheckpointDetection:
    """Test is_lora_adapter_checkpoint function."""

    def test_detects_valid_adapter(self, tmp_path):
        """Test detection of valid PEFT adapter checkpoint."""
        from areal.experimental.engine.archon_lora_checkpoint import (
            is_lora_adapter_checkpoint,
        )

        config = {"peft_type": "LORA", "r": 8, "lora_alpha": 16}
        config_path = tmp_path / "adapter_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        assert is_lora_adapter_checkpoint(str(tmp_path))

    def test_rejects_missing_config(self, tmp_path):
        """Test rejection when no adapter_config.json exists."""
        from areal.experimental.engine.archon_lora_checkpoint import (
            is_lora_adapter_checkpoint,
        )

        assert not is_lora_adapter_checkpoint(str(tmp_path))

    def test_rejects_non_lora_config(self, tmp_path):
        """Test rejection of non-LoRA adapter type."""
        from areal.experimental.engine.archon_lora_checkpoint import (
            is_lora_adapter_checkpoint,
        )

        config = {"peft_type": "PREFIX_TUNING"}
        config_path = tmp_path / "adapter_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        assert not is_lora_adapter_checkpoint(str(tmp_path))

    def test_handles_invalid_json(self, tmp_path):
        """Test handling of invalid JSON config file."""
        from areal.experimental.engine.archon_lora_checkpoint import (
            is_lora_adapter_checkpoint,
        )

        config_path = tmp_path / "adapter_config.json"
        config_path.write_text("not valid json {{{")

        assert not is_lora_adapter_checkpoint(str(tmp_path))


class TestStateDictRoundTrip:
    """Test state dict round-trip conversion with LoRA keys."""

    def setup_method(self):
        mock_config = Mock()
        mock_config.tie_word_embeddings = False
        self.adapter = Qwen2StateDictAdapter(mock_config)

    def test_lora_state_dict_roundtrip(self):
        """Test that LoRA keys survive HF -> Archon -> HF round-trip."""
        hf_state = {
            "model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(8, 64),
            "model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(64, 8),
            "model.layers.0.self_attn.v_proj.lora_A.weight": torch.randn(8, 64),
            "model.layers.0.self_attn.v_proj.lora_B.weight": torch.randn(64, 8),
        }

        # HF -> Archon
        archon_state = self.adapter.from_hf(hf_state)
        assert "layers.0.attention.wq.lora_a.weight" in archon_state
        assert "layers.0.attention.wv.lora_b.weight" in archon_state

        # Archon -> HF
        hf_state_back = self.adapter.to_hf(archon_state)

        assert set(hf_state.keys()) == set(hf_state_back.keys())
        for key in hf_state:
            assert torch.allclose(hf_state[key], hf_state_back[key])

    def test_mixed_base_and_lora_roundtrip(self):
        """Test round-trip with both base and LoRA keys."""
        hf_state = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
            "model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(8, 64),
            "model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(64, 8),
            "model.norm.weight": torch.randn(64),
        }

        archon_state = self.adapter.from_hf(hf_state)
        hf_state_back = self.adapter.to_hf(archon_state)

        assert set(hf_state.keys()) == set(hf_state_back.keys())
