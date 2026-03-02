"""Tests for LoRA adapter checkpoint I/O and PEFT format conversion.

Test Coverage:
1. State dict adapter LoRA key conversion (unit tests)
2. PEFT adapter config generation (unit tests)
3. Checkpoint save/load round-trip (integration tests)
4. PEFT compatibility (optional, requires PEFT library)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

# Import LoRA components
from areal.experimental.models.archon.lora.adapter import get_adapter_params
from areal.experimental.models.archon.lora.lora_linear import LoRALinear
from areal.experimental.models.archon.qwen2.model.state_dict_adapter import (
    Qwen2StateDictAdapter,
)

# Try to import PEFT for compatibility tests
try:
    # Check if PEFT is available (imports will be used in skipped manual tests)
    import peft  # noqa: F401
    import transformers  # noqa: F401

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class TestStateDictAdapterLoRAKeys:
    """Test LoRA key conversion in Qwen2StateDictAdapter."""

    def setup_method(self):
        """Create mock config and adapter."""
        from unittest.mock import Mock

        mock_config = Mock()
        mock_config.tie_word_embeddings = False
        self.adapter = Qwen2StateDictAdapter(mock_config)

    def test_qwen2_lora_key_conversion_attention(self):
        """Test attention LoRA key conversion (wq, wk, wv, wo)."""
        # Test q_proj
        hf_key = "model.layers.0.self_attn.q_proj.lora_A.weight"
        archon_key = self.adapter._convert_key_from_hf(hf_key)
        assert archon_key == "layers.0.attention.wq.lora_a.weight"

        # Test reverse conversion
        hf_key_back = self.adapter._convert_key_to_hf(archon_key)
        assert hf_key_back == hf_key

        # Test k_proj
        hf_key = "model.layers.5.self_attn.k_proj.lora_B.weight"
        archon_key = self.adapter._convert_key_from_hf(hf_key)
        assert archon_key == "layers.5.attention.wk.lora_b.weight"

        # Test v_proj
        hf_key = "model.layers.10.self_attn.v_proj.lora_A.weight"
        archon_key = self.adapter._convert_key_from_hf(hf_key)
        assert archon_key == "layers.10.attention.wv.lora_a.weight"

        # Test o_proj
        hf_key = "model.layers.15.self_attn.o_proj.lora_B.weight"
        archon_key = self.adapter._convert_key_from_hf(hf_key)
        assert archon_key == "layers.15.attention.wo.lora_b.weight"

    def test_qwen2_lora_key_conversion_mlp(self):
        """Test MLP LoRA key conversion (w1, w2, w3)."""
        # Test gate_proj (w1)
        hf_key = "model.layers.0.mlp.gate_proj.lora_A.weight"
        archon_key = self.adapter._convert_key_from_hf(hf_key)
        assert archon_key == "layers.0.feed_forward.w1.lora_a.weight"

        # Test down_proj (w2)
        hf_key = "model.layers.5.mlp.down_proj.lora_B.weight"
        archon_key = self.adapter._convert_key_from_hf(hf_key)
        assert archon_key == "layers.5.feed_forward.w2.lora_b.weight"

        # Test up_proj (w3)
        hf_key = "model.layers.10.mlp.up_proj.lora_A.weight"
        archon_key = self.adapter._convert_key_from_hf(hf_key)
        assert archon_key == "layers.10.feed_forward.w3.lora_a.weight"

    def test_qwen2_lora_key_conversion_output(self):
        """Test output layer LoRA key conversion."""
        hf_key = "lm_head.lora_A.weight"
        archon_key = self.adapter._convert_key_from_hf(hf_key)
        assert archon_key == "output.lora_a.weight"

        hf_key = "lm_head.lora_B.weight"
        archon_key = self.adapter._convert_key_from_hf(hf_key)
        assert archon_key == "output.lora_b.weight"

    def test_qwen2_lora_roundtrip(self):
        """Test that Archon -> HF -> Archon conversion preserves keys."""
        archon_keys = [
            "layers.0.attention.wq.lora_a.weight",
            "layers.0.attention.wq.lora_b.weight",
            "layers.5.feed_forward.w2.lora_a.weight",
            "output.lora_a.weight",
        ]

        for original_key in archon_keys:
            # Archon -> HF
            hf_key = self.adapter._convert_key_to_hf(original_key)
            assert hf_key is not None, f"Failed to convert {original_key} to HF"

            # HF -> Archon
            roundtrip_key = self.adapter._convert_key_from_hf(hf_key)
            assert roundtrip_key == original_key, (
                f"Round-trip failed: {original_key} -> {hf_key} -> {roundtrip_key}"
            )

    def test_case_conversion(self):
        """Verify that lora_a <-> lora_A and lora_b <-> lora_B conversion works."""
        # Lowercase to uppercase (Archon -> HF)
        archon_key = "layers.0.attention.wq.lora_a.weight"
        hf_key = self.adapter._convert_key_to_hf(archon_key)
        assert "lora_A" in hf_key, f"Expected lora_A in {hf_key}"
        assert "lora_a" not in hf_key, f"Should not have lora_a in {hf_key}"

        # Uppercase to lowercase (HF -> Archon)
        hf_key = "model.layers.0.self_attn.q_proj.lora_B.weight"
        archon_key = self.adapter._convert_key_from_hf(hf_key)
        assert "lora_b" in archon_key, f"Expected lora_b in {archon_key}"
        assert "lora_B" not in archon_key, f"Should not have lora_B in {archon_key}"

    def test_all_target_modules_covered(self):
        """Verify all supported modules have LoRA mappings."""
        expected_modules = ["wq", "wk", "wv", "wo", "w1", "w2", "w3", "output"]

        for module in expected_modules:
            # Test lora_a mapping exists
            if module == "output":
                archon_key = f"{module}.lora_a.weight"
            else:
                archon_key = f"layers.0.attention.{module}.lora_a.weight"
                if module in ("w1", "w2", "w3"):
                    archon_key = f"layers.0.feed_forward.{module}.lora_a.weight"

            hf_key = self.adapter._convert_key_to_hf(archon_key)
            assert hf_key is not None, (
                f"Missing LoRA mapping for module {module} (lora_a)"
            )

            # Test lora_b mapping exists
            archon_key = archon_key.replace("lora_a", "lora_b")
            hf_key = self.adapter._convert_key_to_hf(archon_key)
            assert hf_key is not None, (
                f"Missing LoRA mapping for module {module} (lora_b)"
            )


class TestAdapterConfig:
    """Test PEFT adapter configuration generation."""

    def test_create_adapter_config(self):
        """Test that adapter config has correct PEFT structure."""
        from dataclasses import dataclass
        from unittest.mock import Mock

        # Create LoRA config
        @dataclass
        class LoRAConfig:
            rank: int
            alpha: float
            target_modules: list[str]

        lora_config = LoRAConfig(rank=8, alpha=16.0, target_modules=["wq", "wv"])

        # Create state dict adapter
        mock_model_config = Mock()
        mock_model_config.tie_word_embeddings = False
        adapter = Qwen2StateDictAdapter(mock_model_config)

        config = adapter.create_peft_adapter_config(
            lora_config=lora_config,
            base_model_path="Qwen/Qwen2-0.5B",
        )

        # Verify required PEFT fields
        assert config["peft_type"] == "LORA"
        assert config["task_type"] == "CAUSAL_LM"
        assert config["r"] == 8
        assert config["lora_alpha"] == 16
        assert config["base_model_name_or_path"] == "Qwen/Qwen2-0.5B"
        assert config["lora_dropout"] == 0.0
        assert config["fan_in_fan_out"] is False
        assert config["bias"] == "none"

    def test_target_modules_conversion(self):
        """Test that Archon module names are converted to PEFT names."""
        from dataclasses import dataclass
        from unittest.mock import Mock

        # Create LoRA config
        @dataclass
        class LoRAConfig:
            rank: int
            alpha: float
            target_modules: list[str]

        lora_config = LoRAConfig(
            rank=8, alpha=16.0, target_modules=["wq", "wv", "w1", "output"]
        )

        # Create state dict adapter
        mock_model_config = Mock()
        mock_model_config.tie_word_embeddings = False
        adapter = Qwen2StateDictAdapter(mock_model_config)

        config = adapter.create_peft_adapter_config(
            lora_config=lora_config,
            base_model_path=None,
        )

        expected_peft_modules = ["q_proj", "v_proj", "gate_proj", "lm_head"]
        assert config["target_modules"] == expected_peft_modules

        # Verify all mappings exist
        for archon_name, peft_name in adapter.to_peft_module_map.items():
            assert isinstance(archon_name, str)
            assert isinstance(peft_name, str)


class TestCheckpointIO:
    """Integration tests for checkpoint save/load."""

    def test_save_load_adapter_roundtrip(self):
        """Test that save -> load preserves adapter weights."""
        from unittest.mock import Mock

        # Create simple model with LoRA
        model = nn.Sequential(
            LoRALinear(64, 32, rank=8, alpha=16.0),
            nn.ReLU(),
            LoRALinear(32, 16, rank=8, alpha=16.0),
        )

        # Initialize with random weights
        for module in model.modules():
            if isinstance(module, LoRALinear):
                with torch.no_grad():
                    module.lora_a.weight.uniform_(-0.1, 0.1)
                    module.lora_b.weight.uniform_(-0.1, 0.1)

        # Create mock engine
        mock_engine = Mock()
        mock_engine.model = model
        mock_config = Mock()
        mock_config.rank = 8
        mock_config.alpha = 16.0
        mock_engine.lora_config = mock_config

        # Create state dict adapter
        mock_model_config = Mock()
        mock_model_config.tie_word_embeddings = False
        adapter = Qwen2StateDictAdapter(mock_model_config)
        mock_engine.state_dict_adapter = adapter

        # Save adapter
        with tempfile.TemporaryDirectory() as tmpdir:
            from areal.experimental.engine.archon_lora_checkpoint import (
                save_lora_adapter,
            )

            save_lora_adapter(mock_engine, tmpdir, base_model_path="test/model")

            # Verify files created
            assert os.path.exists(os.path.join(tmpdir, "adapter_model.safetensors"))
            assert os.path.exists(os.path.join(tmpdir, "adapter_config.json"))

            # Save original weights
            original_weights = {
                name: param.data.clone()
                for name, param in get_adapter_params(model).items()
            }

            # Modify weights
            for module in model.modules():
                if isinstance(module, LoRALinear):
                    module.lora_a.weight.zero_()
                    module.lora_b.weight.zero_()

            # Load adapter
            from areal.experimental.engine.archon_lora_checkpoint import (
                load_lora_adapter,
            )

            load_lora_adapter(mock_engine, tmpdir, strict=True)

            # Verify weights restored
            loaded_weights = get_adapter_params(model)
            for name, original_weight in original_weights.items():
                loaded_weight = loaded_weights[name]
                torch.testing.assert_close(
                    loaded_weight, original_weight, rtol=1e-7, atol=1e-7
                )

    def test_adapter_checkpoint_structure(self):
        """Test that adapter checkpoint has correct PEFT structure."""
        from unittest.mock import Mock

        model = nn.Sequential(LoRALinear(64, 32, rank=8, alpha=16.0))

        mock_engine = Mock()
        mock_engine.model = model
        mock_config = Mock()
        mock_config.rank = 8
        mock_config.alpha = 16.0
        mock_engine.lora_config = mock_config

        mock_model_config = Mock()
        mock_model_config.tie_word_embeddings = False
        adapter = Qwen2StateDictAdapter(mock_model_config)
        mock_engine.state_dict_adapter = adapter

        with tempfile.TemporaryDirectory() as tmpdir:
            from areal.experimental.engine.archon_lora_checkpoint import (
                save_lora_adapter,
            )

            save_lora_adapter(mock_engine, tmpdir)

            # Check adapter_model.safetensors
            weights_path = os.path.join(tmpdir, "adapter_model.safetensors")
            assert os.path.exists(weights_path)

            weights = load_file(weights_path)
            # Verify keys have PEFT prefix
            for key in weights.keys():
                assert key.startswith("base_model.model.")

            # Check adapter_config.json
            config_path = os.path.join(tmpdir, "adapter_config.json")
            assert os.path.exists(config_path)

            with open(config_path) as f:
                config = json.load(f)
            assert config["peft_type"] == "LORA"
            assert config["r"] == 8
            assert config["lora_alpha"] == 16

    def test_load_strict_mode(self):
        """Test that strict=True raises on missing keys."""
        from unittest.mock import Mock

        model = nn.Sequential(
            LoRALinear(64, 32, rank=8, alpha=16.0),
            LoRALinear(32, 16, rank=8, alpha=16.0),
        )

        mock_engine = Mock()
        mock_engine.model = model
        mock_config = Mock()
        mock_config.rank = 8
        mock_config.alpha = 16.0
        mock_engine.lora_config = mock_config

        mock_model_config = Mock()
        mock_model_config.tie_word_embeddings = False
        adapter = Qwen2StateDictAdapter(mock_model_config)
        mock_engine.state_dict_adapter = adapter

        with tempfile.TemporaryDirectory() as tmpdir:
            from areal.experimental.engine.archon_lora_checkpoint import (
                load_lora_adapter,
                save_lora_adapter,
            )

            # Save only first module's adapter
            model_partial = nn.Sequential(model[0])
            mock_engine.model = model_partial
            save_lora_adapter(mock_engine, tmpdir)

            # Try to load into full model (missing keys)
            mock_engine.model = model

            # Strict mode should raise
            with pytest.raises(ValueError, match="Missing keys"):
                load_lora_adapter(mock_engine, tmpdir, strict=True)

            # Non-strict mode should succeed with warning
            load_lora_adapter(mock_engine, tmpdir, strict=False)

    def test_is_lora_adapter_checkpoint(self):
        """Test adapter checkpoint detection."""
        from areal.experimental.engine.archon_lora_checkpoint import (
            is_lora_adapter_checkpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty directory - not an adapter
            assert not is_lora_adapter_checkpoint(tmpdir)

            # Create adapter_config.json with LORA type
            config_path = Path(tmpdir) / "adapter_config.json"
            with open(config_path, "w") as f:
                json.dump({"peft_type": "LORA"}, f)

            assert is_lora_adapter_checkpoint(tmpdir)

            # Wrong peft_type - not a LoRA adapter
            with open(config_path, "w") as f:
                json.dump({"peft_type": "PREFIX_TUNING"}, f)

            assert not is_lora_adapter_checkpoint(tmpdir)

    def test_peft_prefix_handling(self):
        """Test that base_model.model. prefix is added/stripped correctly."""
        from unittest.mock import Mock

        model = nn.Sequential(LoRALinear(64, 32, rank=8, alpha=16.0))

        mock_engine = Mock()
        mock_engine.model = model
        mock_config = Mock()
        mock_config.rank = 8
        mock_config.alpha = 16.0
        mock_engine.lora_config = mock_config

        mock_model_config = Mock()
        mock_model_config.tie_word_embeddings = False
        adapter = Qwen2StateDictAdapter(mock_model_config)
        mock_engine.state_dict_adapter = adapter

        with tempfile.TemporaryDirectory() as tmpdir:
            from areal.experimental.engine.archon_lora_checkpoint import (
                save_lora_adapter,
            )

            save_lora_adapter(mock_engine, tmpdir)

            # Load weights and verify prefix
            weights_path = os.path.join(tmpdir, "adapter_model.safetensors")
            weights = load_file(weights_path)

            # All keys should have base_model.model. prefix
            for key in weights.keys():
                assert key.startswith("base_model.model."), (
                    f"Key {key} missing PEFT prefix"
                )

            # Load and verify prefix is stripped
            from areal.experimental.engine.archon_lora_checkpoint import (
                load_lora_adapter,
            )

            load_lora_adapter(mock_engine, tmpdir, strict=True)
            # If loading succeeds without error, prefix was stripped correctly


@pytest.mark.skipif(not PEFT_AVAILABLE, reason="PEFT library not installed")
class TestPEFTCompatibility:
    """Test compatibility with PEFT library (optional)."""

    def test_load_archon_adapter_with_peft(self):
        """Test that adapters saved by Archon can be loaded by PEFT."""
        pytest.skip("Requires PEFT library and base model - manual test only")
        # This test requires a real base model and is marked for manual testing
        # Uncomment and modify for actual testing:
        #
        # from unittest.mock import Mock
        # model = nn.Sequential(LoRALinear(4096, 4096, rank=8, alpha=16.0))
        # mock_engine = Mock()
        # mock_engine.model = model
        # ...
        # save_lora_adapter(mock_engine, "/tmp/archon_adapter")
        #
        # base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
        # peft_model = PeftModel.from_pretrained(base_model, "/tmp/archon_adapter")
        # # If this succeeds, Archon adapter is PEFT-compatible!
