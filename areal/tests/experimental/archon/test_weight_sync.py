"""Weight synchronization tests for Archon Engine.

These tests verify that Archon weight conversion to HuggingFace format
is correct, which is critical for SGLang weight sync.

Run tests:
    pytest areal/tests/experimental/archon/test_weight_sync.py -v

Note: These tests require GPU and are marked as slow.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from areal.tests.experimental.archon.utils import (
    DualEngineFixture,
)

# Skip if no CUDA available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@pytest.fixture(scope="module")
def engines():
    """Fixture to provide initialized engines."""
    fixture = DualEngineFixture()
    fixture.setup()
    yield fixture
    fixture.teardown()


class TestWeightSync:
    """Test suite for verifying Archon weight sync to SGLang."""

    @pytest.mark.slow
    def test_archon_weight_name_conversion(self, engines: DualEngineFixture):
        """Verify that Archon state_dict_adapter correctly converts weight names to HF format.

        This is critical because SGLang expects HuggingFace format weight names.
        If conversion is wrong, weight updates will fail silently.
        """
        archon_engine = engines.archon_engine

        # Get the state dict adapter
        adapter = archon_engine.state_dict_adapter
        assert adapter is not None, "state_dict_adapter should be initialized"

        # Expected HF name patterns
        hf_patterns = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]

        # Collect all converted names
        converted_names = set()
        archon_names = []

        for name, param in archon_engine.model.named_parameters():
            archon_names.append(name)
            # Convert to HF format
            hf_pairs = adapter.convert_single_to_hf(name, param.data)
            for hf_name, _ in hf_pairs:
                converted_names.add(hf_name)

        print("\n[Weight Name Conversion]")
        print(f"  Total Archon parameters: {len(archon_names)}")
        print(f"  Total converted HF names: {len(converted_names)}")
        print(f"  Sample Archon names: {archon_names[:5]}")
        print(f"  Sample converted HF names: {list(converted_names)[:5]}")

        # Verify expected patterns exist
        missing_patterns = []
        for pattern in hf_patterns:
            if not any(pattern in name for name in converted_names):
                missing_patterns.append(pattern)

        if missing_patterns:
            print(f"  WARNING: Missing HF patterns: {missing_patterns}")

        # Basic sanity check: converted names should look like HF format
        for hf_name in list(converted_names)[:10]:
            # HF names typically start with "model." or "lm_head"
            assert hf_name.startswith("model.") or hf_name.startswith("lm_head"), (
                f"Unexpected HF name format: {hf_name}"
            )

    @pytest.mark.slow
    def test_archon_all_params_converted(self, engines: DualEngineFixture):
        """Verify that ALL Archon parameters are converted (no missing weights)."""
        archon_engine = engines.archon_engine
        adapter = archon_engine.state_dict_adapter

        unconverted_params = []
        converted_count = 0

        for name, param in archon_engine.model.named_parameters():
            hf_pairs = adapter.convert_single_to_hf(name, param.data)
            if not hf_pairs:
                unconverted_params.append(name)
            else:
                converted_count += len(hf_pairs)

        print("\n[Parameter Conversion Coverage]")
        print(f"  Converted parameters: {converted_count}")
        print(f"  Unconverted parameters: {len(unconverted_params)}")

        if unconverted_params:
            print(f"  Unconverted names: {unconverted_params[:10]}")
            pytest.fail(
                f"Found {len(unconverted_params)} unconverted parameters. "
                f"These weights will NOT be synced to SGLang!"
            )

    @pytest.mark.slow
    def test_archon_hf_weight_shape_match(self, engines: DualEngineFixture):
        """Verify that converted weights have correct shapes matching HF model."""
        archon_engine = engines.archon_engine
        adapter = archon_engine.state_dict_adapter

        # Load HF model to get expected shapes
        hf_model = AutoModelForCausalLM.from_pretrained(
            engines.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        hf_state_dict = hf_model.state_dict()
        del hf_model  # Free memory

        # Compare shapes
        shape_mismatches = []

        for name, param in archon_engine.model.named_parameters():
            hf_pairs = adapter.convert_single_to_hf(name, param.data)
            for hf_name, hf_tensor in hf_pairs:
                if hf_name in hf_state_dict:
                    expected_shape = hf_state_dict[hf_name].shape
                    actual_shape = hf_tensor.shape
                    if expected_shape != actual_shape:
                        shape_mismatches.append(
                            {
                                "archon_name": name,
                                "hf_name": hf_name,
                                "expected": expected_shape,
                                "actual": actual_shape,
                            }
                        )

        print("\n[Weight Shape Verification]")
        print(f"  Shape mismatches: {len(shape_mismatches)}")

        if shape_mismatches:
            for m in shape_mismatches[:5]:
                print(
                    f"    {m['hf_name']}: expected {m['expected']}, got {m['actual']}"
                )
            pytest.fail(
                f"Found {len(shape_mismatches)} shape mismatches. "
                f"Weight sync will fail or produce wrong results!"
            )

    @pytest.mark.slow
    def test_weight_values_after_conversion(self, engines: DualEngineFixture):
        """Verify that weight VALUES are preserved after conversion (not just names/shapes)."""
        archon_engine = engines.archon_engine
        adapter = archon_engine.state_dict_adapter

        # Load HF model
        hf_model = AutoModelForCausalLM.from_pretrained(
            engines.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        hf_state_dict = hf_model.state_dict()
        del hf_model

        # Compare values for a few key weights
        value_diffs = []
        checked_count = 0

        for name, param in archon_engine.model.named_parameters():
            hf_pairs = adapter.convert_single_to_hf(name, param.data)
            for hf_name, hf_tensor in hf_pairs:
                if hf_name in hf_state_dict:
                    expected = hf_state_dict[hf_name]
                    if expected.shape == hf_tensor.shape:
                        # Move tensors to same device for comparison
                        expected_cpu = expected.float().cpu()
                        hf_tensor_cpu = hf_tensor.float().cpu()
                        diff = (expected_cpu - hf_tensor_cpu).abs()
                        max_diff = diff.max().item()
                        if max_diff > 1e-4:
                            value_diffs.append(
                                {
                                    "hf_name": hf_name,
                                    "max_diff": max_diff,
                                }
                            )
                        checked_count += 1

            if checked_count >= 20:  # Check first 20 weights
                break

        print("\n[Weight Value Verification]")
        print(f"  Checked weights: {checked_count}")
        print(f"  Value mismatches (diff > 1e-4): {len(value_diffs)}")

        if value_diffs:
            for v in value_diffs[:5]:
                print(f"    {v['hf_name']}: max_diff={v['max_diff']:.6f}")

        # Values should match since we loaded from same checkpoint
        assert len(value_diffs) == 0, (
            f"Found {len(value_diffs)} weight value mismatches after conversion!"
        )
