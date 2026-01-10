"""Weight synchronization tests for Archon Engine.

These tests verify that Archon weight conversion to HuggingFace format
is correct, which is critical for SGLang weight sync.

Run tests:
    pytest areal/tests/experimental/archon/test_weight_sync.py -v

Note: These tests require GPU. They are included in CI as weight sync
correctness is critical for SGLang integration.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from areal.experimental.models.archon import get_supported_model_types
from areal.tests.experimental.archon.utils import (
    DualEngineFixture,
    get_model_path_for_type,
)

# Skip if no CUDA available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

# Get all supported model types for parametrization
SUPPORTED_MODEL_TYPES = sorted(get_supported_model_types())


@pytest.fixture(scope="module", params=SUPPORTED_MODEL_TYPES)
def engines(request):
    """Fixture to provide initialized engines for each supported model type."""
    model_type = request.param
    model_path = get_model_path_for_type(model_type)

    if model_path is None:
        pytest.skip(f"No model path configured for model type: {model_type}")

    fixture = DualEngineFixture(model_path=model_path)
    fixture.setup()
    fixture.model_type = model_type  # Store for test identification
    yield fixture
    fixture.teardown()


class TestWeightSync:
    """Test suite for verifying Archon weight sync to SGLang.

    These tests ensure 100% correctness of weight conversion, which is critical
    because any mismatch will cause silent failures in SGLang weight sync.
    """

    def test_archon_weight_name_conversion(self, engines: DualEngineFixture):
        """Verify that Archon state_dict_adapter correctly converts weight names to HF format.

        This is critical because SGLang expects HuggingFace format weight names.
        If conversion is wrong, weight updates will fail silently.
        """
        archon_engine = engines.archon_engine
        model_type = getattr(engines, "model_type", "unknown")

        # Get the state dict adapter
        adapter = archon_engine.state_dict_adapter
        assert adapter is not None, "state_dict_adapter should be initialized"

        # Check if model uses tied embeddings
        tie_word_embeddings = getattr(
            archon_engine.model_config, "tie_word_embeddings", False
        )

        # Expected HF name patterns (common across Qwen models)
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
        ]
        # lm_head.weight is only expected when embeddings are not tied
        if not tie_word_embeddings:
            hf_patterns.append("lm_head.weight")

        # Collect all converted names
        converted_names = set()
        archon_names = []

        for name, param in archon_engine.model.named_parameters():
            archon_names.append(name)
            # Convert to HF format (use _get_full_tensor to handle DTensor from FSDP2)
            tensor = archon_engine._get_full_tensor(param)
            hf_pairs = adapter.convert_single_to_hf(name, tensor)
            for hf_name, _ in hf_pairs:
                converted_names.add(hf_name)

        print(f"\n[Weight Name Conversion - {model_type}]")
        print(f"  Total Archon parameters: {len(archon_names)}")
        print(f"  Total converted HF names: {len(converted_names)}")
        print(f"  tie_word_embeddings: {tie_word_embeddings}")
        print(f"  Sample Archon names: {archon_names[:5]}")
        print(f"  Sample converted HF names: {list(converted_names)[:5]}")

        # Verify expected patterns exist (must be exact match, not substring)
        missing_patterns = []
        for pattern in hf_patterns:
            if pattern not in converted_names:
                missing_patterns.append(pattern)

        if missing_patterns:
            pytest.fail(
                f"[{model_type}] Missing required HF weight names: {missing_patterns}"
            )

        # Basic sanity check: ALL converted names should look like HF format
        for hf_name in converted_names:
            # HF names typically start with "model." or "lm_head"
            assert hf_name.startswith("model.") or hf_name.startswith("lm_head"), (
                f"[{model_type}] Unexpected HF name format: {hf_name}"
            )

    def test_archon_all_params_converted(self, engines: DualEngineFixture):
        """Verify that ALL Archon parameters are converted (no missing weights)."""
        archon_engine = engines.archon_engine
        model_type = getattr(engines, "model_type", "unknown")
        adapter = archon_engine.state_dict_adapter

        unconverted_params = []
        converted_count = 0

        for name, param in archon_engine.model.named_parameters():
            # Use _get_full_tensor to handle DTensor from FSDP2
            tensor = archon_engine._get_full_tensor(param)
            hf_pairs = adapter.convert_single_to_hf(name, tensor)
            if not hf_pairs:
                unconverted_params.append(name)
            else:
                converted_count += len(hf_pairs)

        print(f"\n[Parameter Conversion Coverage - {model_type}]")
        print(f"  Converted parameters: {converted_count}")
        print(f"  Unconverted parameters: {len(unconverted_params)}")

        if unconverted_params:
            print(f"  Unconverted names: {unconverted_params[:10]}")
            pytest.fail(
                f"[{model_type}] Found {len(unconverted_params)} unconverted parameters. "
                f"These weights will NOT be synced to SGLang!"
            )

    def test_archon_hf_weight_completeness(self, engines: DualEngineFixture):
        """Verify bidirectional completeness: all HF weights are covered by Archon conversion.

        This test checks that:
        1. Every Archon param converts to valid HF names
        2. Every HF model param is produced by Archon conversion (no missing weights)
        """
        archon_engine = engines.archon_engine
        model_type = getattr(engines, "model_type", "unknown")
        adapter = archon_engine.state_dict_adapter

        # Load HF model to get all expected weight names
        hf_model = AutoModelForCausalLM.from_pretrained(
            engines.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        hf_weight_names = set(hf_model.state_dict().keys())
        tie_word_embeddings = getattr(hf_model.config, "tie_word_embeddings", False)
        del hf_model

        # Collect all converted names from Archon
        archon_converted_names = set()
        for name, param in archon_engine.model.named_parameters():
            tensor = archon_engine._get_full_tensor(param)
            hf_pairs = adapter.convert_single_to_hf(name, tensor)
            for hf_name, _ in hf_pairs:
                archon_converted_names.add(hf_name)

        # Check: HF weights not covered by Archon
        missing_in_archon = hf_weight_names - archon_converted_names

        # If tie_word_embeddings is enabled, lm_head.weight is shared with embed_tokens
        # so it's expected to be missing from the converted weights
        if tie_word_embeddings and "lm_head.weight" in missing_in_archon:
            missing_in_archon.discard("lm_head.weight")

        # Check: Archon produces weights not in HF (unexpected extras)
        extra_in_archon = archon_converted_names - hf_weight_names

        print(f"\n[Weight Completeness Check - {model_type}]")
        print(f"  HF model weights: {len(hf_weight_names)}")
        print(f"  Archon converted weights: {len(archon_converted_names)}")
        print(f"  Missing in Archon: {len(missing_in_archon)}")
        print(f"  Extra in Archon: {len(extra_in_archon)}")

        if missing_in_archon:
            print(f"  Missing weights: {sorted(missing_in_archon)[:10]}")
            pytest.fail(
                f"[{model_type}] {len(missing_in_archon)} HF weights not produced by Archon: "
                f"{sorted(missing_in_archon)[:5]}..."
            )

        if extra_in_archon:
            print(f"  Extra weights: {sorted(extra_in_archon)[:10]}")
            pytest.fail(
                f"[{model_type}] {len(extra_in_archon)} unexpected weights from Archon: "
                f"{sorted(extra_in_archon)[:5]}..."
            )

    def test_archon_hf_weight_shape_match(self, engines: DualEngineFixture):
        """Verify that ALL converted weights have correct shapes matching HF model."""
        archon_engine = engines.archon_engine
        model_type = getattr(engines, "model_type", "unknown")
        adapter = archon_engine.state_dict_adapter

        # Load HF model to get expected shapes
        hf_model = AutoModelForCausalLM.from_pretrained(
            engines.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        hf_state_dict = hf_model.state_dict()
        del hf_model  # Free memory

        # Compare shapes for ALL weights
        shape_mismatches = []
        checked_count = 0

        for name, param in archon_engine.model.named_parameters():
            # Use _get_full_tensor to handle DTensor from FSDP2
            tensor = archon_engine._get_full_tensor(param)
            hf_pairs = adapter.convert_single_to_hf(name, tensor)
            for hf_name, hf_tensor in hf_pairs:
                if hf_name in hf_state_dict:
                    expected_shape = hf_state_dict[hf_name].shape
                    actual_shape = hf_tensor.shape
                    checked_count += 1
                    if expected_shape != actual_shape:
                        shape_mismatches.append(
                            {
                                "archon_name": name,
                                "hf_name": hf_name,
                                "expected": expected_shape,
                                "actual": actual_shape,
                            }
                        )

        print(f"\n[Weight Shape Verification - {model_type}]")
        print(f"  Weights checked: {checked_count}")
        print(f"  Shape mismatches: {len(shape_mismatches)}")

        if shape_mismatches:
            for m in shape_mismatches[:5]:
                print(
                    f"    {m['hf_name']}: expected {m['expected']}, got {m['actual']}"
                )
            pytest.fail(
                f"[{model_type}] Found {len(shape_mismatches)} shape mismatches. "
                f"Weight sync will fail or produce wrong results!"
            )

    def test_weight_values_100_percent(self, engines: DualEngineFixture):
        """Verify that ALL weight values are preserved after conversion.

        This test checks 100% of weights, not just a sample. This is critical
        because even a single corrupted weight can cause model degradation.
        """
        archon_engine = engines.archon_engine
        model_type = getattr(engines, "model_type", "unknown")
        adapter = archon_engine.state_dict_adapter

        # Load HF model
        hf_model = AutoModelForCausalLM.from_pretrained(
            engines.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        hf_state_dict = hf_model.state_dict()
        del hf_model

        # Compare values for ALL weights
        value_diffs = []
        checked_count = 0
        total_elements = 0

        for name, param in archon_engine.model.named_parameters():
            # Use _get_full_tensor to handle DTensor from FSDP2
            tensor = archon_engine._get_full_tensor(param)
            hf_pairs = adapter.convert_single_to_hf(name, tensor)
            for hf_name, hf_tensor in hf_pairs:
                if hf_name in hf_state_dict:
                    expected = hf_state_dict[hf_name]
                    if expected.shape == hf_tensor.shape:
                        # Move tensors to same device for comparison
                        expected_cpu = expected.float().cpu()
                        hf_tensor_cpu = hf_tensor.float().cpu()
                        diff = (expected_cpu - hf_tensor_cpu).abs()
                        max_diff = diff.max().item()
                        total_elements += expected.numel()
                        checked_count += 1

                        # Use strict tolerance for weight sync
                        if max_diff > 1e-5:
                            value_diffs.append(
                                {
                                    "hf_name": hf_name,
                                    "max_diff": max_diff,
                                    "mean_diff": diff.mean().item(),
                                }
                            )

        print(f"\n[Weight Value Verification (100%) - {model_type}]")
        print(f"  Weights checked: {checked_count}")
        print(f"  Total elements checked: {total_elements:,}")
        print(f"  Value mismatches (diff > 1e-5): {len(value_diffs)}")

        if value_diffs:
            for v in value_diffs[:10]:
                print(
                    f"    {v['hf_name']}: max_diff={v['max_diff']:.2e}, "
                    f"mean_diff={v['mean_diff']:.2e}"
                )
            pytest.fail(
                f"[{model_type}] Found {len(value_diffs)} weight value mismatches! "
                f"Weight sync will produce incorrect model behavior."
            )

        # Values should match exactly since we loaded from same checkpoint
        assert checked_count > 0, f"[{model_type}] No weights were checked!"
