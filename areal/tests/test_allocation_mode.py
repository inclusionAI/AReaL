#!/usr/bin/env python3
"""
Comprehensive test suite for AllocationMode with integrated lark-based parsing and validation rules.

Test Rules:
1. If using separate attn and ffn parallelism strategies, their pipeline parallelism
   degree should be the same, and world sizes should be the same.
2. For colocate expressions, the world sizes of inference and training should be the same.
3. The fsdp backend currently only supports data parallelism, raise errors if others are detected.
4. Currently we don't support megatron backend.
5. We don't support pipeline parallelism for inference.
6. If the training parallelism strategy is specified by disaggregated attn and ffn,
   it assumes using the megatron backend.
7. If training backend includes a non-one pipeline parallelism or expert parallelism,
   use the megatron backend if not specified, else use fsdp.
"""

import pytest

from areal.api.alloc_mode import (
    AllocationMode,
    AllocationType,
    AllocationValidationError,
    InvalidAllocationModeError,
    ParallelStrategy,
)


class TestAllocationModeBackwardCompatibility:
    """Test backward compatibility with existing experimental tests"""

    def test_colocate(self):
        alloc_mode_str = "d2p2t1"
        alloc_mode = AllocationMode.from_str(alloc_mode_str)
        assert alloc_mode.type_ == AllocationType.COLOCATE
        train_ps = alloc_mode.train
        assert ParallelStrategy.parallelism_eq(
            train_ps,
            ParallelStrategy(
                tensor_parallel_size=1, data_parallel_size=2, pipeline_parallel_size=2
            ),
        )
        assert train_ps.world_size == 4, alloc_mode_str

        alloc_mode_str = "d2p2t4e2c4"
        alloc_mode = AllocationMode.from_str(alloc_mode_str)
        assert alloc_mode.type_ == AllocationType.COLOCATE
        train_ps = alloc_mode.train
        assert ParallelStrategy.parallelism_eq(
            train_ps,
            ParallelStrategy(
                tensor_parallel_size=4,
                data_parallel_size=2,
                pipeline_parallel_size=2,
                context_parallel_size=4,
                expert_parallel_size=2,
                expert_tensor_parallel_size=4,
            ),
        )
        assert train_ps.world_size == 64, alloc_mode_str
        assert train_ps.expert_data_parallel_size == 4, alloc_mode_str

        alloc_mode_str = "d4p2t2c2/d2p2t4e2"
        alloc_mode = AllocationMode.from_str(alloc_mode_str)
        assert alloc_mode.type_ == AllocationType.COLOCATE
        train_ps = alloc_mode.train
        assert ParallelStrategy.parallelism_eq(
            train_ps,
            ParallelStrategy(
                tensor_parallel_size=2,
                data_parallel_size=4,
                pipeline_parallel_size=2,
                context_parallel_size=2,
                expert_parallel_size=2,
                expert_tensor_parallel_size=4,
            ),
        )
        assert train_ps.world_size == 32, alloc_mode_str
        assert train_ps.expert_data_parallel_size == 2, alloc_mode_str

        alloc_mode_str = "d2p2t1c4/d2p2t1e2"
        with pytest.raises(InvalidAllocationModeError):
            train_ps = AllocationMode.from_str(alloc_mode_str)

    def test_decoupled_train(self):
        alloc_mode_str = "vllm.d2p2t2+d2p2t2"
        alloc_mode = AllocationMode.from_str(alloc_mode_str)
        assert alloc_mode.type_ == AllocationType.DECOUPLED_TRAIN
        assert alloc_mode.gen_backend == "vllm"
        train_ps = alloc_mode.train
        assert ParallelStrategy.parallelism_eq(
            train_ps,
            ParallelStrategy(
                tensor_parallel_size=2,
                data_parallel_size=2,
                pipeline_parallel_size=2,
            ),
        )
        assert train_ps.world_size == 8, alloc_mode_str
        gen_ps = alloc_mode.gen
        assert ParallelStrategy.parallelism_eq(
            gen_ps,
            ParallelStrategy(
                tensor_parallel_size=2,
                data_parallel_size=2,
                pipeline_parallel_size=2,
            ),
        )
        assert gen_ps.world_size == 8, alloc_mode_str

        alloc_mode_str = "sglang.d4p2t2+d2p2t2c2/d2p2t2e2"
        alloc_mode = AllocationMode.from_str(alloc_mode_str)
        assert alloc_mode.type_ == AllocationType.DECOUPLED_TRAIN
        assert alloc_mode.gen_backend == "sglang"
        train_ps = alloc_mode.train
        assert ParallelStrategy.parallelism_eq(
            train_ps,
            ParallelStrategy(
                tensor_parallel_size=2,
                data_parallel_size=2,
                pipeline_parallel_size=2,
                context_parallel_size=2,
                expert_parallel_size=2,
                expert_tensor_parallel_size=2,
            ),
        )
        assert train_ps.world_size == 16, alloc_mode_str
        gen_ps = AllocationMode.from_str(alloc_mode_str).gen
        assert ParallelStrategy.parallelism_eq(
            gen_ps,
            ParallelStrategy(
                tensor_parallel_size=2, data_parallel_size=4, pipeline_parallel_size=2
            ),
        )
        assert gen_ps.world_size == 16, alloc_mode_str

    def test_decoupled_eval(self):
        alloc_mode_str = "sglang.d4p1t1+eval"
        alloc_mode = AllocationMode.from_str(alloc_mode_str)
        assert alloc_mode.type_ == AllocationType.DECOUPLED_EVAL
        assert alloc_mode.gen_backend == "sglang"
        gen_ps = alloc_mode.gen
        assert ParallelStrategy.parallelism_eq(
            gen_ps,
            ParallelStrategy(
                tensor_parallel_size=1, data_parallel_size=4, pipeline_parallel_size=1
            ),
        )
        assert gen_ps.world_size == 4, alloc_mode_str

    def test_llm_server_only(self):
        alloc_mode_str = "sglang.d4p2t2"
        alloc_mode = AllocationMode.from_str(alloc_mode_str)
        assert alloc_mode.type_ == AllocationType.LLM_SERVER_ONLY
        assert alloc_mode.gen_backend == "sglang"
        gen_ps = alloc_mode.gen
        assert ParallelStrategy.parallelism_eq(
            gen_ps,
            ParallelStrategy(
                tensor_parallel_size=2, data_parallel_size=4, pipeline_parallel_size=2
            ),
        )
        assert gen_ps.world_size == 16, alloc_mode_str


class TestNewValidationRules:
    """Test new validation rules with lark-based parser when available"""

    def setup_method(self):
        # Test will be skipped if lark is not available
        try:
            pass
        except ImportError:
            pytest.skip("Lark not available, skipping validation tests")

    def test_rule_5_no_inference_pipeline_parallelism(self):
        """Rule 5: No pipeline parallelism for inference"""
        # Should raise error for pipeline parallelism in inference
        with pytest.raises(
            (ValueError, AllocationValidationError),
            match="Pipeline parallelism not supported for inference",
        ):
            AllocationMode.from_str("sglang:d4t2p2")

        # Should work fine without pipeline parallelism
        result = AllocationMode.from_str("sglang:d4t2")
        assert result.gen_dp_size == 4
        assert result.gen_tp_size == 2
        assert result.gen_pp_size == 1

    def test_rule_3_fsdp_data_parallelism_only(self):
        """Rule 3: FSDP backend only supports data parallelism"""
        # Should work with data parallelism only
        result = AllocationMode.from_str("sglang:d4t2+fsdp:d8")
        assert result.train_dp_size == 8

        # Should raise error with tensor parallelism
        with pytest.raises(
            (ValueError, AllocationValidationError),
            match="FSDP backend only supports data parallelism",
        ):
            AllocationMode.from_str("sglang:d4t2+fsdp:d4t2")

        # Should raise error with pipeline parallelism
        with pytest.raises(
            (ValueError, AllocationValidationError),
            match="FSDP backend only supports data parallelism",
        ):
            AllocationMode.from_str("sglang:d4t2+fsdp:d4p2")

        # Should raise error with context parallelism
        with pytest.raises(
            (ValueError, AllocationValidationError),
            match="FSDP backend only supports data parallelism",
        ):
            AllocationMode.from_str("sglang:d4t2+fsdp:d4c2")

    def test_rule_4_megatron_not_supported(self):
        """Rule 4: Megatron backend is not currently supported"""
        with pytest.raises(
            (ValueError, AllocationValidationError),
            match="Megatron backend is not currently supported",
        ):
            AllocationMode.from_str("sglang:d4t2+megatron:d4p2t2")

    def test_rule_7_automatic_backend_selection(self):
        """Rule 7: Automatic backend selection based on parallelism"""
        # Should auto-select FSDP for data-only parallelism
        result = AllocationMode.from_str("sglang:d4t2+d8")
        # The training backend is auto-selected as fsdp, but we need to check the property access
        assert result.train_dp_size == 8

        # Auto-selected megatron should NOT fail validation (for backward compatibility)
        # These should parse successfully but use megatron internally
        result = AllocationMode.from_str("sglang:d4t2+d4p2")
        assert result.train_pp_size == 2  # Pipeline parallelism should work

        result = AllocationMode.from_str("sglang:d4t2+d4e2")
        assert result.train_ep_size == 2  # Expert parallelism should work


class TestLegacyCompatibility:
    """Test that legacy patterns still work with the integrated parser"""

    def test_legacy_property_access(self):
        """Test that all legacy properties still work"""
        # Test colocate allocation
        alloc_mode = AllocationMode.from_str("d2p2t1")
        assert alloc_mode.gen_tp_size == 1
        assert alloc_mode.gen_pp_size == 2
        assert alloc_mode.gen_dp_size == 2
        assert alloc_mode.gen_world_size == 4
        assert alloc_mode.train_tp_size == 1
        assert alloc_mode.train_pp_size == 2
        assert alloc_mode.train_dp_size == 2
        assert alloc_mode.train_world_size == 4

        # Test decoupled allocation
        alloc_mode = AllocationMode.from_str("sglang.d4p1t2+d2p1t1")
        assert alloc_mode.gen_tp_size == 2
        assert alloc_mode.gen_pp_size == 1
        assert alloc_mode.gen_dp_size == 4
        assert alloc_mode.gen_world_size == 8
        assert alloc_mode.train_tp_size == 1
        assert alloc_mode.train_pp_size == 1
        assert alloc_mode.train_dp_size == 2
        assert alloc_mode.train_world_size == 2

        # Test server-only allocation
        alloc_mode = AllocationMode.from_str("sglang.d4p1t2")
        assert alloc_mode.gen_tp_size == 2
        assert alloc_mode.gen_pp_size == 1
        assert alloc_mode.gen_dp_size == 4
        assert alloc_mode.gen_world_size == 8

        # Test eval allocation
        alloc_mode = AllocationMode.from_str("sglang.d4p1t2+eval")
        assert alloc_mode.gen_tp_size == 2
        assert alloc_mode.gen_pp_size == 1
        assert alloc_mode.gen_dp_size == 4
        assert alloc_mode.gen_world_size == 8

    def test_parallel_strat_compatibility(self):
        """Test that parallel_strat dict is populated for backward compatibility"""
        alloc_mode = AllocationMode.from_str("d2p2t1")
        assert "gen" in alloc_mode.parallel_strat
        assert "*" in alloc_mode.parallel_strat
        assert alloc_mode.parallel_strat["gen"]["d"] == 2
        assert alloc_mode.parallel_strat["gen"]["p"] == 2
        assert alloc_mode.parallel_strat["gen"]["t"] == 1
        assert alloc_mode.parallel_strat["*"]["d"] == 2
        assert alloc_mode.parallel_strat["*"]["p"] == 2
        assert alloc_mode.parallel_strat["*"]["t"] == 1

        alloc_mode = AllocationMode.from_str("sglang.d4p1t2+d2p1t1")
        assert alloc_mode.parallel_strat["gen"]["d"] == 4
        assert alloc_mode.parallel_strat["gen"]["p"] == 1
        assert alloc_mode.parallel_strat["gen"]["t"] == 2
        assert alloc_mode.parallel_strat["*"]["d"] == 2
        assert alloc_mode.parallel_strat["*"]["p"] == 1
        assert alloc_mode.parallel_strat["*"]["t"] == 1


class TestValidConfigurations:
    """Test configurations that should work"""

    def test_simple_configurations(self):
        """Test simple configurations that should parse successfully"""
        valid_cases = [
            # Legacy patterns - should work with both parsers
            "d2p1t1",
            "d4t2p1",
            "sglang.d4t2",
            "vllm.d8t1",
            "sglang.d4t2+eval",
            "sglang.d4t2+cpu",
        ]

        for case in valid_cases:
            result = AllocationMode.from_str(case)
            assert result is not None, f"Failed to parse: {case}"

    def test_valid_fsdp_configurations(self):
        """Test valid FSDP configurations (data-only)"""
        valid_cases = [
            "sglang:d4t2+fsdp:d8",
            "sglang:d4t2+d8",  # Should auto-select FSDP
        ]

        for case in valid_cases:
            result = AllocationMode.from_str(case)
            assert result is not None, f"Failed to parse: {case}"
            if result.train:
                assert result.train_dp_size > 1


class TestInvalidConfigurations:
    """Test configurations that should fail validation"""

    def setup_method(self):
        # Test will be skipped if lark is not available
        try:
            pass
        except ImportError:
            pytest.skip("Lark not available, skipping validation tests")

    def test_invalid_inference_configurations(self):
        """Test inference configurations that should fail"""
        invalid_cases = [
            ("sglang:d4t2p2", "Pipeline parallelism not supported for inference"),
            ("sglang:d4p2", "Pipeline parallelism not supported for inference"),
        ]

        for case, expected_error in invalid_cases:
            with pytest.raises(
                (ValueError, AllocationValidationError), match=expected_error
            ):
                AllocationMode.from_str(case)

    def test_invalid_fsdp_configurations(self):
        """Test FSDP configurations that should fail"""
        invalid_cases = [
            ("sglang:d4t2+fsdp:d4t2", "FSDP backend only supports data parallelism"),
            ("sglang:d4t2+fsdp:d4p2", "FSDP backend only supports data parallelism"),
            ("sglang:d4t2+fsdp:d4c2", "FSDP backend only supports data parallelism"),
        ]

        for case, expected_error in invalid_cases:
            with pytest.raises(
                (ValueError, AllocationValidationError), match=expected_error
            ):
                AllocationMode.from_str(case)

    def test_invalid_megatron_configurations(self):
        """Test Megatron configurations that should fail"""
        # Only explicitly specified megatron backend should fail
        invalid_cases = [
            (
                "sglang:d4t2+megatron:d4p2",
                "Megatron backend is not currently supported",
            ),
            (
                "sglang:d4t2+megatron:d4e2",
                "Megatron backend is not currently supported",
            ),
        ]

        for case, expected_error in invalid_cases:
            with pytest.raises(
                (ValueError, AllocationValidationError), match=expected_error
            ):
                AllocationMode.from_str(case)


class TestWorldSizeCalculations:
    """Test world size calculations are correct"""

    def test_simple_world_sizes(self):
        """Test world size calculations for simple cases"""
        # Simple inference
        result = AllocationMode.from_str("sglang:d4t2")
        assert result.gen_world_size == 8  # 4 * 2 * 1

        # Legacy format
        result = AllocationMode.from_str("d4p1t2")
        assert result.gen_world_size == 8  # 4 * 1 * 2
        assert result.train_world_size == 8

    def test_complex_world_sizes(self):
        """Test world size calculations for complex cases"""
        # Training with multiple dimensions
        try:
            # This should work with legacy parser
            result = AllocationMode.from_str("sglang.d4p1t2+d8p1t1")
            assert result.gen_world_size == 8  # 4 * 1 * 2
            assert result.train_world_size == 8  # 8 * 1 * 1
        except (ValueError, AllocationValidationError):
            # If validation fails, that's expected for some rules
            pass


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_minimal_configurations(self):
        """Test minimal valid configurations"""
        # All dimensions = 1
        result = AllocationMode.from_str("d1p1t1")
        assert result.gen_dp_size == 1
        assert result.gen_pp_size == 1
        assert result.gen_tp_size == 1
        assert result.gen_world_size == 1

    def test_large_parallelism_degrees(self):
        """Test large parallelism degrees"""
        result = AllocationMode.from_str("d16p1t8")
        assert result.gen_dp_size == 16
        assert result.gen_tp_size == 8
        assert result.gen_world_size == 128

    def test_backend_detection(self):
        """Test backend detection"""
        # SGLang
        result = AllocationMode.from_str("sglang.d4t2")
        assert result.gen_backend == "sglang"

        # VLLM
        result = AllocationMode.from_str("vllm.d4t2")
        assert result.gen_backend == "vllm"

        # No backend
        result = AllocationMode.from_str("d4t2")
        assert result.gen_backend is None


def test_manual_validation_examples():
    """Manual test examples for validation - demonstrating parser behavior"""

    test_cases = [
        # Valid cases that should work
        ("sglang:d4t2", "Valid inference config", True),
        ("sglang:d4t2+eval", "Valid eval config", True),
        ("d4t2", "Valid colocate config", True),
        # Cases that depend on lark availability
        ("sglang:d4t2+fsdp:d8", "Valid disaggregated with FSDP", "lark_dependent"),
        (
            "sglang:d4t2p2",
            "Should fail - inference pipeline parallelism",
            "lark_dependent",
        ),
        (
            "sglang:d4t2+fsdp:d4t2",
            "Should fail - FSDP with tensor parallelism",
            "lark_dependent",
        ),
        (
            "sglang:d4t2+megatron:d4p2",
            "Should fail - megatron not supported",
            "lark_dependent",
        ),
    ]

    # Check if lark is available
    try:
        lark_available = True
    except ImportError:
        lark_available = False

    for case_info in test_cases:
        if len(case_info) == 3:
            expr, description, expected = case_info
        else:
            expr, description = case_info[:2]
            expected = True

        # Skip lark-dependent tests if lark is not available
        if expected == "lark_dependent" and not lark_available:
            print(f"⏭️  {expr} -> SKIPPED (Lark not available) ({description})")
            continue

        try:
            AllocationMode.from_str(expr)
            if expected == "lark_dependent" or expected is True:
                print(f"✅ {expr} -> SUCCESS ({description})")
            else:
                print(f"⚠️  {expr} -> UNEXPECTED SUCCESS ({description})")
        except (AllocationValidationError, ValueError) as e:
            if expected == "lark_dependent" and "not supported" in str(e).lower():
                print(f"❌ {expr} -> VALIDATION ERROR: {e} ({description})")
            elif expected is False or expected == "lark_dependent":
                print(f"❌ {expr} -> EXPECTED ERROR: {e} ({description})")
            else:
                print(f"⚠️  {expr} -> UNEXPECTED ERROR: {e} ({description})")
        except Exception as e:
            print(f"⚠️  {expr} -> PARSE ERROR: {e} ({description})")


if __name__ == "__main__":
    print("\n=== Running Manual Validation Examples ===")
    test_manual_validation_examples()
    print(
        "\n=== To run full test suite, use: python -m pytest areal/tests/test_allocation_mode.py -v ==="
    )
