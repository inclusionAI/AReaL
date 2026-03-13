"""Tests for Megatron parameter name conversion and regex patterns.

Verifies that pre-compiled regex patterns produce identical results to
the original inline patterns, and that conversion functions handle
all supported parameter name variants correctly.
"""

import re

import pytest

from areal.engine.megatron_utils.megatron import (
    _RE_DECODER_LAYERS,
    _RE_EXPERT_WEIGHT,
    _RE_MTP_LAYERS,
    _RE_SHARED_EXPERT,
    _RE_TE_EXPERT_WEIGHT,
)


class TestPreCompiledRegexPatterns:
    """Ensure pre-compiled patterns match the original inline patterns exactly."""

    @pytest.mark.parametrize(
        "name,expected_layer,expected_rest",
        [
            (
                "module.module.decoder.layers.0.self_attention.linear_proj.weight",
                "0",
                "self_attention.linear_proj.weight",
            ),
            (
                "module.module.decoder.layers.31.mlp.linear_fc1.weight",
                "31",
                "mlp.linear_fc1.weight",
            ),
            (
                "module.module.decoder.layers.127.mlp.experts.linear_fc1.weight3",
                "127",
                "mlp.experts.linear_fc1.weight3",
            ),
        ],
    )
    def test_decoder_layers_pattern(self, name, expected_layer, expected_rest):
        match = _RE_DECODER_LAYERS.match(name)
        assert match is not None
        layer_idx, rest = match.groups()
        assert layer_idx == expected_layer
        assert rest == expected_rest

    @pytest.mark.parametrize(
        "name",
        [
            "module.module.embedding.word_embeddings.weight",
            "module.module.output_layer.weight",
            "module.module.decoder.final_layernorm.weight",
        ],
    )
    def test_decoder_layers_pattern_no_match(self, name):
        assert _RE_DECODER_LAYERS.match(name) is None

    @pytest.mark.parametrize(
        "rest,expected_type,expected_idx",
        [
            ("mlp.experts.linear_fc1.weight0", "linear_fc1", "0"),
            ("mlp.experts.linear_fc2.weight7", "linear_fc2", "7"),
            ("mlp.experts.linear_fc1.weight15", "linear_fc1", "15"),
        ],
    )
    def test_expert_weight_pattern(self, rest, expected_type, expected_idx):
        match = _RE_EXPERT_WEIGHT.match(rest)
        assert match is not None
        expert_type, expert_idx = match.groups()
        assert expert_type == expected_type
        assert expert_idx == expected_idx

    @pytest.mark.parametrize(
        "rest,expected_suffix",
        [
            ("mlp.shared_experts.linear_fc1.weight", "linear_fc1.weight"),
            ("mlp.shared_experts.linear_fc2.weight", "linear_fc2.weight"),
        ],
    )
    def test_shared_expert_pattern(self, rest, expected_suffix):
        match = _RE_SHARED_EXPERT.match(rest)
        assert match is not None
        assert match.groups()[0] == expected_suffix

    @pytest.mark.parametrize(
        "name,expected_layer",
        [
            ("module.module.mtp.layers.0.transformer_layer.mlp.linear_fc1.weight", "0"),
            ("module.module.mtp.layers.3.self_attention.linear_proj.weight", "3"),
        ],
    )
    def test_mtp_layers_pattern(self, name, expected_layer):
        match = _RE_MTP_LAYERS.match(name)
        assert match is not None
        layer_idx, _ = match.groups()
        assert layer_idx == expected_layer

    @pytest.mark.parametrize(
        "rest,expected_type,expected_idx",
        [
            (
                "transformer_layer.mlp.experts.linear_fc1.weight0",
                "linear_fc1",
                "0",
            ),
            (
                "transformer_layer.mlp.experts.linear_fc2.weight3",
                "linear_fc2",
                "3",
            ),
        ],
    )
    def test_te_expert_weight_pattern(self, rest, expected_type, expected_idx):
        match = _RE_TE_EXPERT_WEIGHT.match(rest)
        assert match is not None
        expert_type, expert_idx = match.groups()
        assert expert_type == expected_type
        assert expert_idx == expected_idx


class TestRegexConsistency:
    """Verify pre-compiled patterns produce same results as original inline patterns."""

    ORIGINAL_DECODER = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    ORIGINAL_EXPERT = r"mlp.experts\.(.+)\.weight(\d+)"
    ORIGINAL_MTP = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
    ORIGINAL_TE_EXPERT = r"transformer_layer.mlp.experts\.(.+)\.weight(\d+)"
    ORIGINAL_SHARED = r"mlp.shared_experts\.(.+)"

    TEST_NAMES = [
        "module.module.decoder.layers.0.self_attention.linear_qkv.weight",
        "module.module.decoder.layers.31.mlp.linear_fc1.weight",
        "module.module.decoder.layers.5.mlp.experts.linear_fc1.weight3",
        "module.module.embedding.word_embeddings.weight",
        "module.module.mtp.layers.0.transformer_layer.mlp.experts.linear_fc1.weight0",
    ]

    EXPERT_REST_NAMES = [
        "mlp.experts.linear_fc1.weight0",
        "mlp.experts.linear_fc2.weight7",
        "mlp.shared_experts.linear_fc1.weight",
        "self_attention.linear_proj.weight",
    ]

    def test_decoder_layers_consistency(self):
        for name in self.TEST_NAMES:
            original = re.match(self.ORIGINAL_DECODER, name)
            compiled = _RE_DECODER_LAYERS.match(name)
            if original is None:
                assert compiled is None, f"Mismatch for {name}"
            else:
                assert compiled is not None, f"Mismatch for {name}"
                assert original.groups() == compiled.groups(), f"Groups differ for {name}"

    def test_expert_weight_consistency(self):
        for rest in self.EXPERT_REST_NAMES:
            original = re.match(self.ORIGINAL_EXPERT, rest)
            compiled = _RE_EXPERT_WEIGHT.match(rest)
            if original is None:
                assert compiled is None, f"Mismatch for {rest}"
            else:
                assert compiled is not None, f"Mismatch for {rest}"
                assert original.groups() == compiled.groups(), f"Groups differ for {rest}"

    def test_shared_expert_consistency(self):
        for rest in self.EXPERT_REST_NAMES:
            original = re.match(self.ORIGINAL_SHARED, rest)
            compiled = _RE_SHARED_EXPERT.match(rest)
            if original is None:
                assert compiled is None, f"Mismatch for {rest}"
            else:
                assert compiled is not None, f"Mismatch for {rest}"
                assert original.groups() == compiled.groups(), f"Groups differ for {rest}"
