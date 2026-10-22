# SPDX-License-Identifier: Apache-2.0

"""Pipeline partition compatibility for models with restricted layouts."""

from types import SimpleNamespace

import pytest

pytest.importorskip("megatron.core")

from areal.engine.megatron_utils.pipeline_parallel import (
    configure_pipeline_layer_splits,
)
from areal.models.mcore.glm5_next_config import validate_glm5_next_runtime


def make_transformer_config():
    return SimpleNamespace(
        num_layers=45,
        hidden_size=32,
        ffn_hidden_size=96,
        num_attention_heads=4,
        context_parallel_size=1,
        pipeline_model_parallel_layout=None,
    )


@pytest.mark.parametrize("model_type", ["glm5_next", "glm5_next_text"])
def test_glm53_pp3_preserves_standard_layout(model_type):
    config = make_transformer_config()
    result = configure_pipeline_layer_splits(
        SimpleNamespace(pipeline_parallel_size=3, virtual_pipeline_parallel_size=1),
        SimpleNamespace(model_type=model_type),
        config,
    )
    assert result is config
    assert result.pipeline_model_parallel_layout is None
    validate_glm5_next_runtime(result)


def test_glm53_explicit_unsupported_layout_is_not_silently_removed():
    config = make_transformer_config()
    config.pipeline_model_parallel_layout = "Et*15|t*15|t*15L"
    result = configure_pipeline_layer_splits(
        SimpleNamespace(pipeline_parallel_size=3, virtual_pipeline_parallel_size=1),
        SimpleNamespace(model_type="glm5_next"),
        config,
    )
    with pytest.raises(ValueError, match="standard pipeline layer layout"):
        validate_glm5_next_runtime(result)


def test_other_models_still_receive_automatic_pipeline_layout():
    result = configure_pipeline_layer_splits(
        SimpleNamespace(pipeline_parallel_size=3, virtual_pipeline_parallel_size=1),
        SimpleNamespace(model_type="qwen3", vocab_size=128),
        make_transformer_config(),
    )
    assert result.pipeline_model_parallel_layout is not None
