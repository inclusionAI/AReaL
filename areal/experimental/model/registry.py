import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import TransformerConfig
from transformers import PretrainedConfig

from areal.experimental.model.qwen3 import (
    hf_to_mcore_config_qwen3_dense,
    make_mcore_layer_specs_qwen3_dense,
)


# Model registry for different architectures
def hf_to_mcore_config(
    hf_config: PretrainedConfig, dtype: torch.dtype
) -> TransformerConfig:
    assert len(hf_config.architectures) == 1
    architecture = hf_config.architectures[0]
    if architecture == "Qwen3ForCausalLM":
        return hf_to_mcore_config_qwen3_dense(hf_config, dtype)
    else:
        raise ValueError(
            f"Architecture not registered for config conversion: {architecture}."
        )


def make_mcore_layer_specs(hf_config: PretrainedConfig, tf_config: TransformerConfig):
    assert len(hf_config.architectures) == 1
    architecture = hf_config.architectures[0]
    if architecture == "Qwen3ForCausalLM":
        return make_mcore_layer_specs_qwen3_dense(tf_config, use_te=True)
    else:
        raise ValueError(
            f"Architecture not registered for config conversion: {architecture}."
        )


def make_mcore_model(
    hf_config: PretrainedConfig, tf_config: TransformerConfig
) -> GPTModel:
    transformer_layer_spec = make_mcore_layer_specs(hf_config, tf_config)
    rope_scaling_args = {}
    if hf_config.rope_scaling is not None:
        assert (
            hf_config.rope_scaling["type"] == "linear"
        ), "only linear scaling is supported for now"
        rope_scaling_args["seq_len_interpolation_factor"] = hf_config.rope_scaling[
            "factor"
        ]

    return GPTModel(
        config=tf_config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=hf_config.vocab_size,
        max_sequence_length=hf_config.max_position_embeddings,
        pre_process=True,  # TODO: pipeline parallel
        post_process=True,  # TODO: pipeline parallel
        share_embeddings_and_output_weights=False,  # TODO: implement share output weights
        position_embedding_type="rope",
        rotary_base=hf_config.rope_theta,
        **rope_scaling_args,
        # vp_stage=None TODO: virtual pipeline parallel
    )
