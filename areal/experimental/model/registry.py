import dataclasses
from typing import Optional, Tuple

import torch
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig as MCoreDDPConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import TransformerConfig
from transformers import AutoConfig, PretrainedConfig

from areal.experimental.api.cli_args import MegatronEngineConfig
from areal.experimental.model.qwen3 import (
    hf_to_mcore_config_qwen3_dense,
    make_mcore_layer_specs_qwen3_dense,
)
from areal.utils import logging

logger = logging.getLogger("areal.experimental.model.registry")


# Model registry for different architectures
def make_hf_and_mcore_config(
    hf_path: str, dtype: torch.dtype, bridge=None
) -> Tuple[PretrainedConfig, TransformerConfig]:
    if bridge is not None:
        hf_config = bridge.hf_config
        hf_config._name_or_path = hf_path
        return hf_config, bridge.config
    else:
        hf_config: PretrainedConfig = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=hf_path,
            trust_remote_code=True,
        )
        assert len(hf_config.architectures) == 1
        architecture = hf_config.architectures[0]
        if architecture == "Qwen3ForCausalLM":
            return hf_config, hf_to_mcore_config_qwen3_dense(hf_config, dtype)
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
    hf_config: PretrainedConfig,
    tf_config: TransformerConfig,
    mcore_config: Optional[MegatronEngineConfig] = None,
    bridge=None,
) -> GPTModel:
    if bridge is not None:
        model = bridge.get_model(
            # TODO: Add DDP options when supporting training
            wrap_with_ddp=mcore_config.wrap_with_ddp,
            ddp_config=dataclasses.asdict(mcore_config.ddp),
            use_torch_fsdp2=mcore_config.use_torch_fsdp2,
            use_custom_fsdp=mcore_config.use_custom_fsdp,
            fp16=tf_config.fp16,
            bf16=tf_config.bf16,
            use_precision_aware_optimizer=mcore_config.use_precision_aware_optimizer,
            overlap_param_gather_with_optimizer_step=mcore_config.overlap_param_gather_with_optimizer_step,
        )[0]
        # TODO: when supporting VPP, model should be a list
        return model
    else:
        transformer_layer_spec = make_mcore_layer_specs(hf_config, tf_config)
        rope_scaling_args = {}
        if hf_config.rope_scaling is not None:
            assert (
                hf_config.rope_scaling["type"] == "linear"
            ), "only linear scaling is supported for now"
            rope_scaling_args["seq_len_interpolation_factor"] = hf_config.rope_scaling[
                "factor"
            ]

        model = GPTModel(
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
        ddp_config = MCoreDDPConfig(**dataclasses.asdict(mcore_config.ddp))
        return DDP(
            config=tf_config,
            ddp_config=ddp_config,
            module=model,
            disable_bucketing=False,
        )
