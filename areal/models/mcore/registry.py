import dataclasses

import torch
from megatron.core import tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig as MCoreDDPConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import TransformerConfig
from transformers import AutoConfig, PretrainedConfig

from areal.api.cli_args import MegatronEngineConfig
from areal.models.mcore.qwen3 import (
    hf_to_mcore_config_qwen3_dense,
    make_mcore_layer_specs_qwen3_dense,
)
from areal.utils import logging

logger = logging.getLogger("MCoreRegistry")


class ValueHead(torch.nn.Linear):
    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        *,
        config: TransformerConfig,
        bias: bool = False,
    ) -> None:
        super().__init__(in_features=input_size, out_features=output_size, bias=bias)
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel:
            self.weight.sequence_parallel = True

        self.weight.data.normal_(mean=0.0, std=0.02)
        if bias:
            self.bias.data.zero_()

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        logits = super().forward(input_)
        logits = logits.float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(
                logits, tensor_parallel_output_grad=False
            )
        return logits, None


def _replace_output_layer_with_value_head(
    model: GPTModel,
    tf_config: TransformerConfig,
) -> None:
    """Replace model's output_layer with ValueHead.

    This function can be used on any GPTModel instance, whether created
    via mbridge or directly. After replacement:
    - model.output_layer becomes a ValueHead instance
    - model.vocab_size is set to 1

    Args:
        model: The GPTModel instance to modify
        tf_config: Transformer configuration containing hidden_size and SP settings
    """
    if not hasattr(model, "output_layer"):
        raise ValueError(
            "Model does not have output_layer. Ensure post_process=True when creating GPTModel."
        )

    dtype = tf_config.params_dtype

    model.output_layer = ValueHead(
        input_size=tf_config.hidden_size,
        output_size=1,
        config=tf_config,
        bias=False,
    ).to(dtype=dtype)

    model.vocab_size = 1


def unwrap_to_gpt_model(model: torch.nn.Module) -> GPTModel:
    """Unwraps a model to the underlying GPTModel instance."""
    _model = model
    while not isinstance(_model, GPTModel) and hasattr(_model, "module"):
        _model = _model.module
    if not isinstance(_model, GPTModel):
        raise TypeError(f"Model could not be unwrapped to GPTModel. Got {type(_model)}")
    return _model


# Model registry for different architectures
def make_hf_and_mcore_config(
    hf_path: str, dtype: torch.dtype, bridge=None
) -> tuple[PretrainedConfig, TransformerConfig]:
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
    mcore_config: MegatronEngineConfig | None = None,
    bridge=None,
    is_critic: bool = False,
) -> list[GPTModel | DDP]:
    if bridge is not None:
        models = bridge.get_model(
            # TODO: Add DDP options when supporting training
            wrap_with_ddp=mcore_config.wrap_with_ddp,
            ddp_config=dataclasses.asdict(mcore_config.ddp),
            use_torch_fsdp2=mcore_config.use_torch_fsdp2,
            use_custom_fsdp=mcore_config.use_custom_fsdp,
            fp16=tf_config.fp16,
            bf16=tf_config.bf16,
            use_precision_aware_optimizer=mcore_config.use_precision_aware_optimizer,
            overlap_param_gather_with_optimizer_step=mcore_config.overlap_param_gather_with_optimizer_step,
        )
        models = list(models)

        # Replace output_layer with ValueHead for critic models
        if is_critic:
            for model in models:
                _model = unwrap_to_gpt_model(model)
                _replace_output_layer_with_value_head(_model, tf_config)

        return models
    else:
        if (
            mcore_config is not None
            and mcore_config.virtual_pipeline_parallel_size is not None
            and mcore_config.virtual_pipeline_parallel_size > 1
        ):
            raise NotImplementedError(
                "Virtual pipeline parallelism requires mbridge-backed models."
            )
        transformer_layer_spec = make_mcore_layer_specs(hf_config, tf_config)
        rope_scaling_args = {}
        if hf_config.rope_scaling is not None:
            if hf_config.rope_scaling["type"] != "linear":
                raise NotImplementedError(
                    f"Rope scaling type {hf_config.rope_scaling['type']} not supported yet."
                )
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

        # Replace output_layer with ValueHead for critic models
        if is_critic:
            _replace_output_layer_with_value_head(model, tf_config)

        if mcore_config.wrap_with_ddp:
            ddp_config = MCoreDDPConfig(**dataclasses.asdict(mcore_config.ddp))
            wrapped = DDP(
                config=tf_config,
                ddp_config=ddp_config,
                module=model,
                disable_bucketing=False,
            )
            return [wrapped]
        return [model]
