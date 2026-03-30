# Copyright 2025 HuggingFace Inc., the KVCache.AI team, Approaching AI, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any

import torch
from transformers import BitsAndBytesConfig

from ...accelerator.helper import get_current_device
from ...config.model_args import ModelArguments
from ...utils import logging
from ...utils.packages import check_version
from ...utils.plugin import BasePlugin


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer

logger = logging.get_logger(__name__)


class QuantizationPlugin(BasePlugin):
    r"""Plugin for model quantization."""

    def __call__(
        self,
        init_kwargs: dict[str, Any] = None,
        config: "PretrainedConfig" = None,
        tokenizer: "PreTrainedTokenizer" = None,
        model_args: "ModelArguments" = None,
        is_trainable: bool = False,
    ) -> dict[str, Any]:
        return super().__call__(
            init_kwargs, config=config, tokenizer=tokenizer, model_args=model_args, is_trainable=is_trainable
        )


@QuantizationPlugin("auto").register()
def quantization_auto(
    init_kwargs: dict[str, Any],
    **kwargs,
) -> dict[str, Any]:
    """Automatic quantization selection, only support bnb currently.

    Args:
        init_kwargs (dict[str, Any]): The kwargs for model initialization.
        **kwargs: Keyword arguments containing the model.

    Returns:
        dict[str, Any]: The updated kwargs for model initialization.
    """
    model_args: ModelArguments = kwargs.get("model_args", None)
    quant_config = model_args.quant_config

    quantization_bit = quant_config.get("quantization_bit", None)
    if quantization_bit is not None:
        logger.info_rank0(f"Loading {quantization_bit}-bit quantized model.")
        if quantization_bit in [8, 4]:
            return quantization_with_bnb(init_kwargs, **kwargs)
        else:
            raise ValueError(f"Unsupported quantization bit: {quantization_bit} for auto quantization.")
    logger.warning_rank0("No quantization method applied.")
    return init_kwargs


@QuantizationPlugin("bnb").register()
def quantization_with_bnb(
    init_kwargs: dict[str, Any],
    model_args: "ModelArguments" = None,
    **kwargs,
) -> dict[str, Any]:
    r"""Quantization with BNB."""
    logger.info_rank0("Using Bitsandbytes quantization.")
    quantization_bit = model_args.quant_config.get("quantization_bit", None)
    if quantization_bit is None:
        logger.warning_rank0("quantization_bit is not specified, default to 8-bit quantization.")
        quantization_bit = 4
    assert quantization_bit in [8, 4], "Bitsandbytes only accepts 4-bit or 8-bit quantization."
    if quantization_bit == 8:
        check_version("bitsandbytes>=0.37.0", mandatory=True)
        init_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization_bit == 4:
        check_version("bitsandbytes>=0.39.0", mandatory=True)
        init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_args.quant_config.get("compute_dtype", torch.float16),
            bnb_4bit_use_double_quant=model_args.quant_config.get("double_quantization", True),
            bnb_4bit_quant_type=model_args.quant_config.get("quantization_type", "nf4"),
            bnb_4bit_quant_storage=model_args.quant_config.get(
                "compute_dtype", torch.float16
            ),  # crucial for fsdp+qlora
        )
    else:
        raise ValueError("Bitsandbytes only accepts 4-bit or 8-bit quantization.")

    # TODO: improve deepspeed zero3 and fsdp detection.
    if kwargs.get("is_trainable", False):
        logger.info_rank0("Detected inference mode, setting device_map for bitsandbytes quantization.")
        init_kwargs["device_map"] = {"": get_current_device()}  # change auto device map for inference
    else:
        logger.info_rank0("Detected training mode, skip setting device_map for bitsandbytes quantization.")
        if model_args.quant_config.get("quantization_bit") != 4:
            raise ValueError("Only 4-bit quantized model can use fsdp+qlora or auto device map.")

        check_version("bitsandbytes>=0.43.0", mandatory=True)

    logger.info_rank0(f"Quantizing model to {model_args.quant_config.get('quantization_bit')} bit with bitsandbytes.")
    return init_kwargs
