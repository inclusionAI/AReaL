# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import functools
import os
import re

import torch

# Import all HuggingFace model implementations.
import realhf.api.from_hf
import realhf.base.logging as logging
from realhf.api.core.model_api import HF_MODEL_FAMILY_REGISTRY
from realhf.base.importing import import_module
from realhf.base.pkg_version import is_available, is_version_less
from realhf.impl.model.conversion.hf_registry import HFModelRegistry
from realhf.impl.model.nn.real_llm_api import ReaLModel

logger = logging.getLogger("model init")

# Import all model implementations.
_p = re.compile(r"^(?!.*__init__).*\.py$")
_filepath = os.path.dirname(__file__)
import_module(os.path.join(_filepath, "interface"), _p)
import_module(os.path.join(_filepath, "nn"), _p)

# NOTE: skip importing vLLM for now to avoid an
# "invalid device context" issue for the 25.01 image
if is_available("vllm") and is_version_less("vllm", "0.6.4"):
    import realhf.impl.model.backend.vllm

import realhf.impl.model.backend.inference
import realhf.impl.model.backend.megatron
import realhf.impl.model.backend.mock_train
import realhf.impl.model.backend.sglang

# Set PyTorch JIT options, following Megatron-LM.
if torch.cuda.is_available():
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(True)
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    # torch._C._jit_set_nvfuser_enabled(True)  # disable the deprecated warning
    torch._C._debug_set_autodiff_subgraph_inlining(False)

# Add HuggingFace hooks to ReaLModel.
_HF_REGISTRIES = {}


def _load_from_hf(
    model: ReaLModel, registry_name, load_dir: str, init_critic_from_actor: bool
):
    r = _HF_REGISTRIES[registry_name]
    return r.load(model, load_dir, init_critic_from_actor)


def _save_to_hf(model: ReaLModel, registry_name, tokenizer, save_dir: str):
    r = _HF_REGISTRIES[registry_name]
    r.save(model, tokenizer, save_dir)


def _config_from_hf(registry_name, hf_config=None, model_path=None, is_critic=False):
    r = _HF_REGISTRIES[registry_name]
    return r.config_from_hf(hf_config, model_path, is_critic)


def _config_to_hf(registry_name, config):
    r = _HF_REGISTRIES[registry_name]
    return r.config_to_hf(config)


def _make_real_config(registry_name):
    r = _HF_REGISTRIES[registry_name]
    if r.real_config_maker is not None:
        return r.real_config_maker()
    raise NotImplementedError(
        f"`real_config_maker` not implemented for {registry_name}. "
        f"Please implement and register `real_config_maker` "
        f"in realhf.api.from_hf.{registry_name} to make customized ReaLModelConfig."
    )


for name, helpers in HF_MODEL_FAMILY_REGISTRY.items():
    _HF_REGISTRIES[name] = r = HFModelRegistry(**helpers)

    _load_from_hf_ = functools.partialmethod(_load_from_hf, name)
    setattr(ReaLModel, f"from_{name}", _load_from_hf_)

    _save_to_hf_ = functools.partialmethod(_save_to_hf, name)
    setattr(ReaLModel, f"to_{name}", _save_to_hf_)

    _config_from_hf_ = functools.partial(_config_from_hf, name)
    setattr(ReaLModel, f"config_from_{name}", staticmethod(_config_from_hf_))

    _config_to_hf_ = functools.partial(_config_to_hf, name)
    setattr(ReaLModel, f"config_to_{name}", staticmethod(_config_to_hf_))

    # make a ReaLModelConfig from only parameters related to model size, used for testing
    _make_real_config_ = functools.partial(_make_real_config, name)
    setattr(ReaLModel, f"make_{name}_config", staticmethod(_make_real_config_))
