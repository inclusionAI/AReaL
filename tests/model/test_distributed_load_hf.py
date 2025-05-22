# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import json
import os
import pathlib
import shutil
import uuid
from typing import *

import pytest
import torch
import torch.distributed as dist
import transformers

from realhf.api.core.model_api import ReaLModelConfig
from realhf.base import constants, logging
from realhf.base.testing import (
    LocalMultiProcessTest,
    clear_name_resolve,
    init_global_constants,
)

logger = logging.getLogger("tests.test_saveload")


def _load_all_pytorch_bin(path: pathlib.Path):
    if os.path.exists(path / "pytorch_model.bin.index.json"):
        with open(path / "pytorch_model.bin.index.json", "r") as f:
            hf_sd_mapping = json.load(f)["weight_map"]
        sd = {}
        for fn in hf_sd_mapping.values():
            sd.update(torch.load(path / fn, map_location="cpu"))
    else:
        sd = torch.load(path / "pytorch_model.bin", map_location="cpu")
    return sd


def _save_then_load(
    tmp_path: pathlib.Path,
    model_family_name: str,
    is_critic: bool,
    init_critic_from_actor: bool,
    pp_dp_tp: Tuple,
    device: torch.device,
):
    # NOTE: import here to avoid initializing CUDA context in the main process
    from realhf.impl.model.nn.real_llm_api import ReaLModel

    # os.environ["REAL_SAVE_MAX_SHARD_SIZE_BYTE"] = str(int(1e6))

    model_name = f"saveload_test_{model_family_name}"
    num_pp, num_dp, num_tp = pp_dp_tp
    init_global_constants(
        num_dp=num_dp,
        num_tp=num_tp,
        num_pp=num_pp,
        model_name=model_name,
    )
    assert dist.get_world_size() == 8, dist.get_world_size()
    assert tmp_path.exists()
    init_save_path = tmp_path / "init"
    real_save_path = tmp_path / "real"
    real_save_path2 = tmp_path / "real2"

    with constants.model_scope(model_name):
        tokenizer = None
        mconfig: ReaLModelConfig = getattr(
            ReaLModel, f"make_{model_family_name}_config"
        )()
        mconfig.is_critic = is_critic
        if mconfig.n_kv_heads % num_tp != 0:
            return

        # load from hf model or create a new critic model
        model = ReaLModel(mconfig, dtype=torch.float32, device=device)
        model.instantiate()
        # sync initialized parameters
        getattr(model, f"to_{model_family_name}")(tokenizer, init_save_path)
        dist.barrier()
        model = getattr(model, f"from_{model_family_name}")(
            init_save_path, init_critic_from_actor=False
        )
        sd1 = model.state_dict()

        # save ReaLModel (e.g., after SFT)
        getattr(model, f"to_{model_family_name}")(tokenizer, real_save_path)
        dist.barrier()
        file_size = 0
        for fn in os.listdir(real_save_path):
            if fn.endswith(".bin"):
                file_size += os.path.getsize(os.path.join(real_save_path, fn))

        # load ReaLModel (e.g., before PPO, RW)
        model = ReaLModel(mconfig, dtype=torch.float32, device=device)
        model._instantiation_hooks.append(
            lambda: getattr(model, f"from_{model_family_name}")(
                real_save_path, init_critic_from_actor
            )
        )
        model.instantiate()
        dist.barrier()
        sd2 = model.state_dict()
        for k, v in sd2.items():
            if init_critic_from_actor and k == f"{mconfig.n_layers + 1}.weight":
                continue
            assert torch.allclose(v, sd1[k]), (
                k,
                v.flatten()[:10],
                sd1[k].flatten()[:10],
            )

        # Load saved ReaLModel using HF APIs.
        if not is_critic:
            hf_model = transformers.AutoModelForCausalLM.from_pretrained(
                real_save_path,
                trust_remote_code=True,
                force_download=True,
            )
            dist.barrier()
            _hf_sd = hf_model.state_dict()
            sd3 = _load_all_pytorch_bin(real_save_path)
            if model_family_name != "gpt2":
                for k, v in sd3.items():
                    if k.endswith(".rotary_emb.inv_freq"):
                        continue
                    assert torch.allclose(v.cpu(), _hf_sd[k]), k
            else:
                for k, v in sd3.items():
                    if k.endswith(".attn.bias"):
                        continue
                    assert torch.allclose(v.cpu(), _hf_sd[f"transformer.{k}"]), k

        # save again, check size
        getattr(model, f"to_{model_family_name}")(tokenizer, real_save_path2)
        dist.barrier()
        file_size2 = 0
        for fn in os.listdir(real_save_path2):
            if fn.endswith(".bin"):
                file_size2 += os.path.getsize(os.path.join(real_save_path2, fn))
        assert file_size2 == file_size, (file_size, file_size2)
        dist.barrier()


@pytest.mark.parametrize(
    "model_family_name",
    ["gemma", "gpt2", "llama", "qwen2", "mistral", "mixtral", "qwen3"],
)
@pytest.mark.parametrize("is_critic", [True, False])
@pytest.mark.parametrize("init_critic_from_actor", [True, False])
@pytest.mark.parametrize("pp_dp_tp", [(4, 2, 1), (2, 2, 2), (1, 2, 4), (1, 8, 1)])
@pytest.mark.distributed
def test_save_then_load(
    tmp_path: pathlib.Path,
    model_family_name: str,
    is_critic: bool,
    init_critic_from_actor: bool,
    pp_dp_tp: Tuple,
):
    if model_family_name == "gpt2" and pp_dp_tp[-1] > 1:
        # GPT-2 has an odd vocabulary size, so it doesn't work
        # with tensor-model parallelism.
        return
    if not is_critic and init_critic_from_actor:
        return
    expr_name = uuid.uuid4()
    trial_name = uuid.uuid4()
    test_impl = LocalMultiProcessTest(
        world_size=8,
        func=_save_then_load,
        expr_name=expr_name,
        trial_name=trial_name,
        dist_backend="gloo",
        model_family_name=model_family_name,
        is_critic=is_critic,
        init_critic_from_actor=init_critic_from_actor,
        pp_dp_tp=pp_dp_tp,
        tmp_path=tmp_path,
        device="cpu",
    )
    test_impl.launch()
