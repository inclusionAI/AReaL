import asyncio
from dataclasses import asdict
import os
import subprocess
import sys
import time
import uuid

import pytest
import requests
import torch
from tensordict import TensorDict

from arealite.api.cli_args import (
    GenerationHyperparameters,
    InferenceEngineConfig,
    SGLangConfig,
)
from arealite.api.io_struct import FinetuneSpec, LLMRequest, LLMResponse
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import name_resolve, network, seeding

EXPR_NAME = "test_sglang_engine"
TRIAL_NAME = "trial_0"
MODEL_PATH = "/storage/testing/models/Qwen__Qwen3-1.7B/"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "Qwen/Qwen2-0.5B"
PORT = 13887
DIST_PORT = 15887
HOST = network.gethostip()

from arealite.engine.sglang_local import LocalSGLangEngine

async def main():
    from realhf.base import seeding
    seeding.set_random_seed(1, EXPR_NAME)
    config = InferenceEngineConfig(experiment_name=EXPR_NAME, trial_name=TRIAL_NAME)
    engine_args = SGLangConfig.build_args(
        sglang_config=SGLangConfig(mem_fraction_static=0.3),
        model_path=MODEL_PATH,
        tp_size=1,
        base_gpu_id=0,
        dist_init_addr=f"{HOST}:{DIST_PORT}",
        served_model_name=MODEL_PATH,
        skip_tokenizer_init=False,
    )

    engine = LocalSGLangEngine(config, engine_args=engine_args)
    engine.initialize(None, None)

    print("SGLang Local Engine initialized", flush=True)
    req = LLMRequest(
        rid=str(uuid.uuid4()),
        text="hello! how are you today",
        gconfig=GenerationHyperparameters(max_new_tokens=16),
    )
    resp = await engine.agenerate(req)
    assert isinstance(resp, LLMResponse)
    assert resp.input_tokens == req.input_ids
    assert (
        len(resp.output_logprobs)
        == len(resp.output_tokens)
        == len(resp.output_versions)
    )
    assert isinstance(resp.completions, str)
    time.sleep(15)
    engine.destroy()

if __name__ == "__main__":
    asyncio.run(main())
