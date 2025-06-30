# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import os
import uuid

import pytest

from arealite.api.cli_args import (
    EngineBackendConfig,
    EngineConfig,
    GenerationHyperparameters,
    LLMClientConfig,
    LLMServiceConfig,
    OptimizerConfig,
    SGLangConfig,
    TrainingArgs,
)
from arealite.api.engine_api import EngineFactory
from arealite.api.io_struct import FinetuneSpec, LLMRequest, LLMResponse
from arealite.api.llm_client_api import LLMClient
from arealite.api.llm_server_api import LLMServerFactory
from realhf.base import constants, name_resolve, seeding

EXPR_NAME = "test_sglang_client"
TRIAL_NAME = "test_sglang_client"
MODEL_PATH = "Qwen/Qwen2-0.5B"


@pytest.fixture(scope="module")
def args():
    args = TrainingArgs(experiment_name=EXPR_NAME, trial_name=TRIAL_NAME)
    seeding.set_random_seed(args.seed, EXPR_NAME)
    name_resolve.reconfigure(args.cluster.name_resolve)
    yield args
    name_resolve.reset()


@pytest.fixture(scope="module")
def sglang_server(args):
    server_args = LLMServiceConfig(model_path=MODEL_PATH)
    server_args.sglang = SGLangConfig(mem_fraction_static=0.3)
    server = LLMServerFactory(args).make_server(server_args)
    server._startup()
    yield
    server._graceful_exit(0)


@pytest.fixture(scope="module")
def sglang_client(args, sglang_server):
    from arealite.system.sglang_client import SGLangClient

    args.rollout.server_backend = "sglang"
    args.rollout.model_path = MODEL_PATH
    llm_client = LLMClientConfig()
    client = SGLangClient(args, client_config=llm_client)
    yield client


@pytest.mark.asyncio
async def test_sglang_generate(sglang_client):
    req = LLMRequest(
        rid=str(uuid.uuid4()),
        text="hello! how are you today",
        gconfig=GenerationHyperparameters(max_new_tokens=16),
    )
    resp = await sglang_client.agenerate(req)
    assert isinstance(resp, LLMResponse)
    assert resp.input_tokens == req.input_ids
    assert (
        len(resp.output_logprobs)
        == len(resp.output_tokens)
        == len(resp.output_versions)
    )
    assert isinstance(resp.completion, str)


@pytest.mark.asyncio
async def test_sglang_update_weights_from_disk(sglang_client: LLMClient):
    servers = sglang_client.get_healthy_servers()
    assert len(servers) == 1
    await sglang_client.aupdate_weights_from_disk(
        server_info=servers[0], path=MODEL_PATH
    )


@pytest.fixture(scope="module")
def engine(sglang_server):
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "7777"
    engine_config = EngineConfig(
        path=MODEL_PATH,
        gradient_checkpointing=False,
        optimizer=OptimizerConfig(),
        backend=EngineBackendConfig(type="fsdp"),
    )

    mock_args = TrainingArgs(n_nodes=1, n_gpus_per_node=1)

    engine_factory = EngineFactory(mock_args)
    engine = engine_factory.make_engine(engine_config)
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=100, train_batch_size=2)
    engine.init_distributed(None, ft_spec)
    print("✓ Engine created successfully")
    yield engine


def test_sglang_update_weights_from_distributed(
    engine, sglang_server, sglang_client: LLMClient
):
    engine.update_weights_to(sglang_client)
