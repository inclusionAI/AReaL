import uuid

from omegaconf import OmegaConf

from arealite.api.cli_args import (
    GenerationHyperparameters,
    LLMClientConfig,
    TrainingArgs,
)
from arealite.api.io_struct import LLMRequest, LLMResponse
from realhf.base import name_resolve

args = OmegaConf.load("arealite/config/async_ppo.yaml")
default_args = OmegaConf.structured(TrainingArgs)
args = OmegaConf.merge(default_args, args)
args: TrainingArgs = OmegaConf.to_object(args)
name_resolve.reconfigure(args.cluster.name_resolve)

from arealite.impl.sglang_client import SGLangClient

client_cfg = LLMClientConfig(
    server_backend="sglang",
    tokenizer_path="/storage/testing/models/Qwen__Qwen3-1.7B/",
    gen_timeout=1800,
)
client = SGLangClient(args, client_config=client_cfg)
req = LLMRequest(
    rid=str(uuid.uuid4()),
    text="hello! how are you today",
    gconfig=GenerationHyperparameters(max_new_tokens=16),
)
resp = client.generate(req)
assert isinstance(resp, LLMResponse)
assert resp.input_tokens == req.input_ids
assert len(resp.output_logprobs) == len(resp.output_tokens) == len(resp.output_versions)
assert isinstance(resp.completion, str)
