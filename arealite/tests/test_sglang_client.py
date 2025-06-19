import uuid

from omegaconf import OmegaConf

from arealite.api.cli_args import (
    GenerationHyperparameters,
    LLMClientConfig,
    TrainingArgs,
)
from arealite.api.io_struct import LLMRequest, LLMResponse
from arealite.impl.sglang_client import SGLangClient
from realhf.base import name_resolve

args = TrainingArgs(experiment_name="test_rollout", trial_name="test_rollout")
name_resolve.reconfigure(args.cluster.name_resolve)
args.rollout.llm_client = LLMClientConfig(
    server_backend="sglang",
    tokenizer_path="Qwen/Qwen2-0.5B",
    request_timeout=10,
)
client = SGLangClient(args, client_config=args.rollout.llm_client)
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
