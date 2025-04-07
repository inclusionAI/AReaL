# Copyright 2025 Ant Group Inc.

import asyncio
import dataclasses
import json
import os
import sys
import time
import traceback
from typing import Dict, List, Tuple

import aiohttp
import requests
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import transformers
from tqdm.asyncio import tqdm

from realhf.api.cli_args import SGLangConfig
from realhf.api.core import data_api
from realhf.api.core.model_api import (
    APIGenerateInput,
    APIGenerateOutput,
    FinetuneSpec,
    GenerationHyperparameters,
    LLMAPIClient,
    Model,
    ModelBackend,
    PipelinableEngine,
    register_backend,
)
from realhf.base import (
    cluster,
    constants,
    gpu_utils,
    logging,
    network,
    pkg_version,
    seeding,
)

logger = logging.getLogger("SGLang backend")

SGLANG_INIT_TIMEOUT = 300


def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


if pkg_version.is_available("sglang"):
    if pkg_version.is_version_greater_or_equal("sglang", "0.4.4"):
        SGLANG_TOKEN_OUTPUT_IDENTIFIER = "output_ids"
    else:
        SGLANG_TOKEN_OUTPUT_IDENTIFIER = "token_ids"


class SGLangAPIClient(LLMAPIClient):

    async def _do_generate(
        self, req: APIGenerateInput, stream: bool = False
    ) -> APIGenerateOutput:
        gconfig = req.gconfig
        sample_params = {
            "n": gconfig.n,
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_new_tokens": gconfig.max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": req.stop_token_ids,
        }
        payload = {
            "input_ids": req.prompt_ids,
            "sampling_params": sample_params,
            "return_logprob": req.return_logprob,
            "stream": stream,
        }

        output = APIGenerateOutput.from_input(req)

        # The following code is partially adopted from sglang/bench_serving.py
        output_ids = []
        output_logprobs = []
        finish_reason = {}
        ttft = 0.0
        latency = float("inf")
        st = time.perf_counter()
        most_recent_timestamp = st
        timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=None)
        try:
            async with self.session.post(
                url=self.generate_url,
                json=payload,
                timeout=timeout,
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data[SGLANG_TOKEN_OUTPUT_IDENTIFIER]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                output_ids = data[SGLANG_TOKEN_OUTPUT_IDENTIFIER]
                                finish_reason = data["meta_info"]["finish_reason"]
                                output_logprobs = data["meta_info"][
                                    "output_token_logprobs"
                                ]

                    assert finish_reason["type"] in ["length", "stop"], finish_reason
                    output.output_logprobs = [x[0] for x in output_logprobs]
                    output.output_ids = output_ids
                    output.no_eos = finish_reason["type"] == "length"
                    output.success = True
                    output.latency = latency
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception as e:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            raise RuntimeError(
                f"SGLang generation request fails:\n{output.error}"
            ) from e

        return output

    async def async_update_weights_from_disk(self, path):
        timeout = aiohttp.ClientTimeout(total=300, connect=30, sock_read=None)
        async with self.session.post(
            url=self.update_weights_url,
            json=dict(model_path=path),
            timeout=timeout,
        ) as response:
            if response.status != 200:
                raise RuntimeError("Update weights failed.")


def sglang_server_process(server_args_dict):

    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import kill_process_tree

    if pkg_version.is_version_less("sglang", "0.4.3"):
        from sglang.srt.server import launch_server

        server_args_dict.pop("enable_nccl_nvls")
        server_args_dict.pop("triton_attention_num_kv_splits")
        server_args_dict.pop("cuda_graph_bs")
        server_args_dict.pop("enable_memory_saver")
        server_args_dict.pop("allow_auto_truncate")
        server_args_dict.pop("file_storage_path")
    else:
        from sglang.srt.entrypoints.http_server import launch_server

    server_args = ServerArgs(**server_args_dict)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        map(str, list(range(gpu_utils.gpu_count())))
    )

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


class SGLangGenerationEngine(PipelinableEngine):

    def __init__(
        self,
        server_args_dict: Dict,
        hybrid_train: bool,
        request_timeout: int = 1800,
    ):
        if constants.model_parallel_rank() != 0:
            dist.barrier(group=constants.model_parallel_cpu_group())
            return
        # Start the serving process
        self.server_proc = mp.Process(
            target=sglang_server_process,
            args=(server_args_dict,),
        )
        self.server_proc.start()

        self.base_url = f"http://{server_args_dict['host']}:{server_args_dict['port']}"

        self.api_urls = {
            "generate": f"{self.base_url}/generate",
            "offload_weights": f"{self.base_url}/offload_weights",
            "init_kv_cache": f"{self.base_url}/init_kv_cache",
            "clear_kv_cache": f"{self.base_url}/clear_kv_cache",
            "init_model_weights": f"{self.base_url}/init_model_weights",
            "update_weights_from_disk": f"{self.base_url}/update_weights_from_disk",
        }

        asyncio.run(self.wait_server())

        self.request_timeout = request_timeout

        # offload weights/cache
        self.hybrid_train = hybrid_train

        dist.barrier(group=constants.model_parallel_cpu_group())

    def __del__(self):
        if hasattr(self, "server_proc"):
            from sglang.srt.utils import kill_process_tree

            self.server_proc.terminate()

            kill_process_tree(os.getpid())

    # NOTE: A placeholder function.
    def train(self, mode: bool = True):
        return self

    # NOTE: A placeholder function.
    def eval(self):
        return self

    async def wait_server(self):
        # Wait until the server is launched
        from sglang.srt.utils import kill_process_tree
        from sglang.utils import get_exception_traceback

        success = False
        for _ in range(SGLANG_INIT_TIMEOUT):
            await asyncio.sleep(1)
            try:
                res = requests.get(
                    self.base_url + "/get_model_info", timeout=5, headers={}
                )
                assert res.status_code == 200, f"{res=}, {res.text=}"
                success = True
                break
            except (AssertionError, requests.exceptions.RequestException):
                last_traceback = get_exception_traceback()
                pass
        if not success:
            logger.error(f"Initialization failed. warmup error: {last_traceback}")
            kill_process_tree(os.getpid())
            return

    async def async_generate(
        self,
        input_: data_api.SequenceSample,
        mb_spec: data_api.MicroBatchSpec,
        tokenizer: transformers.PreTrainedTokenizerFast,
        gconfig: GenerationHyperparameters = dataclasses.field(
            default_factory=GenerationHyperparameters
        ),
        stream: bool = False,
        disable_tqdm: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:

        pbar = None if disable_tqdm else tqdm(total=input_.bs * gconfig.n)

        async with SGLangAPIClient(
            generate_url=self.api_urls["generate"],
            update_weights_url=self.api_urls["update_weights_from_disk"],
        ) as client:
            tasks = []
            input_queries = []
            for d in input_.unpack():
                if len(d.seqlens["packed_input_ids"]) > 1:
                    raise RuntimeError(
                        f"sglang backend does not support grouped generation "
                        f"for now. Group size {len(d.seqlens['packed_input_ids'])}."
                    )

                prompt_token_ids = d.data["packed_input_ids"].cpu().numpy().tolist()
                qid = d.ids[0]
                for group_idx in range(gconfig.n):
                    req = APIGenerateInput(
                        qid=qid,
                        group_idx=group_idx,
                        prompt_ids=prompt_token_ids,
                        input_ids=prompt_token_ids,
                        gconfig=gconfig.new(n=1),
                        stop_token_ids=[tokenizer.pad_token_id, tokenizer.eos_token_id],
                        return_logprob=True,
                    )
                    input_queries.append((qid, group_idx))
                    tasks.append(
                        client.async_add_generate_request(
                            req,
                            stream=stream,
                        )
                    )

            outputs = {}
            for r in asyncio.as_completed(tasks):
                out = await r
                outputs[(out.qid, out.group_idx)] = out
                if pbar:
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

            results: List[APIGenerateOutput] = [outputs[key] for key in input_queries]

        # Build the output: generated token ids, generated token scores,
        # and logits mask (which will always be None in sglang).
        batch_token_ids = []
        batch_logprobs = []
        max_seqlen = -1
        for x in results:
            max_seqlen = max(max_seqlen, len(x.output_ids))
            batch_token_ids.append(x.output_ids)
            batch_logprobs.append(x.output_logprobs)

        # To be consistent with our internal implementation,
        # we should pad generated tokens and logprobs
        batch_token_ids = [
            t + [tokenizer.pad_token_id] * (max_seqlen - len(t))
            for t in batch_token_ids
        ]
        batch_logprobs = [p + [0.0] * (max_seqlen - len(p)) for p in batch_logprobs]

        return (
            torch.tensor(
                batch_token_ids, dtype=torch.long, device=constants.current_device()
            ),
            torch.tensor(
                batch_logprobs, dtype=torch.float32, device=constants.current_device()
            ),
            None,
        )

    def generate(
        self,
        input_: data_api.SequenceSample,
        mb_spec: data_api.MicroBatchSpec,
        tokenizer: transformers.PreTrainedTokenizerFast,
        gconfig: GenerationHyperparameters = dataclasses.field(
            default_factory=GenerationHyperparameters
        ),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
        if gconfig.min_new_tokens != 0:
            raise RuntimeError(
                "NOTE: passing in an arbitrary `min_new_tokens` will lead to a bug for SGLang v0.4.3 "
                "because we force to skip_tokenizer_init."
            )
        if constants.model_parallel_rank() != 0:
            dist.barrier(group=constants.model_parallel_cpu_group())
            return None, None, None

        results = asyncio.run(
            self.async_generate(
                input_=input_,
                mb_spec=mb_spec,
                tokenizer=tokenizer,
                gconfig=gconfig,
            )
        )
        dist.barrier(group=constants.model_parallel_cpu_group())
        return results

    def update_weights_from_disk(self, path):
        if constants.model_parallel_rank() != 0:
            dist.barrier(group=constants.model_parallel_cpu_group())
            return

        async def _fn():
            async with SGLangAPIClient(
                generate_url=self.api_urls["generate"],
                update_weights_url=self.api_urls["update_weights_from_disk"],
            ) as client:
                await client.async_update_weights_from_disk(path)

        asyncio.run(_fn())
        dist.barrier(group=constants.model_parallel_cpu_group())


@dataclasses.dataclass
class SGLangGenerationBackend(ModelBackend, SGLangConfig):
    model_path: str = ""
    dtype: str = "float16"

    def _initialize(self, model: Model, spec: FinetuneSpec) -> Model:
        if constants.pipe_parallel_world_size() != 1:
            raise RuntimeError("SGLang does not support pipe parallel size > 1.")
        if constants.model_parallel_world_size() > cluster.spec.n_gpus_per_node:
            raise RuntimeError(
                "AReaL's SGLang integration does not support model parallel size > n_gpus_per_node."
            )

        additional_args = dataclasses.asdict(self)
        additional_args.pop("hybrid_train")
        additional_args["random_seed"] = seeding.get_seed()

        # For simplicity, we let all DP ranks have different ports.
        ports = [None for _ in range(constants.data_parallel_world_size())]
        while any(port is None for port in ports) or len(set(ports)) != len(ports):
            dist.all_gather_object(
                ports,
                network.find_free_port(low=20000, high=40000),
                group=constants.data_parallel_group(),
            )
        additional_args["port"] = ports[constants.data_parallel_rank()]

        server_args_dict = dict(
            host="localhost",
            # Model and tokenizer
            tokenizer_path=self.model_path,
            tokenizer_mode="auto",
            load_format="auto",
            trust_remote_code=True,
            kv_cache_dtype="auto",
            device="cuda",
            served_model_name=f"{constants.experiment_name()}/{constants.trial_name()}/{constants.model_name().role}",
            is_embedding=False,
            skip_tokenizer_init=True,
            # Other runtime options
            tp_size=constants.model_parallel_world_size(),
            # Because we have set CUDA_VISIBLE_DEVICES to a single GPU in each process
            base_gpu_id=int(os.environ["CUDA_VISIBLE_DEVICES"]),
            file_storage_path=os.path.join(
                constants.SGLANG_CACHE_PATH,
                f"sglang_storage{constants.data_parallel_rank()}",
            ),
            # Data parallelism
            dp_size=1,  # TODO: check whether we require SGLang dp
            load_balance_method="round_robin",
            # Expert parallelism
            ep_size=1,  # TODO: check
            # Multi-node distributed serving
            dist_init_addr=None,
            nnodes=1,
            node_rank=0,
            **additional_args,
        )

        model.module = SGLangGenerationEngine(
            server_args_dict,
            hybrid_train=self.hybrid_train,
        )
        model.backend_name = "sglang"
        return model

    def load(self, model: Model, load_dir: str):
        model.module.update_weights_from_disk(load_dir)


register_backend("sglang", SGLangGenerationBackend)
