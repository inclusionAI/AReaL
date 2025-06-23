# Copyright 2025 Ant Group Inc.

import asyncio
import dataclasses
import json
import os
import socket
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
    constants,
    datapack,
    gpu_utils,
    logging,
    name_resolve,
    names,
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
            "input_ids": req.input_ids,
            "sampling_params": sample_params,
            "return_logprob": req.return_logprob,
            "stream": stream,
        }

        assert not stream, "streaming mode not yet implemented"
        outputs = [APIGenerateOutput.from_input(req) for _ in range(gconfig.n)]
        most_recent_timestamps = [time.perf_counter() for _ in range(gconfig.n)]
        output_idx = 0

        # The following code is partially adopted from sglang/bench_serving.py
        st = time.perf_counter()
        async with self.session.post(url=self.generate_url, json=payload) as response:
            response.raise_for_status()
            async for chunk_bytes in response.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue

                chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                latency = time.perf_counter() - st
                if chunk == "[DONE]":
                    pass
                else:
                    datas = json.loads(chunk)
                    if not isinstance(datas, list):
                        datas = [datas]
                    for data in datas:

                        output = outputs[output_idx]
                        timestamp = time.perf_counter()
                        # First token
                        if output.ttft == float("inf"):
                            ttft = time.perf_counter() - st
                            output.ttft = ttft
                        # Decoding phase
                        else:
                            output.itl.append(
                                timestamp - most_recent_timestamps[output_idx]
                            )

                        most_recent_timestamps[output_idx] = timestamp
                        output.output_ids = [data[SGLANG_TOKEN_OUTPUT_IDENTIFIER]]
                        finish_reason = data["meta_info"]["finish_reason"]
                        if req.return_logprob:
                            output.output_logprobs = [
                                [
                                    x[0]
                                    for x in data["meta_info"]["output_token_logprobs"]
                                ]
                            ]
                        assert finish_reason["type"] in [
                            "length",
                            "stop",
                        ], finish_reason
                        output.no_eos = [finish_reason["type"] == "length"]
                        output.latency = latency

                        output_idx += 1

        return APIGenerateOutput.concat(outputs)

    async def async_update_weights_from_disk(self, path, retries=5):
        for _ in range(retries):
            async with self.session.post(
                url=self.update_weights_url,
                json=dict(model_path=path),
            ) as resp:
                if resp.status == 200:
                    res = await resp.json()
                    success = res["success"]
                    if success:
                        return
                    logger.warning(
                        f"Update weights failed: {res['message']}. Retrying."
                    )
                logger.warning(f"Update weights failed: {resp.reason}. Retrying.")
            time.sleep(0.1)
        raise RuntimeError("Update weights failed.")


def sglang_server_process(server_args_dict):

    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import kill_process_tree

    if pkg_version.is_version_less("sglang", "0.4.4"):
        server_args_dict.pop("log_requests_level")

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
        logger.info(f"SGLang Server Args: {server_args}")
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
        if constants.tensor_parallel_rank() != 0:
            dist.barrier(group=constants.tensor_parallel_cpu_group())
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

        self.wait_server()

        if server_args_dict["enable_metrics"]:
            dp_rank = constants.data_parallel_rank()
            pp_rank = constants.pipe_parallel_rank()
            tp_rank = constants.tensor_parallel_rank()
            metric_server_name = f"d{dp_rank}p{pp_rank}t{tp_rank}"
            key = names.metric_server(
                constants.experiment_name(),
                constants.trial_name(),
                "sglang",
                metric_server_name,
            )
            host_ip = server_args_dict["host"]
            host_port = server_args_dict["port"]
            address = f"{host_ip}:{host_port}"
            name_resolve.add(key, address, keepalive_ttl=None, delete_on_exit=True)
            logger.info(f"SGLang {metric_server_name} metrics URL: {address}")

        self.request_timeout = request_timeout

        # offload weights/cache
        self.hybrid_train = hybrid_train

        dist.barrier(group=constants.tensor_parallel_cpu_group())

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

    def wait_server(self):
        # Wait until the server is launched
        from sglang.srt.utils import kill_process_tree
        from sglang.utils import get_exception_traceback

        success = False
        for _ in range(SGLANG_INIT_TIMEOUT):
            time.sleep(1)
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
            for d in input_.unpack():
                if len(d.seqlens["packed_input_ids"]) > 1:
                    raise RuntimeError(
                        f"sglang backend does not support grouped generation "
                        f"for now. Group size {len(d.seqlens['packed_input_ids'])}."
                    )

                prompt_token_ids = d.data["packed_input_ids"].cpu().numpy().tolist()
                qid = d.ids[0]
                req = APIGenerateInput(
                    qid=qid,
                    prompt_ids=prompt_token_ids,
                    input_ids=prompt_token_ids,
                    gconfig=gconfig,
                    stop_token_ids=[tokenizer.pad_token_id, tokenizer.eos_token_id],
                    return_logprob=True,
                )
                tasks.append(
                    client.async_add_generate_request(
                        req,
                        stream=stream,
                    )
                )

            outputs = {}
            for r in asyncio.as_completed(tasks):
                out = await r
                outputs[out.qid] = out
                if pbar:
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

            results: List[APIGenerateOutput] = [outputs[key] for key in input_.ids]

        # Build the output: generated token ids, generated token scores,
        # and logits mask (which will always be None in sglang).
        batch_token_ids = []
        batch_logprobs = []
        max_seqlen = -1
        for x in results:
            max_seqlen = max(max_seqlen, max(x.output_lens))
            batch_token_ids += x.output_ids
            batch_logprobs += x.output_logprobs

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
        if constants.tensor_parallel_rank() != 0:
            dist.barrier(group=constants.tensor_parallel_cpu_group())
            return None, None, None

        def run_in_thread():
            # Create a new event loop for this thread
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(
                    self.async_generate(
                        input_=input_,
                        mb_spec=mb_spec,
                        tokenizer=tokenizer,
                        gconfig=gconfig,
                    )
                )
            finally:
                new_loop.close()

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            results = future.result()
        dist.barrier(group=constants.tensor_parallel_cpu_group())
        return results

    def update_weights_from_disk(self, path):
        if constants.tensor_parallel_rank() != 0:
            dist.barrier(group=constants.tensor_parallel_cpu_group())
            return

        resp = requests.post(
            url=self.api_urls["update_weights_from_disk"],
            json=dict(model_path=path),
        )
        resp.raise_for_status()
        res = resp.json()
        assert res["success"]
        dist.barrier(group=constants.tensor_parallel_cpu_group())


@dataclasses.dataclass
class SGLangGenerationBackend(ModelBackend, SGLangConfig):
    model_path: str = ""

    def _initialize(self, model: Model, spec: FinetuneSpec) -> Model:
        if constants.pipe_parallel_world_size() != 1:
            raise RuntimeError("SGLang does not support pipe parallel size > 1.")
        if constants.tensor_parallel_world_size() > torch.cuda.device_count():
            raise RuntimeError(
                "AReaL's SGLang integration does not support model parallel size > torch.cuda.device_count()."
            )

        additional_args = dataclasses.asdict(self)
        additional_args.pop("hybrid_train")
        additional_args["random_seed"] = seeding.get_seed()

        # For simplicity, we let all DP ranks have different ports.
        ports = [None for _ in range(constants.data_parallel_world_size())]
        while any(port is None for port in ports) or len(
            set(datapack.flat2d(ports))
        ) != len(datapack.flat2d(ports)):
            dist.all_gather_object(
                ports,
                network.find_multiple_free_ports(
                    2,
                    low=10000,
                    high=60000,
                    experiment_name=constants.experiment_name(),
                    trial_name=constants.trial_name(),
                    lockfile_root=os.path.join(
                        constants.get_cache_path(self.args), "ports"
                    ),
                ),
                group=constants.data_parallel_group(),
            )
        api_server_port, dist_port = ports[constants.data_parallel_rank()]
        additional_args["port"] = api_server_port

        host_ip = socket.gethostbyname(socket.gethostname())
        server_args_dict = dict(
            host="localhost" if not self.enable_metrics else host_ip,
            # Model and tokenizer
            tokenizer_path=self.model_path,
            tokenizer_mode="auto",
            load_format="auto",
            trust_remote_code=True,
            device="cuda",
            served_model_name=f"{constants.experiment_name()}/{constants.trial_name()}/{constants.model_name().role}",
            is_embedding=False,
            skip_tokenizer_init=True,
            # Other runtime options
            tp_size=constants.tensor_parallel_world_size(),
            # Because we have set CUDA_VISIBLE_DEVICES to a single GPU in each process
            base_gpu_id=int(os.environ["CUDA_VISIBLE_DEVICES"]),
            # Data parallelism
            dp_size=1,  # TODO: check whether we require SGLang dp
            load_balance_method="round_robin",
            # Expert parallelism
            ep_size=1,  # TODO: check
            # Multi-node distributed serving
            dist_init_addr=f"{network.gethostip()}:{dist_port}",
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
