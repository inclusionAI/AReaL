# Copyright 2025 Ant Group Inc.

import dataclasses
import functools
import time
from typing import Dict, List, Optional, Tuple

import torch
import transformers

try:
    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs
    from vllm.inputs.data import TokensPrompt
    from vllm.sampling_params import SamplingParams
    from vllm.utils import Counter

    from realhf.impl.model.backend.thirdparty.vllm import (
        GPUExecutor_,
        LLMEngine_,
        init_vllm,
    )
except ModuleNotFoundError:

    class LLM:
        pass

    class LLMEngine_:
        pass


from realhf.api.cli_args import vLLMConfig
from realhf.api.core import data_api, model_api
from realhf.base import constants, logging, seeding

logger = logging.getLogger("vLLM backend")


class vLLMGenerationEngine(model_api.PipelinableEngine, LLM):
    def __init__(self, llm_engine: LLMEngine_, hybrid_train: bool):
        # NOTE: vLLM's `LLM` class exactly assigns the following
        # two attributes.
        self.llm_engine = llm_engine
        self.request_counter = Counter()

        self.dtype = llm_engine.model_executor.model_config.dtype
        self.device = llm_engine.model_executor.device_config.device

        self.hybrid_train = hybrid_train
        if self.hybrid_train:
            self.llm_engine.model_executor.clear_kv_cache()

    # NOTE: A placeholder function.
    def train(self, mode: bool = True):
        return self

    # NOTE: A placeholder function.
    def eval(self):
        return self

    def update_weights_from_disk(self, path):
        self.llm_engine.model_executor.update_weights(path)

    # A wraper over vLLM's LLM.generate() function.
    def generate(
        self,
        input_: data_api.SequenceSample,
        mb_spec: data_api.MicroBatchSpec,
        tokenizer: transformers.PreTrainedTokenizerFast,
        gconfig: model_api.GenerationHyperparameters = dataclasses.field(
            default_factory=model_api.GenerationHyperparameters
        ),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
        # init kv cache
        tik = time.perf_counter()
        if self.hybrid_train:
            self.llm_engine._initialize_kv_caches()

        if constants.parallelism_rank() == 0:
            if not gconfig.force_no_logits_mask:
                logger.warning("vLLM does not returns the logits mask.")

        # Unpack the input and convert prompts into lists of integers.
        prompts = []
        sample_params = []
        for d in input_.unpack():
            if len(d.seqlens["packed_input_ids"]) > 1:
                raise RuntimeError(
                    f"vLLM backend does not support grouped generation "
                    f"for now. Group size {len(d.seqlens['packed_input_ids'])}."
                )
            max_num_seqs = self.llm_engine.scheduler_config.max_num_seqs
            if max_num_seqs < gconfig.n:
                n_replicas = (gconfig.n + max_num_seqs - 1) // max_num_seqs
                sp_ns = [max_num_seqs for _ in range(n_replicas - 1)] + [
                    gconfig.n - max_num_seqs * (n_replicas - 1)
                ]
            else:
                n_replicas = 1
                sp_ns = [gconfig.n]
            prompts += [
                TokensPrompt(
                    prompt_token_ids=d.data["packed_input_ids"].cpu().numpy().tolist()
                )
            ] * n_replicas
            sample_params += [
                SamplingParams(
                    n=n,
                    top_p=gconfig.top_p,
                    top_k=gconfig.top_k,
                    max_tokens=gconfig.max_new_tokens,
                    min_tokens=gconfig.min_new_tokens,
                    temperature=0.0 if gconfig.greedy else gconfig.temperature,
                    detokenize=False,
                    logprobs=0,
                )
                for n in sp_ns
            ]

        # TODO: find a way to get the GPU tensors.
        req_outputs = LLM.generate(
            self,
            prompts=prompts,
            sampling_params=sample_params,
            use_tqdm=True,
        )

        # Build the output: generated token ids, generated token scores,
        # and logits mask (which will always be None in vLLM).
        batch_token_ids = []
        batch_logprobs = []
        max_seqlen = -1
        for req_output in req_outputs:
            for output in req_output.outputs:
                max_seqlen = max(max_seqlen, len(output.token_ids))
                batch_token_ids.append(list(output.token_ids))
                assert len(output.logprobs) == len(output.token_ids)
                logprobs = []
                for t, logp in zip(output.token_ids, output.logprobs):
                    logprobs.append(logp[t].logprob)
                batch_logprobs.append(logprobs)

        # To be consistent with our internal implementation,
        # we should pad generated tokens and logprobs
        batch_token_ids = [
            t + [tokenizer.pad_token_id] * (max_seqlen - len(t))
            for t in batch_token_ids
        ]
        batch_logprobs = [p + [0.0] * (max_seqlen - len(p)) for p in batch_logprobs]

        # clear kv cache and offload model weights
        if self.hybrid_train:
            tik = time.perf_counter()
            self.llm_engine.model_executor.offload_weights()
            self.llm_engine.model_executor.clear_kv_cache()
            if constants.parallelism_rank() == 0:
                logger.info(f"Clear KV cache time: {time.perf_counter() - tik}s")

        return (
            torch.tensor(batch_token_ids, dtype=torch.long, device=self.device),
            torch.tensor(batch_logprobs, dtype=torch.float32, device=self.device),
            None,
        )


@dataclasses.dataclass
class vLLMGenerationBackend(vLLMConfig, model_api.ModelBackend):
    model_path: str = ""

    def _initialize(
        self, model: model_api.Model, spec: model_api.FinetuneSpec
    ) -> model_api.Model:

        init_vllm()

        if constants.pipe_parallel_world_size() != 1:
            raise NotImplementedError(
                "vLLM does not support pipeline parallelism for now."
            )

        engine_kwargs = dict(
            # Basic config.
            model=self.model_path,
            tokenizer=self.model_path,
            tokenizer_mode="auto",
            skip_tokenizer_init=False,
            trust_remote_code=True,
            max_model_len=self.max_model_len,
            seed=seeding.get_seed(),
            dtype=getattr(torch, self.dtype),
            kv_cache_dtype=self.kv_cache_type,
            device=constants.current_device(),
            # Parallelism.
            tensor_parallel_size=constants.tensor_parallel_world_size(),
            pipeline_parallel_size=constants.pipe_parallel_world_size(),
            # KV cahce and scheduling.
            num_scheduler_steps=self.num_scheduler_steps,
            multi_step_stream_outputs=self.multi_step_stream_outputs,
            block_size=self.block_size,
            swap_space=self.swap_space,
            cpu_offload_gb=self.cpu_offload_gb,
            max_num_seqs=self.max_num_seqs,
            # max_num_batched_tokens=bs * 1024,
            # enable_chunked_prefill=False,
            # Our default system-wise configs.
            max_seq_len_to_capture=self.max_seq_len_to_capture,
            enable_prefix_caching=self.enable_prefix_caching,
            gpu_memory_utilization=self.gpu_memory_utilization,
            disable_sliding_window=self.disable_sliding_window,
            enable_chunked_prefill=self.enable_chunked_prefill,
            disable_custom_all_reduce=True,
            disable_async_output_proc=False,
            disable_log_stats=False,
            worker_use_ray=False,
            enforce_eager=self.enforce_eager,
        )
        for k, v in self.additional_engine_args.items():
            if k in engine_kwargs:
                logger.warning(f"Overriding {k} from {engine_kwargs[k]} to {v}")
            engine_kwargs[k] = v
        engine_args = EngineArgs(**engine_kwargs)
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()
        # just a random name
        engine_config.parallel_config.distributed_executor_backed = "realhf"

        executor_class = GPUExecutor_

        # Create the LLM engine.
        # By default, KV caches will be initialized during LLMEngine initialization.
        # We should release them first and then re-initialize them upon each
        # generation call.
        llm_engine = LLMEngine_(
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            stat_loggers=None,
        )
        model.module = vLLMGenerationEngine(
            llm_engine,
            hybrid_train=self.hybrid_train,
        )
        model.backend_name = "vllm"
        return model


model_api.register_backend("vllm", vLLMGenerationBackend)
