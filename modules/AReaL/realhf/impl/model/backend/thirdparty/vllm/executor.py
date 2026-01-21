# Copyright 2025 Ant Group Inc.

import functools
import gc
import os
import time
from typing import Dict, List, Optional, Set, Tuple, Union

import pynvml
import torch
import torch.distributed as dist
import torch.nn as nn
from vllm.executor.gpu_executor import GPUExecutor
from vllm.model_executor.model_loader.loader import (
    device_loading_context,
    get_model_loader,
)
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.worker.model_runner import GPUModelRunnerBase, ModelRunner
from vllm.worker.multi_step_model_runner import MultiStepModelRunner
from vllm.worker.multi_step_worker import MultiStepWorker
from vllm.worker.worker import Worker, _check_if_gpu_supports_dtype

from realhf.base import constants, logging

from .custom_cache_manager import maybe_set_triton_cache_manager

logger = logging.getLogger("vllm executor")


# Update weights patch
def _update_weights_model_runner(self: ModelRunner, path: str) -> nn.Module:
    target_device = torch.device(self.device_config.device)
    loader = get_model_loader(self.load_config)
    self.model_config.model = path

    if getattr(self.model, "_offloaded", None):
        self.model = loader.load_model(
            model_config=self.model_config,
            device_config=self.device_config,
            lora_config=self.lora_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            cache_config=self.cache_config,
        )
        return self.model.eval()

    model_config = self.model_config
    model = self.model
    with set_default_torch_dtype(model_config.dtype):
        # Skip model initialization here.
        model.load_weights(loader._get_all_weights(self.model_config, model))
        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                # When quant methods need to process weights after loading
                # (for repacking, quantizing, etc), they expect parameters
                # to be on the global target device. This scope is for the
                # case where cpu offloading is used, where we will move the
                # parameters onto device for processing and back off after.
                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)
    return model.eval()


def _update_weights_multi_step_runner(
    self: MultiStepModelRunner, path: str
) -> nn.Module:
    return _update_weights_model_runner(self._base_model_runner, path)


def _update_weights_worker(self, path):
    if isinstance(self, MultiStepWorker):
        return _update_weights_multi_step_runner(self.model_runner, path)
    elif isinstance(self, Worker):
        return _update_weights_model_runner(self.model_runner, path)
    else:
        raise NotImplementedError(f"Unsupported worker type: {type(self)}")


# Offload patch
def _offload_weights_model_runner(self: ModelRunner):
    for p in self.model.parameters():
        p.data = p.data.new()
    self.model._offloaded = True


def _offload_weights_worker(self):
    if isinstance(self, MultiStepWorker):
        return _offload_weights_model_runner(self.model_runner._base_model_runner)
    elif isinstance(self, Worker):
        return _offload_weights_model_runner(self.model_runner)
    else:
        raise NotImplementedError(f"Unsupported worker type: {type(self)}")


class GPUExecutor_(GPUExecutor):

    def _init_executor(self):
        # NOTE: Difference from vLLM:
        # 1. Relax the assertion of TP == 1
        # 2. Unroll worker.load_model() (or model_runner.load_model())

        # Create worker.
        # We use the name `driver_worker` to comfort vLLM's naming.
        # The `driver_worker` may not be the actual driver.

        # workaround for https://github.com/vllm-project/vllm/issues/6103
        if constants.tp_and_pp_world_size() > 1:
            maybe_set_triton_cache_manager()

        self.driver_worker = self._create_worker(
            local_rank=0,
            rank=constants.tp_and_pp_rank(),
        )
        # Patch an `update_weights` method.
        setattr(self.driver_worker.__class__, "update_weights", _update_weights_worker)
        setattr(
            self.driver_worker.__class__, "offload_weights", _offload_weights_worker
        )

        # Init device.
        # Unroll the init_device method to skip the following two operations:
        # 1) initializing the distributed environment
        # 2) setting the random seed, because we have set it in model workers
        if self.driver_worker.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.driver_worker.device = torch.device(
                f"cuda:{self.driver_worker.local_rank}"
            )
            torch.cuda.set_device(self.driver_worker.device)

            _check_if_gpu_supports_dtype(self.driver_worker.model_config.dtype)
            gc.collect()
            torch.cuda.empty_cache()
            self.driver_worker.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.driver_worker.device_config.device}"
            )

        # Load model.
        self.driver_worker.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        # NOTE: Difference from vLLM:
        # 1. We use dist.all_reduce to get the minimum number of blocks
        #    across all TP + PP parallel ranks.
        # 2. We set the init GPU memory as the total GPU memory, such that
        #    vLLM will allocate (total_mem * max_util - cur_mem) for KV caches.
        #    Check the worker class in vLLM for details.
        free_mem, total_mem = torch.cuda.mem_get_info()
        used_mem = total_mem - free_mem
        self.driver_worker.init_gpu_memory = total_mem
        if (
            total_mem * self.driver_worker.cache_config.gpu_memory_utilization
            < used_mem
        ):
            raise torch.cuda.OutOfMemoryError(
                f"Not enough space of vLLM KV caches. "
                f"Used memory: {used_mem / 1024**3:.2f} GB, total memory: {total_mem / 1024**3:.2f} GB, "
                f"max_utilization: {self.driver_worker.cache_config.gpu_memory_utilization}, "
                f"max possible memory usage: {self.driver_worker.cache_config.gpu_memory_utilization * total_mem / 1024**3:.2f}."
            )
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_blocks = self.driver_worker.determine_num_available_blocks()

        # Take the minimum across all ranks.
        num_blocks = torch.tensor(
            num_blocks, device=constants.current_device(), dtype=torch.long
        )
        # NOTE: this is the TP + PP group
        dist.all_reduce(
            num_blocks,
            op=dist.ReduceOp.MIN,
            group=constants.tp_and_pp_group(),
        )

        return int(num_blocks[0]), int(num_blocks[1])

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        # NOTE: Difference from vLLM: we log memory usage here.
        logger.info(
            "Rank %d, # GPU blocks: %d, # CPU blocks: %d",
            dist.get_rank(),
            num_gpu_blocks,
            num_cpu_blocks,
        )

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        torch.cuda.synchronize()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        before_mem = float(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
        tik = time.perf_counter()

        self.driver_worker.initialize_cache(
            num_gpu_blocks=num_gpu_blocks, num_cpu_blocks=num_cpu_blocks
        )

        # Add memory logging here.
        torch.cuda.synchronize()
        tok = time.perf_counter()
        after_mem = float(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
        is_dp_head = (
            constants.is_last_pipe_stage() and constants.tensor_parallel_rank() == 0
        )
        if is_dp_head:
            logger.info(
                f"vLLM DP rank {constants.data_parallel_rank()} "
                f"KV cache memory: {before_mem / 1024**2:.2f} MB"
                f" -> {after_mem / 1024**2:.2f} MB, "
                f"Initializing KV cache time consumption: {tok - tik:.2f} seconds"
            )

    def clear_kv_cache(self) -> None:
        # Log the memory usage before clearing the cache
        torch.cuda.synchronize()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        before_mem = float(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
        tik = time.perf_counter()

        # Clear the cache
        self.driver_worker.gpu_cache = None  # Optional[List[List[torch.Tensor]]]
        self.driver_worker.cache_engine = None
        gc.collect()
        torch.cuda.empty_cache()

        # Log the memory usage after clearing the cache
        torch.cuda.synchronize()
        tok = time.perf_counter()
        after_mem = float(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
        is_dp_head = (
            constants.is_last_pipe_stage() and constants.tensor_parallel_rank() == 0
        )
        if is_dp_head:
            logger.info(
                f"vLLM DP rank {constants.data_parallel_rank()} "
                f"KV cache memory: {before_mem / 1024**2:.2f} MB"
                f" -> {after_mem / 1024**2:.2f} MB, "
                f"Clearing KV cache time consumption: {tok - tik:.2f} seconds"
            )

    def update_weights(self, path):
        return self.driver_worker.update_weights(path)

    def offload_weights(self):
        return self.driver_worker.offload_weights()
