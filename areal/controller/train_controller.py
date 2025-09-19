import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List

import cloudpickle
import torch
from torch import Tensor

from areal.api.cli_args import TrainControllerConfig
from areal.api.controller_api import TrainController
from areal.api.engine_api import TrainEngine
from areal.api.io_struct import AllocationMode, SaveLoadMeta, WeightUpdateMeta
from areal.controller.utils import create_engine_with_retry
from areal.dataset.distributed_batch_memory import DistributedBatchMemory
from areal.extension.asystem.remote_megatron_engine import RemoteMegatronInitConfig
from areal.scheduler.base import (
    ContainerSpec,
    Scheduler,
    ScheduleStrategy,
    SchedulingConfig,
)
from realhf.base import logging, stats_tracker

logger = logging.getLogger("DistributedTrainController")


class DistributedTrainController(TrainController):
    def __init__(
        self,
        train_engine: TrainEngine,
        config: TrainControllerConfig,
        scheduler: Scheduler,
        *args,
        **kwargs,
    ):
        super().__init__(train_engine, config, scheduler)
        self.allocate_mode = AllocationMode.from_str(config.allocation_mode)
        self.role = kwargs.get("role", "train")
        self.group_size = config.group_size
        self.world_size = self.allocate_mode.train_world_size
        self.dp_size = self.allocate_mode.train_dp_size
        self.tp_size = self.allocate_mode.train_tp_size
        self.pp_size = self.allocate_mode.train_pp_size
        self.enable_colocate_mode = self.config.enable_colocate_mode
        self.storage_prefix = config.storage_prefix

    def initialize(self, *args, **kwargs):
        """Initialize environments for distributed training and load models."""
        scheduling = self.train_engine.get_scheduling_config()
        scheduling_config = SchedulingConfig(replicas=self.world_size)

        target = kwargs.get("colocation_with")
        scheduling_config.schedule_strategy = (
            ScheduleStrategy(type="colocation", uid=target.uid) if target else None
        )
        logger.info(f"scheduling config: {scheduling_config}")

        areal_path = os.getenv("REAL_PACKAGE_PATH", "")
        engine_path = os.getenv("ENGINE_PATH", "")
        workerSpec = ContainerSpec(
            cpu=0,
            mem=0,
            gpu=scheduling.gpu,
            cmd=f"bash {areal_path}/areal/scheduler/scripts/launch-worker.sh".format(
                areal_path=areal_path
            ),
            env_vars=(
                scheduling.env_vars.copy() if scheduling.env_vars is not None else {}
            ),
            portCount=1,
        )
        workerSpec.env_vars["REAL_PACKAGE_PATH"] = areal_path
        workerSpec.env_vars["WORKER_IMAGE"] = (
            "/storage/openpsi/images/areal-25.01-sglang-bf16-editable-metrics-xccl-20250716.sif"
        )
        workerSpec.env_vars["WORKER_LOG_DIR"] = (
            "{storage_prefix}/experiments/logs/root/{experiment_name}/{trial_name}".format(
                storage_prefix=self.storage_prefix,
                experiment_name=self.config.experiment_name,
                trial_name=self.config.trial_name,
            )
        )
        workerSpec.env_vars["WORKER_TYPE"] = f"{self.role}-worker"

        engineSpec = ContainerSpec(
            cpu=0,
            mem=0,
            gpu=0,
            cmd=f"bash {areal_path}/areal/scheduler/scripts/launch-hybrid-server.sh".format(
                areal_path=areal_path
            ),
            env_vars=(
                scheduling.env_vars.copy() if scheduling.env_vars is not None else {}
            ),
            portCount=1,
        )
        engineSpec.env_vars["ENGINE_PACKAGE_PATH"] = engine_path
        engineSpec.env_vars["WORKER_IMAGE"] = (
            "/storage/openpsi/images/hybrid-engine-13680179-20250912200412.sif"
        )
        engineSpec.env_vars["WORKER_LOG_DIR"] = (
            "{storage_prefix}/experiments/logs/root/{experiment_name}/{trial_name}".format(
                storage_prefix=self.storage_prefix,
                experiment_name=self.config.experiment_name,
                trial_name=self.config.trial_name,
            )
        )
        engineSpec.env_vars["WORKER_TYPE"] = f"{self.role}-engine"
        engineSpec.env_vars["WORK_MODE"] = "TRAINING"
        engineSpec.env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        engineSpec.env_vars["USE_MAX_V2"] = "1"
        engineSpec.env_vars["DISCOVERY_CONFIG_CENTER_TYPE"] = "FILE"
        engineSpec.env_vars["NCCL_CUMEM_ENABLE"] = "0"
        engineSpec.env_vars["NCCL_NVLS_ENABLE"] = "0"
        engineSpec.env_vars["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        engineSpec.env_vars["NCCL_DEBUG"] = "WARNING"
        engineSpec.env_vars["ASTRA_SHARED_PATH"] = (
            f"{self.storage_prefix}/astate_shared_storage"
        )
        engineSpec.env_vars["NVTE_FUSED_ATTN"] = "0"
        engineSpec.env_vars["NCCL_MAX_NCHANNELS"] = "16"
        engineSpec.env_vars["NCCL_DEBUG_SUBSYS"] = "INIT,TUNING,GRAPH"
        engineSpec.env_vars["USE_AREAL_LITE"] = "1"

        engineSpec.env_vars["NCCL_SOCKET_IFNAME"] = "bond0"
        engineSpec.env_vars["GLOO_SOCKET_IFNAME"] = "eth0"
        engineSpec.env_vars["NCCL_NET_PLUGIN"] = ""
        engineSpec.env_vars["NCCL_IB_GID_INDEX"] = "3"
        engineSpec.env_vars["NCCL_IB_TIMEOUT"] = "22"
        engineSpec.env_vars["NCCL_IB_RETRY_CNT"] = "7"
        engineSpec.env_vars["NCCL_IB_SL"] = "5"
        engineSpec.env_vars["NCCL_IB_TC"] = "136"
        engineSpec.env_vars["NCCL_IB_HCA"] = "mlx5_bond"
        engineSpec.env_vars["NCCL_SET_THREAD_NAME"] = "1"
        engineSpec.env_vars["NCCL_IB_QPS_PER_CONNECTION"] = "8"
        engineSpec.env_vars["NCCL_SET_THREAD_NAME"] = "1"
        # FIXME @fenghui @xuantai: if not add CUDA_LAUNCH_BLOCKING, bailing max expr(960 gpu) will hang in train batch stage
        engineSpec.env_vars["CUDA_LAUNCH_BLOCKING"] = "1"

        scheduling_config.specs.append(workerSpec)
        scheduling_config.specs.append(engineSpec)

        self.uid = self.scheduler.create_workers(self.role, scheduling_config)

        self.workers = self.scheduler.get_workers(self.role, timeout=1800)
        self.rank_info = {}
        server_addrs = [
            f"{worker.ip}:{worker.ports[0]}" for worker in self.workers if worker.ports
        ]
        # FIXME: @chucai
        time.sleep(60)
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            futures = [
                executor.submit(
                    partial(
                        create_engine_with_retry,
                        self.scheduler.create_engine,
                        worker.id,
                        self.train_engine,
                        RemoteMegatronInitConfig(
                            server_addrs=server_addrs,
                            global_rank=index,
                            world_size=self.world_size,
                            enable_colocate_mode=self.enable_colocate_mode,
                        ),
                    )
                )
                for index, worker in enumerate(self.workers)
            ]
            try:
                for worker_index, future in enumerate(futures):
                    rank_info = future.result()
                    self.rank_info[worker_index] = rank_info
                    logger.info(f"worker_index: {worker_index}, rank_info: {rank_info}")
            except KeyboardInterrupt:
                for f in futures:
                    f.cancel()
                raise
            except Exception as e:
                for f in futures:
                    f.cancel()
                raise RuntimeError(
                    f"Failed to initialize worker_index: {worker_index}, error: {e}"
                )
        # todo: 不能写死remote megatron, 让engine抽象出接口

    def destroy(self):
        """Destroy the engine and release GPU memory."""

    def _rpc_call(self, method, *args, **kwargs):
        logger.info(
            f"start to  rpc call, method: {method}, args: {args}, kwargs: {kwargs}"
        )
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            futures = [
                executor.submit(
                    self.scheduler.call_engine, worker.id, method, *args, **kwargs
                )
                for worker in self.workers
            ]

            results = []
            try:
                for future in futures:
                    result = future.result()  # 可加异常处理
                    results.append(result)
            except KeyboardInterrupt:
                logger.info("receive ctrl+c, terminating all initialization tasks...")
                # 取消所有未完成的future
                for f in futures:
                    f.cancel()
                raise  # 重新抛出异常，主程序能感知
            except Exception as e:
                for f in futures:
                    f.cancel()
                raise RuntimeError(f"{method} failed, error: {e}")

    def upload_weights(self, meta: WeightUpdateMeta):
        """Upload weights to the inference engine."""
        return self._rpc_call("upload_weights", meta)

    def save(self, meta: SaveLoadMeta):
        """Save model weights (and optimizer states) for later use."""
        return self._rpc_call("save", meta)

    def load(self, meta: SaveLoadMeta):
        """Load model weights and optimizer states from a file."""
        return self._rpc_call("load", meta)

    def step_lr_scheduler(self):
        """Step learning rate scheduler.

        Since PPO uses minibatch updates, this method just need to be called once after a few train_batch calls.
        It is separated from train_batch to allow for more flexible scheduling.
        """
        return self._rpc_call("step_lr_scheduler")

    def notify_event(self, event: str, global_step: int) -> None:
        """Notify workers about training start/end events.

        Args:
            event: "train_start" or "train_end"
            global_step: Current global step
        """
        self._rpc_call("notify_event", event, global_step)
        return None

    def train_distributed_batch(
        self, input_: DistributedBatchMemory
    ) -> Dict[str, float]:
        """Update the model with a batch of data and a loss function."""
        logger.info(f"start to train_distributed_batch")
        with (stats_tracker.record_timing("train_distributed_batch_data_split"),):
            batches = input_._split_by_seqlen_ffd_helper(self.group_size, self.dp_size)

        self._calc_metrics(batches)

        serialized_data = [
            cloudpickle.dumps(("train_distributed_batch", [batch], {}))
            for batch in batches
        ]
        futures = []
        results = []
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            for index, worker in enumerate(self.workers):
                # 32卡 tp8pp2
                # 0-7: tp0-7, pp0, dp0
                # 8-15: tp0-7, pp0, dp1
                # 16-23: tp0-7, pp1, dp0
                # 24-31: tp0-7, pp1, dp1

                rank_info = self.rank_info[index]
                dp_rank = rank_info["dp_rank"]
                batch_data = serialized_data[dp_rank]
                futures.append(
                    executor.submit(
                        self.scheduler.call_engine_with_serialized_data,
                        worker.id,
                        batch_data,
                    )
                )
            try:
                for future in futures:
                    result = future.result()
                    results.append(result)
            except KeyboardInterrupt:
                for f in futures:
                    f.cancel()
                raise
            except Exception as e:
                for f in futures:
                    f.cancel()
                raise RuntimeError(f"train_distributed_batch failed, error: {e}")

        # 处理多个minibatch返回的结果
        for worker_result in results:
            if len(worker_result) > 1:  # 处理多个minibatch的情况
                for minibatch in worker_result:
                    stats_tracker.scalar(**minibatch)
            else:  # 保持对单个结果的兼容
                stats_tracker.scalar(**worker_result[0])

        return

    def compute_logprobs_with_distributed(
        self, input_: DistributedBatchMemory
    ) -> Tensor:
        """Update the model with a batch of data and a loss function."""
        logger.info(f"start to compute_logprobs_with_distributed")
        with (
            stats_tracker.record_timing("compute_logprobs_with_distributed_data_split"),
        ):
            batches = input_.split(self.dp_size)
            serialized_data = [
                cloudpickle.dumps(("compute_logprobs_with_distributed", [batch], {}))
                for batch in batches
            ]

        futures = []
        results = []
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            for index, worker in enumerate(self.workers):
                rank_info = self.rank_info[index]
                dp_rank = rank_info["dp_rank"]
                batch_data = serialized_data[dp_rank]
                futures.append(
                    executor.submit(
                        self.scheduler.call_engine_with_serialized_data,
                        worker.id,
                        batch_data,
                    )
                )
            try:
                for future in futures:
                    results.append(future.result())
            except KeyboardInterrupt:
                for f in futures:
                    f.cancel()
                raise
            except Exception as e:
                for f in futures:
                    f.cancel()
                raise RuntimeError(f"compute_logp failed, error: {e}")

        # cat tensor from dp head with padding
        tensors_from_dp_heads = results[: self.dp_size]
        if not tensors_from_dp_heads:
            return torch.tensor([])

        # Find max length in dim 1
        max_len = max(t.shape[1] for t in tensors_from_dp_heads)
        max_len_all = max(t.shape[1] for t in results)
        assert max_len_all == max_len
        # Pad all tensors to max length
        padded_tensors = []
        for t in tensors_from_dp_heads:
            pad_size = max_len - t.shape[1]
            padded = torch.nn.functional.pad(t, (0, pad_size), value=0.0)
            padded_tensors.append(padded)

        # Concatenate along batch dimension
        concatenated_result = torch.cat(padded_tensors, dim=0)
        return concatenated_result

    @torch.no_grad()
    def eval_batch(
        self,
        input_: Dict,
    ) -> torch.Tensor | None:
        """Evaluate the model using the forward pass and loss function."""
        raise NotImplementedError()

    @torch.no_grad()
    def forward(
        self,
        input_: Dict,
        output_seqlens: List[List[int]] | None = None,
    ) -> Any | None:
        """Run the forward pass or inference on the model. Note that it is gradient-free."""
        raise NotImplementedError()

    def _calc_metrics(self, batch_inputs):
        # seqlen std
        seqlens = [td["seqlen"].sum().item() for td in batch_inputs]
        seqlen_std = torch.tensor(seqlens).float().std().item()
        stats_tracker.scalar(**{"seqlen_std": seqlen_std})
