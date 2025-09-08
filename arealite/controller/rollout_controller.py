import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import torch
from tensordict import TensorDict, stack

from arealite.api.cli_args import RemoteHybridInferenceConfig, RolloutControllerConfig
from arealite.api.controller_api import RolloutController
from arealite.api.engine_api import InferenceEngine
from arealite.api.io_struct import AllocationMode, WeightUpdateMeta
from arealite.api.workflow_api import RolloutWorkflow
from arealite.controller.utils import create_engine_with_retry, group_avg_torch
from arealite.dataset.distributed_batch_memory import DistributedBatchMemory
from arealite.extension.asystem.remote_hybrid_inference_worker import (
    RemoteHybridInferenceWorker,
    RemoteHypidInferenceInitConfig,
)
from arealite.extension.asystem.remote_sglang_engine import (
    RemoteSGLangEngine,
    RemoteSGLangInitConfig,
)
from arealite.scheduler.base import (
    ContainerSpec,
    Scheduler,
    ScheduleStrategy,
    SchedulingConfig,
    Worker,
)
from arealite.utils.data import concat_padded_tensors
from realhf.api.core.data_api import RL_TASKS
from realhf.base import logging, stats_tracker

logger = logging.getLogger("DistributedRolloutController")



class DistributedRolloutController(RolloutController):
    # RolloutController可以通过同名接口调用所有InferenceEngine的方法
    # 除此之外没有别的方法了
    # 虽然方法相同，但是传数据集的参数类型不同:
    #   Engine data: List[Dict[str, Any]]
    #   Controller data: DistributedBatch
    def __init__(
        self,
        inf_engine: InferenceEngine,
        config: RolloutControllerConfig,
        scheduler: Scheduler,
        *args,
        **kwargs,
    ):
        super().__init__(inf_engine, config, scheduler)
        self.allocate_mode = AllocationMode.from_str(config.allocation_mode)
        self.dp_world_size = (
            self.allocate_mode.gen_world_size // self.allocate_mode.gen_dp_size
        )
        self.role = kwargs.get("role", "rollout")
        self.enable_colocate_mode = self.config.enable_colocate_mode
        self.storage_prefix = config.storage_prefix

    def _rpc_call(self, method, batches=None, *args, **kwargs):
        """
        执行 RPC 调用，支持每个 worker 的参数不同，或使用通用参数。

        :param method: 要调用的方法名
        :param batches: 包含每个 worker 特定参数的列表（可选）
        :param args: 通用参数
        :param kwargs: 通用关键字参数
        :return: 所有调用的结果列表
        """
        logger.info(
            f"start to rpc call, method: {method}, args: {args}, kwargs: {kwargs}, self.allocate_mode.gen_dp_size: {self.allocate_mode.gen_dp_size}"
        )
        futures = []
        results = []

        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            for i in range(self.allocate_mode.gen_dp_size):
                # 获取当前 master worker 的地址
                master_worker = self.workers[self.dp_world_size * i]

                logger.info(
                    f"RPC calling worker {master_worker.id} for method {method}"
                )

                # 如果 batches 为空，使用通用参数；否则使用 batch 中的特定参数
                if batches and i < len(batches):
                    batch = batches[i]
                    futures.append(
                        executor.submit(
                            self.scheduler.call_engine,
                            master_worker.id,
                            method,
                            batch,  # 使用 batch 中的参数
                            *args,
                            **kwargs,
                        )
                    )
                else:
                    futures.append(
                        executor.submit(
                            self.scheduler.call_engine,
                            master_worker.id,
                            method,
                            *args,  # 使用通用参数
                            **kwargs,
                        )
                    )

            try:
                for future in futures:
                    result = future.result()  # 可加异常处理
                    results.append(result)
            except KeyboardInterrupt:
                logger.info(
                    "receive ctrl+c, terminating all initialization tasks...")
                for f in futures:
                    f.cancel()
                raise

        return results

    def initialize(self, *args, **kwargs):
        """Initialize environments for distributed inference and load models."""

        # 1个engine代表一组sglang实例，存在以下部署场景
        # 一、一组sglang实例跨机部署，如tp8pp2 跨2台机器（每台机器8卡）
        # 二、多组sglang实例在同一台机器部署，如tp4pp1 2个实例可以部署在1台机器上（每台机器8卡）

        scheduling = self.inf_engine.get_scheduling_config()

        # replicas = self.allocate_mode.gen_world_size * node_count if scheduling.gpu >= n_gpu_per_node else self.allocate_mode.gen_world_size
        scheduling_config = SchedulingConfig(
            replicas=self.allocate_mode.gen_world_size)
        target = kwargs.get("colocation_with")
        scheduling_config.schedule_strategy = (
            ScheduleStrategy(type="colocation", uid=target.uid) if target else None
        )

        arealite_path = os.getenv("REAL_PACKAGE_PATH", "")
        engine_path = os.getenv("ENGINE_PATH", "")
        workerSpec = ContainerSpec(
            cpu=0,
            mem=0,
            gpu=scheduling.gpu,
            cmd=f"bash {arealite_path}/arealite/scheduler/scripts/launch-worker.sh".format(
                arealite_path=arealite_path
            ),
            env_vars=(
                scheduling.env_vars.copy() if scheduling.env_vars is not None else {}
            ),
            portCount=1,
        )
        workerSpec.env_vars["REAL_PACKAGE_PATH"] = arealite_path
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
            cmd=f"bash {arealite_path}/arealite/scheduler/scripts/launch-hybrid-server.sh".format(
                arealite_path=arealite_path
            ),
            env_vars=(
                scheduling.env_vars.copy() if scheduling.env_vars is not None else {}
            ),
            portCount=3,
        )
        engineSpec.env_vars["ENGINE_PACKAGE_PATH"] = engine_path
        engineSpec.env_vars["WORKER_IMAGE"] = (
            "/storage/openpsi/images/hybrid-engine-13680179-20250905155406.sif"
        )
        engineSpec.env_vars["WORKER_LOG_DIR"] = (
            "{storage_prefix}/experiments/logs/root/{experiment_name}/{trial_name}".format(
                storage_prefix=self.storage_prefix,
                experiment_name=self.config.experiment_name,
                trial_name=self.config.trial_name,
            )
        )
        engineSpec.env_vars["WORKER_TYPE"] = f"{self.role}-engine"
        engineSpec.env_vars["WORK_MODE"] = "GENERATION"
        engineSpec.env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        engineSpec.env_vars["USE_MAX_V2"] = "1"
        engineSpec.env_vars["DISCOVERY_CONFIG_CENTER_TYPE"] = "FILE"
        engineSpec.env_vars["NCCL_CUMEM_ENABLE"] = "0"
        engineSpec.env_vars["NCCL_NVLS_ENABLE"] = "0"
        engineSpec.env_vars["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        engineSpec.env_vars["NCCL_DEBUG"] = "WARNING"
        engineSpec.env_vars["NVTE_FUSED_ATTN"] = "0"
        engineSpec.env_vars["NCCL_MAX_NCHANNELS"] = "16"
        engineSpec.env_vars["ASTRA_SHARED_PATH"] = (
            f"{self.storage_prefix}/astate_shared_storage"
        )
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
        # sglang need to specify NCCL version
        engineSpec.env_vars["SGLANG_NCCL_SO_PATH"]="/opt/conda/lib/python3.10/site-packages/nvidia/nccl-amem/lib/libnccl.so.2"

        scheduling_config.specs.append(workerSpec)
        scheduling_config.specs.append(engineSpec)

        self.uid = self.scheduler.create_workers(self.role, scheduling_config)

        self.workers = self.scheduler.get_workers(self.role, timeout=1800)
        # 如果1个实例跨机部署，返回的server_addrs是engine实例数的整数倍;e.g. dp2tp8pp2, 需要2个engine，返回了4个server_addrs， 只有index==0|2的才是真正的服务地址
        # 如果多个实例同机部署，返回的server_addrs和engine实例数等长
        worker_addrs = [
            f"{worker.ip}:{worker.ports[0]}" for worker in self.workers if worker.ports
        ]
        assert len(worker_addrs) % self.allocate_mode.gen_dp_size == 0

        futures = []

        master_addr = ""

        self.exiting = threading.Event()

        # FIXME: @chucai
        time.sleep(60)
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            # for i in range(self.allocate_mode.gen_dp_size):
            for index, worker in enumerate(self.workers):
                if index % self.dp_world_size != 0:
                    continue
                master_worker = self.workers[index]
                main_server_addrs = [
                    f"{worker.ip}:{worker.ports[0]}"
                    for worker in self.workers[index : index + self.dp_world_size]
                    if worker.ports
                ]
                free_addrs = [
                    [
                        f"{worker.ip}:{port}"
                        for worker in self.workers[index : index + self.dp_world_size]
                        for port in worker.ports[1:]
                    ]
                ]

                if isinstance(self.inf_engine, RemoteSGLangEngine):
                    init_config = RemoteSGLangInitConfig(
                        main_server_addrs=main_server_addrs,
                        sglang_addrs_list=free_addrs,
                    )
                elif isinstance(self.inf_engine, RemoteHybridInferenceWorker):
                    if index == 0:
                        master_addr = free_addrs[0][0]
                    init_config = RemoteHypidInferenceInitConfig(
                        main_server_addrs=main_server_addrs,
                        free_addrs=free_addrs,
                        world_size=self.allocate_mode.gen_world_size,
                        global_ranks=list(
                            range(index, index+self.dp_world_size)),
                        master_addr=master_addr,
                        enable_colocate_mode=self.enable_colocate_mode,
                    )

                futures.append(
                    executor.submit(
                        partial(
                            create_engine_with_retry,
                            self.scheduler.create_engine,
                            master_worker.id,
                            self.inf_engine,
                            init_config,
                        )
                    )
                )
            try:
                for future in futures:
                    future.result()  # 可加异常处理
            except KeyboardInterrupt:
                logger.info(
                    "receive ctrl+c, terminating all initialization tasks...")
                for f in futures:
                    f.cancel()
                raise

    def destroy(self):
        self.exiting.set()

    def __del__(self):
        self.destroy()

    def update_weights(self, meta: WeightUpdateMeta) -> None:
        """Update weights in the inference engine."""
        self._rpc_call("update_weights", None, meta)
        return

    def notify_event(self, event: str, global_step: int) -> None:
        """Notify workers about inference start/end events.

        Args:
            event: "rollout_start" or "rollout_end"
            global_step: Current global step
        """
        self._rpc_call("notify_event", None, event, global_step)
        return None

    def submit(self, data: List[Dict[str, Any]], workflow: RolloutWorkflow) -> None:
        """Asynchronously submit a request to the inference engine. Exits immediately."""
        batches = self.split_list(data, self.allocate_mode.gen_dp_size)

        for index in range(self.allocate_mode.gen_dp_size):
            master_worker = self.workers[self.dp_world_size * index]
            self.scheduler.call_engine(
                master_worker.id, "submit", batches[index], workflow
            )

    def wait(self, count: int, timeout: int) -> TensorDict:
        """Wait for a specified number of requests to complete, with a timeout."""
        batch_count = count // len(self.workers)
        assert count % self.allocate_mode.gen_dp_size == 0
        results = []
        for index in range(self.allocate_mode.gen_dp_size):
            master_worker = self.workers[self.dp_world_size * index]
            result = self.scheduler.call_engine(
                master_worker.id, "wait", batch_count, timeout
            )
            results.append(result)
        res = stack(results, dim=0)
        return res

    def _thread_loop(
        self, func: Callable, sleep_time: float = 0.1, recycle: Callable = None
    ):
        try:
            while not self.exiting.is_set():
                func()
                time.sleep(sleep_time)
        except Exception:
            traceback.print_exc()
        finally:
            # Cancel remaining tasks
            if recycle is not None:
                recycle()

    def _start_wait_at_least_no_concat_loop(
        self,
        master_worker: Worker,
        callback: Callable[[List[TensorDict]], None],
        batch_count: int,
        timeout: float,
        no_response_timeout: float = None,
    ) -> None:
        assert batch_count >= 0
        assert timeout >= 0

        if no_response_timeout is None:
            no_response_timeout = timeout * 20

        def _wait_at_least_no_concat_callable():
            try:
                res: List[TensorDict] = self.scheduler.call_engine(
                    master_worker.id, "wait_at_least_no_concat", batch_count, timeout
                )
                if res is None or len(res) == 0:
                    res: List[TensorDict] = self.scheduler.call_engine(
                        master_worker.id,
                        "wait_at_least_no_concat",
                        1,
                        no_response_timeout,
                    )
            except TimeoutError as e:
                logger.debug(
                    f"Timeout while waiting for results from worker {master_worker.id}: {e}"
                )
            except Exception as e:
                logger.error(
                    f"Error while waiting for results from worker {master_worker.id}: {e}"
                )
                raise e

            callback(res)

        t = threading.Thread(
            target=partial(
                self._thread_loop, _wait_at_least_no_concat_callable, sleep_time=0
            )
        )
        t.start()

    def register_callback_to_all_worker(
        self, method: str, callback: Callable, **kwargs
    ):
        if method == "wait_at_least_no_concat":
            assert "batch_count" in kwargs and "timeout" in kwargs
            for index in range(self.allocate_mode.gen_dp_size):
                master_worker = self.workers[self.dp_world_size * index]
                self._start_wait_at_least_no_concat_loop(
                    master_worker, callback, batch_count=kwargs["batch_count"], timeout=kwargs["timeout"], no_response_timeout=kwargs.get("no_response_timeout", None))
        else:
            raise ValueError(
                f"Unsupported method {method} for registering callback.")

    def abort_all_requests(self) -> None:
        """Abort all pending requests in the inference engine."""
        self._rpc_call("abort_all_requests")
        logger.info("All pending requests have been aborted.")
        return None

    def rollout_distributed_batch(
        self, data: DistributedBatchMemory, workflow: RolloutWorkflow
    ) -> DistributedBatchMemory:  # batcsize=16
        """Submit a batch of requests to the inference engine and wait for the results."""
        batches = data.split(self.allocate_mode.gen_dp_size)
        assert len(self.workers) % self.dp_world_size == 0
        results = self._rpc_call(
            "rollout_distributed_batch", batches, workflow)

        batchdata = DistributedBatchMemory(None)
        for dataset in results:
            batchdata = batchdata.merge(dataset)

        return batchdata

    def rollout(
        self, data: List[Dict[str, Any]], workflow: RolloutWorkflow
    ) -> TensorDict:
        batches = self.split_list(data, self.allocate_mode.gen_dp_size)

        results = self._rpc_call("rollout", batches, workflow)
        assert len(results) > 0
        size = int(results[0]["input_ids"].shape[0])
        bs = size * len(results)

        padded = concat_padded_tensors(results)
        if isinstance(padded, dict):
            padded = TensorDict(padded, batch_size=[bs])
        return padded

    def split_list(self, lst, n):
        if n <= 0:
            raise ValueError("n must larger than 0")
        chunk_size, rem = divmod(len(lst), n)
        result = []
        index = 0
        for i in range(n):
            current_size = chunk_size + 1 if i < rem else chunk_size
            result.append(lst[index : index + current_size])
            index += current_size
        return result

