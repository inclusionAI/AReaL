import asyncio
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import sglang as sgl
import torch.distributed as dist
from tensordict import TensorDict

from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import LLMRequest, LLMResponse, RolloutStat, WeightUpdateMeta
from realhf.base import logging, name_resolve, names, pkg_version

if TYPE_CHECKING:
    from areal.api.workflow_api import RolloutWorkflow
logger = logging.getLogger(__name__)

if pkg_version.is_available("sglang"):
    if pkg_version.is_version_greater_or_equal("sglang", "0.4.4"):
        SGLANG_TOKEN_OUTPUT_IDENTIFIER = "output_ids"
    else:
        SGLANG_TOKEN_OUTPUT_IDENTIFIER = "token_ids"

ROLLOUT_POLL_WAIT_TIME = 0.4
RID_CACHE_SIZE = 128

"""
Local SGLang Inference Engine
SGLangEngine currently only supports single-controller. Cannot be used in SPMD
"""


class SGLangEngine(InferenceEngine):

    def __init__(
        self,
        config: InferenceEngineConfig,
        engine_args: Optional[Dict[str, Any]] = None,
    ):
        config.max_concurrent_rollouts = (
            config.max_concurrent_rollouts or config.consumer_batch_size
        )
        self.config = config
        self.engine_args = engine_args or {}

        qsize = config.queue_size or config.max_concurrent_rollouts * 10
        self.input_queue = Queue(maxsize=qsize)
        self.output_queue = Queue(maxsize=qsize)
        self.result_cache = []

        self.exiting = threading.Event()
        self.lock = threading.Lock()

        self.rollout_stat = RolloutStat()

        self._version = 0

    def initialize(self, addr: str | None, ft_spec: Optional[Dict[str, Any]] = None):
        self.engine = sgl.Engine(**self.engine_args)

        self.rollout_thread = threading.Thread(target=self._rollout_thread)
        self.rollout_thread.start()

    def destroy(self):
        self.exiting.set()
        self.rollout_thread.join()

        if hasattr(self, "engine") and self.engine is not None:
            try:
                self.engine.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down engine: {e}")

    def set_version(self, version):
        with self.lock:
            self._version = version

    def get_version(self):
        with self.lock:
            return self._version

    def _rollout_thread(self):
        """Thread that runs the rollout loop."""
        try:
            asyncio.run(self._rollout_thread_async())
        except Exception as e:
            traceback.print_exc()
            raise e

    async def _rollout_thread_async(self):
        data = None

        rollout_tasks: Dict[str, asyncio.Task] = {}
        rid = 0

        try:
            while not self.exiting.is_set():
                # Load next data from controller
                if data is None:
                    try:
                        data, workflow = self.input_queue.get_nowait()
                        logger.info(f"Get data from puller: {data}")
                    except Empty:
                        logger.debug(f"No data from puller stream.")

                # Check capacity
                if dist.is_initialized():
                    world_size = dist.get_world_size()
                else:
                    world_size = 1

                cannot_rollout_reason = []
                capacity = max(1, self.config.max_concurrent_rollouts // world_size)
                can_rollout = len(rollout_tasks) < capacity
                if not can_rollout:
                    cannot_rollout_reason.append(
                        f"Exceeding capacity: # running tasks {len(rollout_tasks)} >= capacity {capacity}"
                    )

                # Staleness control
                version = self.get_version()
                ofp = self.config.max_head_offpolicyness
                with self.lock:
                    sample_cnt = self.rollout_stat.accepted + self.rollout_stat.running
                expected_version = sample_cnt // self.config.consumer_batch_size
                not_staled = expected_version <= ofp + version
                can_rollout &= not_staled
                if not not_staled:
                    cannot_rollout_reason.append(
                        f"Staled: expected version ({expected_version}) = "
                        f"global sample cnt ({sample_cnt}) // batch size ({self.config.consumer_batch_size}), "
                        f"current latest version {version}, "
                        f"offpolicyness {self.config.max_head_offpolicyness}."
                    )

                if not can_rollout:
                    logger.debug(
                        f"Cannot submit new rollouts. "
                        + "\n".join(cannot_rollout_reason)
                    )

                # Create new rollout task
                if can_rollout and data is not None:
                    task = asyncio.create_task(
                        workflow.arun_episode(self, data), name=str(rid)
                    )
                    rollout_tasks[str(rid)] = task

                    with self.lock:
                        self.rollout_stat.submitted += 1
                        self.rollout_stat.running += 1
                        logger.info(
                            f"Submit rollout rid {rid}. "
                            f"Submit: {self.rollout_stat.submitted}, "
                            f"running: {self.rollout_stat.running}, "
                            f"accepted: {self.rollout_stat.accepted}."
                        )

                    rid += 1
                    data = None

                # Wait for rollout completion
                tasks = list(rollout_tasks.values())
                done = []
                if tasks:
                    done, _ = await asyncio.wait(
                        tasks,
                        timeout=ROLLOUT_POLL_WAIT_TIME,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                else:
                    await asyncio.sleep(ROLLOUT_POLL_WAIT_TIME)

                # Collect done results
                for task in done:
                    traj = await task
                    traj: TensorDict
                    task_rid = task.get_name()
                    rollout_tasks.pop(task_rid)
                    self.rollout_stat.accepted += 1

                    try:
                        self.output_queue.put_nowait(traj)
                    except Full:
                        raise RuntimeError(
                            "Output queue full. Please increase queue_size."
                        )

                    with self.lock:
                        self.rollout_stat.running -= 1
                        logger.info(
                            f"Finish rollout {task_rid}. "
                            f"Submit: {self.rollout_stat.submitted}, "
                            f"running: {self.rollout_stat.running}, "
                            f"accepted: {self.rollout_stat.accepted}."
                        )
        finally:
            # Cancel remaining tasks
            for task in rollout_tasks.values():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    async def agenerate(self, req: LLMRequest) -> LLMResponse:
        """Async version of generate using local sglang engine."""
        if not hasattr(self, "engine") or self.engine is None:
            raise RuntimeError(
                "Local SGLang engine is not initialized, cannot generate."
            )

        # Prepare request payload
        gconfig = req.gconfig
        stop_token_ids = gconfig.stop_token_ids

        if gconfig.n_samples != 1:
            raise ValueError(
                "LocalSGLangEngine does not support n_samples > 1. "
                "Please call generate for multiple times with n_samples = 1."
            )
        sample_params = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_new_tokens": gconfig.max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": stop_token_ids,
        }

        completions = ""
        prompt = req.text if req.text else None
        input_ids = req.input_ids if req.input_ids else None

        # Make request
        start_time = time.perf_counter()
        accumulated_output_tokens = []
        accumulated_output_logprobs = []
        accumulated_versions = []
        stop_reason = "length"
        while (
            stop_reason != "stop"
            and len(accumulated_output_tokens) < gconfig.max_new_tokens
        ):

            try:
                outputs = await self.engine.async_generate(
                    prompt=prompt,
                    input_ids=input_ids,
                    sampling_params=sample_params,
                    return_logprob=True,
                )

                completions += outputs["text"]
                if prompt is None:
                    prompt = outputs["text"]
                else:
                    prompt += outputs["text"]

                meta_info = outputs["meta_info"]
                output_tokens = [x[1] for x in meta_info["output_token_logprobs"]]
                output_logprobs = [x[0] for x in meta_info["output_token_logprobs"]]

                finish_reason = meta_info.get("finish_reason", {})
                stop_reason = finish_reason.get("type", "length")

                accumulated_output_tokens.extend(output_tokens)
                accumulated_output_logprobs.extend(output_logprobs)
                accumulated_versions.extend([-1] * len(output_tokens))

            except Exception as e:
                raise RuntimeError(f"Local SGLang engine generation failed: {e}")

        latency = time.perf_counter() - start_time

        return LLMResponse(
            completions=completions,
            input_tokens=req.input_ids if req.input_ids else [],
            output_tokens=accumulated_output_tokens,
            output_logprobs=accumulated_output_logprobs,
            output_versions=accumulated_versions,
            stop_reason=stop_reason,
            latency=latency,
            ttft=latency,
        )

    def update_weights(self, meta):
        executor = ThreadPoolExecutor(max_workers=1)
        return executor.submit(self._update_weights, meta)

    def _update_weights(self, meta: WeightUpdateMeta):
        if not hasattr(self, "engine") or self.engine is None:
            raise RuntimeError(
                "Local SGLang engine is not initialized, cannot update weights."
            )
        if meta.type == "disk":
            try:
                update_name = names.update_weights_from_disk(
                    self.config.experiment_name,
                    self.config.trial_name,
                    meta.model_version,
                )
                save_timestamp = int(name_resolve.wait(update_name, timeout=120))
                load_timestamp = time.time_ns()
                logger.info(
                    f"Begin update weights from {meta.path}, responded in {(load_timestamp - save_timestamp)/1e6:.2f} ms"
                )
                # Update weights from disk,
                self.engine.update_weights_from_disk(model_path=meta.path)

                logger.info(
                    f"Loading weights done in {(time.time_ns() - load_timestamp)/1e6:.2f} ms"
                )
                self.set_version(meta.model_version)
            except Exception as e:
                logger.error(f"Failed to update weights: {e}")
                raise
        else:
            raise NotImplementedError(f"Unsupported weight update type: {meta.type}")

    def submit(self, data: Dict[str, Any], workflow: "RolloutWorkflow") -> None:
        try:
            self.input_queue.put_nowait((data, workflow))
        except Full:
            raise RuntimeError("Input queue full. Please increase queue_size.")

    def wait(self, count: int, timeout: float, should_accept: Callable) -> TensorDict:
        tik = time.perf_counter()
        accepted = len(self.result_cache)
        while (
            accepted < count
            and not self.exiting.is_set()
            and time.perf_counter() - tik < timeout
        ):
            try:
                result = self.output_queue.get(timeout=ROLLOUT_POLL_WAIT_TIME)
                if should_accept(result):
                    self.result_cache.append(result)
                    accepted += 1
                else:
                    with self.lock:
                        self.rollout_stat.accepted -= 1
            except Empty:
                time.sleep(ROLLOUT_POLL_WAIT_TIME)
        if self.exiting.is_set():
            raise RuntimeError("Rollout engine is exiting, cannot wait for results.")
        if accepted < count:
            raise TimeoutError(
                f"Timed out waiting for {count} rollouts, " f"only received {accepted}."
            )
        results, self.result_cache = (
            self.result_cache[:count],
            self.result_cache[count:],
        )
        return TensorDict.cat(results, dim=0)

    def rollout(
        self, data: List[Dict[str, Any]], workflow: "RolloutWorkflow"
    ) -> TensorDict:
        """Submit a batch of requests to the inference engine and wait for the results."""
        for item in data:
            self.submit(item, workflow)
        return self.wait(
            count=len(data),
            timeout=self.config.request_timeout,
            should_accept=lambda x: True,
        )
