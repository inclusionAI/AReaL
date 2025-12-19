import asyncio
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import ray
import ray.exceptions
import torch
from ray.runtime_env import RuntimeEnv
from ray.util.placement_group import (
    PlacementGroup,
    placement_group,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from areal.api.cli_args import BaseExperimentConfig
from areal.api.scheduler_api import Job, Scheduler, SchedulingSpec, Worker
from areal.scheduler.exceptions import (
    EngineCallError,
    WorkerCreationError,
    WorkerFailedError,
    WorkerNotFoundError,
    WorkerTimeoutError,
)
from areal.scheduler.rpc.ray_rpc_server import RayRPCServer
from areal.utils import logging
from areal.utils.launcher import get_env_vars
from areal.utils.ray import get_placement_group_master_ip_and_port

logger = logging.getLogger("RayScheduler")


def ray_resource_type():
    if torch.cuda.is_available():
        return "GPU"

    from areal.platforms import is_npu_available

    if is_npu_available:
        return "NPU"

    return "CPU"


@dataclass
class RayWorkerInfo:
    worker: Worker
    actor: ray.actor.ActorHandle
    role: str
    placement_group: PlacementGroup
    bundle_index: int | None
    created_at: float
    env_vars: dict[str, str] = field(default_factory=dict)


class RayScheduler(Scheduler):
    def __init__(
        self,
        startup_timeout: float = 30.0,
        *,
        exp_config: BaseExperimentConfig | None = None,
    ):
        self.exp_config = exp_config
        self.startup_timeout = startup_timeout

        self._workers: dict[str, list[RayWorkerInfo]] = defaultdict(list)
        self._placement_groups: list[PlacementGroup] = []

    def _prepare_worker_specs(
        self, role: str, num_workers: int, schedulings: list[SchedulingSpec] | None
    ) -> list[SchedulingSpec]:
        if not schedulings:
            raise WorkerCreationError(
                role, "Invalid configuration", "Tasks SchedulingSpec must be provided"
            )
        if len(schedulings) == 1:
            return [schedulings[0]] * num_workers

        if len(schedulings) == num_workers:
            return schedulings

        raise WorkerCreationError(
            role,
            "Invalid Configuration",
            f"schedulings length ({len(schedulings)}) must be 1 or equal to replicas ({num_workers})",
        )

    def _bundle_spec(self, cpu: int, gpu: int, mem: int) -> dict:
        """
        define a bundle dict for a given cpu, gpu, mem requirement
        """
        device = ray_resource_type()
        if device == "CPU" and gpu > 0:
            raise ValueError(
                f"Current detected device is CPU but specified number of GPUs is {gpu}"
            )
        return {
            "CPU": cpu,
            device: float(gpu),
            "memory": mem * 1024 * 1024,  # convert mb to bytes
        }

    def _create_bundle_list_gpu(self, cpu: int, gpu: int, mem: int) -> list[dict]:
        """
        for dividing out resources so that 1 bundle can be contained on 1 node and creates a list of bundles
        """
        bundle_list = []

        n_gpus_per_node = self.exp_config.cluster.n_gpus_per_node

        if n_gpus_per_node == 0 and gpu > 0:
            raise ValueError(
                f"Requested {gpu} GPUs but number of GPUs per node is {n_gpus_per_node}"
            )

        if gpu < n_gpus_per_node:
            return [self._bundle_spec(cpu, gpu, mem)]

        gpu_remaining_to_be_assigned = gpu

        while gpu_remaining_to_be_assigned > 0:
            # do not want to take all gpus in node if we do not need that many
            gpu_in_bundle = min(gpu_remaining_to_be_assigned, n_gpus_per_node)

            # for scaling the amount of cpu and memory relative to gpu in bundle
            resource_per_node_multiplier = min(gpu_in_bundle / gpu, 1)
            cpu_in_bundle = math.ceil(cpu * resource_per_node_multiplier)
            mem_in_bundle = math.ceil(mem * resource_per_node_multiplier)

            bundle_list.append(
                self._bundle_spec(cpu_in_bundle, gpu_in_bundle, mem_in_bundle)
            )
            gpu_remaining_to_be_assigned -= gpu_in_bundle

        return bundle_list

    def _actor_resource_spec(self, cpu: int, gpu: int, mem: int) -> dict:
        """
        create a dictionary for passing into ray actor options specifying resource requirements
        """

        device = ray_resource_type()
        if device == "CPU" and gpu > 0:
            raise ValueError(
                f"Current detected device is CPU but specified number of GPUs is {gpu}"
            )

        return {
            "num_cpus": cpu,
            "resources": {device: float(gpu)},
            "memory": mem * 1024 * 1024,
        }

    def _sum_resource_spec(
        self, schedulings: list[SchedulingSpec]
    ) -> tuple[int, int, int]:
        num_cpu = sum(spec.cpu for spec in schedulings)
        num_gpu = sum(spec.gpu for spec in schedulings)
        num_mem = sum(spec.mem for spec in schedulings)

        return (num_cpu, num_gpu, num_mem)

    def _ping_workers(self, role: str, timeout: float | None = None):
        worker_info_list = self._workers[role]
        timeout = timeout if timeout is not None else self.startup_timeout
        refs = [wi.actor.ping.remote() for wi in worker_info_list]

        ref_to_worker = {ref: wi for wi, ref in zip(worker_info_list, refs)}

        pending = refs
        while pending:
            ready, pending = ray.wait(pending, num_returns=1, timeout=timeout)
            # ray.wait timed out
            if len(ready) == 0:
                raise WorkerTimeoutError(role, timeout)

            ref = ready[0]

            try:
                # get to determine if this is a failed actor
                ray.get(ref)
            except ray.exceptions.GetTimeoutError:
                failed_worker = ref_to_worker[ref]
                raise WorkerTimeoutError(failed_worker.worker.id, timeout)
            except ray.exceptions.RayActorError:
                failed_worker = ref_to_worker[ref]
                raise WorkerFailedError(failed_worker.worker.id, -1)

    def _create_rollout_workers(
        self, role: str, schedulings: list[SchedulingSpec]
    ) -> tuple[list[RayWorkerInfo], list[str]]:
        """
        Crate rollout workers, assuming 1 worker per rollout instance.

        Parameters
        ---------
        role: str
        schedulings: list[SchedulingSpec]

        Returns
        --------
        Tuple[list[RayWorkerInfo], list[str]]
            List of RayWorkerInfo of created workers
            List of worker IDs created
        """

        worker_info_list: list[RayWorkerInfo] = []
        worker_ids: list[str] = []

        # create placement_groups
        for idx, spec in enumerate(schedulings):
            worker_id = f"{role}/{idx}"

            bundles = [self._bundle_spec(spec.cpu, spec.gpu, spec.mem)]
            pg = placement_group(bundles, strategy="PACK")

            try:
                ray.get(pg.ready(), timeout=self.startup_timeout)
            except ray.exceptions.GetTimeoutError:
                logger.error(
                    f"Ray placement group timeout for train role {role}\n"
                    f"ray.nodes(): {ray.nodes()}"
                    f"bundles: {bundles}"
                )
                raise
            self._placement_groups.append(pg)

            master_ip, master_port = get_placement_group_master_ip_and_port(
                pg, placement_group_bundle_index=0
            )

            # define resources to actor
            options = self._actor_resource_spec(spec.cpu, spec.gpu, spec.mem)

            env = get_env_vars(
                self.exp_config,
                ",".join([f"{k}={v}" for k, v in spec.env_vars.items()]),
            )

            if spec.env_vars:
                env.update(spec.env_vars)

            actor = RayRPCServer.options(
                **options,
                name=worker_id,
                runtime_env=RuntimeEnv(env_vars=env),
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=0,
                    placement_group_capture_child_tasks=True,
                ),
            ).remote()

            # 0 needed to pad the list as the trainer takes index 1 for ports
            worker_ports = ["0", str(master_port)]
            worker = Worker(
                id=worker_id, ip=master_ip, worker_ports=worker_ports, engine_ports=[]
            )

            wi = RayWorkerInfo(
                worker=worker,
                actor=actor,
                role=role,
                placement_group=pg,
                bundle_index=0,
                created_at=time.time(),
                env_vars=env,
            )

            worker_info_list.append(wi)
            worker_ids.append(worker_id)

        return worker_info_list, worker_ids

    def _create_train_workers(
        self, role: str, schedulings: list[SchedulingSpec]
    ) -> tuple[list[RayWorkerInfo], list[str]]:
        """
        Create workers for training roles. One PG per role with multiple bundles.
        Assume 1 ray worker per train rank.

        Parameters
        ---------
        role: str
        schedulings: list[SchedulingSpec]

        Returns
        --------
        Tuple[list[RayWorkerInfo], list[str]]
            List of RayWorkerInfo of created workers
            List of worker IDs created
        """
        # build bundles
        sum_cpu, sum_gpu, sum_mem = self._sum_resource_spec(schedulings)
        bundles: list[dict[str, float]] = self._create_bundle_list_gpu(
            sum_cpu, sum_gpu, sum_mem
        )

        pg = placement_group(bundles=bundles, strategy="PACK")

        try:
            ray.get(pg.ready(), timeout=self.startup_timeout)
        except ray.exceptions.GetTimeoutError:
            logger.error(
                f"Ray placement group timeout for train role {role}\n"
                f"ray.nodes(): {ray.nodes()}"
                f"bundles: {bundles}"
            )
            raise

        self._placement_groups.append(pg)

        master_ip, master_port = get_placement_group_master_ip_and_port(
            pg, placement_group_bundle_index=0
        )

        worker_info_list: list[RayWorkerInfo] = []
        worker_ids: list[str] = []

        for idx, spec in enumerate(schedulings):
            worker_id = f"{role}/{idx}"

            options = self._actor_resource_spec(spec.cpu, spec.gpu, spec.mem)

            env = get_env_vars(
                self.exp_config,
                ",".join([f"{k}={v}" for k, v in spec.env_vars.items()]),
            )

            if spec.env_vars:
                env.update(spec.env_vars)

            actor = RayRPCServer.options(
                **options,
                name=worker_id,
                runtime_env=RuntimeEnv(env_vars=env),
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_capture_child_tasks=True
                ),
            ).remote()

            worker_ports = ["0", str(master_port)]
            worker = Worker(
                id=worker_id, ip=master_ip, worker_ports=worker_ports, engine_ports=[]
            )

            wi = RayWorkerInfo(
                worker=worker,
                actor=actor,
                role=role,
                placement_group=pg,
                bundle_index=None,  # decided by ray
                created_at=time.time(),
                env_vars=env,
            )
            worker_info_list.append(wi)
            worker_ids.append(worker_id)

        return worker_info_list, worker_ids

    def create_workers(self, job: Job, *args, **kwargs) -> list[str]:
        """
        Create worker actors.

        Parameters
        --------
        job: Job
            Job configuration with role, replicas, tasks, scheduling strategy
        *args
            Additional arguments (UNUSED)
        **kwargs
            Additional keyword arguments (UNUSED)

        Returns
        --------
        list[str]
            List of worker IDs created (e.g., ["rollout/0", "rollout/1])

        Raises
        --------
        WorkerCreationError
            If worker creation fails
        """
        role = job.role
        if role in self._workers:
            raise WorkerCreationError(
                role,
                "Worker group already exists",
                f"Use delete_workers('{role}') first to remove existing workers.",
            )

        num_workers = job.replicas
        if num_workers == 0:
            raise WorkerCreationError(
                role, "Invalud configuration", "replicas must be greater than 0"
            )

        schedulings = self._prepare_worker_specs(role, num_workers, job.tasks)

        strategy = job.scheduling_strategy
        if strategy is None:
            strategy_type = "separation"
        else:
            strategy_type = strategy.type or "separation"
            if strategy_type == "colocation":
                raise WorkerCreationError(
                    role,
                    "Unavailable strategy type",
                    "RayScheduler only supports separation strategy",
                )

        if role == "rollout":
            worker_info_list, worker_ids = self._create_rollout_workers(
                role, schedulings
            )
        else:
            worker_info_list, worker_ids = self._create_train_workers(role, schedulings)

        self._workers[role].extend(worker_info_list)

        self._ping_workers(role, self.startup_timeout)

        if self.exp_config is not None:
            for rank, wi in enumerate(worker_info_list):
                try:
                    wi.actor.configure.remote(self.exp_config, wi.role, rank)
                except Exception as e:
                    logger.error(
                        f"Configure failed on worker {wi.worker.id}: {e}", exc_info=True
                    )
                    self._cleanup_workers(worker_info_list)
                    raise WorkerCreationError(
                        role, "Worker configuration failed", str(e)
                    )

        return worker_ids

    def get_workers(self, role: str, timeout: float | None = None) -> list[Worker]:
        if role not in self._workers:
            raise WorkerNotFoundError(role)

        worker_info_list = self._workers[role]

        self._ping_workers(role, timeout)

        return [wi.worker for wi in worker_info_list]

    def delete_workers(self, role: str | None = None):
        """
        Delete workers and clean up resources

        Parameters
        --------
        role: str, optional
            Specific worker role to delete, or None to delete all
        """
        if role is None:
            roles = list(self._workers.keys())
            for r in roles:
                self.delete_workers(r)
            return

        if role not in self._workers:
            logger.warning(f"Worker role '{role}' not found, skipping deletion")
            return

        workers = self._workers[role]
        logger.info(f"Deleting {len(workers)} workers for role '{role}'")

        self._cleanup_workers(workers)

        del self._workers[role]

        logger.info(f"Successfully deleted workers for role '{role}'")

    def _cleanup_workers(self, workers: list[RayWorkerInfo]):
        for wi in workers:
            actor = wi.actor
            try:
                actor.destroy.remote()
            except Exception:
                logger.warning(
                    f"Could not destroy remote actor {actor}, force killing actor"
                )
                ray.kill(actor, no_restart=True)

            try:
                remove_placement_group(wi.placement_group)
            except Exception:
                logger.warning(f"Could not remove placement group {wi.placement_group}")
            if wi.placement_group in self._placement_groups:
                self._placement_groups.remove(wi.placement_group)

    def _get_worker_info_by_id(self, worker_id: str) -> RayWorkerInfo | None:
        for worker_info_list in self._workers.values():
            for wi in worker_info_list:
                if wi.worker.id == worker_id:
                    return wi
        return None

    async def set_worker_env(self, worker_id: str, env: dict[str, str]) -> None:
        wi = self._get_worker_info_by_id(worker_id)
        if wi is None:
            raise WorkerNotFoundError(worker_id)
        if not env:
            return

        await wi.actor.set_env.remote(env)
        wi.env_vars.update(env)

    async def create_engine(self, worker_id: str, engine: str, *args, **kwargs) -> Any:
        wi = self._get_worker_info_by_id(worker_id)
        if wi is None:
            raise WorkerNotFoundError(worker_id)

        if not isinstance(engine, str):
            raise WorkerCreationError(
                worker_id, f"Engine must be a string import path, got {type(engine)}"
            )
        await wi.actor.create_engine.remote(engine, *args, **kwargs)

    def call_engine(
        self,
        worker_id: str,
        method: str,
        *args,
        http_timeout: float = 7200.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs,
    ) -> Any:
        wi = self._get_worker_info_by_id(worker_id)
        if wi is None:
            raise WorkerNotFoundError(worker_id)

        last_error: str | None = None

        for attempt in range(1, max_retries + 1):
            try:
                ref = wi.actor.call.remote(method, *args, **kwargs)
                result = ray.get(ref, timeout=http_timeout)
                if attempt > 1:
                    logger.info(
                        f"Method '{method}' on '{worker_id}' "
                        f"succeeded after {attempt} attempts"
                    )
                return result
            except ray.exceptions.GetTimeoutError as e:
                last_error = f"Timeout: {e}"
            except ray.exceptions.RayActorError as e:
                raise WorkerFailedError(worker_id, -1, str(e)) from e
            except ray.exceptions.RayTaskError as e:
                raise EngineCallError(worker_id, method, str(e), attempt) from e
            except EngineCallError:
                raise
            except Exception as e:
                last_error = f"Ray call failed: {e}"

            # Retry with exponential backoff
            if attempt < max_retries:
                delay = retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Method '{method}' failed on worker '{worker_id}' "
                    f"(attempt {attempt}/{max_retries}): {last_error}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        raise EngineCallError(
            worker_id, method, last_error or "Max retries exceeded", attempt=max_retries
        )

    async def async_call_engine(
        self,
        worker_id: str,
        method: str,
        *args,
        http_timeout: float = 7200.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs,
    ) -> Any:
        wi = self._get_worker_info_by_id(worker_id)
        if wi is None:
            raise WorkerNotFoundError(worker_id)

        last_error: str | None = None

        for attempt in range(1, max_retries + 1):
            try:
                ref = wi.actor.call.remote(method, *args, **kwargs)
                result = await ref
                if attempt > 1:
                    logger.info(
                        f"Method '{method}' on '{worker_id}' "
                        f"succeeded after {attempt} attempts"
                    )
                return result
            except ray.exceptions.GetTimeoutError as e:
                last_error = f"Timeout: {e}"
            except ray.exceptions.RayActorError as e:
                raise WorkerFailedError(worker_id, -1, str(e)) from e
            except ray.exceptions.RayTaskError as e:
                raise EngineCallError(worker_id, method, str(e), attempt) from e
            except EngineCallError:
                raise
            except Exception as e:
                last_error = f"Ray async call failed: {e}"

            # Retry with exponential backoff
            if attempt < max_retries:
                delay = retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Method '{method}' failed on worker '{worker_id}' "
                    f"(attempt {attempt}/{max_retries}): {last_error}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

        raise EngineCallError(
            worker_id, method, last_error or "Max retries exceeded", attempt=max_retries
        )

    def __del__(self):
        try:
            self.delete_workers()
        except Exception:
            pass
