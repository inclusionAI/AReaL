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

from areal.api.cli_args import (
    BaseExperimentConfig,
    SchedulingSpec,
    SchedulingStrategyType,
)
from areal.api.scheduler_api import Job, Scheduler, Worker
from areal.scheduler.exceptions import (
    EngineCallError,
    WorkerCreationError,
    WorkerFailedError,
    WorkerNotFoundError,
    WorkerTimeoutError,
)
from areal.scheduler.rpc.ray_rpc_server import RayRPCServer
from areal.utils import logging
from areal.utils.launcher import get_env_vars, get_thread_env_vars
from areal.utils.ray import get_placement_group_master_ip_and_port

logger = logging.getLogger("RayScheduler")


def ray_resource_type():
    if torch.cuda.is_available():
        return "GPU"

    from areal.infra.platforms import is_npu_available

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
        self._worker_info_by_id: dict[str, RayWorkerInfo] = {}
        self._placement_groups: list[PlacementGroup] = []

        # Colocation tracking: colocated roles reuse workers from target role
        self._colocated_roles: dict[str, str] = {}  # colocated_role -> target_role

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
        device_resource = device
        if device == "CPU":
            return {
                "CPU": cpu,
                "memory": mem * 1024**3,  # convert gb to bytes
            }
        return {
            "CPU": cpu,
            device_resource: float(gpu),
            "memory": mem * 1024**3,  # convert gb to bytes
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

        res = {
            "num_cpus": cpu,
            "memory": mem * 1024**3,
        }
        if device == "CPU":
            return res

        # Use 0.9 GPUs to allow forked workers
        if device == "GPU":
            res["num_gpus"] = float(gpu) * 0.9
            return res

        return {
            "num_cpus": cpu,
            "resources": {device: float(gpu) * 0.9},
            "memory": mem * 1024**3,
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

    def _create_placement_group(self, role: str, bundles: list[dict]) -> PlacementGroup:
        """Helper to create and wait for a placement group."""
        pg = placement_group(bundles=bundles, strategy="PACK")
        try:
            ray.get(pg.ready(), timeout=self.startup_timeout)
        except ray.exceptions.GetTimeoutError:
            logger.error(
                f"Ray placement group timeout for role {role}\n"
                f"ray.nodes(): {ray.nodes()}"
                f"bundles: {bundles}"
            )
            raise
        self._placement_groups.append(pg)
        return pg

    def _build_env_vars(self, spec: SchedulingSpec) -> dict[str, str]:
        """Helper to build environment variables for a worker."""
        additional_envs_str = None
        if spec.env_vars:
            additional_envs_str = ",".join(f"{k}={v}" for k, v in spec.env_vars.items())
        env = get_env_vars(additional_envs_str)
        thread_env = get_thread_env_vars(
            cpus_per_task=spec.cpu,
            existing_env_vars=spec.env_vars,
        )
        env.update(thread_env)
        return env

    def _create_ray_workers(
        self, role: str, schedulings: list[SchedulingSpec]
    ) -> tuple[list[RayWorkerInfo], list[str]]:
        """Create Ray workers with individual placement groups per worker.

        Each worker gets its own placement group with exclusive GPU access.
        This ensures proper GPU isolation and supports forked workers sharing
        the same PG/GPU.
        """
        worker_info_list: list[RayWorkerInfo] = []
        worker_ids: list[str] = []

        # Create one PG per worker with explicit bundle_index=0
        placement_groups = []
        bundle_indices: list[int] = []
        for spec in schedulings:
            bundles = [self._bundle_spec(spec.cpu, spec.gpu, spec.mem)]
            pg = self._create_placement_group(role, bundles)
            placement_groups.append(pg)
            bundle_indices.append(0)  # Always use bundle_index=0

        master_ip, master_port = get_placement_group_master_ip_and_port(
            placement_groups[0], placement_group_bundle_index=0
        )

        for idx, (spec, pg, bundle_idx) in enumerate(
            zip(schedulings, placement_groups, bundle_indices)
        ):
            worker_id = f"{role}/{idx}"
            env = self._build_env_vars(spec)
            options = self._actor_resource_spec(spec.cpu, spec.gpu, spec.mem)

            # Build scheduling strategy with explicit bundle index
            strategy_kwargs: dict[str, Any] = {
                "placement_group": pg,
                "placement_group_capture_child_tasks": True,
                "placement_group_bundle_index": bundle_idx,  # Always 0
            }

            actor = RayRPCServer.options(
                **options,
                name=worker_id,
                runtime_env=RuntimeEnv(env_vars=env),
                scheduling_strategy=PlacementGroupSchedulingStrategy(**strategy_kwargs),
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
                bundle_index=bundle_idx,
                created_at=time.time(),
                env_vars=env,
            )
            worker_info_list.append(wi)
            worker_ids.append(worker_id)

        return worker_info_list, worker_ids

    def _create_forked_workers_internal(
        self,
        role: str,
        target_role: str,
        target_workers: list[RayWorkerInfo],
        schedulings,
    ) -> list[str]:
        """Create forked workers on same placement groups as target workers.

        Since each target worker has its own PG with bundle_index=0, forked workers
        share the exact same GPU by using the same PG and bundle_index=0.

        Main workers use num_gpus=0.9, leaving 0.1 for forked workers.
        Using num_gpus=0.01 allows up to 10 forked workers per target if needed.

        Parameters
        ----------
        role : str
            Role name for the forked workers
        target_role : str
            Target role to fork from
        target_workers : list[RayWorkerInfo]
            List of target worker infos to fork from
        schedulings : list[SchedulingSpec]
            Scheduling specs for the forked workers

        Returns
        -------
        list[str]
            List of forked worker IDs
        """
        worker_info_list: list[RayWorkerInfo] = []
        worker_ids: list[str] = []

        for idx, (target_wi, spec) in enumerate(zip(target_workers, schedulings)):
            worker_id = f"{role}/{idx}"

            # Reuse placement group from target worker
            pg = target_wi.placement_group
            bundle_idx = target_wi.bundle_index  # Should always be 0 now

            # Build scheduling strategy for same placement group
            strategy_kwargs: dict[str, Any] = {
                "placement_group": pg,
                "placement_group_capture_child_tasks": True,
                "placement_group_bundle_index": bundle_idx,  # Same as target (0)
            }

            # Use 0.01 GPU to share with target worker (which uses 0.9)
            # This allows multiple forked workers per target if needed
            device = ray_resource_type()
            additional_options = {}
            if spec.gpu > 0:
                if device == "GPU":
                    additional_options = dict(num_gpus=0.01)
                else:
                    additional_options = {"resources": {device: 0.01}}
            actor = RayRPCServer.options(
                **additional_options,
                num_cpus=0,  # Minimal CPU allocation for forked actor
                name=worker_id,
                runtime_env=RuntimeEnv(env_vars=target_wi.env_vars),
                scheduling_strategy=PlacementGroupSchedulingStrategy(**strategy_kwargs),
            ).remote()

            # Build Worker object with same IP/ports as target
            worker_ports = ray.get(
                target_wi.actor.alloc_ports.remote(
                    count=len(target_wi.worker.worker_ports)
                )
            )

            worker = Worker(
                id=worker_id,
                ip=target_wi.worker.ip,
                worker_ports=worker_ports,
                engine_ports=[],
            )

            wi = RayWorkerInfo(
                worker=worker,
                actor=actor,
                role=role,
                placement_group=pg,  # Same PG as target
                bundle_index=bundle_idx,
                created_at=time.time(),
                env_vars=target_wi.env_vars.copy(),
            )
            worker_info_list.append(wi)
            worker_ids.append(worker_id)

        # Register forked workers
        self._workers[role] = worker_info_list
        for wi in worker_info_list:
            self._worker_info_by_id[wi.worker.id] = wi

        # Ping forked workers to ensure they're ready
        self._ping_workers(role, self.startup_timeout)

        # Configure if exp_config available
        if self.exp_config is not None:
            for rank, wi in enumerate(worker_info_list):
                try:
                    wi.actor.configure.remote(self.exp_config, wi.role, rank)
                except Exception as e:
                    logger.error(
                        f"Configure failed on forked worker {wi.worker.id}: {e}",
                        exc_info=True,
                    )
                    self._cleanup_forked_workers(worker_info_list)
                    raise WorkerCreationError(
                        role, "Forked worker configuration failed", str(e)
                    )

        logger.info(
            f"Role '{role}' forked from '{target_role}': "
            f"created {len(worker_ids)} new actors on same placement groups"
        )

        return worker_ids

    def _cleanup_forked_workers(self, workers: list[RayWorkerInfo]):
        """Clean up forked workers without removing placement groups.

        Unlike _cleanup_workers, this doesn't remove placement groups since
        forked workers share placement groups with target workers.
        """
        for wi in workers:
            actor = wi.actor
            try:
                actor.destroy.remote()
            except Exception:
                logger.warning(
                    f"Could not destroy forked actor {actor}, force killing actor"
                )
                ray.kill(actor, no_restart=True)
            # Remove from worker_info_by_id
            self._worker_info_by_id.pop(wi.worker.id, None)

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
        if role in self._workers or role in self._colocated_roles:
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
        strategy_type = SchedulingStrategyType(strategy.type)
        colocate_role = strategy.target
        logger.info(
            f"Creating {num_workers} workers for role '{role}' "
            f"(strategy: {strategy_type}, colocate_with: {colocate_role})"
        )

        # Handle colocation: reuse existing workers from target role
        if strategy_type == SchedulingStrategyType.colocation:
            if not colocate_role:
                raise WorkerCreationError(
                    role,
                    "Invalid strategy",
                    "Colocation strategy requires target role to be specified",
                )
            if colocate_role not in self._workers:
                raise WorkerNotFoundError(
                    f"Cannot colocate with role '{colocate_role}' - role not found"
                )

            target_workers = self._workers[colocate_role]
            if num_workers != len(target_workers):
                raise WorkerCreationError(
                    role,
                    "Replica count mismatch",
                    f"Colocated role must have same replica count as target "
                    f"({num_workers} != {len(target_workers)})",
                )

            # Check if fork mode is enabled
            if strategy.fork:
                # Fork mode: spawn new actors on same placement groups
                worker_ids = self._create_forked_workers_internal(
                    role, colocate_role, target_workers, schedulings
                )
                self._colocated_roles[role] = colocate_role
                return worker_ids

            # Reuse existing workers - no new actors spawned
            worker_ids = [w.worker.id for w in target_workers]
            self._colocated_roles[role] = colocate_role

            logger.info(
                f"Role '{role}' colocated with '{colocate_role}': "
                f"reusing workers {worker_ids}"
            )
            return worker_ids

        if strategy_type != SchedulingStrategyType.separation:
            raise ValueError(f"Unknown scheduling strategy type: {strategy_type}")
        # Non-colocated: spawn new worker actors
        worker_info_list, worker_ids = self._create_ray_workers(role, schedulings)

        self._workers[role].extend(worker_info_list)

        for wi in worker_info_list:
            self._worker_info_by_id[wi.worker.id] = wi

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
        # Check if this is a colocated role
        if role in self._colocated_roles:
            # If forked role (has its own workers), use those
            if role in self._workers:
                worker_info_list = self._workers[role]
                self._ping_workers(role, timeout)
                return [wi.worker for wi in worker_info_list]
            # Otherwise delegate to target role
            target_role = self._colocated_roles[role]
            return self.get_workers(target_role, timeout)

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
            # Delete colocated roles first (they're just mappings)
            colocated_roles = list(self._colocated_roles.keys())
            for r in colocated_roles:
                self.delete_workers(r)
            # Then delete actual worker roles
            roles = list(self._workers.keys())
            for r in roles:
                self.delete_workers(r)
            return

        # Handle colocated role
        if role in self._colocated_roles:
            # Check if this is a forked role (has its own workers)
            if role in self._workers:
                # Forked role: clean up the spawned actors (but not placement groups)
                workers = self._workers[role]
                logger.info(
                    f"Cleaning up {len(workers)} forked actors for role '{role}'"
                )
                self._cleanup_forked_workers(workers)
                del self._workers[role]
            else:
                logger.info(f"Removing colocated role '{role}' mapping")
            # Remove colocated mapping
            del self._colocated_roles[role]
            return

        if role not in self._workers:
            logger.warning(f"Worker role '{role}' not found, skipping deletion")
            return

        workers = self._workers[role]
        logger.info(f"Deleting {len(workers)} workers for role '{role}'")

        self._cleanup_workers(workers)

        del self._workers[role]

        logger.info(f"Successfully deleted workers for role '{role}'")

    def fork_workers(
        self,
        role: str,
        target_role: str,
        command: str | None = None,
    ) -> list[str]:
        """Fork new worker processes from existing workers.

        Creates new Ray actors colocated with existing workers of the target role.
        The forked workers share the same placement groups as their target workers.

        Note: The `command` parameter is ignored for RayScheduler since Ray actors
        always run the RayRPCServer. For custom module behavior, use LocalScheduler.

        Parameters
        ----------
        role : str
            Role name for the new forked workers (e.g., "proxy")
        target_role : str
            Role of existing workers to fork from (e.g., "rollout")
        command : str, optional
            Custom module path (ignored for Ray - Ray actors always run RayRPCServer)

        Returns
        -------
        list[str]
            List of worker IDs created (e.g., ["proxy/0", "proxy/1"])
        """
        if command is not None:
            logger.warning(
                f"RayScheduler.fork_workers: 'command' parameter is ignored. "
                f"Ray actors always use RayRPCServer. Got command='{command}'"
            )

        if target_role not in self._workers:
            raise WorkerNotFoundError(f"Target role '{target_role}' not found for fork")
        target_workers = self._workers[target_role]

        schedulings = []
        for target_wi in target_workers:
            # Use minimal resources for forked workers
            schedulings.append(SchedulingSpec(cpu=0, mem=0, gpu=1, port_count=1))

        worker_ids = self._create_forked_workers(
            role, target_role, target_workers, schedulings
        )
        self._colocated_roles[role] = target_role
        return worker_ids

    def _cleanup_workers(self, workers: list[RayWorkerInfo]):
        # Kill actors first
        for wi in workers:
            actor = wi.actor
            try:
                # Asynchronously destroy actor
                actor.destroy.remote()
            except Exception:
                logger.warning(
                    f"Could not destroy remote actor {actor}, force killing actor"
                )
                ray.kill(actor, no_restart=True)

        # Collect unique placement groups and remove them
        unique_pgs = {wi.placement_group for wi in workers}
        for pg in unique_pgs:
            try:
                remove_placement_group(pg)
            except Exception:
                logger.warning(f"Could not remove placement group {pg}")
            if pg in self._placement_groups:
                self._placement_groups.remove(pg)

    def _get_worker_info_by_id(self, worker_id: str) -> RayWorkerInfo | None:
        return self._worker_info_by_id.get(worker_id, None)

    async def set_worker_env(self, worker_id: str, env: dict[str, str]) -> None:
        wi = self._get_worker_info_by_id(worker_id)
        if wi is None:
            raise WorkerNotFoundError(worker_id)
        if not env:
            return

        await wi.actor.set_env.remote(env)
        wi.env_vars.update(env)

    async def create_engine(
        self,
        worker_id: str,
        engine: str,
        engine_name: str | None = None,
        *args,
        **kwargs,
    ) -> Any:
        wi = self._get_worker_info_by_id(worker_id)
        if wi is None:
            raise WorkerNotFoundError(worker_id)

        if not isinstance(engine, str):
            raise WorkerCreationError(
                worker_id, f"Engine must be a string import path, got {type(engine)}"
            )
        # Pass engine_name to support multiple engines per worker (colocation)
        await wi.actor.create_engine.remote(
            engine, *args, engine_name=engine_name, **kwargs
        )

    def call_engine(
        self,
        worker_id: str,
        method: str,
        engine_name: str | None = None,
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
                # Pass engine_name to support multiple engines per worker (colocation)
                ref = wi.actor.call.remote(
                    method, *args, engine_name=engine_name, **kwargs
                )
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
        engine_name: str | None = None,
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
                # Pass engine_name to support multiple engines per worker (colocation)
                ref = wi.actor.call.remote(
                    method, *args, engine_name=engine_name, **kwargs
                )
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
        # delete in case delete_workers is not called from controllers
        # explicit shutdown is by directly calling delete_workers
        try:
            self.delete_workers()
        except Exception:
            pass
