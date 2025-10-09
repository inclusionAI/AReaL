"""Launch AReaL experiments on SkyPilot-managed clusters.

This launcher mirrors the semantics of the Ray and Slurm launchers while
delegating provisioning and task execution to the SkyPilot Python SDK.
"""

from __future__ import annotations

import re
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import yaml

from areal.api.alloc_mode import AllocationMode, AllocationType
from areal.api.cli_args import (
    ClusterSpecConfig,
    LauncherConfig,
    RecoverConfig,
    SGLangConfig,
    SkyPilotLauncherConfig,
    parse_cli_args,
    to_structured_cfg,
    vLLMConfig,
)
from areal.utils import logging, name_resolve, names
from areal.utils.launcher import (
    JobException,
    JobInfo,
    JobState,
    get_env_vars,
    validate_config_for_distributed_launcher,
    wait_llm_server_addrs,
)
from areal.utils.recover import check_if_recover

logger = logging.getLogger("SkyPilotLauncher")


try:
    import sky
    from sky import JobStatus as SkyJobStatus
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "SkyPilot launcher requires the `skypilot` package. "
        "Install it via `pip install -U skypilot`."
    ) from exc


SKY_TO_JOB_STATE: Dict[SkyJobStatus, JobState] = {
    SkyJobStatus.INIT: JobState.PENDING,
    SkyJobStatus.PENDING: JobState.PENDING,
    SkyJobStatus.SETTING_UP: JobState.PENDING,
    SkyJobStatus.RUNNING: JobState.RUNNING,
    SkyJobStatus.SUCCEEDED: JobState.COMPLETED,
    SkyJobStatus.FAILED: JobState.FAILED,
    SkyJobStatus.FAILED_SETUP: JobState.FAILED,
    SkyJobStatus.FAILED_DRIVER: JobState.FAILED,
    SkyJobStatus.CANCELLED: JobState.CANCELLED,
}


SKY_WAIT_CHECK_TIME_INTERVAL = 5  # seconds


def _readable_cluster_name(experiment_name: str, trial_name: str) -> str:
    slug = f"areal-{experiment_name}-{trial_name}"
    return re.sub(r"[^a-zA-Z0-9-]", "-", slug).lower()


def _parse_key_value_pairs(value: Optional[str]) -> Dict[str, str]:
    if not value:
        return {}
    result = {}
    for chunk in value.split(","):
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"Environment/secret entry '{chunk}' must be in KEY=VALUE format."
            )
        key, val = chunk.split("=", 1)
        result[key.strip()] = val.strip()
    return result


def _parse_yaml_like(value: Optional[str]) -> Any:
    if not value:
        return None
    return yaml.safe_load(value)


def _default_workdir(skypilot_cfg: SkyPilotLauncherConfig) -> str:
    if skypilot_cfg.workdir:
        return skypilot_cfg.workdir
    return str(Path.cwd())


RunSpec = Union[str, Callable[[int, List[str]], str]]


class SkyPilotLauncher:
    def __init__(
        self,
        experiment_name: str,
        trial_name: str,
        total_nodes: int,
        skypilot_cfg: SkyPilotLauncherConfig,
    ):
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.total_nodes = total_nodes
        self.skypilot_cfg = skypilot_cfg
        self.cluster_name = skypilot_cfg.name or _readable_cluster_name(
            experiment_name, trial_name
        )
        self._cluster_ready = False
        self.jobs: Dict[int, JobInfo] = {}
        self._job_meta: Dict[int, Dict[str, Any]] = {}
        self._job_groups: Dict[str, set[int]] = {}
        self.ensure_cluster()

    @staticmethod
    def _build_resources(cfg: SkyPilotLauncherConfig) -> sky.Resources:
        kwargs: Dict[str, Any] = {}
        if cfg.infra:
            kwargs["infra"] = cfg.infra
        if cfg.accelerators:
            kwargs["accelerators"] = cfg.accelerators
        if cfg.accelerator_args:
            kwargs["accelerator_args"] = _parse_yaml_like(cfg.accelerator_args)
        if cfg.cpus:
            kwargs["cpus"] = cfg.cpus
        if cfg.memory:
            kwargs["memory"] = cfg.memory
        if cfg.instance_type:
            kwargs["instance_type"] = cfg.instance_type
        if cfg.use_spot is not None:
            kwargs["use_spot"] = cfg.use_spot
        if cfg.disk_size:
            kwargs["disk_size"] = cfg.disk_size
        if cfg.disk_tier:
            kwargs["disk_tier"] = cfg.disk_tier
        if cfg.network_tier:
            kwargs["network_tier"] = cfg.network_tier
        if cfg.ports:
            kwargs["ports"] = cfg.ports
        if cfg.image_id:
            kwargs["image_id"] = _parse_yaml_like(cfg.image_id) or cfg.image_id
        if cfg.labels:
            kwargs["labels"] = _parse_key_value_pairs(cfg.labels)
        if cfg.any_of:
            kwargs["any_of"] = _parse_yaml_like(cfg.any_of)
        if cfg.ordered:
            kwargs["ordered"] = _parse_yaml_like(cfg.ordered)
        if cfg.job_recovery:
            kwargs["job_recovery"] = _parse_yaml_like(cfg.job_recovery)
        if cfg.autostop:
            kwargs["autostop"] = _parse_yaml_like(cfg.autostop) or cfg.autostop
        if cfg.volumes:
            kwargs["volumes"] = _parse_yaml_like(cfg.volumes)
        return sky.Resources(**kwargs)

    def _base_task(
        self,
        name: str,
        num_nodes: int,
        run: RunSpec,
        extra_envs: Optional[Dict[str, str]] = None,
    ) -> sky.Task:
        base_envs = _parse_key_value_pairs(self.skypilot_cfg.envs)
        secrets = _parse_key_value_pairs(self.skypilot_cfg.secrets)
        if secrets:
            base_envs.update(secrets)
        if extra_envs:
            base_envs.update(extra_envs)
        workdir = _default_workdir(self.skypilot_cfg)
        file_mounts = None
        if self.skypilot_cfg.file_mounts:
            file_mounts = _parse_yaml_like(self.skypilot_cfg.file_mounts)
        resources = self._build_resources(self.skypilot_cfg)
        task_kwargs: Dict[str, Any] = {
            "name": name,
            "num_nodes": num_nodes,
            "run": run,
            "workdir": workdir,
        }
        if base_envs:
            task_kwargs["envs"] = base_envs
        if file_mounts:
            task_kwargs["file_mounts"] = file_mounts
        task = sky.Task(**task_kwargs)
        task.set_resources(resources)
        return task

    def ensure_cluster(self) -> None:
        if self._cluster_ready:
            return
        provision_task = self._base_task(
            name=f"{self.cluster_name}-provision",
            num_nodes=self.total_nodes,
            run="echo '[SkyPilot] Cluster ready for AReaL launching.'",  # noqa: E501
        )
        logger.info(
            "Launching/repairing SkyPilot cluster '%s' with %d node(s).",
            self.cluster_name,
            self.total_nodes,
        )
        req_id = sky.launch(provision_task, cluster_name=self.cluster_name)
        sky.stream_and_get(req_id)
        self._cluster_ready = True

    @property
    def run_name(self) -> str:
        return f"{self.experiment_name}_{self.trial_name}"

    def submit(self, job_name: str, task: sky.Task) -> int:
        job_ids = self.submit_array(job_name, [task])
        return job_ids[0]

    def submit_array(self, job_name: str, tasks: List[sky.Task]) -> List[int]:
        assert tasks, "Tasks list cannot be empty."
        job_ids: List[int] = []
        for idx, task in enumerate(tasks):
            derived_name = job_name if len(tasks) == 1 else f"{job_name}:{idx}"
            task.name = derived_name
            logger.info("Submitting SkyPilot task '%s'", derived_name)
            request_id = sky.exec(task, cluster_name=self.cluster_name)
            job_id, _ = sky.get(request_id)
            self._register_job(job_id, derived_name)
            job_ids.append(job_id)
        return job_ids

    def stop(self, job_name: str, force: bool = False) -> None:
        job_ids = list(self._job_groups.get(job_name, set()))
        if not job_ids:
            return
        logger.info("Stopping jobs %s (ids=%s)", job_name, job_ids)
        sky.cancel(self.cluster_name, job_ids=job_ids)
        for job_id in job_ids:
            self._remove_job(job_id)

    def stop_all(self, force: bool = False) -> None:
        job_ids = list(self.jobs.keys())
        if not job_ids:
            return
        logger.info("Stopping all SkyPilot jobs: %s", job_ids)
        sky.cancel(self.cluster_name, job_ids=job_ids)
        for job_id in job_ids:
            self._remove_job(job_id)

    def find(self, job_name: str) -> Optional[JobInfo]:
        self._update_all()
        job_ids = list(self._job_groups.get(job_name, set()))
        if not job_ids:
            return None
        return self.jobs[job_ids[0]]

    def find_all(self, job_name_regex: str = ".*") -> List[JobInfo]:
        self._update_all()
        pattern = re.compile(job_name_regex)
        results: List[JobInfo] = []
        for job_id, info in self.jobs.items():
            base = self._job_meta[job_id]["base"]
            if pattern.fullmatch(base):
                results.append(info)
        return results

    def wait(
        self,
        timeout: Optional[int] = None,
        check_status: Tuple[JobState, ...] = (
            JobState.CANCELLED,
            JobState.FAILED,
            JobState.NOT_FOUND,
        ),
        remove_status: Tuple[JobState, ...] = (JobState.COMPLETED,),
        update: bool = False,
        job_names: Optional[Iterable[str]] = None,
    ) -> None:
        deadline = None if timeout is None else time.time() + timeout
        target_ids = self._select_job_ids(job_names)
        pending = set(target_ids)
        if not pending:
            return
        logger.info("Waiting for jobs %s", sorted(pending))
        while pending:
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(
                    f"Timeout waiting for jobs {sorted(pending)} to finish."
                )
            self._update_all()
            for job_id in list(pending):
                info = self.jobs.get(job_id)
                if info is None:
                    pending.discard(job_id)
                    continue
                state = info.state
                base = self._job_meta[job_id]["base"]
                if state in check_status:
                    raise JobException(
                        run_name=self.run_name,
                        worker_type=base,
                        host=self.cluster_name,
                        reason=state,
                    )
                if state in remove_status:
                    logger.info(
                        "Job %s (id=%s) reached %s", info.name, job_id, state.name
                    )
                    pending.discard(job_id)
                    if update:
                        self._remove_job(job_id)
            if pending:
                time.sleep(SKY_WAIT_CHECK_TIME_INTERVAL)

    def _register_job(self, job_id: int, job_name: str) -> None:
        base = job_name.split(":", maxsplit=1)[0]
        self.jobs[job_id] = JobInfo(
            name=job_name,
            state=JobState.PENDING,
            host=self.cluster_name,
        )
        self._job_meta[job_id] = {"name": job_name, "base": base}
        self._job_groups.setdefault(base, set()).add(job_id)

    def _remove_job(self, job_id: int) -> None:
        info = self._job_meta.pop(job_id, None)
        self.jobs.pop(job_id, None)
        if info is None:
            return
        base = info["base"]
        group = self._job_groups.get(base)
        if group and job_id in group:
            group.remove(job_id)
            if not group:
                self._job_groups.pop(base, None)

    def _select_job_ids(self, job_names: Optional[Iterable[str]]) -> List[int]:
        if job_names is None:
            return list(self.jobs.keys())
        selected: List[int] = []
        for base in job_names:
            selected.extend(list(self._job_groups.get(base, set())))
        return selected

    def _update_all(self) -> None:
        if not self.jobs:
            return
        job_ids = list(self.jobs.keys())
        try:
            status_request = sky.job_status(self.cluster_name, job_ids=job_ids)
            statuses = sky.get(status_request)
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.warning("Failed to query SkyPilot job status: %s", exc)
            return
        for job_id in job_ids:
            info = self.jobs.get(job_id)
            if info is None:
                continue
            status = statuses.get(job_id)
            if status is None:
                info.state = JobState.NOT_FOUND
            else:
                info.state = SKY_TO_JOB_STATE.get(status, JobState.NOT_FOUND)


def _quoted_cmd(parts: List[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def _build_sglang_task(
    launcher: SkyPilotLauncher,
    job_name: str,
    config_args: List[str],
    allocation: AllocationMode,
    sglang_cfg: SGLangConfig,
    n_nodes: int,
    gpus_per_node: int,
    env_vars: Dict[str, str],
) -> sky.Task:
    assert allocation.gen_backend == "sglang"
    base_seed = sglang_cfg.random_seed
    n_sglang_servers = allocation.gen.dp_size
    n_servers_per_node = max(n_sglang_servers // max(n_nodes, 1), 1)
    cross_nodes = allocation.gen_instance_size > gpus_per_node

    def run_generator(node_rank: int, host_ips: List[str]) -> str:
        args = list(config_args)
        args.append(f"sglang.random_seed={base_seed + node_rank * n_servers_per_node}")
        cmd = _quoted_cmd(["python", "-m", "areal.launcher.sglang_server", *args])
        exports: List[str] = []
        if cross_nodes:
            exports.append(f"export AREAL_SGLANG_MULTI_NODE_RANK={node_rank}")
            exports.append(f"export AREAL_SGLANG_MULTI_NODE_MASTER_ADDR={host_ips[0]}")
            # TODO: find free port
            exports.append("export AREAL_SGLANG_MULTI_NODE_MASTER_PORT=17901")
        if exports:
            return " && ".join(exports + [cmd])
        return cmd

    return launcher._base_task(  # pylint: disable=protected-access
        name=job_name,
        num_nodes=n_nodes,
        run=run_generator,
        extra_envs=env_vars,
    )


def _build_vllm_task(
    launcher: SkyPilotLauncher,
    job_name: str,
    config_args: List[str],
    allocation: AllocationMode,
    vllm_cfg: vLLMConfig,
    n_nodes: int,
    env_vars: Dict[str, str],
) -> sky.Task:
    assert allocation.gen_backend == "vllm"
    base_seed = vllm_cfg.seed

    def run_generator(node_rank: int, _host_ips: List[str]) -> str:
        args = list(config_args)
        args.append(f"vllm.seed={base_seed + node_rank}")
        cmd = _quoted_cmd(["python", "-m", "areal.launcher.vllm_server", *args])
        return cmd

    return launcher._base_task(  # pylint: disable=protected-access
        name=job_name,
        num_nodes=n_nodes,
        run=run_generator,
        extra_envs=env_vars,
    )


def _build_trainer_task(
    launcher: SkyPilotLauncher,
    job_name: str,
    trainer_entry: str,
    trainer_args: List[str],
    allocation: AllocationMode,
    n_nodes: int,
    gpus_per_node: int,
    env_vars: Dict[str, str],
    is_eval_only: bool,
) -> sky.Task:

    if is_eval_only:

        cmd = _quoted_cmd(["python", trainer_entry, *trainer_args])
        return launcher._base_task(
            job_name,
            num_nodes=1,
            run=cmd,
            extra_envs=env_vars,
        )

    # TODO: find free port
    rendezvous_port = 29501

    def run_generator(node_rank: int, host_ips: List[str]) -> str:
        master_addr = host_ips[0]
        torchrun_cmd = [
            "torchrun",
            "--nnodes",
            str(n_nodes),
            "--nproc-per-node",
            str(gpus_per_node),
            "--rdzv_backend",
            "c10d",
            "--rdzv_endpoint",
            f"{master_addr}:{rendezvous_port}",
            "--node_rank",
            str(node_rank),
            trainer_entry,
            *trainer_args,
        ]
        cmd = _quoted_cmd(torchrun_cmd)
        return cmd

    return launcher._base_task(
        job_name,
        num_nodes=n_nodes,
        run=run_generator,
        extra_envs=env_vars,
    )


def skypilot_main(config, run_id: int = 0):
    config.launcher = to_structured_cfg(config.launcher, LauncherConfig)
    config.recover = to_structured_cfg(config.recover, RecoverConfig)
    config.cluster = to_structured_cfg(config.cluster, ClusterSpecConfig)
    config.launcher.skypilot = to_structured_cfg(
        config.launcher.skypilot, SkyPilotLauncherConfig
    )

    is_recover_run = check_if_recover(config.recover, run_id)
    validate_config_for_distributed_launcher(config)

    name_resolve.reconfigure(config.cluster.name_resolve)
    name_resolve.clear_subtree(
        names.trial_root(
            experiment_name=config.experiment_name, trial_name=config.trial_name
        )
    )

    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    launcher = SkyPilotLauncher(
        experiment_name=config.experiment_name,
        trial_name=config.trial_name,
        total_nodes=config.cluster.n_nodes,
        skypilot_cfg=config.launcher.skypilot,
    )
    launcher.ensure_cluster()

    trainer_entry = sys.argv[1]
    trainer_args = sys.argv[2:]

    llm_job_name: Optional[str] = None
    llm_addrs: List[str] = []

    try:
        gpus_per_node = config.cluster.n_gpus_per_node
        llm_backend = allocation_mode.gen_backend
        if llm_backend == "sglang":
            llm_job_name = f"{launcher.cluster_name}-sglang"
            config.sglang = to_structured_cfg(config.sglang, SGLangConfig)
            n_llm_nodes = max(
                (allocation_mode.gen.world_size + gpus_per_node - 1)
                // max(gpus_per_node, 1),
                1,
            )
            llm_env = get_env_vars(
                config.cluster.cluster_name,
                config.launcher.inference_server_env_vars,
            )
            task = _build_sglang_task(
                launcher=launcher,
                job_name=llm_job_name,
                config_args=list(trainer_args),
                allocation=allocation_mode,
                sglang_cfg=config.sglang,
                n_nodes=n_llm_nodes,
                gpus_per_node=gpus_per_node,
                env_vars=llm_env,
            )
            launcher.submit(llm_job_name, task)
        elif llm_backend == "vllm":
            llm_job_name = f"{launcher.cluster_name}-vllm"
            config.vllm = to_structured_cfg(config.vllm, vLLMConfig)
            n_llm_nodes = max(
                (allocation_mode.gen.world_size + gpus_per_node - 1)
                // max(gpus_per_node, 1),
                1,
            )
            llm_env = get_env_vars(
                config.cluster.cluster_name,
                config.launcher.inference_server_env_vars,
            )
            task = _build_vllm_task(
                launcher=launcher,
                job_name=llm_job_name,
                config_args=list(trainer_args),
                allocation=allocation_mode,
                vllm_cfg=config.vllm,
                n_nodes=n_llm_nodes,
                env_vars=llm_env,
            )
            launcher.submit(llm_job_name, task)

        if llm_job_name is not None:
            llm_addrs = wait_llm_server_addrs(
                experiment_name=config.experiment_name,
                trial_name=config.trial_name,
                n_rollout_servers=allocation_mode.gen.dp_size,
            )

        if allocation_mode.type_ == AllocationType.LLM_SERVER_ONLY:
            if llm_job_name is None:
                logger.warning(
                    "Allocation mode is LLM_SERVER_ONLY but no LLM job launched."
                )
            else:
                launcher.wait(
                    job_names=[llm_job_name],
                    check_status=(
                        JobState.FAILED,
                        JobState.CANCELLED,
                        JobState.NOT_FOUND,
                    ),
                    remove_status=(JobState.COMPLETED,),
                    update=False,
                )
            return

        trainer_env = dict(
            **get_env_vars(
                config.cluster.cluster_name,
                config.launcher.trainer_env_vars,
            ),
            AREAL_RECOVER_RUN=str(int(is_recover_run)),
        )
        if llm_addrs:
            trainer_env["AREAL_LLM_SERVER_ADDRS"] = ",".join(llm_addrs)

        if allocation_mode.type_ == AllocationType.DECOUPLED_EVAL:
            trainer_nodes = 1
            gpus_per_node = 0
        else:
            trainer_nodes = max(
                config.cluster.n_nodes
                - (allocation_mode.gen.world_size // config.cluster.n_gpus_per_node),
                1,
            )
            gpus_per_node = config.cluster.n_gpus_per_node

        trainer_job_name = f"{launcher.cluster_name}-trainer"
        trainer_task = _build_trainer_task(
            launcher=launcher,
            job_name=trainer_job_name,
            trainer_entry=trainer_entry,
            trainer_args=trainer_args,
            allocation=allocation_mode,
            n_nodes=trainer_nodes,
            gpus_per_node=gpus_per_node,
            env_vars=trainer_env,
            is_eval_only=allocation_mode.type_ == AllocationType.DECOUPLED_EVAL,
        )
        launcher.submit(trainer_job_name, trainer_task)

        launcher.wait(
            job_names=[trainer_job_name],
            check_status=(
                JobState.FAILED,
                JobState.CANCELLED,
                JobState.NOT_FOUND,
            ),
            remove_status=(JobState.COMPLETED,),
            update=True,
        )
    except (KeyboardInterrupt, TimeoutError, JobException) as exc:
        logger.error("SkyPilot launcher encountered an error: %s", exc)
        launcher.stop_all()
        recoverable_states = {JobState.FAILED}
        if (
            isinstance(exc, JobException)
            and exc.reason in recoverable_states
            and run_id < config.recover.retries
            and config.recover.mode in ("auto", "fault")
        ):
            time.sleep(10)
            skypilot_main(config, run_id=run_id + 1)
        else:
            raise
    finally:
        if llm_job_name is not None:
            try:
                launcher.stop(llm_job_name)
            except Exception as cancel_exc:  # pragma: no cover
                logger.warning("Failed to cancel LLM server job: %s", cancel_exc)


def main():
    config, _ = parse_cli_args(sys.argv[1:])
    skypilot_main(config, run_id=0)


if __name__ == "__main__":
    main()
