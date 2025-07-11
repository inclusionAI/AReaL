# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import os
import sys

from arealite.api.cli_args import prepare_training_args
from arealite.api.io_struct import AllocationMode
from arealite.api.llm_server_api import LLMServiceRegistry
from realhf.base import constants, name_resolve, names
from realhf.scheduler.client import JobException, JobState
from realhf.scheduler.client import make as make_scheduler


def main():
    cfg, config_file = prepare_training_args(sys.argv[1:])
    if cfg.shutdown_server_on_exit:
        name_resolve.clear_subtree(
            names.trial_root(
                experiment_name=cfg.experiment_name, trial_name=cfg.trial_name
            )
        )

    # Launch inference and training jobs
    alloc_mode = AllocationMode.from_str(cfg.allocation_mode)
    assert cfg.mode == "local"
    scheduler = make_scheduler(cfg)
    BASE_ENVIRONS = constants.get_env_vars(cfg)
    for k, v in BASE_ENVIRONS.items():
        os.environ[k] = v

    # discover existing servers
    existing_servers = LLMServiceRegistry(
        cfg.experiment_name, cfg.trial_name
    ).get_healthy_servers()
    # Launch LLM servers.
    if len(existing_servers) == 0:
        n_gpus_per_instance = alloc_mode.gen_pp_size * alloc_mode.gen_tp_size
        servers_to_launch = alloc_mode.gen_dp_size - len(existing_servers)
        scheduler.submit_array(
            worker_type="llm_server",
            cmd=f"python3 arealite/cli/launch_server.py --config {str(config_file)}",
            count=servers_to_launch,
            cpu=cfg.cpu_per_inf_proc * n_gpus_per_instance,
            gpu=n_gpus_per_instance,
            mem=cfg.mem_per_inf_proc * n_gpus_per_instance,
            env_vars=BASE_ENVIRONS,
            container_image=cfg.cluster.gpu_infer_image,
        )
    # Launch trainers.
    scheduler.submit(
        worker_type="trainer",
        cmd=f"torchrun --nnodes 1 --nproc-per-node {alloc_mode.train_world_size} arealite/cli/launch_trainer.py --config {str(config_file)}",
        cpu=cfg.cpu_per_train_proc * alloc_mode.train_world_size,
        gpu=alloc_mode.train_world_size,
        mem=cfg.cpu_per_train_proc * cfg.mem_per_train_proc,
        container_image=cfg.cluster.gpu_image,
        nodelist=cfg.nodelist,
        exclude=cfg.exclude,
        env_vars=BASE_ENVIRONS,
        hostfile=False,
        multiprog=False,
    )

    # Waiting for the job.
    try:
        scheduler.wait(
            check_status=(
                JobState.CANCELLED,
                JobState.FAILED,
                JobState.NOT_FOUND,
                JobState.COMPLETED,
            ),
            remove_status=(),
        )
    except (KeyboardInterrupt, JobException, TimeoutError):
        kill_signal = (
            "SIGKILL" if cfg.mode == "slurm" else "SIGTERM"
        )  # use sigkill to terminate slurm jobs
        if cfg.shutdown_server_on_exit:
            scheduler.stop_all(kill_signal)
        else:
            scheduler.stop("trainer")


if __name__ == "__main__":
    main()
