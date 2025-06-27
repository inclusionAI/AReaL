# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import argparse
import os
from pathlib import Path

from hydra import compose as hydra_compose
from hydra import initialize as hydra_init
from omegaconf import OmegaConf

from arealite.api.cli_args import TrainingArgs
from arealite.api.io_struct import AllocationMode
from arealite.api.llm_server_api import LLMServiceRegistry
from realhf.base import constants, logging, name_resolve, names
from realhf.scheduler.client import JobException, JobState
from realhf.scheduler.client import make as make_scheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="The path of the main configuration file", required=True
    )
    args, overrides = parser.parse_known_args()

    # Initialize hydra config
    config_file = Path(args.config).absolute()
    assert config_file.exists()
    relpath = Path(os.path.relpath(str(config_file), Path(__file__).parent.absolute()))
    hydra_init(config_path=str(relpath.parent), job_name="app", version_base=None)
    cfg = hydra_compose(
        config_name=str(relpath.name).rstrip(".yaml"),
        overrides=overrides,
    )

    # Merge with the default configuration
    default_cfg = OmegaConf.structured(TrainingArgs)
    cfg = OmegaConf.merge(default_cfg, cfg)
    cfg: TrainingArgs = OmegaConf.to_object(cfg)

    constants.set_experiment_trial_names(cfg.experiment_name, cfg.trial_name)
    # NOTE: do not reset the name resolve repo to discover previous LLM servers
    name_resolve.reconfigure(cfg.cluster.name_resolve)

    # Launch inference and training jobs
    alloc_mode = AllocationMode.from_str(cfg.allocation_mode)
    assert cfg.mode == "local"
    scheduler = make_scheduler(cfg)
    jobs = []
    BASE_ENVIRONS = constants.get_env_vars(cfg)
    for k, v in BASE_ENVIRONS.items():
        os.environ[k] = v

    # discover existing servers
    existing_servers = LLMServiceRegistry(
        cfg.experiment_name, cfg.trial_name
    ).get_healthy_servers()
    # Launch LLM servers.
    if len(existing_servers) < alloc_mode.gen_dp_size:
        n_gpus_per_instance = alloc_mode.gen_pp_size * alloc_mode.gen_tp_size
        jobs += scheduler.submit_array(
            worker_type="llm_server",
            cmd=f"python3 arealite/cli/launch_server.py --config {str(config_file)}",
            count=len(existing_servers) - alloc_mode.gen_dp_size,
            cpu=cfg.cpu_per_inf_proc * n_gpus_per_instance,
            gpu=n_gpus_per_instance,
            mem=cfg.mem_per_inf_proc * n_gpus_per_instance,
            env_vars=BASE_ENVIRONS,
            container_image=cfg.cluster.gpu_infer_image,
        )
    # Launch trainers.
    jobs += scheduler.submit(
        worker_type="trainer",
        cmd=f"torchrun --nnodes 1 --nproc-per-node {alloc_mode.train_world_size} arealite/cli/launch_trainer.py --config {str(config_file)}",
        count=1,
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
    except (KeyboardInterrupt, JobException, TimeoutError) as e:
        kill_signal = (
            "SIGKILL" if cfg.mode == "slurm" else "SIGTERM"
        )  # use sigkill to terminate slurm jobs
        if cfg.shutdown_server_on_exit:
            scheduler.stop_all(kill_signal)
        else:
            scheduler.stop("trainer")


if __name__ == "__main__":
    main()
