# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import argparse
import getpass
import os
import re
import time
import uuid
from typing import Dict, List, Optional

import realhf.api.core.system_api as config_package
import realhf.scheduler.client as sched_client
import realhf.system as system
from realhf.api.core.system_api import ExpStatus
from realhf.base import constants, logging, name_resolve, names, recover
from realhf.scheduler.client import JobException, JobState
from realhf.scheduler.evaluator import AutomaticEvaluator
from realhf.version import get_full_version_with_dirty_description

logger = logging.getLogger("main", "system")

CONTROLLER_TIME_LIMIT = None


def scheduler_mode(mode: str) -> str:
    return mode if mode == "slurm" else "local"


def _submit_workers(
    sched: sched_client.SchedulerClient,
    expr_name: str,
    trial_name: str,
    debug: bool,
    worker_type: str,
    scheduling_configs: List[config_package.TasksGroup],
    environs: Dict[str, str],
) -> List[str]:
    if len(scheduling_configs) == 0:
        return []

    scheduled_jobs = []
    for sch_cfg in scheduling_configs:
        if sch_cfg is None:
            continue
        job_environs = {**environs, **sch_cfg.scheduling.env_vars}
        cmd = sched_client.remote_worker_cmd(expr_name, trial_name, debug, worker_type)

        logger.debug(f"Scheduling worker {worker_type}, {scheduling_configs}")

        nodelist = sch_cfg.scheduling.nodelist
        exclude = sch_cfg.scheduling.exclude
        container_image = sch_cfg.scheduling.container_image

        scheduled_jobs.append(
            sched.submit_array(
                worker_type=worker_type,
                cmd=cmd,
                count=sch_cfg.count,
                cpu=sch_cfg.scheduling.cpu,
                gpu=sch_cfg.scheduling.gpu,
                mem=sch_cfg.scheduling.mem,
                container_image=container_image,
                nodelist=nodelist,
                exclude=exclude,
                env_vars=job_environs,
                hostfile=True,
                multiprog=True,
                begin=sch_cfg.scheduling.begin,
                deadline=sch_cfg.scheduling.deadline,
                time_limit=sch_cfg.scheduling.time_limit,
            ),
        )
    return scheduled_jobs


def main_start(args, job_group_id: str = "", recover_count: int = 0):
    if not job_group_id:
        job_group_id = str(uuid.uuid4())
    logger.info(f"AReaL Version: {get_full_version_with_dirty_description()}")
    logger.info(f"AReaL Job Group ID: {job_group_id}")
    logger.info(f"AReaL Job Group Index (recover count): {recover_count}")
    if recover_count == 0:
        constants.set_experiment_trial_names(args.experiment_name, args.trial_name)
    experiment = config_package.make_experiment(args.experiment_name)

    # Run initial_setup to go through all sanity checks.
    try:
        exp_cfg = experiment.initial_setup()
        assert isinstance(exp_cfg, config_package.ExperimentConfig)
        exp_cfg.lazy_init()
    except Exception as e:
        raise RuntimeError("Experiment initial setup failed.") from e

    evaluator = (
        AutomaticEvaluator(exp_cfg, exp_cfg.evaluator, exp_cfg.wandb, exp_cfg.swanlab)
        if exp_cfg.auto_eval
        else None
    )

    if args.mode == "local":
        assert (
            args.recover_mode == "disabled"
        ), "Recover mode is not supported for local runs!"
    # handle args
    args.ignore_worker_error = (
        args.ignore_worker_error and args.recover_mode == "disabled"
    )
    trial_name = args.trial_name or f"test-{getpass.getuser()}"
    expr_name = args.experiment_name
    is_recover_run = False
    if args.recover_mode == "resume":
        is_recover_run = True
    if args.recover_mode == "fault":
        is_recover_run = recover_count > 0
    if args.recover_mode == "auto":
        try:
            recover.discover_ckpt(experiment)
            is_recover_run = True
        except recover.InValidRecoverCkpt as e:
            logger.warning(
                "Invalid recover checkpoint when recover_mode='auto'. "
                "Running the experiment from scratch with fault tolerance. "
                f"Err message: {e}"
            )
            is_recover_run = False
    if is_recover_run:
        recover_ckpt_path, model_ckpt_dirs, recover_info = recover.discover_ckpt(
            experiment
        )
        logger.info(f"Will load recover info from {recover_ckpt_path}.")
        logger.info(f"Will load model checkpoints from {model_ckpt_dirs}.")
        logger.info(
            f"Training will start from: epoch {recover_info.recover_start.epoch + 1} "
            f"epoch step {recover_info.recover_start.epoch_step + 1} "
            f"global step {recover_info.recover_start.global_step + 1}."
        )
    save_recover_states = args.recover_mode != "disabled"

    # set env vars
    BASE_ENVIRONS = constants.get_env_vars(
        experiment,
        REAL_MODE=args.mode.upper(),
        REAL_RECOVER_RUN="1" if is_recover_run else "0",
        REAL_SAVE_RECOVER_STATES="1" if save_recover_states else "0",
    )
    for k, v in BASE_ENVIRONS.items():
        os.environ[k] = v

    # setup experiments
    sched = sched_client.make(
        experiment,
        evaluator=evaluator,
        job_group_id=job_group_id,
        job_group_index=recover_count,
    )

    setup = experiment.scheduling_setup()

    logger.info(f"Resetting name resolving repo...")

    try:
        name_resolve.clear_subtree(
            names.trial_root(
                experiment_name=args.experiment_name, trial_name=args.trial_name
            )
        )
        # NOTE: During fault recovery, the NFS-based name resolve may encounter
        # unexpected OS errors when making or removing directories. Sleeping for
        # a while to avoid such errors.
        time.sleep(5)
    except Exception as e:
        logger.warning(f"Resetting name resolving repo failed.")
        raise e
    logger.info(f"Resetting name resolving repo... Done.")

    logger.info(
        f"Running configuration: {experiment.__class__.__name__}. "
        f"The current recover retry: {recover_count + 1}/{args.recover_retries}"
    )

    # Schedule controller
    if args.mode == "ray":
        controller_type = "ray"
    else:
        controller_type = "zmq"
    # For ray mode, the controller will start all remote workers.
    sched.submit_array(
        worker_type="ctl",
        cmd=sched_client.control_cmd(
            expr_name,
            trial_name,
            args.debug,
            args.ignore_worker_error,
            controller_type,
        ),
        count=1,
        cpu=1,
        gpu=0,
        mem=1024,
        env_vars=BASE_ENVIRONS,
        container_image=setup.controller_image,
        time_limit=CONTROLLER_TIME_LIMIT,
    )

    if args.mode != "ray":
        workers_configs = ((k, getattr(setup, k)) for k in system.WORKER_TYPES)

        for name, scheduling_setup in workers_configs:
            if not isinstance(scheduling_setup, list):
                scheduling_setup = [scheduling_setup]
            # For local or slurm mode, launch all workers.
            # For ray mode, nothing to do because workers will be
            # started by the controller, rather than the scheduler.
            _submit_workers(
                sched,
                expr_name,
                trial_name,
                args.debug,
                name,
                scheduling_setup,
                BASE_ENVIRONS,
            )

    try:
        sched.wait(
            check_status=(
                JobState.CANCELLED,
                JobState.FAILED,
                JobState.NOT_FOUND,
                JobState.COMPLETED,
            ),
            remove_status=(),
        )
    except (KeyboardInterrupt, JobException, TimeoutError) as e:
        recover_states = [
            JobState.CANCELLED,
            JobState.FAILED,
            JobState.NOT_FOUND,
        ]
        reason = e.reason if isinstance(e, JobException) else None
        recover_this = (
            args.recover_mode in ["auto", "fault"]
            and recover_count < args.recover_retries
        )
        recover_this = recover_this and reason in recover_states

        # Check whether this exception is caused by experiment finish.
        name = names.experiment_status(
            constants.experiment_name(), constants.trial_name()
        )
        try:
            exp_status = name_resolve.get(name)
            recover_this = recover_this and exp_status != str(ExpStatus.COMPLETE)
            if exp_status == str(ExpStatus.COMPLETE):
                logger.warning("*" * 100)
                logger.warning(
                    "*"
                    + f"Will not recover because the experiment has completed! Congrats!".center(
                        98, " "
                    )
                    + "*"
                )
                logger.warning("*" * 100)
        except name_resolve.NameEntryNotFoundError:
            raise name_resolve.NameEntryNotFoundError(
                f"Experiment status not found during recover. "
                "This indicates that the master worker is not running. Exit the recover loop."
            )

        kill_signal = (
            "SIGKILL" if args.mode == "slurm" else "SIGTERM"
        )  # use sigkill to terminate slurm jobs
        # Recovering should use SIGKILL as well, since we
        # are not calling exit hook on workers.
        # Using SIGINT might cause some workers to be stuck,
        # leaving RUNNING state workers in the slurm cluster
        sched.stop_all(kill_signal)
        if recover_this:
            logger.warning(
                f"Recovering from error {e} after {args.recover_after} seconds. Recover count: {recover_count+1}, "
                f"total recover count {args.recover_retries}"
            )
            time.sleep(args.recover_after)
            main_start(args, job_group_id=job_group_id, recover_count=recover_count + 1)
        else:
            raise e


def main_stop(args):
    experiment = config_package.make_experiment(args.experiment_name)
    sched = sched_client.make(experiment)
    sched.find_all()
    sched.stop_all()


def main_find_config(args):
    exp_names = [
        x for x in config_package.ALL_EXPERIMENT_CLASSES if re.match(args.regex, x)
    ]
    if len(exp_names) == 0:
        print("No matched experiment names.")
    if len(exp_names) > 20:
        response = input(f"Found {len(exp_names)} experiments, list all?(y/n)")
        if response != "y":
            return
    for exp_name in exp_names:
        print(exp_name)


def main():
    parser = argparse.ArgumentParser(prog="ReaLHF")
    subparsers = parser.add_subparsers(dest="cmd", help="sub-command help")
    subparsers.required = True

    subparser = subparsers.add_parser("start", help="starts an experiment")
    subparser.add_argument(
        "--experiment_name",
        "-e",
        type=str,
        required=True,
        help="name of the experiment",
    )
    subparser.add_argument(
        "--trial_name",
        "-f",
        type=str,
        default=None,
        help="trial name; by default uses '<USER>-test'",
    )
    subparser.add_argument(
        "--mode",
        default="slurm",
        choices=["local", "slurm", "ray"],
    )
    subparser.add_argument(
        "--schedule_strategy",
        default="empty_first",
        choices=["empty_first", "allocated_first"],
        help="Schedule strategy for scheduler. Currently only effective in slurm mode. "
        "In slurm mode, jobs are scheduled in pack to avoid fragmentation. "
        "Specifically, the scheduler will avoid scheduling master worker/ctl jobs on different nodes from model workers; "
        "'empty_first': schedule jobs to the node with least allocated resources first; "
        "'allocated_first': schedule jobs to the node with most allocated resources first. ",
    )
    subparser.add_argument(
        "--partition",
        default="dev",
        help="slurm partition to schedule the trial",
    )
    subparser.add_argument("--ignore_worker_error", action="store_true")
    subparser.add_argument(
        "--debug",
        action="store_true",
        help="If True, activate all assertions in the code.",
    )
    subparser.add_argument(
        "--recover_mode",
        required=False,
        default="disabled",
        choices=["disabled", "auto", "resume", "fault"],
        help="Recover mode, 'auto': automatically recover the last failed run, "
        "otherwise run from sratch with fault tolerance; "
        "'fault': run from scratch with fault tolerance; "
        "'resume': force to load the last failed run and run it without fault tolerance; "
        "'disabled': do nothing when error occurs. ",
    )
    subparser.add_argument(
        "--recover_retries",
        type=int,
        required=False,
        default=1,
        help="Total number of trials for the system to recover automatically when a worker fails. "
        "Only effective when recover_mode is 'auto' or 'fault'",
    )
    subparser.add_argument(
        "--recover_after",
        type=int,
        required=False,
        default=10,
        help="Number of seconds to wait before recovering.",
    )
    subparser.add_argument(
        "--allocation_mode",
        type=str,
        required=False,
        default="pipe_model",
        choices=["manual", "heuristic", "pipe_model", "pipe_data"],
        help="Mode of GPU resource/model parallel strategy allocation.",
    )
    subparser.set_defaults(ignore_worker_error=False)
    subparser.set_defaults(func=main_start)

    subparser = subparsers.add_parser(
        "stop", help="stops an experiment. only slurm experiment is supported."
    )
    subparser.add_argument(
        "--experiment_name",
        "-e",
        type=str,
        required=True,
        help="name of the experiment",
    )
    subparser.add_argument(
        "--trial_name", "-f", type=str, required=True, help="name of the trial"
    )
    subparser.add_argument(
        "--mode",
        default="slurm",
        choices=["local", "slurm", "ray"],
    )
    subparser.set_defaults(func=main_stop)

    subparser = subparsers.add_parser(
        "find_config", help="find configuration by matching regular expression."
    )
    subparser.add_argument("--regex", "-r", type=str, required=True)
    subparser.set_defaults(func=main_find_config)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
