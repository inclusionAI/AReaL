# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import asyncio
import importlib
import os
import traceback
from typing import Type

import realhf.base.logging as logging

logger = logging.getLogger("system")

# NOTE: Workers are configured in the following order.
# Take special care when adding a new worker type.
WORKER_TYPES = [
    "generation_server",
    "gserver_manager",
    "model_worker",
    "master_worker",
    "rollout_worker",
]


def load_worker(worker_type: str) -> Type:
    assert worker_type in WORKER_TYPES, f"Invalid worker type {worker_type}"
    module = importlib.import_module(worker_type_to_module(worker_type))
    class_name = worker_type_to_class_name(worker_type)
    return getattr(module, class_name)


def worker_type_to_module(worker_type: str):
    return "realhf.system." + worker_type


def worker_type_to_class_name(worker_type: str):
    return "".join([w.capitalize() for w in worker_type.split("_")])


def run_worker(
    worker_type, experiment_name, trial_name, worker_name, worker_server_type
):
    """Run one worker
    Args:
        worker_type: string, one of the worker types listed above,
        experiment_name: string, the experiment this worker belongs to,
        trial_name: string, the specific trial this worker belongs to,
        worker_name: name given to the worker, typically "<worker_type>/<worker_index>"
        worker_server_type: string, either 'zmq' or 'ray'.
    """
    worker_class = load_worker(worker_type)
    make_server_fn = getattr(
        importlib.import_module("realhf.system.worker_control"), "make_server"
    )
    server = make_server_fn(
        type_=worker_server_type,
        experiment_name=experiment_name,
        trial_name=trial_name,
        worker_name=worker_name,
    )
    worker = worker_class(server=server)
    try:
        if worker_type in ["rollout_worker", "master_worker"]:
            asyncio.run(worker.run_async())
        else:
            worker.run()
    except Exception as e:
        logger.error("Worker %s failed with exception: %s", worker_name, e)
        logger.error(traceback.format_exc())
        raise e


def make_controller(type_, experiment_name, trial_name):
    module = importlib.import_module("realhf.system.controller")
    if type_ == "zmq":
        control_module = importlib.import_module("realhf.system.worker_control")
        panel = getattr(control_module, "make_control")(
            "zmq", experiment_name, trial_name
        )
        return getattr(module, "Controller")(experiment_name, trial_name, panel)
    elif type_ == "ray":
        return getattr(module, "RayController")(experiment_name, trial_name)
    else:
        raise NotImplementedError(f"Unknown controller type {type_}.")
