# Copyright 2025 Ant Group Inc.
import asyncio
import functools
import multiprocessing as mp
from typing import *

import pytest

from realhf.api.core.system_api import Experiment, register_experiment
from realhf.base import constants, logging, testing
from realhf.system.worker_base import WorkerServerStatus
from tests.fixtures import *

logger = logging.getLogger("tests.experiments.utils", "benchmark")


@pytest.fixture(params=["llama"])
def model_class(request):
    return request.param


def run_model_worker(cfg, mw, barrier, expr_name=None):
    constants.set_force_cpu(True)
    # Register all datasets and models
    import realhf.impl.dataset  # isort: skip
    import realhf.impl.model  # isort: skip
    from realhf.api.core import system_api
    from realhf.system.model_worker import ModelWorker

    system_api.ALL_EXPERIMENT_CLASSES = {}
    register_experiment(expr_name or testing._DEFAULT_EXPR_NAME, lambda: cfg)
    constants.set_experiment_trial_names(
        mw.worker_info.experiment_name, mw.worker_info.trial_name
    )

    worker = ModelWorker()
    logger.info("Configuring model worker...")
    worker.configure(mw.worker_info, setup_id=0)
    logger.info("Configuring model worker... Done.")
    barrier.wait()
    initd = False
    while worker.status != WorkerServerStatus.PAUSED:
        if not initd:
            logger.info("Running model worker lazy initialization...")
        worker._poll()
        if not initd:
            logger.info("Running model worker lazy initialization... Done.")
            initd = True


def run_test_exp(
    exp_cfg: Experiment,
    expr_name=None,
    trial_name=None,
):
    constants.set_force_cpu(True)
    # Register all datasets and models
    import realhf.impl.dataset  # isort: skip
    import realhf.impl.model  # isort: skip
    from realhf.api.core import system_api
    from realhf.system.master_worker import MasterWorker

    system_api.ALL_EXPERIMENT_CLASSES = {}
    register_experiment(expr_name or testing._DEFAULT_EXPR_NAME, lambda: exp_cfg)

    # Get worker configurations
    exp_setup = exp_cfg.initial_setup()
    exp_setup.set_worker_information(
        expr_name or testing._DEFAULT_EXPR_NAME,
        trial_name or testing._DEFAULT_TRIAL_NAME,
    )

    # Initialize the master worker
    mas = MasterWorker()
    logger.info("Configuring master worker...")
    mas.configure(setup_id=0, worker_info=exp_setup.master_worker[0].worker_info)
    logger.info("Configuring master worker... Done.")

    # Run model workers in subprocesses
    barrier = mp.Barrier(len(exp_setup.model_worker))
    testcase = testing.LocalMultiProcessTest(
        world_size=len(exp_setup.model_worker),
        func=[
            functools.partial(
                run_model_worker,
                cfg=exp_cfg,
                mw=mw,
                barrier=barrier,
                expr_name=expr_name,
            )
            for mw in exp_setup.model_worker
        ],
        expr_name=expr_name or testing._DEFAULT_EXPR_NAME,
        trial_name=trial_name or testing._DEFAULT_TRIAL_NAME,
        timeout_secs=300,
        setup_dist_torch=False,
    )
    testcase.start()

    # Run the master worker.
    async def run_master_worker():
        initd = False
        for _ in range(int(1e4)):
            if mas.status == WorkerServerStatus.PAUSED:
                break
            if not initd:
                logger.info("Running master worker lazy initialization...")
            await mas._poll_async()
            if not initd:
                logger.info("Running master worker lazy initialization... Done.")
                initd = True

    asyncio.run(run_master_worker())
    testcase.wait(timeout=0.1)
