# Copyright 2025 Ant Group Inc.

import os
import time
import uuid
from typing import *

import pytest

from realhf.api.cli_args import (
    ClusterSpecConfig,
    ExperimentSaveEvalControl,
    MFCConfig,
    ModelTrainEvalConfig,
    ParallelismConfig,
    PromptAnswerDatasetConfig,
    PromptOnlyDatasetConfig,
)
from realhf.base import logging, name_resolve, testing
from realhf.experiments.common.null_exp import NullPPOConfig, NullSFTConfig
from tests.experiments.utils import run_test_exp
from tests.fixtures import *

logger = logging.getLogger("test buffer recover")


@pytest.fixture(params=["llama"])
def model_class(request):
    return request.param


@pytest.fixture(params=[300])
def math_code_dataset_with_size(request, save_path):
    size = request.param
    max_prompt_len = 8
    max_resp_len = 8
    dataset = []
    for i in range(size):
        prompt_len = random.randint(1, max_prompt_len)
        d = dict(
            query_id=str(uuid.uuid4()),
            prompt=generate_random_sentence(prompt_len),
            task=random.choice(["math", "code"]),
        )
        if d["task"] == "math":
            d["solutions"] = [generate_random_sentence(max_resp_len)]
        elif d["task"] == "code":
            d["input_output"] = json.dumps(dict(inputs=["the\n"], outputs=["the\n"]))
        dataset.append(d)
        with open(str(save_path / "math_code_dataset.jsonl"), "a") as f:
            f.write(json.dumps(d) + "\n")
    return dataset, len(dataset)


@pytest.mark.parametrize("dp", [4])
@pytest.mark.parametrize("bs", [63])
def test_buffer_recover(
    bs,
    tmp_path_factory,
    math_code_dataset_with_size,
    tokenizer,
    save_path,
    cpu_hf_model,
    dp,
):
    _, dataset_size = math_code_dataset_with_size
    # Setup experiment env. Should be done before any other operations.
    expr_name = str(uuid.uuid4())
    trial_name = str(uuid.uuid4())
    constants.set_experiment_trial_names(expr_name, trial_name)

    exp_cfg = NullPPOConfig(
        experiment_name=expr_name,
        trial_name=trial_name,
        mode="local",
        # allocation_mode=f"m1d{dp}p1",
        nodelist="slurmd-01",
        allocation_mode="manual",
        inf=MFCConfig(
            device_mesh="slurmd-01:0,1,2,3,4,5,6,7",
            parallel=ParallelismConfig(
                tensor_parallel_size=2,
                pipeline_parallel_size=2,
                data_parallel_size=dp // 2,
            ),
        ),
        train=MFCConfig(
            device_mesh="slurmd-01:8,9,10,11,12,13,14,15",
            parallel=ParallelismConfig(
                tensor_parallel_size=2,
                pipeline_parallel_size=2,
                data_parallel_size=dp // 2,
            ),
        ),
        n_nodes=1,
        n_gpus_per_node=dp * 4,
        model=ModelTrainEvalConfig(
            path=str(save_path),
            init_from_scratch=True,
            backend="mock_train",
        ),
        dataset=PromptOnlyDatasetConfig(
            path=str(save_path / "math_code_dataset.jsonl"),
            max_prompt_len=128,
            train_bs_n_seqs=bs,
            fill_to_max_length=False,
        ),
        dataset_filter_threshold=-1,
        dataset_max_filter_percentage=0.05,
        exp_ctrl=ExperimentSaveEvalControl(
            total_train_epochs=100,
            save_freq_steps=2,
            benchmark_steps=0,
        ),
        cluster=ClusterSpecConfig(
            fileroot=str(tmp_path_factory.mktemp("buffer-recover")),
            n_gpus_per_node=16,
        ),
    )

    os.environ["REAL_SAVE_RECOVER_STATES"] = "1"
    os.environ["REAL_RECOVER_RUN"] = "1"
    os.environ["REAL_MASTER_BUFFER_SIZE"] = str(int(dataset_size * 1.5))
    constants.set_force_cpu(True)
    # Register all datasets and models
    import realhf.impl.dataset  # isort: skip
    import realhf.impl.model  # isort: skip
    from realhf.system.master_worker import MasterWorker
    from realhf.system.worker_base import WorkerServerStatus

    step_each_run = 4
    n_data = exp_cfg.exp_ctrl.total_train_epochs * dataset_size
    total_steps = (
        n_data + exp_cfg.dataset.train_bs_n_seqs - 1
    ) // exp_cfg.dataset.train_bs_n_seqs
    for i in range((total_steps + step_each_run - 1) // step_each_run):
        exp_cfg.exp_ctrl.benchmark_steps += step_each_run
        run_test_exp(exp_cfg, expr_name, trial_name)
