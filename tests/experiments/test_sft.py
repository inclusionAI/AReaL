# Copyright 2025 Ant Group Inc.

import os
from typing import *

import pytest

from realhf.api.cli_args import (
    ClusterSpecConfig,
    ExperimentSaveEvalControl,
    ModelTrainEvalConfig,
    PromptAnswerDatasetConfig,
)
from realhf.base import testing
from realhf.experiments.common.sft_exp import SFTConfig
from tests.experiments.utils import run_test_exp
from tests.fixtures import *


@pytest.fixture(params=["llama"])
def model_class(request):
    return request.param


@pytest.mark.skipif(
    os.cpu_count() < 32 or testing.get_free_mem_gb() < 50,
    reason="Testing with larger parallelization degrees requires more CPU cores and memory.",
)
@pytest.mark.parametrize(
    "dp,pp,tp",
    [
        (2, 2, 2),
        (1, 2, 4),
        (2, 4, 1),
        (2, 1, 4),
    ],
)
def test_sft_xl(tmp_path_factory, tokenizer, save_path, cpu_hf_model, dp, pp, tp):
    test_sft(
        tmp_path_factory,
        tokenizer,
        save_path,
        cpu_hf_model,
        dp,
        pp,
        tp,
    )


@pytest.mark.parametrize(
    "dp,pp,tp",
    [
        (1, 1, 1),
        (2, 1, 1),
        (1, 2, 1),
        (1, 1, 2),
    ],
)
def test_sft(tmp_path_factory, tokenizer, save_path, cpu_hf_model, dp, pp, tp):

    # Setup experiment env. Should be done before any other operations.
    constants.set_experiment_trial_names(
        testing._DEFAULT_EXPR_NAME, testing._DEFAULT_TRIAL_NAME
    )

    minbs = 32
    exp_cfg = SFTConfig(
        exp_ctrl=ExperimentSaveEvalControl(eval_freq_steps=2),
        experiment_name=testing._DEFAULT_EXPR_NAME,
        trial_name=testing._DEFAULT_TRIAL_NAME,
        mode="local",
        allocation_mode=f"m{tp}d{dp}p{pp}",
        n_nodes=1,
        n_gpus_per_node=tp * dp * pp,
        model=ModelTrainEvalConfig(
            path=str(save_path),
            init_from_scratch=True,
            backend="mock_train",
        ),
        dataset=PromptAnswerDatasetConfig(
            train_path=str(save_path / "dataset.jsonl"),
            valid_path=str(save_path / "dataset.jsonl"),
            max_seqlen=128,
            train_bs_n_seqs=minbs,
            valid_bs_n_seqs=minbs,
            fill_to_max_length=False,
        ),
        cluster=ClusterSpecConfig(fileroot=str(tmp_path_factory.mktemp("sft"))),
    )

    run_test_exp(exp_cfg)
