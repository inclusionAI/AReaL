# Copyright 2025 Ant Group Inc. All Rights Reserved.

import os
import shutil
import uuid
from typing import *

import pytest
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader

from realhf.api.core.data_api import (
    DatasetUtility,
    MicroBatchSpec,
    SequenceSample,
    load_hf_tokenizer,
)
from realhf.api.core.model_api import FinetuneSpec, Model
from realhf.base import constants, name_resolve, network, testing
from tests.fixtures import *


@pytest.fixture(params=[testing.TESTING_DATASET_SIZE])
def math_code_dataset(request, save_path):
    size = request.param
    max_prompt_len = 8
    dataset = []
    for i in range(size):
        prompt_len = random.randint(1, max_prompt_len)
        n_pairs = random.randint(1, 5)
        if random.random() < 0.5:
            d = dict(
                task="code",
                query_id=str(uuid.uuid4()),
                prompt=generate_random_sentence(prompt_len),
                problem_id=str(uuid.uuid4()),
                input_output=json.dumps(
                    {"inputs": ["1\n"] * 8, "outputs": ["1\n"] * 8}
                ),
                solutions=json.dumps(
                    ["```python\ninput()\nimport time\ntime.sleep(1e-3)\nprint(1)\n```"]
                    * 3
                ),
                difficulty=random.random() * 10,
            )
        else:
            d = dict(
                task="math",
                query_id=str(uuid.uuid4()),
                prompt=generate_random_sentence(prompt_len),
                answers=["\\boxed{-\\frac{2}{3}}"],
                solutions=["\\boxed{-\\frac{2}{3}}"],
            )
        dataset.append(d)
    with open(str(save_path / "math_code_dataset.jsonl"), "w") as f:
        f.write("\n".join([json.dumps(d) for d in dataset]))
    return dataset


@pytest.mark.parametrize(
    "tokenizer_path", ["/storage/openpsi/models/Qwen__Qwen2-1.5B-Instruct/"]
)
def test_multi_task_reward_interface(save_path, tokenizer_path, math_code_dataset):
    from realhf.api.cli_args import NameResolveConfig
    from realhf.impl.dataset.math_code_dataset import MATHCodePromptDataset

    name_resolve.reconfigure(
        NameResolveConfig("nfs", f"/tmp/areal/{str(uuid.uuid4())}/")
    )
    dist.init_process_group(
        rank=0, world_size=1, init_method=f"tcp://localhost:{network.find_free_port()}"
    )
    testing.init_global_constants()

    dataset = MATHCodePromptDataset(
        DatasetUtility(
            seed=0,
            dp_rank=0,
            world_size=1,
            tokenizer=load_hf_tokenizer(tokenizer_path),
        ),
        max_length=512,
        dataset_path=str(save_path / "math_code_dataset.jsonl"),
    )
    dataloader = DataLoader(
        dataset,
        collate_fn=SequenceSample.gather,
        # NOTE: This is *NOT* the actual batch size for training.
        # It is just a proper size to load data to workers.
        batch_size=4,
        shuffle=True,
    )
    from realhf.impl.model.interface.math_rw_interface import MultiTaskRewardInterface

    with constants.model_scope(testing.MODEL_NAME):
        interface = MultiTaskRewardInterface(
            dataset_path=str(save_path / "math_code_dataset.jsonl"),
            tokenizer_path=tokenizer_path,
            group_size=1,
            check_verifier_status=False,
        )
        model = Model(
            name="test",
            module=None,
            tokenizer=load_hf_tokenizer(tokenizer_path),
            device=torch.device("cpu"),
            ft_spec=FinetuneSpec(
                total_train_epochs=1, dataset_size=100, train_batch_size=3
            ),
        )

        for d in dataloader:
            d = interface.mock("inference", model, d)
            rewards = interface.inference(model, d, mb_spec=MicroBatchSpec())
            d.update_(rewards)
            assert rewards.data["rewards"].all(), rewards.data["rewards"]
        dist.destroy_process_group()
