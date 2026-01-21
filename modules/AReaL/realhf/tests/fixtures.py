# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import json
import random
import uuid

import pytest
import torch
import torch.distributed as dist
import transformers
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from transformers import PreTrainedTokenizerFast

from realhf.base import constants, logging, testing


@pytest.fixture
def save_path(tmpdir_factory: pytest.TempdirFactory):
    return tmpdir_factory.mktemp("save_path")


def generate_random_sentence(length):
    # A predefined list of common English words
    # fmt: off
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "and", "then", "runs", "away", "from", "big", "scary", "bear",
        "in", "the", "forest", "during", "sunny", "day", "while", "birds",
        "sing", "beautiful", "songs", "under", "blue", "sky", "with", "white",
        "clouds", "floating", "gently",
    ]
    # fmt: on

    # Randomly select words to form a sentence
    sentence = " ".join(random.choices(words, k=length)) + "\n"

    return sentence


@pytest.fixture(params=[testing.TESTING_DATASET_SIZE])
def dataset(request, save_path):
    size = request.param
    max_prompt_len = 8
    max_resp_len = 8
    dataset = []
    for i in range(size):
        prompt_len = random.randint(1, max_prompt_len)
        n_pairs = random.randint(1, 5)
        qid = str(uuid.uuid4())
        d = dict(
            id=qid,
            query_id=qid,
            prompt=generate_random_sentence(prompt_len),
            solutions=["\\boxed{xxxxx}"],
            answer=generate_random_sentence(random.randint(1, max_resp_len)),
            pos_answers=[
                generate_random_sentence(random.randint(1, max_resp_len))
                for _ in range(n_pairs)
            ],
            neg_answers=[
                generate_random_sentence(random.randint(1, max_resp_len))
                for _ in range(n_pairs)
            ],
            task="math",
        )
        dataset.append(d)
    with open(str(save_path / "dataset.jsonl"), "w") as f:
        for d in dataset:
            f.write(json.dumps(d) + "\n")
    return dataset


@pytest.fixture
def tokenizer(dataset, save_path):

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordPieceTrainer(
        vocab_size=testing.TESTING_MODEL_VOCAB_SIZE - 2,
        min_frequency=0,
        special_tokens=[],
    )

    data = [d["prompt"] + d["answer"] for d in dataset]
    # Train the tokenizer on the sample data
    tokenizer.train_from_iterator(data, trainer)
    tokenizer.save(str(save_path / "tokenizer.json"))

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(save_path / "tokenizer.json")
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]", "eos_token": "[EOS]"})

    return tokenizer


def maybe_prepare_cpu_env(max_prompt_len: int):
    if not dist.is_initialized():
        # for parametrized runs
        dist.init_process_group(
            "gloo", rank=0, world_size=1, init_method="tcp://localhost:7777"
        )
        testing.init_global_constants(
            num_dp=1,
            num_tp=1,
            num_pp=1,
            sequence_parallel=False,
            max_prompt_len=max_prompt_len,
        )
        assert dist.get_world_size() == 1, dist.get_world_size()


@pytest.fixture
def mconfig(model_class):
    from realhf.impl.model.nn.real_llm_api import ReaLModel

    mconfig = getattr(ReaLModel, f"make_{model_class}_config")()
    return mconfig


@pytest.fixture
def cpu_real_model(model_class, mconfig, save_path):
    # Import here to avoid CUDA initialization
    from realhf.impl.model.nn.real_llm_api import add_helper_functions

    max_prompt_len = mconfig.n_positions
    maybe_prepare_cpu_env(max_prompt_len)
    with constants.model_scope(testing.MODEL_NAME):
        from realhf.impl.model.nn.real_llm_api import ReaLModel

        model = ReaLModel(mconfig, dtype=torch.float32, device="cpu")
        add_helper_functions(model)
        model.instantiate()
        model.eval()
        getattr(model, f"to_{model_class}")(None, save_path)
    return model


@pytest.fixture
def cpu_hf_model(model_class, mconfig, save_path):
    from realhf.impl.model.nn.real_llm_api import ReaLModel

    hf_config = getattr(ReaLModel, f"config_to_{model_class}")(mconfig)
    hf_model = transformers.AutoModelForCausalLM.from_config(hf_config)
    hf_model.eval()
    hf_model.save_pretrained(save_path)
    return hf_model
