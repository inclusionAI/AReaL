import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml
from sh import Command

from tests.utils import get_dataset_path, get_model_path

from areal.api.cli_args import GRPOConfig, load_expr_config

# Each (training_backend, inference_backend) pair maps to a config file and pytest mark.
_SGLANG_CASES = [
    pytest.param("fsdp", "sglang", id="fsdp-sglang", marks=pytest.mark.sglang),
    pytest.param("megatron", "sglang", id="megatron-sglang", marks=pytest.mark.sglang),
    pytest.param("archon", "sglang", id="archon-sglang", marks=pytest.mark.sglang),
]
_VLLM_CASES = [
    pytest.param("fsdp", "vllm", id="fsdp-vllm", marks=pytest.mark.vllm),
    pytest.param("megatron", "vllm", id="megatron-vllm", marks=pytest.mark.vllm),
    pytest.param("archon", "vllm", id="archon-vllm", marks=pytest.mark.vllm),
]


@pytest.mark.parametrize(("backend", "inference"), _SGLANG_CASES + _VLLM_CASES)
def test_grpo(tmp_path: Path, backend: str, inference: str) -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Select config file based on inference backend
    if inference == "vllm":
        config_path = os.path.join(base_dir, f"config_{backend}_vllm.yaml")
    else:
        config_path = os.path.join(base_dir, f"config_{backend}.yaml")

    # Wrap over the original config to use local models/datasets if possible
    config, _ = load_expr_config(["--config", config_path], GRPOConfig)

    # Use get_model_path to check local or download from HuggingFace
    local_model_path = config.actor.path.replace("/", "__")
    model_path = get_model_path(
        os.path.join("/storage/openpsi/models", local_model_path),
        config.actor.path,
    )
    config.actor.path = model_path
    config.ref.path = model_path
    config.tokenizer_path = model_path

    # Set inference engine model path based on backend
    if inference == "vllm":
        config.vllm.model = model_path
    else:
        config.sglang.model_path = model_path

    # Use get_dataset_path to check local or download from HuggingFace
    local_dataset_path = config.train_dataset.path.replace("/", "__")
    dataset_path = get_dataset_path(
        os.path.join("/storage/openpsi/data", local_dataset_path),
        config.train_dataset.path,
    )
    config.train_dataset.path = dataset_path

    # save new config
    os.makedirs(os.path.join(tmp_path, "config"), exist_ok=True)
    with open(os.path.join(tmp_path, "config", "config.yaml"), "w") as f:
        yaml.dump(
            asdict(config),
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    cmd = (
        Command("python")
        .bake(m="areal.infra.launcher.local")
        .bake(os.path.join(base_dir, "entrypoint.py"))
    )

    cmd(
        f"cluster.fileroot={tmp_path}",
        config=os.path.join(tmp_path, "config", "config.yaml"),
        _err=sys.stderr,
        _out=sys.stdout,
        _env=os.environ,
        _ok_code=1,  # AReaL exits with code 1 even when successful.
    )

    with open(os.path.join(tmp_path, "rewards.json")) as f:
        rewards: list[float] = json.load(f)

    assert rewards[-1] > 0.6
