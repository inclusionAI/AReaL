import json
import os
import sys
from dataclasses import asdict
from typing import List

import yaml
from sh import Command

from areal.api.cli_args import GRPOConfig, load_expr_config


def test_grpo(tmp_path: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Wrap over the original config to use local models/datasets if possible
    config, _ = load_expr_config(
        ["--config", os.path.join(base_dir, f"config.yaml")], GRPOConfig
    )
    config: GRPOConfig

    local_model_path = config.actor.path.replace("/", "__")
    local_model_path = os.path.join("/storage/openpsi/models", local_model_path)
    if os.path.exists(local_model_path):
        config.actor.path = local_model_path
        config.ref.path = local_model_path
        config.tokenizer_path = local_model_path
        config.sglang.model_path = local_model_path

    local_dataset_path = config.train_dataset.path.replace("/", "__")
    local_dataset_path = os.path.join("/storage/openpsi/data", local_dataset_path)
    if os.path.exists(local_dataset_path):
        config.train_dataset.path = local_dataset_path

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
        .bake(m="areal.launcher.local")
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
        rewards: List[float] = json.load(f)

    assert rewards[-1] > 0.6
