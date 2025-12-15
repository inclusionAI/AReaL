import json
import os
import sys
from dataclasses import asdict

import pytest
import yaml
from sh import Command

from areal.api.cli_args import SFTConfig, load_expr_config
from areal.tests.utils import get_dataset_path, get_model_path


@pytest.mark.parametrize("backend", ["fsdp", "megatron"])
def test_sft(tmp_path: str, backend: str) -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, f"config_{backend}.yaml")
    ref_losses_path = os.path.join(base_dir, f"ref_losses_{backend}.json")

    # Wrap over the original config to use local models/datasets if possible
    config, _ = load_expr_config(
        ["--config", config_path], SFTConfig
    )

    # Use get_model_path to check local or download from HuggingFace
    local_model_path = config.model.path.replace("/", "__")
    model_path = get_model_path(
        os.path.join("/storage/openpsi/models", local_model_path),
        config.model.path,
    )
    config.model.path = model_path
    config.tokenizer_path = model_path

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

    with open(os.path.join(tmp_path, "losses.json")) as f:
        losses: list[float] = json.load(f)

    with open(ref_losses_path) as f:
        ref_losses: list[float] = json.load(f)

    # Refer to https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close
    assert all(
        loss == pytest.approx(ref_loss, rel=1.6e-2, abs=1e-5)
        for loss, ref_loss in zip(losses, ref_losses)
    )
