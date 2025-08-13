import json
import os
import sys
from typing import List

import pytest
from sh import Command

import areal.api.cli_args as cli_args
from areal.api.cli_args import SFTConfig
from areal.utils.stats_logger import StatsLogger


def test_sft():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    cmd = (
        Command("python")
        .bake(m="areal.launcher.local")
        .bake(os.path.join(base_dir, "entrypoint.py"))
    )

    config_path = os.path.join(base_dir, f"config.yaml")
    config, _ = cli_args.load_expr_config(
        ["--config", config_path],
        SFTConfig,
    )

    losses_path = os.path.join(
        StatsLogger.get_log_path(config.stats_logger), "losses.json"
    )
    if os.path.exists(losses_path):
        os.remove(losses_path)

    cmd(
        config=config_path,
        _err=sys.stderr,
        _out=sys.stdout,
        _env=os.environ,
        _ok_code=1,  # AReaL exits with code 1 even when successful.
    )

    with open(losses_path) as f:
        losses: List[float] = json.load(f)

    with open(os.path.join(base_dir, "ref_losses.json")) as f:
        ref_losses: List[float] = json.load(f)

    # Refer to https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close
    assert all(
        loss == pytest.approx(ref_loss, rel=1.6e-2, abs=1e-5)
        for loss, ref_loss in zip(losses, ref_losses)
    )
