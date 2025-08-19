import json
import os
import sys
from typing import List

import pytest
import sympy
import torch
from sh import Command

import areal.api.cli_args as cli_args
from areal.api.cli_args import SFTConfig
from areal.utils.stats_logger import StatsLogger


def build_alloc_mode(device_count: int) -> str:
    assert device_count > 0

    primes = sorted(
        [
            prime
            for prime, exp in sympy.factorint(device_count).items()
            for _ in range(exp)
        ],
        reverse=True,
    )

    d, p, t = 1, 1, 1
    for prime in primes:
        if d <= p and d <= t:
            d *= prime
        elif p <= t:
            p *= prime
        else:
            t *= prime

    return f"d{d}p{p}t{t}"


@pytest.mark.parametrize("config_name", ["gsm8k"])
def test_sft(config_name: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    cmd = (
        Command("python")
        .bake(m="areal.launcher.local")
        .bake(os.path.join(base_dir, "entrypoint.py"))
    )

    config_path = os.path.join(base_dir, f"{config_name}.yaml")
    config, _ = cli_args.load_expr_config(
        ["--config", config_path],
        SFTConfig,
    )

    loss_avg_list_path = os.path.join(
        StatsLogger.get_log_path(config.stats_logger),
        "loss_avg_list.json",
    )
    if os.path.exists(loss_avg_list_path):
        os.remove(loss_avg_list_path)

    device_count = torch.cuda.device_count()
    cmd(
        f"cluster.n_gpus_per_node={device_count}",
        f"allocation_mode={build_alloc_mode(device_count)}",
        config=config_path,
        _err=sys.stderr,
        _out=sys.stdout,
        _env=os.environ,
        _ok_code=1,  # AReaL exits with code 1 even when successful.
    )

    with open(loss_avg_list_path) as f:
        loss_avg_list: List[float] = json.load(f)

    with open(os.path.join(base_dir, "loss_avg_list_ref.json")) as f:
        loss_avg_list_ref: List[float] = json.load(f)

    # Compare the first 20 loss/avg elements.
    # Refer to https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close
    assert all(
        loss_avg == pytest.approx(loss_avg_ref, rel=1.6e-2, abs=1e-5)
        for loss_avg, loss_avg_ref in zip(loss_avg_list[:20], loss_avg_list_ref[:20])
    )
