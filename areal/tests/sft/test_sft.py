import json
import os
import sys
from typing import Dict, List

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


@pytest.mark.parametrize(
    "config_path",
    [
        "areal/tests/sft/gsm8k.yaml",
    ],
)
def test_sft(config_path: str):
    cmd = (
        Command("python")
        .bake(m="areal.launcher.local")
        .bake("areal/tests/sft/entrypoint.py")
    )

    config, _ = cli_args.load_expr_config(
        ["--config", config_path],
        SFTConfig,
    )
    stats_path = os.path.join(
        StatsLogger.get_log_path(config.stats_logger),
        "stats.json",
    )
    if os.path.exists(stats_path):
        os.remove(stats_path)

    cmd(
        f"cluster.n_gpus_per_node={torch.cuda.device_count()}",
        f"allocation_mode={build_alloc_mode(torch.cuda.device_count())}",
        config=config_path,
        _err=sys.stderr,
        _out=sys.stdout,
        _env=os.environ,
        _ok_code=1,  # AReaL exits with code 1 even when successful.
    )

    with open(
        os.path.join(
            StatsLogger.get_log_path(config.stats_logger),
            "stats.json",
        ),
        "r",
    ) as f:
        stats: List[Dict[str, float]] = json.load(f)

    assert all(stat["loss/avg"] >= stats[-1]["loss/avg"] for stat in stats[:20])
