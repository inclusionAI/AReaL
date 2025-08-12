import json
import os
import sys
from typing import Dict, List

import numpy as np
from sh import Command

import areal.api.cli_args as cli_args
from areal.api.cli_args import SFTConfig
from areal.utils.stats_logger import StatsLogger

CONFIG_PATH = "areal/tests/sft/gsm8k.yaml"


def test_sft():
    cmd = (
        Command("python")
        .bake(m="areal.launcher.local")
        .bake("areal/tests/sft/entrypoint.py")
    )

    cmd(
        config=CONFIG_PATH,
        _err=sys.stderr,
        _out=sys.stdout,
        _env=os.environ,
        _ok_code=1,
    )

    config, _ = cli_args.load_expr_config(
        ["--config", CONFIG_PATH],
        SFTConfig,
    )

    with open(
        os.path.join(
            StatsLogger.get_log_path(config.stats_logger),
            "stats.json",
        ),
        "r",
    ) as f:
        stats: List[Dict[str, float]] = json.load(f)

    slope, _ = np.polyfit(
        list(range(len(stats))),
        [stat["loss/avg"] for stat in stats],
        1,
    )

    assert slope < 0, f"Loss should be decreasing, but slope is {slope}"
