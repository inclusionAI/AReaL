import json
import os
import sys
from typing import List

import pytest
from sh import Command


def test_grpo(tmp_path: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    cmd = (
        Command("python")
        .bake(m="areal.launcher.local")
        .bake(os.path.join(base_dir, "entrypoint.py"))
    )

    cmd(
        f"cluster.fileroot={tmp_path}",
        config=os.path.join(base_dir, f"config.yaml"),
        _err=sys.stderr,
        _out=sys.stdout,
        _env=os.environ,
        _ok_code=1,  # AReaL exits with code 1 even when successful.
    )

    with open(os.path.join(tmp_path, "rewards.json")) as f:
        rewards: List[float] = json.load(f)

    with open(os.path.join(base_dir, "ref_rewards.json")) as f:
        ref_rewards: List[float] = json.load(f)

    # Refer to https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close
    assert all(
        reward == pytest.approx(ref_reward, rel=1.6e-2, abs=1e-5) or reward > ref_reward
        for reward, ref_reward in zip(rewards, ref_rewards)
    )
