import importlib.util
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

if "verl" not in sys.modules:
    verl_stub = types.ModuleType("verl")
    verl_stub.DataProto = object
    sys.modules["verl"] = verl_stub

if "deepscaler" not in sys.modules:
    sys.modules["deepscaler"] = types.ModuleType("deepscaler")
if "deepscaler.rewards" not in sys.modules:
    sys.modules["deepscaler.rewards"] = types.ModuleType("deepscaler.rewards")
if "deepscaler.rewards.math_rewardv2" not in sys.modules:
    math_rewardv2_stub = types.ModuleType("deepscaler.rewards.math_rewardv2")
    math_rewardv2_stub.deepscaler_reward_fn = lambda **_kwargs: ({"reward": 0.0, "second_reward": 0.0}, {})
    sys.modules["deepscaler.rewards.math_rewardv2"] = math_rewardv2_stub

MODULE_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, MODULE_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


local_reward_manager = _load_module("test_local_reward_manager", "verl/trainer/reward_manager.py")
server_reward_manager = _load_module("test_server_reward_manager", "verl/trainer/reward_manager_with_server.py")


def _cfg():
    return {
        "subtask_beta": 0.5,
        "trial_beta": 0.25,
        "parallel_ratio_beta": 0.75,
        "latency_alpha": 0.1,
        "group_shaping_eps": 1e-8,
    }


def _sample_scores():
    return [
        {"reward": 1.0, "second_reward": 0.0},
        {"reward": 1.0, "second_reward": 0.0},
        {"reward": -1.0, "second_reward": 0.0},
        {"reward": 2.0, "second_reward": 0.0},
    ]


def _sample_extra_infos():
    return [
        {
            "correct": True,
            "subtask_ratio": 0.2,
            "trial_ratio": 0.4,
            "parallel_ratio": 0.6,
            "acceleration_ratio": 0.1,
        },
        {
            "correct": True,
            "subtask_ratio": 0.6,
            "trial_ratio": 0.2,
            "parallel_ratio": 0.8,
            "acceleration_ratio": 0.3,
        },
        {
            "correct": False,
            "subtask_ratio": 0.5,
            "trial_ratio": 0.5,
            "parallel_ratio": 0.5,
            "acceleration_ratio": 0.5,
        },
        {
            "correct": True,
            "subtask_ratio": 0.5,
            "trial_ratio": 0.5,
            "parallel_ratio": 0.5,
            "acceleration_ratio": 0.5,
        },
    ]


def test_parallel_bonus_is_group_normalized_and_correct_only():
    scores, extra_infos = local_reward_manager._apply_groupwise_parallel_bonus(
        _sample_scores(),
        _sample_extra_infos(),
        np.array(["g1", "g1", "g2", "g2"], dtype=object),
        _cfg(),
    )

    expected_bonus_0 = (
        0.5 * -1.0
        + 0.25 * 1.0
        + 0.75 * -1.0
        + 0.1 * -1.0
    )
    expected_bonus_1 = (
        0.5 * 1.0
        + 0.25 * -1.0
        + 0.75 * 1.0
        + 0.1 * 1.0
    )

    assert math.isclose(extra_infos[0]["parallel_bonus_subtask_z"], -1.0, abs_tol=1e-6)
    assert math.isclose(extra_infos[1]["parallel_bonus_subtask_z"], 1.0, abs_tol=1e-6)
    assert math.isclose(extra_infos[2]["parallel_bonus_subtask_z"], 0.0, abs_tol=1e-6)
    assert math.isclose(extra_infos[3]["parallel_bonus_subtask_z"], 0.0, abs_tol=1e-6)

    assert math.isclose(extra_infos[0]["parallel_rewardv2_bonus"], expected_bonus_0, abs_tol=1e-6)
    assert math.isclose(extra_infos[1]["parallel_rewardv2_bonus"], expected_bonus_1, abs_tol=1e-6)
    assert math.isclose(extra_infos[2]["parallel_rewardv2_bonus"], 0.0, abs_tol=1e-6)
    assert math.isclose(extra_infos[3]["parallel_rewardv2_bonus"], 0.0, abs_tol=1e-6)

    assert math.isclose(scores[0]["reward"], 1.0 + expected_bonus_0, abs_tol=1e-6)
    assert math.isclose(scores[1]["reward"], 1.0 + expected_bonus_1, abs_tol=1e-6)
    assert math.isclose(scores[2]["reward"], -1.0, abs_tol=1e-6)
    assert math.isclose(scores[3]["reward"], 2.0, abs_tol=1e-6)


def test_server_and_local_parallel_bonus_helpers_match():
    local_scores, local_infos = local_reward_manager._apply_groupwise_parallel_bonus(
        _sample_scores(),
        _sample_extra_infos(),
        np.array(["g1", "g1", "g2", "g2"], dtype=object),
        _cfg(),
    )
    server_scores, server_infos = server_reward_manager._apply_groupwise_parallel_bonus(
        _sample_scores(),
        _sample_extra_infos(),
        np.array(["g1", "g1", "g2", "g2"], dtype=object),
        _cfg(),
    )

    assert local_scores == server_scores
    assert local_infos == server_infos


def test_reward_manager_writes_final_reward_to_last_valid_token(monkeypatch):
    canned_results = [
        (0, {"reward": 1.0, "second_reward": 0.0}, 2, "seq-0", {
            "correct": True,
            "subtask_ratio": 0.2,
            "trial_ratio": 0.4,
            "parallel_ratio": 0.6,
            "acceleration_ratio": 0.1,
        }),
        (1, {"reward": -1.0, "second_reward": 0.0}, 3, "seq-1", {
            "correct": False,
            "subtask_ratio": 0.8,
            "trial_ratio": 0.1,
            "parallel_ratio": 0.9,
            "acceleration_ratio": 0.4,
        }),
    ]

    def fake_process_item(args):
        return canned_results[args[0]]

    monkeypatch.setattr(local_reward_manager, "process_item", fake_process_item)

    class FakeItem:
        def __init__(self):
            self.batch = {}
            self.non_tensor_batch = {}

    class FakeData:
        def __init__(self):
            self.batch = {
                "responses": torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long),
            }
            self.non_tensor_batch = {
                "uid": np.array(["g1", "g1"], dtype=object),
            }
            self._items = [FakeItem(), FakeItem()]

        def __len__(self):
            return len(self._items)

        def __getitem__(self, idx):
            return self._items[idx]

    manager = local_reward_manager.RewardManager(tokenizer=None, num_examine=0, config=_cfg())
    data = FakeData()
    result = manager(data, return_dict=True)

    reward_tensor = result["reward_tensor"]["main_reward_tensor"]
    extra_info = result["reward_extra_info"]

    expected_bonus = (
        0.5 * -1.0
        + 0.25 * 1.0
        + 0.75 * -1.0
        + 0.1 * -1.0
    )

    assert reward_tensor.shape == torch.Size([2, 3])
    assert torch.allclose(reward_tensor[0], torch.tensor([0.0, 1.0 + expected_bonus, 0.0]))
    assert torch.allclose(reward_tensor[1], torch.tensor([0.0, 0.0, -1.0]))
    assert extra_info["parallel_rewardv2_bonus"] == [pytest.approx(expected_bonus), pytest.approx(0.0)]
