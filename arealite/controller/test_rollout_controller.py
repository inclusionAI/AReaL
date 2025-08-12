import unittest
from unittest.mock import MagicMock

import torch
from tensordict import TensorDict

from arealite.api.cli_args import RolloutControllerConfig
from arealite.api.controller_api import RolloutController
from arealite.base import logging
from arealite.controller.rollout_controller import DistributedRolloutController
from arealite.dataset.distributed_batch_memory import DistributedBatchMemory


logger = logging.getLogger("TestDistributedRolloutController")


def are_tensordicts_equal(td1: TensorDict, td2: TensorDict, strict: bool = True) -> bool:
    # 检查键集合是否一致
    if set(td1.keys()) != set(td2.keys()):
        return False

    for key in td1.keys():
        val1 = td1[key]
        val2 = td2[key]

        # 若值为 TensorDict，递归比较
        if isinstance(val1, TensorDict) or isinstance(val2, TensorDict):
            if not (isinstance(val1, TensorDict) and isinstance(val2, TensorDict)):
                return False  # 类型不一致
            if not are_tensordicts_equal(val1, val2, strict):
                return False
        else:
            # 普通张量比较
            if strict:
                if not torch.equal(val1, val2):
                    return False
            else:
                if not torch.allclose(val1, val2):
                    return False
    return True

class TestDistributedRolloutController(unittest.TestCase):
    def setUp(self):
        # 构造一个包含 Dict[str, Tensor] 的列表，用于测试
        self.data = [
            {"input_ids": torch.arange(i+1)} for i in range(8)
        ]
        self.ctl = DistributedRolloutController(None, RolloutControllerConfig(allocation_mode="gen:d8t4p1,train:d8t1p4"),None)

    def test_split_list(self):
        # 8 个元素分成 2 组，每组 4 个
        n = 2
        results = self.ctl.split_list(self.data, n)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0]), 4)
        self.assertEqual(len(results[1]), 4)
        logger.info("")
        logger.info(results)

        expected0 = [{"input_ids": torch.arange(i+1)} for i in range(4)]
        expected1 = [{"input_ids": torch.arange(i+5)} for i in range(4)]

        for i in range(4):
            self.assertTrue(are_tensordicts_equal(results[0][i], expected0[i]))

        for i in range(4):
            self.assertTrue(are_tensordicts_equal(results[1][i], expected1[i]))

    def test_rollout(self):
        self.ctl._rpc_call = MagicMock()

        # 构造模拟的_rpc_call返回结果
        result1 = TensorDict({
            "input_ids": torch.randn(256, 15571),
            "attention_mask": torch.ones(256, 15571, dtype=torch.long),
            "rewards": torch.randn(256),
            "seqlen": torch.full((256,), 15571, dtype=torch.long)
        }, batch_size=256)

        result2 = TensorDict({
            "input_ids": torch.randn(256, 15602),
            "attention_mask": torch.ones(256, 15602, dtype=torch.long),
            "rewards": torch.randn(256),
            "seqlen": torch.full((256,), 15602, dtype=torch.long)
        }, batch_size=256)

        self.ctl._rpc_call.return_value = [result1, result2]
        data = [{"dummy": "dummy"}]
        workflow = MagicMock()
        padded = self.ctl.rollout(data, workflow)
        logger.info(f"padded: {padded}")

        self.assertEqual(padded["input_ids"].shape[0], 512)
        self.assertEqual(padded["input_ids"].shape[1], 15602)


if __name__ == '__main__':
    unittest.main()