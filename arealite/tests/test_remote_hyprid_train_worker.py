import unittest

import torch

from arealite.extension.asystem.remote_hybrid_train_worker import pack_logprobs


class TestPackLogprobs(unittest.TestCase):
    def test_pack_logprobs_basic(self):
        """测试pack_logprobs基本功能"""
        # 测试数据：2个样本，长度分别为3和4
        logprobs = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.0],  # 样本1，有效长度3
                [0.4, 0.5, 0.6, 0.7],  # 样本2，有效长度4
            ]
        )
        seqlen = torch.tensor([3, 4])  # 各样本实际长度

        # 执行函数
        packed = pack_logprobs(logprobs, seqlen)

        # 验证结果
        expected = torch.tensor([0.2, 0.3, 0.5, 0.6, 0.7])  # 跳过每个样本第一个元素
        self.assertTrue(torch.allclose(packed, expected))

        # 验证输出形状
        self.assertEqual(packed.shape, (5,))  # (3-1)+(4-1)=5


if __name__ == "__main__":
    unittest.main()
