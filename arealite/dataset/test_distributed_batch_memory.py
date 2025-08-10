import unittest
import torch

from arealite.dataset.distributed_batch_memory import DistributedBatchMemory


class TestSplitByGroups(unittest.TestCase):

    def test_split_by_groups(self):
        batch_size = 512
        group_size = 8
        n = 32

        data = {
            "input_ids": torch.arange(batch_size),
            "other_key": torch.randint(0, 2, (batch_size,))
        }

        memory = DistributedBatchMemory(data)
        batches = memory.split_by_groups(group_size=group_size, n=n)
        print(f"batches: {batches}")
        self.assertEqual(len(batches), n)

        # 每组包含块数
        k = (batch_size // group_size) // n
        first_batch = batches[0]
        input_ids = first_batch["input_ids"]
        other_key = first_batch["other_key"]
        self.assertEqual(input_ids.shape[0], k * group_size)
        self.assertEqual(other_key.shape[0], k * group_size)
        # expected = tensor([0,1,2,3,4,5,6,7,256,257,258,259,260,261,262,263])
        expected = torch.cat([torch.arange(0, 8), torch.arange(256, 264)])
        print(f"input_ids: {input_ids}, expected: {expected}")

        self.assertTrue(torch.equal(input_ids, expected))

    def test_split_by_groups_with_invalid_total_chunks(self):
        # 测试总块数不能被 n 整除的情况
        data = {
            "input_ids": torch.arange(16)
        }

        memory = DistributedBatchMemory(data)

        with self.assertRaises(AssertionError):
            memory.split_by_groups(group_size=4, n=3)  # 16 / 4 = 4 chunks, 4 % 3 != 0

    def test_split_by_groups_with_invalid_group_size(self):
        # 测试 batch_size 不能被 group_size 整除的情况
        data = {
            "input_ids": torch.arange(15)
        }

        memory = DistributedBatchMemory(data)

        with self.assertRaises(AssertionError):
            memory.split_by_groups(group_size=4, n=2)


if __name__ == '__main__':
    unittest.main()