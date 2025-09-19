import logging

import cloudpickle
import torch
from tensordict import TensorDict

from areal.dataset.distributed_batch_memory import DistributedBatchMemory
from areal.scheduler.local import LocalScheduler
from areal.scheduler.test.my_engine import MyEngine


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    sched = LocalScheduler({"type": "local"})
    # sched.create_workers("infer", {"num_workers": 1})

    fields = {
        "attention_mask": torch.zeros((256, 9060), dtype=torch.float32),
        "input_ids": torch.zeros((256, 9060), dtype=torch.int64),
        "logprobs": torch.zeros((256, 9060), dtype=torch.float32),
        "prompt_mask": torch.zeros((256, 9060), dtype=torch.int64),
        "rewards": torch.zeros(256, dtype=torch.int64),
        "seq_no_eos_mask": torch.zeros(256, dtype=torch.bool),
        "seqlen": torch.zeros(256, dtype=torch.int64),
        "task_ids": torch.zeros(256, dtype=torch.int64),
        "versions": torch.zeros((256, 9060), dtype=torch.int64),
    }
    result = TensorDict(fields, batch_size=[256])

    # torch.save(rollout_res, "rollout_res.pt")
    rollout_res_dict = result.to_dict()
    dbm = DistributedBatchMemory(rollout_res_dict)
    a = cloudpickle.dumps(rollout_res_dict)
    b = cloudpickle.dumps(dbm)
    import sys

    print(f"a: {sys.getsizeof(a)}, b: {sys.getsizeof(b)}")
    sched.get_workers("infer")
    engine_obj = MyEngine({"value": 24})
    # assert sched.create_engine(workers[0].id, engine_obj, {"init": 1})
    result = sched.call_engine("", "infer", dbm)
    print("Result:", result)
    assert result == 1024


if __name__ == "__main__":
    main()
