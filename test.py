from areal.controller.rollout_controller import RolloutController
from areal.engine.sglang_local import LocalSGLangEngine
from areal.scheduler.local import LocalScheduler
from areal.api.cli_args import InferenceEngineConfig
from areal.api.alloc_mode import AllocationMode
config = InferenceEngineConfig(
    experiment_name='test',
    trial_name='test',
    max_concurrent_rollouts=16,
    consumer_batch_size=16,
    max_head_offpolicyness=2,
    enable_rollout_tracing=True,
)
rollout = RolloutController(
    inf_engine=LocalSGLangEngine,
    config=config,
    scheduler=LocalScheduler(log_dir='./logs/integration')
)
rollout.initialize(alloc_mode=AllocationMode.from_str("sglang.d1"))
from areal.tests.utils import TestWorkflow
workflow = TestWorkflow()
from torchdata.stateful_dataloader import StatefulDataLoader
from datasets import Dataset
import random
dataset= Dataset.from_dict(dict(random=[random.random() for _ in range(16)]))
print(dataset)
dataloader = StatefulDataLoader(dataset=dataset, batch_size=4, collate_fn=lambda x: x)
for data in dataloader:
    print(data)
result = rollout.prepare_batch(
    dataloader,
    workflow_path="areal.tests.utils.TestWorkflow",
    workflow_kwargs={},
    should_accept_path=None,
)
print(result)
