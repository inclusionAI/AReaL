import asyncio
import copy
from asyncio.queues import QueueEmpty
from unittest.mock import patch

import pytest

from realhf.api.core.config import (
    AgentAbstraction,
    DatasetAbstraction,
    EnvServiceAbstraction,
    ModelName,
)
from realhf.api.core.model_api import (
    BundledGenerationOutputs,
    GenerationHyperparameters,
)
from realhf.api.core.system_api import RolloutWorker as RolloutWorkerConfig
from realhf.api.core.system_api import WorkerInformation
from realhf.base import constants, name_resolve, names, network, testing
from realhf.system.push_pull_stream import NameResolvingZmqPusher
from realhf.system.rollout_worker import RolloutWorker
from tests.fixtures import *

N_PULLERS = 3


class MockPartialRolloutManager:
    def __init__(self, request_queue, reply_queue, **kwargs):
        self.request_queue = request_queue
        self.reply_queue = reply_queue
        self.internal_queue = []

    def get_num_gen_requests(self):
        return len(self.internal_queue)

    async def run_step(self):
        async def poll_fresh_requests():
            for _ in range(8):
                try:
                    qid, prompt_token_ids, gconfig = self.request_queue.get_nowait()
                    assert isinstance(qid, str)
                    assert isinstance(prompt_token_ids, list)
                    assert all(isinstance(x, int) for x in prompt_token_ids)
                    assert isinstance(gconfig, GenerationHyperparameters)
                    self.internal_queue.append(qid)
                except QueueEmpty:
                    await asyncio.sleep(0.01)

        async def poll_old_requests():
            for _ in range(8):
                if random.random() < 0.5 and len(self.internal_queue) > 0:
                    # responses may not return in order
                    idx = random.randint(0, len(self.internal_queue) - 1)
                    qid = self.internal_queue.pop(idx)
                    out = BundledGenerationOutputs(
                        qid=qid,
                        prompt_ids=[1],
                        output_ids=[[2], [3]],
                        seqs=[[1, 2], [1, 3]],
                        logprobs=[[0.0, 0.1], [0.0, 2.0]],
                        no_eos=[True, True],
                        version_start=[0, 1],
                        version_end=[1, 2],
                    )
                    await self.reply_queue.put(out)
                else:
                    await asyncio.sleep(0.01)

        await asyncio.gather(poll_fresh_requests(), poll_old_requests())


@pytest.fixture
def rollout_workers(request):
    testing.clear_name_resolve()
    constants.set_experiment_trial_names(
        testing._DEFAULT_EXPR_NAME, testing._DEFAULT_TRIAL_NAME
    )

    # add name resolve to make zmq pusher happy
    puller_ports = network.find_multiple_free_ports(N_PULLERS)
    for puller_index in range(N_PULLERS):
        name = names.stream_pullers(
            testing._DEFAULT_EXPR_NAME,
            testing._DEFAULT_TRIAL_NAME,
        )
        name_resolve.add_subentry(name, str(puller_index))
        name = names.push_pull_stream(
            testing._DEFAULT_EXPR_NAME,
            testing._DEFAULT_TRIAL_NAME,
            f"puller{puller_index}",
        )
        name_resolve.add(name, f"localhost:{puller_ports[puller_index]}")

    with (
        patch.object(NameResolvingZmqPusher, "push", return_value=None) as mock_push,
        patch(
            "realhf.system.rollout_worker.PartialRolloutManager",
            new=MockPartialRolloutManager,
        ),
    ):

        ms = [RolloutWorker() for _ in range(request.param)]
        yield ms


@pytest.mark.parametrize("rollout_workers", [1, 2, 3], indirect=True)
@pytest.mark.parametrize("offpolicyness", [0, 1, 4])
@pytest.mark.asyncio
async def test_offpolicyness_control(
    rollout_workers, save_path, dataset, offpolicyness
):
    train_batch_size = 8
    config = RolloutWorkerConfig(
        base_seed=0,
        model_name=ModelName("default", 0),
        max_head_offpolicyness=offpolicyness,
        train_batch_size=train_batch_size,
        tokenizer_path="/storage/openpsi/models/Qwen__Qwen2.5-1.5B/",
        new_tokens_per_chunk=1024,
        max_concurrent_rollouts=128,
        env=EnvServiceAbstraction("null"),
        agent=AgentAbstraction("null", dict(episode_length=5, traj_size=5)),
        datasets=[
            DatasetAbstraction(
                "prompt", args=dict(dataset_path=str(save_path / "dataset.jsonl"))
            )
        ],
        worker_info=WorkerInformation(
            experiment_name=testing._DEFAULT_EXPR_NAME,
            trial_name=testing._DEFAULT_TRIAL_NAME,
            worker_type="rollout_worker",
            worker_count=N_PULLERS * 2,
            worker_index=0,
        ),
    )
    for i, m in enumerate(rollout_workers):
        config = copy.deepcopy(config)
        config.worker_info.worker_index = i
        m._configure(config)

    for i in range(10 * (offpolicyness + 1)):
        for m in rollout_workers:
            await m._poll_async()

    # Ensure that data is not overly produced
    for m in rollout_workers:
        assert m.agent.ACT_GET_CNT > 0
        assert (
            (offpolicyness + 1) * train_batch_size >= m.push_stream.push.call_count > 0
        )

    # Increase the model version by 1
    version_name = names.model_version(
        constants.experiment_name(),
        constants.trial_name(),
        config.model_name.role,
    )
    name_resolve.add(version_name, "1")

    # Run the rollout worker again
    for i in range(10 * (offpolicyness + 1)):
        for m in rollout_workers:
            await m._poll_async()

    # The rollout worker should produce new samples
    for m in rollout_workers:
        assert (offpolicyness + 2) * train_batch_size >= m.push_stream.push.call_count
        assert (offpolicyness + 1) * train_batch_size < m.push_stream.push.call_count

    # Final clean up
    name_resolve.delete(version_name)
    for m in rollout_workers:
        await m._exit_async_tasks()
