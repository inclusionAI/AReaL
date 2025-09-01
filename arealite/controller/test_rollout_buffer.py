import pytest
import torch
from tensordict import NonTensorData, TensorDict

from arealite.controller.rollout_buffer import RolloutBuffer


def make_sample(query_id, index_in_group, version=1):
    query_id = str(query_id)
    seq = [10, 22, 33]
    prompt_mask = [1, 1, 0]
    logprobs = [-0.1, -0.2, -0.3]
    reward = 0.5
    task_id = 42
    seq_no_eos_mask = False

    return TensorDict(
        {
            "input_ids": torch.tensor(seq).unsqueeze(0),
            "prompt_mask": torch.tensor(prompt_mask).unsqueeze(0),
            "logprobs": torch.tensor(logprobs).unsqueeze(0),
            "versions": torch.tensor([version]).unsqueeze(0),
            "attention_mask": torch.ones(len(seq)).unsqueeze(0),
            "rewards": torch.tensor([reward]),
            "seqlen": torch.tensor([len(seq)]),
            "task_ids": torch.tensor([task_id]),
            "seq_no_eos_mask": torch.tensor([seq_no_eos_mask]),
            "query_id": NonTensorData([query_id]),
            "index_in_group": NonTensorData([index_in_group]),
            "task": NonTensorData([42]),
            "solutions": NonTensorData([["solution1", "solution2"]]),
        },
        batch_size=[1],
    )


def test_add_and_get_current_size():
    buf = RolloutBuffer(train_batch_size=2, group_size=2, mini_samples_per_group=2)
    sample1 = make_sample(1, 0)
    sample2 = make_sample(1, 1)
    buf.add(sample1)
    assert buf.get_current_size() == 1
    buf.add(sample2)
    assert buf.get_current_size() == 2


def test_add_duplicate_raises():
    buf = RolloutBuffer(group_size=2, mini_samples_per_group=2)
    sample = make_sample(1, 0)
    buf.add(sample)
    with pytest.raises(AssertionError):
        buf.add(sample)


def test_is_sufficient():
    buf = RolloutBuffer(train_batch_size=2, group_size=2, mini_samples_per_group=2)
    buf.add(make_sample(1, 0))
    assert not buf.is_sufficient()
    buf.add(make_sample(1, 1))
    assert buf.is_sufficient()


def test_pop_batched_rollout_res(monkeypatch):
    buf = RolloutBuffer(train_batch_size=2, group_size=2, mini_samples_per_group=2)
    buf.add(make_sample(1, 0))
    buf.add(make_sample(1, 1))
    # Patch concat_padded_tensors to just return the list for test
    monkeypatch.setattr(
        "arealite.controller.rollout_buffer.concat_padded_tensors", lambda x: x
    )
    res = buf.pop_batched_rollout_res()
    assert isinstance(res, list)
    assert len(res) == 2
    assert buf.get_current_size() == 0
    assert buf.ready_to_train_sample_num == 0
    for r in res:
        # These fields should be deleted
        assert "query_id" not in r
        assert "index_in_group" not in r
        assert "task" not in r
        assert "solutions" not in r
        # These fields should remain
        assert "input_ids" in r
        assert "prompt_mask" in r
        assert "logprobs" in r
        assert "versions" in r
        assert "attention_mask" in r
        assert "rewards" in r
        assert "seqlen" in r
        assert "task_ids" in r
        assert "seq_no_eos_mask" in r
        # Check tensor types
        assert isinstance(r["input_ids"], torch.Tensor)
        assert isinstance(r["prompt_mask"], torch.Tensor)
        assert isinstance(r["logprobs"], torch.Tensor)
        assert isinstance(r["versions"], torch.Tensor)
        assert isinstance(r["attention_mask"], torch.Tensor)
        assert isinstance(r["rewards"], torch.Tensor)
        assert isinstance(r["seqlen"], torch.Tensor)
        assert isinstance(r["task_ids"], torch.Tensor)
        assert isinstance(r["seq_no_eos_mask"], torch.Tensor)


def test_expire_stale_samples():
    buf = RolloutBuffer(group_size=2, mini_samples_per_group=2, staleness_version=1)
    buf.add(make_sample(1, 0, version=1))
    buf.add(make_sample(1, 1, version=2))
    buf.expire_stale_samples(current_version=3)
    # Only version=1 should be expired (4-1 > 1)
    assert buf.get_current_size() == 1
    # Only sample with version=2 remains
    remaining = list(buf.buffer.values())[0]
    assert list(remaining.values())[0]["versions"][0][0] == 2

    buf.expire_stale_samples(current_version=4)
    # Now both samples should be expired
    assert buf.get_current_size() == 0


def test_pop_all_cached_samples():
    buf = RolloutBuffer(group_size=2, mini_samples_per_group=2)
    buf.add(make_sample(1, 0))
    buf.add(make_sample(1, 1))
    samples = buf.pop_all_cached_samples()
    assert isinstance(samples, list)
    assert len(samples) == 2
    for s in samples:
        print(s)
        assert "query_id" in s
        assert "index_in_group" in s
        assert "task" in s
        assert "solutions" in s
        assert "previous_ids" in s
        assert "previous_version" in s
        assert "previous_logprobs" in s
        assert "previous_prompt_len" in s
        assert "previous_seq_no_eos_mask" in s
        assert "previous_rewards" in s
        # Check that query_id and index_in_group are extracted from NonTensorData
        assert isinstance(s["query_id"], (list))
        assert len(s["query_id"]) == 1
        assert isinstance(s["query_id"][0], (str))
        assert len(s["index_in_group"]) == 1
        assert isinstance(s["index_in_group"][0], (int))
        assert len(s["task"]) == 1
        assert isinstance(s["task"][0], (int, float))
        assert isinstance(s["solutions"], list)
        assert isinstance(s["previous_ids"], list)
        assert isinstance(s["previous_version"], list)
        assert isinstance(s["previous_logprobs"], list)
        assert isinstance(s["previous_prompt_len"], list)
        assert isinstance(s["previous_prompt_len"][0], int)
        assert isinstance(s["previous_seq_no_eos_mask"], list)
        assert len(s["previous_seq_no_eos_mask"]) == 1
        assert isinstance(s["previous_seq_no_eos_mask"][0], (int, bool))
        assert isinstance(s["previous_rewards"], list)
        assert len(s["previous_rewards"]) == 1
        assert isinstance(s["previous_rewards"][0], (float, int))

    assert buf.get_current_size() == 0
    assert buf.ready_to_train_sample_num == 0
