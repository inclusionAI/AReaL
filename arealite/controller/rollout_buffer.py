from typing import List, Dict, Any
import torch
from tensordict import TensorDict, NonTensorData
from arealite.utils.padding import concat_padded_tensors
import threading

class RolloutBuffer(object):
    """
    A buffer to store rollout samples and aggregate it by group.
    """

    def __init__(self, train_batch_size: int = 512, batch_size_exceeding_num: int = 512, 
                 group_size: int = 8, mini_samples_per_group: int = 8, staleness_version: int = 1):
        self.train_batch_size = train_batch_size
        self.batch_size_exceeding_num = batch_size_exceeding_num
        assert mini_samples_per_group == group_size, "mini_samples_per_group should not equal to group_size does not support yet."
        self.group_size = group_size
        self.mini_samples_per_group = mini_samples_per_group
        self.staleness_version = staleness_version

        self.buffer = {} # key: query_id, value: dict<index_in_group, TensorDict>
        self.current_size = 0
        self.ready_to_train_sample_num = 0

        self.lock_ = threading.Lock()

    def get_current_size(self) -> int:
        """This read only method is not thread safe"""
        return self.current_size
    
    def get_train_batch_size(self) -> int:
        """This read only method is not thread safe"""
        return self.train_batch_size

    def get_ready_to_train_sample_num(self) -> int:
        """This read only method is not thread safe"""
        return self.ready_to_train_sample_num

    def add(self, sample: TensorDict):
        """
        Add a rollout sample to the buffer.
        :param sample: A TensorDict containing the rollout sample. This is the return of 
        PartialRolloutWorkflow.arun_episode, and should only have batch_size=1. All field 
        in the sample are unpacked.
        """
        assert sample.batch_size == torch.Size([1]), "Only support batch_size=1 sample."
        
        query_id = sample["query_id"][0]
        index_in_group = sample["index_in_group"][0]
        with self.lock_:
            assert (query_id not in self.buffer) or (index_in_group not in self.buffer[query_id]), \
                f"Sample with query_id {query_id} and index_in_group {index_in_group} already exists in the buffer."
            if query_id not in self.buffer:
                self.buffer[query_id] = {}
            self.buffer[query_id][index_in_group] = sample
            self.current_size += 1
            if len(self.buffer[query_id]) == self.mini_samples_per_group:
                self.ready_to_train_sample_num += self.mini_samples_per_group
            elif len(self.buffer[query_id]) > self.mini_samples_per_group:
                self.ready_to_train_sample_num += 1
        
    def is_sufficient(self) -> bool:
        with self.lock_:
            return self.ready_to_train_sample_num >= self.train_batch_size

    def current_has(self, size: int) -> bool:
        """
        Return True if rollout buffer has at least `size` samples.
        This method is thread safe.
        """
        with self.lock_:
            return self.current_size >= size

    def pop_batched_rollout_res(self) -> TensorDict:
        """
        Return all samples that are ready to train(num samples in the group > mini_samples_per_group), 
        and remove them from the buffer.
        """
        assert self.is_sufficient(), \
            f"Not enough samples to form a training batch. Current ready_to_train_sample_num: {self.ready_to_train_sample_num}, train_batch_size: {self.train_batch_size}"
        
        with self.lock_:
            results = []
            for query_id in list(self.buffer.keys()):
                group_samples = self.buffer[query_id]
                if len(group_samples) >= self.mini_samples_per_group:
                    for index_in_group, sample in group_samples.items():
                        for key in list(sample.keys()):
                            # del "query_id", "index_in_group", "task", "solutions"
                            if not isinstance(sample[key], torch.Tensor):
                                sample.del_(key)
                        results.append(sample)
                    del self.buffer[query_id]
                    self.current_size -= len(group_samples)
                    self.ready_to_train_sample_num -= len(group_samples)
                if len(results) >= self.train_batch_size:
                    break
        return concat_padded_tensors(results)
    
    def expire_stale_samples(self, current_version: int):
        """
        Remove samples that are older than the staleness version.
        :param current_version: The current version of the model.
        """
        with self.lock_:
            expired_samples = 0
            for query_id in list(self.buffer.keys()):
                group_samples = self.buffer[query_id]
                for index_in_group in list(group_samples.keys()):
                    sample = group_samples[index_in_group]
                    sample_version = sample["versions"][0][0]
                    if sample_version < current_version - self.staleness_version:
                        expired_samples += 1
                        del group_samples[index_in_group]
                if len(group_samples) == 0:
                    del self.buffer[query_id]
            self.current_size -= expired_samples
            self.ready_to_train_sample_num -= expired_samples

    def pop_all_cached_samples(self) -> List[Dict[str, Any]]:
        """
        Return all cached samples in the buffer as dataset format, it will containes 
        "previous_ids", "previous_version", "task", "solutions", "query_id" and "index_in_group".
        Tensors will be converted to lists, and NonTensorData will be converted to their data.
        """
        with self.lock_:
            all_samples = []
            for query_id, group_samples in self.buffer.items():
                for index_in_group, sample in group_samples.items():
                    previous_prompt_len = sample["prompt_mask"].count_nonzero()
                    previous_prompt_len = previous_prompt_len.item() if isinstance(previous_prompt_len, torch.Tensor) else previous_prompt_len
                    sample_data = {
                        "query_id": [sample["query_id"][0]],
                        "index_in_group": [sample["index_in_group"][0]],
                        "task": [sample["task"][0]],
                        "solutions": [sample["solutions"][0]],
                        "previous_ids": [sample["input_ids"].squeeze().tolist()],
                        "previous_version": [sample["versions"].squeeze().tolist()],
                        "previous_logprobs": [sample["logprobs"].squeeze().tolist()],
                        "previous_prompt_len": [previous_prompt_len],
                        "previous_seq_no_eos_mask": [sample["seq_no_eos_mask"].item()],
                        "previous_rewards": [sample["rewards"].item()],
                    }
                    all_samples.append(sample_data)
            self.buffer.clear()
            self.current_size = 0
            self.ready_to_train_sample_num = 0
            return all_samples
        
    def save(self, path: str) -> None:
        with self.lock_:
            data = {
                "train_batch_size": self.train_batch_size,
                "batch_size_exceeding_num": self.batch_size_exceeding_num,
                "group_size": self.group_size,
                "mini_samples_per_group": self.mini_samples_per_group,
                "staleness_version": self.staleness_version,
                "current_size": self.current_size,
                "ready_to_train_sample_num": self.ready_to_train_sample_num,
                "buffer": self.buffer
            }
            torch.save(data, path)

    def load(self, path: str) -> None:
        with self.lock_:
            data = torch.load(path)
            self.train_batch_size = data["train_batch_size"]
            self.batch_size_exceeding_num = data["batch_size_exceeding_num"]
            self.group_size = data["group_size"]
            self.mini_samples_per_group = data["mini_samples_per_group"]
            self.staleness_version = data["staleness_version"]
            self.current_size = data["current_size"]
            self.ready_to_train_sample_num = data["ready_to_train_sample_num"]
            self.buffer = data["buffer"]
