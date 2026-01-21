from typing import Any, Dict, List, Union

import torch
from torch import Tensor

from areal.api.controller_api import DistributedBatch
from areal.utils.batch_utils import (
    convert_dict_to_list,
    convert_list_to_dict,
    validate_dict_dataset,
)
from areal.utils.datapack import ffd_allocate
from areal.utils.errors import FrameworkError


class DistributedBatchMemory(DistributedBatch):
    dataset = None

    @classmethod
    def from_dict(cls, dict_dataset: Dict[str, Union[Tensor, Any]]):
        """Create a DistributedBatchMemory from dictionary format dataset.

        Parameters
        ----------
        dict_dataset : Dict[str, Union[Tensor, Any]]
            Dictionary format dataset, where values can be Tensor, scalar, or list types

        Returns
        -------
        DistributedBatchMemory
            New DistributedBatchMemory instance
        """
        validate_dict_dataset(dict_dataset)
        instance = cls.__new__(cls)
        instance.dataset = dict_dataset
        return instance

    @classmethod
    def from_list(cls, list_dataset: List[Dict[str, Union[Tensor, Any]]]):
        """Create a DistributedBatchMemory from list format dataset.

        Parameters
        ----------
        list_dataset : List[Dict[str, Union[Tensor, Any]]]
            List format dataset

        Returns
        -------
        DistributedBatchMemory
            New DistributedBatchMemory instance
        """
        dict_dataset = convert_list_to_dict(list_dataset)
        return cls.from_dict(dict_dataset)

    def chunk(self, dp_size: int) -> list["DistributedBatchMemory"]:
        """Split the dataset across data parallel processes.

        This function preserves the original order of data, ensuring that
        the sequence of samples in the concatenated result matches the
        original dataset order."""
        if not self.dataset:
            raise FrameworkError(
                "FrameworkError",
                "DistributedBatchMemoryError",
                "Cannot split empty dataset",
            )

        total = self._get_total_size()
        part_size = (total + dp_size - 1) // dp_size
        batches = []
        for i in range(dp_size):
            start = i * part_size
            end = min(start + part_size, total)
            split_data = {}
            for k, v in self.dataset.items():
                if isinstance(v, torch.Tensor):
                    split_data[k] = v[start:end].clone()
                elif isinstance(v, list):
                    split_data[k] = v[start:end]
                else:
                    # For scalar values, keep as-is
                    split_data[k] = v
            batch = self.__class__.__new__(self.__class__)
            batch.dataset = split_data
            batches.append(batch)
        return batches

    def chunk_by_ffd(
        self, group_size: int, dp_size: int
    ) -> list["DistributedBatchMemory"]:
        """Split data by sequence length using First Fit Decreasing algorithm

        Parameters
        ----------
        group_size : int
            Size of each group
        dp_size : int
            Number of data parallel processes

        Returns
        -------
        list[DistributedBatchMemory]
            List of DistributedBatchMemory objects
        """
        total_size = self._get_total_size()
        assert (
            total_size % group_size == 0
        ), "tensor length must be devided by group_size"

        # Handle seqlen calculation for both tensor and scalar types
        if "seqlen" in self.dataset.keys():
            seqlen = self.dataset["seqlen"]
            if isinstance(seqlen, torch.Tensor):
                reshaped = seqlen.view(-1, group_size)
                group_total_lens = reshaped.sum(dim=1)
            else:
                # Handle scalar/list case
                seqlen_list = (
                    seqlen if isinstance(seqlen, list) else [seqlen] * total_size
                )
                reshaped = [
                    seqlen_list[i : i + group_size]
                    for i in range(0, len(seqlen_list), group_size)
                ]
                group_total_lens = [sum(group) for group in reshaped]
        elif "attention_mask" in self.dataset.keys():
            attention_mask = self.dataset["attention_mask"]
            if isinstance(attention_mask, torch.Tensor):
                seqlen = attention_mask.sum(1)
                reshaped = seqlen.view(-1, group_size)
                group_total_lens = reshaped.sum(dim=1)
            else:
                # Fallback for scalar types - assume equal length
                group_total_lens = [group_size] * (total_size // group_size)
        else:
            # Fallback when neither seqlen nor attention_mask exists
            group_total_lens = [group_size] * (total_size // group_size)

        unsorted_group_rebalanced_indexs = ffd_allocate(
            group_total_lens, int(1e12), dp_size
        )
        group_rebalanced_indexs = sorted(
            [sorted(g) for g in unsorted_group_rebalanced_indexs]
        )
        batches = []
        for i in range(dp_size):
            indexes = []
            for group_index in group_rebalanced_indexs[i]:
                tmp_indexs = list(
                    range(
                        group_size * group_index, group_size * group_index + group_size
                    )
                )
                indexes.extend(tmp_indexs)
            split_data = {}
            for k, v in self.dataset.items():
                if isinstance(v, torch.Tensor):
                    split_data[k] = v[indexes]
                elif isinstance(v, list):
                    split_data[k] = [v[i] for i in indexes]
                else:
                    # For scalar values, keep as-is (they represent single sample)
                    split_data[k] = v
            batch = self.__class__.__new__(self.__class__)
            batch.dataset = split_data
            batches.append(batch)
        return batches

    def union(self, other: "DistributedBatchMemory") -> "DistributedBatchMemory":
        """Merge another batch with this one"""
        merged_data = {k: v for k, v in self.dataset.items()}
        for k, v in other.dataset.items():
            if k in merged_data:
                if isinstance(merged_data[k], torch.Tensor) and isinstance(
                    v, torch.Tensor
                ):
                    merged_data[k] = torch.cat([merged_data[k], v], dim=0)
                elif isinstance(merged_data[k], list) and isinstance(v, list):
                    merged_data[k] = merged_data[k] + v
                else:
                    # Handle mixed types or scalar values
                    if isinstance(merged_data[k], list):
                        merged_data[k].append(v)
                    else:
                        merged_data[k] = [merged_data[k], v]
            else:
                merged_data[k] = v
        batch = self.__class__.__new__(self.__class__)
        batch.dataset = merged_data
        return batch

    def _get_total_size(self) -> int:
        """Get the total size of the dataset, supporting both tensor and scalar types.

        Returns
        -------
        int
            The total size (batch size) of the dataset
        """
        if not self.dataset:
            return 0

        first_value = next(iter(self.dataset.values()))
        if isinstance(first_value, torch.Tensor):
            return first_value.shape[0]
        elif isinstance(first_value, list):
            return len(first_value)
        else:
            # For scalar values, assume it's a single sample
            return 1

    def get_data(self) -> Dict[str, Union[torch.Tensor, Any]]:
        """Get all data from the DistributedBatchMemory.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary where keys are field names and values are 1D tensors
            containing all values for that field across the entire batch.
            Each tensor is formed by concatenating all individual values
            for the corresponding field.
        """
        if not self.dataset:
            return {}

        # Get all attributes from the first sample
        first_item = self[0]
        attrs = list(first_item.keys())

        # Construct attr -> tensor mapping
        batch_data = {}
        for attr in attrs:
            batch_data[attr] = self[attr]

        return batch_data

    @staticmethod
    def concat(data: list["DistributedBatchMemory"]) -> "DistributedBatchMemory":
        """Concatenate multiple DistributedBatchMemory objects

        Parameters
        ----------
        data : list[DistributedBatchMemory]
            List of DistributedBatchMemory objects to concatenate

        Returns
        -------
        DistributedBatchMemory
            Single concatenated DistributedBatchMemory object
        """
        if not data:
            batch = DistributedBatchMemory.__new__(DistributedBatchMemory)
            batch.dataset = {}
            return batch

        merged_data = {}
        for batch in data:
            for k, v in batch.dataset.items():
                if k in merged_data:
                    if isinstance(merged_data[k], torch.Tensor) and isinstance(
                        v, torch.Tensor
                    ):
                        merged_data[k] = torch.cat([merged_data[k], v], dim=0)
                    elif isinstance(merged_data[k], list) and isinstance(v, list):
                        merged_data[k] = merged_data[k] + v
                    else:
                        # Handle mixed types or scalar values
                        if isinstance(merged_data[k], list):
                            merged_data[k].append(v)
                        else:
                            merged_data[k] = [merged_data[k], v]
                else:
                    merged_data[k] = v
        result = DistributedBatchMemory.__new__(DistributedBatchMemory)
        result.dataset = merged_data
        return result

    def __getstate__(self):
        return {"dataset": self.dataset}

    def __setstate__(self, state):
        self.dataset = state["dataset"]

    def __getitem__(self, key):
        if isinstance(key, int):
            return {k: v[key] for k, v in self.dataset.items()}
        elif isinstance(key, str):
            return self.dataset[key]
        else:
            raise FrameworkError(
                "FrameworkError",
                "DistributedBatchMemoryError",
                "Key must be int or str",
            )

    def __setitem__(self, key, value):
        """Support two assignment methods:
        - str key: update entire attribute tensor
        - int index: requires converting data to list format for update (less efficient, avoid if possible)
        """
        if isinstance(key, str):
            # Update entire attribute tensor or scalar/list value
            if self.dataset:
                expected_total_size = self._get_total_size()
                if isinstance(value, torch.Tensor):
                    if value.shape[0] != expected_total_size:
                        raise FrameworkError(
                            "FrameworkError",
                            "DistributedBatchMemoryError",
                            f"The batch size of the tensor does not match. Expected {expected_total_size}, actual {value.shape[0]}",
                        )
                elif isinstance(value, list):
                    if len(value) != expected_total_size:
                        raise FrameworkError(
                            "FrameworkError",
                            "DistributedBatchMemoryError",
                            f"The batch size of the list does not match. Expected {expected_total_size}, actual {len(value)}",
                        )
            self.dataset[key] = value
        else:
            raise FrameworkError(
                "FrameworkError", "DistributedBatchMemoryError", f"key must be str"
            )

    def __delitem__(self, key):
        """Support two deletion methods:
        - int index: delete sample at specified position
        - str key: delete entire attribute
        """
        if isinstance(key, int):
            # Convert to list format for deletion
            list_dataset = convert_dict_to_list(self.dataset)
            del list_dataset[key]
            self.dataset = convert_list_to_dict(list_dataset)
        elif isinstance(key, str):
            # Delete entire attribute directly
            if key in self.dataset:
                del self.dataset[key]
        else:
            raise FrameworkError(
                "FrameworkError",
                "DistributedBatchMemoryError",
                f"key: {type(key)} must be str or int",
            )

    def __str__(self):
        if not self.dataset:
            return "DistributedBatchMemory<empty>"

        total_size = self._get_total_size()
        keys = list(self.dataset.keys())
        shapes = {}
        for k, v in self.dataset.items():
            if isinstance(v, torch.Tensor):
                shapes[k] = v.shape
            elif isinstance(v, list):
                shapes[k] = f"list[{len(v)}]"
            else:
                shapes[k] = f"scalar({type(v).__name__})"
        return f"DistributedBatchMemory<total_size={total_size}, keys={keys}, shapes={shapes}>"

    def __len__(self):
        """Return the total size."""
        return self._get_total_size()

    def __repr__(self):
        return self.__str__()
