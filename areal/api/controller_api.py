import abc
from typing import Any

import torch


class DistributedBatch(abc.ABC):
    """Abstract base class for data exchange between controller and engine.

    This class defines the interface for handling batched data operations
    between controller and engine components in a distributed environment.
    It supports two modes of data transfer:
    - Memory mode: Full data transfer through memory
    - File mode: Transfer only metadata between controller and engine
    """

    @classmethod
    def from_dict(cls, dataset: dict[str, torch.Tensor | Any]) -> "DistributedBatch":
        """Create a DistributedBatch from a dictionary format dataset.

        Parameters
        ----------
        dataset : Dict[str, Union[torch.Tensor, Any]]
            Dictionary format dataset to convert, supporting Tensor, scalar, and list types

        Returns
        -------
        DistributedBatch
            DistributedBatch instance created from the dictionary
        """
        raise NotImplementedError()

    @classmethod
    def from_list(
        cls, dataset: list[dict[str, torch.Tensor | Any]]
    ) -> "DistributedBatch":
        """Create a DistributedBatch from a list format dataset.

        Parameters
        ----------
        dataset : List[Dict[str, Union[torch.Tensor, Any]]]
            List format dataset to convert, supporting Tensor, scalar, and list types

        Returns
        -------
        DistributedBatch
            DistributedBatch instance created from the list
        """
        raise NotImplementedError()

    def chunk(self, dp_size: int) -> list["DistributedBatch"]:
        """Split the dataset across data parallel processes.

        This function preserves the original order of data, ensuring that
        the sequence of samples in the concatenated result matches the
        original dataset order.

        Parameters
        ----------
        dp_size : int
            Number of data parallel processes

        Returns
        -------
        list[DistributedBatch]
            List of DistributedBatch objects, one for each process
        """
        raise NotImplementedError()

    def chunk_by_ffd(self, group_size: int, dp_size: int) -> list["DistributedBatch"]:
        """Split data by sequence length using First Fit Decreasing algorithm.

        Parameters
        ----------
        group_size : int
            Size of each group
        dp_size : int
            Number of data parallel processes to split into

        Returns
        -------
        list[DistributedBatch]
            List of DistributedBatch objects
        """
        raise NotImplementedError()

    def union_(self, other: "DistributedBatch") -> "DistributedBatch":
        """In-place merge another batch into this one.

        Parameters
        ----------
        other : DistributedBatch
            Another batch to merge into this batch

        Returns
        -------
        DistributedBatch
            Merged batch (self)
        """
        raise NotImplementedError()

    def get_data(self) -> dict[str, torch.Tensor | Any]:
        """Get all data from the DistributedBatch.

        Returns
        -------
        Dict[str, Union[torch.Tensor, Any]]
            Dictionary where keys are field names and values can be Tensor, scalar, or list types
            containing all values for that field across the entire batch.
        """
        raise NotImplementedError()

    @staticmethod
    def concat(data: list["DistributedBatch"]) -> "DistributedBatch":
        """Concatenate multiple batches into a single batch.

        Parameters
        ----------
        data : list[DistributedBatch]
            List of batches to concatenate

        Returns
        -------
        DistributedBatch
            Concatenated batch
        """
        raise NotImplementedError()

    def __getitem__(self, key: int | str):
        """Get an item from the batch.

        Parameters
        ----------
        key : int or str
            Index or key to retrieve

        Returns
        -------
        Dict[str, Union[torch.Tensor, Any]] or Union[torch.Tensor, Any]
            Retrieved item
        """
        raise NotImplementedError()

    def __setitem__(self, key: str, value: torch.Tensor | Any):
        """Set an item in the batch.

        Parameters
        ----------
        key : str
            Key to set
        value : Union[torch.Tensor, Any]
            Value to set (Tensor, scalar, or list)
        """
        raise NotImplementedError()

    def __delitem__(self, key: int | str):
        """Delete an item from the batch.

        Parameters
        ----------
        key : int or str
            Index or key to delete
        """
        raise NotImplementedError()

    def __getstate__(self):
        """Serialize the batch for pickle dump.

        Returns
        -------
        dict
            Dictionary containing the state to be serialized
        """
        raise NotImplementedError()

    def __setstate__(self, state):
        """Restore the batch from pickle load.

        Parameters
        ----------
        state : dict
            Dictionary containing the serialized state
        """
        raise NotImplementedError()
