# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
import json
import os
import random
import time
from contextlib import contextmanager
from functools import lru_cache

# NOTE: We don't sue wildcard importing here because the type
# `Sequence` has a very similar name to `SequenceSample`.
# We don't want to confuse them.
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
import transformers

# NOTE: We only use pandatic dataclasses for SequenceSample
# such that it will perform automatic checks.
from pydantic import Field
from pydantic import dataclasses as pdclasses
from pydantic import field_validator, model_validator

from realhf.api.cli_args import MicroBatchSpec
from realhf.api.core import config as config_api
from realhf.base import constants, datapack, logging, seeding
from realhf.utils import load_hf_or_local_file

logger = logging.getLogger("api.data")

RL_TASKS = ["math", "code", "rlhf", "stem"]


@lru_cache(maxsize=8)
def load_hf_tokenizer(
    model_name_or_path: str,
    fast_tokenizer=True,
    padding_side: Optional[str] = None,
) -> transformers.PreTrainedTokenizerFast:
    kwargs = {}
    if padding_side is not None:
        kwargs["padding_side"] = padding_side
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        fast_tokenizer=fast_tokenizer,
        trust_remote_code=True,
        force_download=True,
        **kwargs,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


@pdclasses.dataclass
class SequenceSplitSpec:
    partitions: Optional[List[Tuple[int, int]]] = None
    sizes: Optional[List[int]] = None

    @model_validator(mode="after")
    def _validate_partitions(self) -> "SequenceSplitSpec":
        if self.partitions is not None:
            bound = 0
            for start, end in self.partitions:
                if start >= end:
                    raise ValueError(f"Partition {start}-{end} is empty.")
                if start != bound:
                    raise ValueError(f"Partition {start}-{end} is not contiguous.")
                bound = end

        if self.sizes is None and self.partitions is None:
            raise ValueError("Either sizes or partitions must be provided.")
        elif self.sizes is not None and self.partitions is not None:
            if len(self.sizes) != len(self.partitions):
                raise ValueError("Sizes and partitions are not the consistent.")
            if self.sizes != [end - start for start, end in self.partitions]:
                raise ValueError("Sizes and partitions are not the consistent.")
        elif self.sizes is None:
            self.sizes = [end - start for start, end in self.partitions]
        elif self.partitions is None:
            offsets = np.cumsum([0] + self.sizes)
            self.partitions = [
                (offsets[i], offsets[i + 1]) for i in range(len(self.sizes))
            ]

        return self


@pdclasses.dataclass(config=dict(arbitrary_types_allowed=True))
class SequenceSample:
    """The data structure used to represent sequence data.

    Each piece of data is assumed to have several "keys" (like a dictionary),
    with each key potentially corresponding to multiple sequences.

    For example, when running PPO, multiple responses can be generated for each prompt.
    If there are 2 prompts, each with 3 responses, the batch might look like:

    .. code-block:: console

        >>> s = SequenceSample(...)
        >>> s.keys
        {'resp', 'prompt'}
        >>> s.seqlens
        {'prompt': [[13], [6]], 'resp': [[6, 17, 15], [13, 15, 13]]}
        >>> s.data
        {'prompt': torch.tensor([...]), 'resp': torch.tensor([...])}

    Key points:

    - Data with different batch indices can have varying lengths (e.g., the first prompt has a length of 13
      while the second has a length of 6).

    - A key (e.g., "response") can correspond to multiple sequences with different lengths.
      Additionally, the number of sequences for each key can differ from the number of sequences for the data.
      For example, the first prompt may have 2 responses, and the second may have 3.

    - Regardless of the batch size or the number of sequences stored for each key,
      the data is concatenated into a 1D tensor. The outer dimension represents the batch size,
      and the inner dimension represents the number of sequences for the key.

    This data structure facilitates easy gathering, splitting,
    and transferring of non-padded batches between different GPUs.

    :param keys: The keys of the data.
    :type keys: Set[str]
    :param trailing_shapes: The trailing shapes of the data,
        excluding the first dimension, which must be the sequence length.
        Used to construct the receiving buffer for data transfer.
    :type trailing_shapes: Dict[str, torch.Size | Tuple | None]
    :param dtypes: The types of the data. Used to construct
        the receiving buffer for data transfer.
    :type dtypes: Dict[str, torch.dtype | None]
    :param ids: Unique identifiers for each piece of data.
        Should be provided in the dataset implementation.
        Used to append new data to the buffer after a model function call.
    :type ids: List[Hashable]
    :param seqlens: The sequence lengths of each sequence in the data. For a given key,
        this should be a list of lists of integers. The outer list represents the batch size,
        while the inner lists represent the sequence lengths for this key.
        Python-native lists are used here because (1) pickling torch.Tensor or numpy array is inefficient,
        and (2) the size of the inner lists can vary across the batch, making 2D arrays impractical.
    :type seqlens: Dict[str, List[List[int]]]
    :param data: The actual concatenated data. If this is None,
        the sample is a metadata-only sample used by the master worker.
        The specification of the data should be consistent with the seqlens,
        dtypes, and trailing_shapes.
    :type data: Optional[Dict[str, torch.Tensor | None]]
    :param metadata: Metadata for the sample. It should be a
        dictionary of lists, provided in the dataset implementation.
        Note that adding metadata can slow down data transfer.
    :type metadata: Dict[str, List[Any]]
    """

    keys: Set[str]
    trailing_shapes: Dict[str, torch.Size | Tuple | None]
    dtypes: Dict[str, torch.dtype | None]

    ids: List[Hashable]

    seqlens: Dict[str, List[List[int]]]

    data: Optional[Dict[str, torch.Tensor | None]] = None

    metadata: Dict[str, List[Any]] = Field(default_factory=dict)

    @field_validator("ids")
    @classmethod
    def _validate_ids(cls, ids: List[Hashable]) -> List[str]:
        ids = list(map(str, ids))
        if len(ids) != len(set(ids)):
            raise ValueError(f"IDs contain duplicates: {ids}.")
        return ids

    @field_validator("trailing_shapes")
    @classmethod
    def _validate_trailing_shapes(
        cls, trailing_shapes: Dict
    ) -> Dict[str, Tuple | None]:
        for k, v in trailing_shapes.items():
            if v is not None:
                trailing_shapes[k] = tuple(v)
        return trailing_shapes

    @field_validator("keys")
    @classmethod
    def _validate_keys_type(cls, keys: Iterable) -> Set[str]:
        keys_ = set(keys)
        if len(keys_) != len(keys):
            raise ValueError(f"Keys contain duplicates: {keys}.")
        return keys_

    @field_validator("seqlens")
    @classmethod
    def _validate_seqlens_device_dtype(
        cls, seqlens: Dict[str, List[torch.Tensor]]
    ) -> Dict[str, List[torch.Tensor]]:
        for k, lens in seqlens.items():
            assert isinstance(lens, list)
            assert all(isinstance(l, list) for l in lens)
            for i, lens_ in enumerate(lens):
                assert all(isinstance(l_, int) for l_ in lens_)
        return seqlens

    @model_validator(mode="after")
    def _validate_list_length(self) -> "SequenceSample":
        cond = True
        l = len(self.ids)
        cond &= all(len(lens) == l for lens in self.seqlens.values())
        if not cond:
            raise ValueError(
                f"Lengths of ids({len(self.ids)})"
                f"/seqlens({self.seqlens}) "
                "are not the same."
            )

        return self

    @model_validator(mode="after")
    def _validate_keys(self) -> "SequenceSample":
        cond = True
        cond &= self.keys == set(self.seqlens.keys())
        cond &= self.keys == set(self.trailing_shapes.keys())
        cond &= self.keys == set(self.dtypes.keys())
        if self.data is not None:
            cond &= self.keys == set(self.data.keys())
        if not cond:
            err = (
                f"Keys are mismatched. "
                f"keys={self.keys}, "
                f"seqlens keys={set(self.seqlens.keys())}, "
                f"trailing_shapes keys={set(self.trailing_shapes.keys())}, "
                f"dtypes keys={set(self.dtypes.keys())}"
            )
            if self.data is not None:
                err += f", data keys={set(self.data.keys())}"
            raise KeyError(err)
        return self

    @model_validator(mode="after")
    def _validate_shapes(self) -> "SequenceSample":
        if self.data is None:
            return self
        acc_seqlen = {
            k: sum(sum(lens) for lens in lens_list)
            for k, lens_list in self.seqlens.items()
        }
        for k, v in self.data.items():
            if v is None:
                continue
            if v.shape != (acc_seqlen[k], *self.trailing_shapes[k]):
                raise ValueError(
                    f"Key: {k}, Data shape {v.shape} does not match "
                    f"configured shape {(acc_seqlen[k], *self.trailing_shapes[k])}."
                )
        return self

    @model_validator(mode="after")
    def _validate_dtypes(self) -> "SequenceSample":
        if self.data is None:
            return self
        for k, v in self.data.items():
            if v is None:
                continue
            if v.dtype != self.dtypes[k]:
                raise ValueError(
                    f"Data dtype {v.dtype} "
                    f"does not match configured "
                    f"dtype {self.dtypes[k]}."
                )
        return self

    @classmethod
    def gather(cls, samples: List["SequenceSample"], keys: Optional[List[str]] = None):
        """Gather a list of SequenceSample objects into a single batch.

        :param samples: A list of SequenceSample objects to be gathered.
        :type samples: List[SequenceSample]
        :param keys: The keys to be gathered. Only a subset of keys can
            be gathered. If None, the keys from the first sample will be
            used.
        :type keys: Optional[List[str]]
        """
        if keys is None:
            keys = samples[0].keys
        else:
            keys = set(keys)

        seqlens = {k: sum([s.seqlens[k] for s in samples], []) for k in keys}
        if samples[0].data is not None:
            data = {
                k: (
                    torch.cat([s.data[k] for s in samples], dim=0)
                    if samples[0].data[k] is not None
                    else None
                )
                for k in keys
            }
        else:
            data = None
        id_ = sum([s.ids for s in samples], [])
        metadata = {
            k: sum([s.metadata[k] for s in samples], []) for k in samples[0].metadata
        }
        with cls.disable_validation():
            return cls(
                keys=keys,
                dtypes={key: samples[0].dtypes[key] for key in keys},
                trailing_shapes={key: samples[0].trailing_shapes[key] for key in keys},
                ids=id_,
                seqlens=seqlens,
                data=data,
                metadata=metadata,
            )

    def _get_split_key(self) -> str:
        acc_seqlen = {k: sum(sum(l) for l in lens) for k, lens in self.seqlens.items()}
        return max(acc_seqlen, key=acc_seqlen.get)

    def split_with_spec(self, spec: SequenceSplitSpec) -> List["SequenceSample"]:
        """Split the data according to the given spec."""
        samples = []
        data_offset = {k: 0 for k in self.keys}
        for start, end in spec.partitions:
            new_seqlens = {
                k: lens_list[start:end] for k, lens_list in self.seqlens.items()
            }
            _data_len = {
                k: sum(sum(lens) for lens in lens_list)
                for k, lens_list in new_seqlens.items()
            }
            if self.data is not None:
                new_data = {
                    k: (
                        v[data_offset[k] : _data_len[k] + data_offset[k]]
                        if v is not None
                        else None
                    )
                    for k, v in self.data.items()
                }
            else:
                new_data = None
            for k in self.keys:
                data_offset[k] += _data_len[k]
            new_id = self.ids[start:end]
            for k, v in self.metadata.items():
                if not isinstance(v, list):
                    raise ValueError(
                        f"Unknown how to split non-list metadata: ({k}, {v})."
                    )
            with self.disable_validation():
                samples.append(
                    SequenceSample(
                        dtypes=self.dtypes,
                        trailing_shapes=self.trailing_shapes,
                        keys=self.keys,
                        ids=new_id,
                        seqlens=new_seqlens,
                        data=new_data,
                        metadata={k: v[start:end] for k, v in self.metadata.items()},
                    )
                )
        return samples

    def split_with_lengths(
        self, mb_spec: MicroBatchSpec, lens: List[int]
    ) -> Tuple[List["SequenceSample"], List[int] | np.ndarray, List[int] | np.ndarray]:
        group_indices = datapack.ffd_allocate(
            lens, mb_spec.max_tokens_per_mb, min_groups=mb_spec.n_mbs
        )
        group_indices = sorted([sorted(g) for g in group_indices])

        forward_indices = datapack.flat2d(group_indices)
        sample = SequenceSample.reorder(self, forward_indices)

        backward_indices = np.zeros(self.bs, dtype=np.int64)
        backward_indices[forward_indices] = np.arange(self.bs)

        spec = SequenceSplitSpec(sizes=[len(group) for group in group_indices])

        return sample.split_with_spec(spec), forward_indices, backward_indices

    def split(
        self, mb_spec: MicroBatchSpec
    ) -> Tuple[List["SequenceSample"], List[int] | np.ndarray, List[int] | np.ndarray]:
        """Split the data into `n_mbs` parts.

        :param mb_spec: The configuration to split the data into.
            `n_mbs` is the minimum number of micro-batches,
            `max_tokens_per_mb` is the maximum number of tokens in each micro-batch.
            If `max_tokens_per_mb` is a large value, defaults to balanced split.
        :type mb_spec: MicroBatchSpec
        """
        lens = [sum(lens) for lens in self.seqlens[self._get_split_key()]]
        return self.split_with_lengths(mb_spec, lens)

    def synced_data_parallel_split(
        self, mb_spec: MicroBatchSpec
    ) -> List["SequenceSample"]:
        mb_inputs, *_ = self.split(mb_spec)
        all_n_mbs = [None for _ in range(constants.data_parallel_world_size())]
        dist.all_gather_object(
            all_n_mbs, len(mb_inputs), group=constants.data_parallel_group()
        )
        if all(mbs == len(mb_inputs) for mbs in all_n_mbs):
            return mb_inputs
        # This method is called when max_tokens_per_mb is given and during training.
        # In this case, we evenly partition sequences across DP ranks,
        # so the recursion will always terminate when n_mbs = bs // dp_size
        return self.synced_data_parallel_split(
            MicroBatchSpec.new(mb_spec, n_mbs=max(all_n_mbs))
        )

    @staticmethod
    def reorder(
        sample: "SequenceSample", indices: List[int] | np.ndarray
    ) -> "SequenceSample":
        assert set(list(indices)) == set(range(sample.bs))
        samples = sample.unpack()
        return SequenceSample.gather([samples[i] for i in indices])

    @staticmethod
    def reorder_output(
        x: torch.Tensor,
        expected_seqlens: List[List[int]],
        forward_indices: List[int] | np.ndarray,
        backward_indices: List[int] | np.ndarray,
    ) -> torch.Tensor:
        assert len(forward_indices) == len(backward_indices) == len(expected_seqlens)
        actual_seqlens = [expected_seqlens[i] for i in forward_indices]

        group_seqlens = [sum(s) for s in actual_seqlens]
        assert x.shape[0] == sum(group_seqlens), (
            x.shape[0],
            group_seqlens,
            len(group_seqlens),
            sum(group_seqlens),
        )
        offsets = [0] + np.cumsum(group_seqlens, axis=0).tolist()
        mbs = [x[s:e] for s, e in zip(offsets[:-1], offsets[1:])]
        return torch.cat([mbs[i] for i in backward_indices])

    def unpack(self):
        """Unpack a batch of data into individual pieces of data."""
        partitions = [(i, i + 1) for i in range(self.bs)]
        return self.split_with_spec(SequenceSplitSpec(partitions=partitions))

    def cuda(self):
        return self.to_device("cuda")

    def cpu(self):
        return self.to_device("cpu")

    def to_device(self, device: torch.device):
        """Move the data to device inplace."""
        if self.data is None:
            return self
        self.data = {
            k: v.to(device) if v is not None else None for k, v in self.data.items()
        }
        return self

    @property
    def bs(self):
        """The batch size or the number of data pieces in the sample."""
        return len(self.ids)

    def meta(self) -> "SequenceSample":
        """Create a new SequenceSample that does not contain any data."""
        with self.disable_validation():
            return SequenceSample(
                keys=self.keys,
                trailing_shapes=self.trailing_shapes,
                dtypes=self.dtypes,
                ids=self.ids,
                data=None,
                seqlens=self.seqlens,
                metadata=self.metadata,
            )

    def update_(self, other: "SequenceSample"):
        """Inplace update data from another SequenceSample.

        Used to amend newly produced data after a model function call.
        """
        self.keys = self.keys.union(other.keys)
        self.trailing_shapes.update(other.trailing_shapes)
        self.dtypes.update(other.dtypes)
        assert self.ids == other.ids, (self.ids, other.ids)
        if self.data is not None:
            self.data.update(other.data)
        self.seqlens.update(other.seqlens)
        self.metadata.update(other.metadata)

    @staticmethod
    def shuffled(sample: "SequenceSample") -> "SequenceSample":
        """Create a shuffled sample.
        Define it as a staticmethod because it is an out-of-place operation.
        (Think about the difference between `sorted` and `l.sort()`).
        """
        seed = seeding.get_shuffle_seed()
        rng = np.random.RandomState(seed)
        indices = np.arange(sample.bs)
        rng.shuffle(indices)
        return SequenceSample.reorder(sample, indices)

    @staticmethod
    def _resolve_seqlen_from_key(key, seqlens: List[int]) -> List[torch.Tensor]:
        if key in [
            "seq_no_eos_mask",
            "greedy_seq_no_eos_mask",
            "loss_mask",
            "rewards",
            "greedy_rewards",
            "base_scores",
            "task_ids",
        ]:
            return [[1] for _ in seqlens]
        elif key in [
            "input_ids",
            "packed_seq",
            "seq",
            "packed_logits_mask",
            "logits_mask",
            "prompt_mask",
            "greedy_prompt_mask",
            "packed_input_ids",
            "greedy_packed_input_ids",
            "values",
            "packed_prompts",
        ]:
            return [[seqlen] for seqlen in seqlens]
        elif key in [
            "packed_logprobs",
            "prox_logp",
            "logprobs",
            "packed_ref_logprobs",
            "ref_logprobs",
            "old_logp",
            "ref_logp",
            "advantages",
            "ppo_loss_mask",
            "kl_rewards",
            "returns",
        ]:
            return [[seqlen - 1] for seqlen in seqlens]
        else:
            raise NotImplementedError(
                f"Seqlen could not be resolved given key {key}. "
                f"Please explicltly construct the `SequenceSample` object"
                " without using the `from_default` method."
            )

    @classmethod
    def from_default(
        cls,
        seqlens: List[int],
        ids: List[Hashable],
        data: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Construct a `SequenceSample` object from default parameters.

        This helper function is intended for cases where each piece of data has
        a single sequence length (e.g., a single response for each prompt).
        The sequence lengths for different keys are resolved automatically
        according to the rules in ``_resolve_seqlen_from_key``. While this function
        can reduce boilerplate code, it may introduce potential bugs, so it should
        be used with caution.

        :param seqlens: The sequence lengths of each piece of data. This represents
            the length of the main attribute (e.g., `packed_input_ids`). Sequence lengths
            for other attributes (e.g., rewards and logprobs) are computed from this parameter.
            It is **NOT** the actual length of rewards or logprobs even if it is the only key
            in the data.
        :type seqlens: List[int]
        :param ids: Unique identifiers for each piece of data.
        :type ids: List[Hashable]
        :param data: The actual data.
        :type data: Dict[str, torch.Tensor]
        :param metadata: Metadata for the sample. Should be a dictionary where each value
            is a list with a length equal to the number of sequence lengths.
        :type metadata: Optional[Dict[str, Any]]
        """
        if metadata is None:
            metadata = {}
        for k, v in metadata.items():
            if not isinstance(v, list) or len(v) != len(seqlens):
                raise ValueError(
                    f"Metadata `{k}` should be a list of length {len(seqlens)}: {v}."
                )
        keys = set(data.keys())
        if isinstance(seqlens[0], list):
            assert len(seqlens[0]) == 1
            seqlens = [seqlen[0] for seqlen in seqlens]
        else:
            assert all(isinstance(seqlen, int) for seqlen in seqlens)
        seqlens = {key: cls._resolve_seqlen_from_key(key, seqlens) for key in keys}
        trailing_shapes = {
            key: data[key].shape[1:] if data[key] is not None else None for key in keys
        }
        dtypes = {
            key: data[key].dtype if data[key] is not None else None for key in keys
        }
        return cls(
            keys=keys,
            ids=ids,
            seqlens=seqlens,
            trailing_shapes=trailing_shapes,
            dtypes=dtypes,
            data=data,
            metadata=metadata,
        )

    def select(self, keys: List[str]):
        """Select a subset of keys inside the SequenceSample."""
        with self.disable_validation():
            keys = set(keys)
            return SequenceSample(
                keys=keys,
                dtypes={key: self.dtypes[key] for key in keys},
                trailing_shapes={key: self.trailing_shapes[key] for key in keys},
                ids=self.ids,
                seqlens={key: self.seqlens[key] for key in keys},
                data=(
                    None if self.data is None else {key: self.data[key] for key in keys}
                ),
                metadata=self.metadata,
            )

    def remap_keys_(self, remap: Dict[str, str]):
        """Inplace remap keys of the data.

        Useful for reusing the same interface implementation in
        different algorithms, where the data can be named differently.
        """
        for k in self.keys:
            if k in remap:
                new_k = remap[k]
                self.seqlens[new_k] = self.seqlens.pop(k)
                self.trailing_shapes[new_k] = self.trailing_shapes.pop(k)
                self.dtypes[new_k] = self.dtypes.pop(k)
                if self.data is not None:
                    self.data[new_k] = self.data.pop(k)
        self.keys = set(remap.get(k, k) for k in self.keys)

    @classmethod
    @contextmanager
    def disable_validation(cls):
        """Disable the expensive pydantic validation within this context.

        Used to accelerate gather/split/transfer operations since we
        have ensured that the data created in datasets and interfaces
        are valid.
        """
        original_init = cls.__init__

        def no_validation_init(self, *args, **kwargs):
            kwargs["keys"] = set(kwargs["keys"])
            self.__dict__.update(kwargs)

        cls.__init__ = no_validation_init
        try:
            yield
        finally:
            cls.__init__ = original_init

    def as_json_compatible(self) -> Dict:
        return dict(
            ids=self.ids,
            keys=list(self.keys),
            trailing_shapes={
                k: tuple(v) if v is not None else None
                for k, v in self.trailing_shapes.items()
            },
            dtypes={k: str(v) if v is not None else v for k, v in self.dtypes.items()},
            seqlens=self.seqlens,
            data={
                k: v.cpu().numpy().tolist() if v is not None else None
                for k, v in self.data.items()
            },
            metadata=self.metadata,
        )

    @classmethod
    def from_json_compatible(cls, data: Dict):
        dtypes = {}
        for k, dtype_str in data["dtypes"].items():
            if dtype_str is not None:
                dtypes[k] = getattr(torch, dtype_str.split(".")[1])
            else:
                dtypes[k] = None
        return cls(
            ids=data["ids"],
            keys=set(data["keys"]),
            trailing_shapes=data["trailing_shapes"],
            dtypes=dtypes,
            seqlens=data["seqlens"],
            data={
                k: torch.tensor(v, dtype=dtypes[k]) if v is not None else v
                for k, v in data["data"].items()
            },
            metadata=data["metadata"],
        )


@dataclasses.dataclass
class DataBatchMeta:
    dp_rank: int
    meta_sample: SequenceSample | None
    birth_times: List


@dataclasses.dataclass
class DatasetUtility:
    seed: int
    dp_rank: int
    world_size: int
    tokenizer: transformers.PreTrainedTokenizerFast

    def __post_init__(self):
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            if self.tokenizer.eos_token_id is None:
                raise ValueError("eos_token_id of tokenizer must be defined.")


def get_shuffle_indices(seed: int, size: int):
    """Generate shuffled indices given seed and (dataset) size."""
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


def load_shuffle_split_dataset(
    util: DatasetUtility,
    dataset_path: Optional[str] = None,
    dataset_builder: Optional[Callable[[], List[Dict[str, str]]]] = None,
):
    if dataset_path is not None:
        dataset_path = load_hf_or_local_file(dataset_path)
        if dataset_path.endswith(".jsonl"):
            with open(dataset_path, "r") as f:
                data = [json.loads(ff) for ff in f]
        else:
            raise NotImplementedError(f"Unknown dataset extension: {dataset_path}")
    else:
        assert dataset_builder is not None
        data = dataset_builder()

    if any("id" not in d for d in data):
        logger.warning(
            f'Key "id" not found in the dataset. Use indices as dataset IDs.'
        )
        for idx, d in enumerate(data):
            d["id"] = idx

    # NOTE: in the original way of seperating data, there is a chance that some DP rank
    # get no data, which will raise error in the dataset tokenizer.
    assert (
        len(data) >= util.world_size
    ), "Dataset size must not be smaller than data parallel world size."
    bins = np.zeros(util.world_size, dtype=np.int64)
    for idx, d in enumerate(data):
        bins[idx % util.world_size] += 1
    dp_indices = np.pad(np.cumsum(bins), (1, 0))
    shuffle_indices = get_shuffle_indices(util.seed, len(data))
    subset_indices = shuffle_indices[
        dp_indices[util.dp_rank] : dp_indices[util.dp_rank + 1]
    ]
    data: List[Dict[str, str]] = [data[i] for i in subset_indices]

    return data


ALL_DATASET_CLASSES = {}


def register_dataset(name, dataset_cls):
    assert name not in ALL_DATASET_CLASSES
    assert "/" not in name
    ALL_DATASET_CLASSES[name] = dataset_cls


def make_dataset(
    cfg: Union[str, config_api.DatasetAbstraction],
    seed: int,
    dp_rank: int,
    world_size: int,
    tokenizer_or_tokenizer_name: Union[transformers.PreTrainedTokenizerFast, str],
) -> torch.utils.data.Dataset:
    if isinstance(cfg, str):
        cfg = config_api.DatasetAbstraction(type_=cfg)

    if isinstance(tokenizer_or_tokenizer_name, str):
        tokenizer = load_hf_tokenizer(tokenizer_or_tokenizer_name)
    else:
        tokenizer = tokenizer_or_tokenizer_name
    util = DatasetUtility(
        seed,
        dp_rank,
        world_size,
        tokenizer,
    )
    dataset_cls = ALL_DATASET_CLASSES[cfg.type_]
    return dataset_cls(util=util, **cfg.args)


def gather_stat(src: List[Dict]) -> Dict:
    cnt, stats = {}, {}
    for reply in src:
        # FIXME: understand why the reply can be None
        if not reply:
            continue
        for k, v in reply.items():
            cnt[k] = cnt.get(k, 0) + 1
            stats[k] = stats.get(k, 0) + v
    res = {k: v / cnt for k, v, cnt in zip(stats.keys(), stats.values(), cnt.values())}
    for k, c in cnt.items():
        if c != len(src):
            logger.warning(f"Gathered `{k}` is not present in every returned stats.")
    for k, v in res.items():
        if any(abs(v - x.get(k, None)) > 1e-4 for x in src):
            logger.warning(
                f"Gathered `{k}` is not all-reduced "
                f"before returning: ({[x.get(k, None) for x in src]}, {v})."
            )
    return res


def tabulate_stats(data: Dict[str, float], col=4, floatfmt=".4e") -> str:
    from tabulate import tabulate

    items = list(data.items())
    # Calculate how many rows we'll need
    row_count = (len(items) + col - 1) // col

    # Reorganize items in column-major order
    column_major = []
    for i in range(row_count):
        row = []
        for j in range(col):
            index = i + j * row_count
            if index < len(items):
                row.extend(items[index])
        column_major.append(row)

    return tabulate(column_major, floatfmt=floatfmt, tablefmt="fancy_grid")
