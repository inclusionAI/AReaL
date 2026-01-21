# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import asyncio
import copy
import time
from dataclasses import dataclass, field
from typing import *

import numpy as np

import realhf.api.core.dfg as dfg
import realhf.base.logging as logging
from realhf.api.core.data_api import SequenceSample

logger = logging.getLogger("buffer")


class BufferFull(Exception):
    pass


@dataclass
class _ReplayEntry:
    reuses_left: int
    receive_time: float
    sample: SequenceSample


BUFFER_KEY_WARN_CACHE = set()


class _TensorDictSequenceBuffer:
    """An thread-unsafe buffer implementation based on list.

    Used as an internal buffer object in asyncio-based SequenceBuffer.
    Can be replaced with a more efficient C++ implementation based on
    std vector.

    Methods starting with _ should be called in a locked context.
    """

    def __init__(self, keys: List[str], max_size: int, reuses: int):
        # Fixed-size storage, storing pointers, but sequeces in dict have variable lengths.
        self.__storage: List[_ReplayEntry] = [None for _ in range(max_size)]

        # Some states of the storage. Read/Write applied to them should be locked.
        self.__has_keys = np.zeros((max_size, len(keys)), dtype=bool)

        self.__keys = keys
        self.__reuses = reuses

    def _update_has_keys(self, indices: List[int]):
        for idx in indices:
            self.__has_keys[idx] = [
                k in self.__storage[idx].sample.keys for k in self.__keys
            ]
            if any(k not in self.__keys for k in self.__storage[idx].sample.keys):
                global BUFFER_KEY_WARN_CACHE
                ck = (
                    tuple(sorted(self.__keys)),
                    tuple(sorted(self.__storage[idx].sample.keys)),
                )
                if ck not in BUFFER_KEY_WARN_CACHE:
                    logger.debug(
                        f"Unexpected keys in the sample. Expected keys from all MFCDef: {self.__keys}. "
                        f"Keys in the current sample: {self.__storage[idx].sample.keys}"
                    )
                    BUFFER_KEY_WARN_CACHE.add(ck)

    def _get_has_keys(self, indices):
        return self.__has_keys[indices, :]

    def put_batch(self, indices: List[int], xs: List[SequenceSample]):
        assert len(indices) == len(xs)
        # Can be parallelized.
        for idx, x in zip(indices, xs):
            self.__storage[idx] = _ReplayEntry(
                reuses_left=self.__reuses,
                receive_time=time.time(),
                sample=x,
            )

    def amend_batch(self, indices: List[int], xs: List[SequenceSample]):
        assert len(indices) == len(xs)
        # Can be parallelized.
        for idx, x in zip(indices, xs):
            self.__storage[idx].sample.update_(x)

    def get_batch(self, indices: List[int]) -> List[_ReplayEntry]:
        # Can be parallelized.
        res = []
        for idx in indices:
            r = self.__storage[idx]
            r.reuses_left -= 1
            res.append(r)
        return res

    def inspect_batch(self, indices: List[int]) -> List[_ReplayEntry]:
        res = []
        for idx in indices:
            r = self.__storage[idx]
            res.append(r)
        return res

    def pop_batch(self, indices: List[int]):
        res = []
        for idx in indices:
            r = self.__storage[idx]
            self.__storage[idx] = None
            self.__has_keys[idx] = False
            res.append(r)
        return res


class AsyncIOSequenceBuffer:

    def __init__(
        self,
        rpcs: List[dfg.MFCDef],
        max_size: int,
    ):
        self.rpcs = rpcs
        self._lock = asyncio.Condition(asyncio.Lock())

        # Buffer indicators, should be locked by self._lock.
        # Put, amend, ready, idle, and empty are mutually exclusive.
        self._is_being_put = np.zeros(max_size, dtype=bool)
        self._is_being_amended = np.zeros(max_size, dtype=bool)
        self._is_being_read = np.zeros(max_size, dtype=bool)
        self._is_idle = np.zeros(max_size, dtype=bool)
        self._is_empty = np.ones(max_size, dtype=bool)

        self._buf_size = 0

        self._birth_time = np.zeros(max_size, dtype=np.int64)

        self._buf_size = 0

        # We allow concurrent amenders and readers.
        self._n_amenders = np.zeros(max_size, dtype=int)
        self._n_readers = np.zeros(max_size, dtype=int)

        self._ready_for_rpcs = np.zeros((max_size, len(rpcs)), dtype=bool)
        self._completed_rpc = np.zeros((max_size, len(rpcs)), dtype=bool)

        self._rpc_data_keys = rpc_data_keys = list(
            set().union(*[rpc.input_keys for rpc in rpcs])
        )
        # We can efficiently compute whether an RPC is ready using this mask
        self._rpc_key_mask = np.stack(
            [
                np.array([k in rpc.input_keys for k in rpc_data_keys], dtype=bool)
                for rpc in rpcs
            ],
            axis=1,
        )
        self._rpc_names = [rpc.name for rpc in rpcs]

        # The internal buffer implementation.
        self.__max_size = max_size
        self.__buffer = _TensorDictSequenceBuffer(
            keys=rpc_data_keys, max_size=max_size, reuses=len(rpcs)
        )

    @property
    def max_size(self) -> int:
        return self.__max_size

    @property
    def size(self) -> int:
        return self._buf_size

    @property
    def lock(self):
        return self._lock

    @property
    def n_rpcs(self):
        return len(self._rpc_names)

    def _assert_valid_indicator(self):
        assert (
            self._is_being_put
            + self._is_being_amended
            + self._is_being_read
            + self._is_idle
        ).sum() == self._buf_size
        assert (self._is_empty.sum() + self._buf_size) == self.__max_size
        assert ((self._n_amenders > 0) == self._is_being_amended).all()
        assert (self._n_amenders >= 0).all()
        assert ((self._n_readers > 0) == self._is_being_read).all()
        assert (self._n_readers >= 0).all()
        assert (self._is_empty[:, None] * self._ready_for_rpcs).sum() == 0
        assert (self._is_empty[:, None] * self._completed_rpc).sum() == 0

    def put_batch_synced(self, samples: List[SequenceSample]):
        self._assert_valid_indicator()

        n = len(samples)

        if n == 0:
            return np.array([], dtype=np.int64)

        indices = np.where(self._is_empty)[0][:n]

        if len(indices) < n:
            raise BufferFull(
                "You are probably using a large dataset. "
                "The default buffer size 1M is not large enough. "
                "Please set a larger buffer size by setting "
                "the environment variable, e.g., REAL_MASTER_BUFFER_SIZE=3000000."
            )
        self._is_empty[indices] = False
        self._is_being_put[indices] = True

        self.__buffer.put_batch(indices, samples)

        self.__buffer._update_has_keys(indices)

        # Set a slight difference in birth time to let the order
        # be deterministic.
        self._birth_time[indices] = time.monotonic_ns() + np.arange(
            len(indices), dtype=np.int64
        )

        has_keys = self.__buffer._get_has_keys(indices)  # [bs, #keys]
        rpc_key_mask = self._rpc_key_mask  # [#keys, #rpcs]
        self._ready_for_rpcs[indices] = (
            has_keys[:, :, None] >= rpc_key_mask[None, :, :]
        ).all(axis=1)

        self._is_being_put[indices] = False
        self._is_idle[indices] = True

        self._buf_size += len(samples)
        if self._buf_size >= 0.95 * self.__max_size:
            logger.warning(
                f"Buffer is 95% full. The current buffer size is {self._buf_size} "
                f"while the maximum size is {self.__max_size}. "
                f"If your dataset has more than 1M sequences, consider enlarge "
                f"the default batch size in the master worker."
            )
        return indices

    async def put_batch(
        self, samples: List[SequenceSample], birth_times: List[int] | None = None
    ):
        n = len(samples)

        if n == 0:
            return np.array([], dtype=np.int64)

        async with self._lock:
            self._assert_valid_indicator()

            indices = np.where(self._is_empty)[0][:n]

            if len(indices) < n:
                raise BufferFull(
                    "You are probably using a large dataset. "
                    "The default buffer size 1M is not large enough. "
                    "Please set a larger buffer size by setting "
                    "the environment variable, e.g., REAL_MASTER_BUFFER_SIZE=3000000."
                )
            self._is_empty[indices] = False
            self._is_being_put[indices] = True

        self.__buffer.put_batch(indices, samples)

        # Set a slight difference in birth time to let the order
        # be deterministic.
        if birth_times is None:
            self._birth_time[indices] = time.monotonic_ns() + np.arange(
                len(indices), dtype=np.int64
            )
        else:
            self._birth_time[indices] = birth_times

        async with self._lock:
            self.__buffer._update_has_keys(indices)

            has_keys = self.__buffer._get_has_keys(indices)  # [bs, #keys]
            rpc_key_mask = self._rpc_key_mask  # [#keys, #rpcs]
            self._ready_for_rpcs[indices] = (
                has_keys[:, :, None] >= rpc_key_mask[None, :, :]
            ).all(axis=1)

            self._is_being_put[indices] = False
            self._is_idle[indices] = True

            self._buf_size += len(samples)
            if self._buf_size >= 0.95 * self.__max_size:
                logger.warning(
                    f"Buffer is 95% full. The current buffer size is {self._buf_size} "
                    f"while the maximum size is {self.__max_size}. "
                    f"If your dataset has more than 1M sequences, consider enlarge "
                    f"the default batch size in the master worker."
                )

            can_do_rpcs = {rpc.name: self._can_do_rpc(rpc) for rpc in self.rpcs}
            logger.debug(f"After putting batch, can do RPCs? {can_do_rpcs}.")

            self._lock.notify(len(self._rpc_names))
        return indices

    async def amend_batch(self, indices: List[int], samples: List[SequenceSample]):
        async with self._lock:
            await self._lock.wait_for(
                lambda: (
                    self._is_idle[indices] | self._is_being_amended[indices]
                ).all(),
            )
            self._assert_valid_indicator()
            self._is_idle[indices] = False
            self._is_being_amended[indices] = True
            self._n_amenders[indices] += 1

        self.__buffer.amend_batch(indices, samples)

        async with self._lock:
            self.__buffer._update_has_keys(indices)

            has_keys = self.__buffer._get_has_keys(indices)  # [bs, #keys]
            rpc_key_mask = self._rpc_key_mask  # [#keys, #rpcs]
            self._ready_for_rpcs[indices] = (
                has_keys[:, :, None] >= rpc_key_mask[None, :, :]
            ).all(axis=1)

            self._n_amenders[indices] -= 1
            self._is_being_amended[indices] = self._n_amenders[indices] > 0
            self._is_idle[indices] = np.logical_not(self._is_being_amended[indices])
            if self._is_idle[indices].any():
                self._lock.notify(len(self._rpc_names))

    def _can_do_rpc(self, rpc: dfg.MFCDef) -> bool:
        rpc_idx = self._rpc_names.index(rpc.name)
        ready_indices = np.nonzero(
            (self._is_idle | self._is_being_read)
            & self._ready_for_rpcs[:, rpc_idx]
            & ~self._completed_rpc[:, rpc_idx]
        )[0]
        if len(ready_indices) < rpc.n_seqs:
            return False
        return True

    async def get_batch_for_rpc(
        self, rpc: dfg.MFCDef
    ) -> Tuple[List[int], SequenceSample]:
        logger.debug(
            f"MFC {rpc.name} is waiting for its input keys: {rpc.input_keys}..."
        )
        rpc_idx = self._rpc_names.index(rpc.name)

        async with self._lock:
            # await self._lock.wait_for(_can_do_rpc)

            while not self._can_do_rpc(rpc):
                await self._lock.wait()

            logger.debug(f"Input keys ({rpc.input_keys}) for MFC {rpc.name} are ready!")
            self._assert_valid_indicator()

            ready_indices = np.nonzero(
                (self._is_idle | self._is_being_read)
                & self._ready_for_rpcs[:, rpc_idx]
                & ~self._completed_rpc[:, rpc_idx]
            )[0]

            # Prioritize old data.
            assert np.all(self._birth_time[ready_indices] > 0)
            indices = ready_indices[
                np.argsort(self._birth_time[ready_indices])[: rpc.n_seqs]
            ]

            self._is_idle[indices] = False
            self._is_being_read[indices] = True
            self._n_readers[indices] += 1

        entries = self.__buffer.get_batch(indices)
        assert all([entry.reuses_left >= 0 for entry in entries])
        pop_indices = [
            idx for idx, entry in zip(indices, entries) if entry.reuses_left == 0
        ]
        # The following call is safe because no more RPC will write to popped data.
        if len(pop_indices) > 0:
            self.__buffer.pop_batch(pop_indices)

        async with self._lock:
            self._n_readers[indices] -= 1
            self._is_being_read[indices] = self._n_readers[indices] > 0
            self._is_idle[indices] = self._n_readers[indices] == 0
            self._completed_rpc[indices, rpc_idx] = True

            assert (self._n_readers[pop_indices] == 0).all()
            assert (self._n_amenders[pop_indices] == 0).all()
            self._is_empty[pop_indices] = True
            self._is_idle[pop_indices] = False
            self._completed_rpc[pop_indices] = False
            self._ready_for_rpcs[pop_indices] = False
            self._buf_size -= len(pop_indices)

            if self._is_idle[indices].any():
                self._lock.notify(len(self._rpc_names))
        return indices, SequenceSample.gather(
            [e.sample for e in entries], keys=rpc.input_keys
        )
