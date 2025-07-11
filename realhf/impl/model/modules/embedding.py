# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

from typing import *

import torch
import torch.nn as nn
from torch.nn import init

from realhf.impl.model.parallelism.tensor_parallel.modules import ParallelEmbedding


class OffsetPositionalEmbedding(nn.Embedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        offset: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.__offset = offset
        super().__init__(
            num_embeddings + self.__offset,
            embedding_dim,
            dtype=dtype,
            device=device,
        )

    def forward(self, position_ids: torch.LongTensor):
        return super().forward(position_ids + self.__offset)


class OffsetParallelPositionalEmbedding(ParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        offset: int,
        init_method=init.xavier_normal_,
        # params_dtype: torch.dtype=torch.float32,
        perform_initialization: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.__offset = offset
        super(OffsetParallelPositionalEmbedding, self).__init__(
            num_embeddings=num_embeddings + offset,
            embedding_dim=embedding_dim,
            init_method=init_method,
            perform_initialization=perform_initialization,
            dtype=dtype,
            device=device,
        )

    def forward(self, input_: torch.LongTensor) -> torch.Tensor:
        return super().forward(input_ + self.__offset)
