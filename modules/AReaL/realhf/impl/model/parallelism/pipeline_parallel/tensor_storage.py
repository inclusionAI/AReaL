# Modified from DeepSpeed.
# Copyright [2025] Microsoft Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import defaultdict
from typing import Any, List, Optional, Tuple

import torch

import realhf.base.logging as logging
import realhf.impl.model.parallelism.pipeline_parallel.p2p as p2p

logger = logging.getLogger("tensor_utils")


def get_shape(tensor):
    return tensor.shape if torch.is_tensor(tensor) else None


def print_data_shapes(name, rank, mbid, x, ys):
    if rank == 0:
        logger.debug(f"{name}: rank {rank} mbid {mbid}")
        logger.debug(
            f"shapes: x.pp_input {get_shape(x.pp_input)}, x.pp_output {get_shape(x.pp_output)},"
            f" x.cu_seqlens {get_shape(x.cu_seqlens)}"
        )
        for i, y in enumerate(ys):
            logger.debug(
                f"shapes: ys[{i}].input_ids {get_shape(y.packed_input_ids)}, "
                f"ys[{i}].k_cache {get_shape(y.k_cache)}, ys[{i}].v_cache {get_shape(y.v_cache)}, "
                f"ys[{i}].cache_seqlens {get_shape(y.cache_seqlens)}"
            )


class TensorBuffer:
    # could store both tensors and other data

    def __init__(self):
        self.tensors = defaultdict(dict)

    def put(self, name: str, mbid: int, x: torch.Tensor):
        self.tensors[name][mbid] = x

    def alloc(
        self,
        name: str,
        mbid: int,
        shape: Tuple[int],
        dtype: torch.dtype,
        device: torch.device,
        require_grads: bool = False,
    ):
        self.tensors[name][mbid] = torch.zeros(
            shape, dtype=dtype, device=device, requires_grad=require_grads
        )
        return self.tensors[name][mbid]

    def get(
        self,
        name: str,
        mbid: int,
        remove: bool = False,
        raise_error: bool = True,
    ):
        try:
            if remove:
                return self.tensors[name].pop(mbid)
            else:
                return self.tensors[name][mbid]
        except KeyError as e:
            if raise_error:
                raise e
            else:
                return None

    def remove(self, name: str, mbid: Optional[int] = None, check_exists: bool = False):
        try:
            if mbid is None:
                del self.tensors[name]
            else:
                self.tensors[name].pop(mbid)
        except KeyError:
            if not check_exists:
                return
            raise KeyError(f"TensorBuffer.remove: key {name} mbid {mbid} not found")

    def check_name(self, name: str):
        return name in self.tensors

    def check_mbid(self, name: str, mbid: int):
        if name not in self.tensors:
            return False
        return mbid in self.tensors[name]

    def clear(self):
        self.tensors = defaultdict(dict)
