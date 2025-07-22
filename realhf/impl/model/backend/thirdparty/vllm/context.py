# Copyright 2025 Ant Group Inc.

import enum
import time
from typing import *

import torch
import torch.distributed
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    _get_unique_name,
    _register_group,
)
from vllm.platforms import current_platform

from realhf.api.core.config import ModelName
from realhf.base import constants, logging, topology

logger = logging.getLogger("vLLM third party init")


class _vLLMGroupType(enum.Enum):
    """VLLM group types.

    In the world of vLLM, there's no data parallel semantics,
    so we should map the `constants.tp_and_pp_group` to the
    world group of vLLM.
    """

    WORLD = 1
    TP = 2
    PP = 3


def _vllm_group_rank(group_type: _vLLMGroupType):
    if group_type == _vLLMGroupType.WORLD:
        return constants.tp_and_pp_rank()
    elif group_type == _vLLMGroupType.TP:
        return constants.tensor_parallel_rank()
    elif group_type == _vLLMGroupType.PP:
        return constants.pipe_parallel_rank()


def _vllm_group_size(group_type: _vLLMGroupType):
    if group_type == _vLLMGroupType.WORLD:
        return constants.tp_and_pp_world_size()
    elif group_type == _vLLMGroupType.TP:
        return constants.tensor_parallel_world_size()
    elif group_type == _vLLMGroupType.PP:
        return constants.pipe_parallel_world_size()


def _vllm_parallel_group(group_type: _vLLMGroupType):
    if group_type == _vLLMGroupType.WORLD:
        return constants.tp_and_pp_group()
    elif group_type == _vLLMGroupType.TP:
        return constants.tensor_parallel_group()
    elif group_type == _vLLMGroupType.PP:
        return constants.pipe_parallel_group()


def _vllm_cpu_parallel_group(group_type: _vLLMGroupType):
    if group_type == _vLLMGroupType.WORLD:
        return constants.grid().ds_model_proc_group_gloo
    elif group_type == _vLLMGroupType.TP:
        return constants.grid().slice_proc_group_gloo
    elif group_type == _vLLMGroupType.PP:
        return constants.grid().pp_proc_group_gloo


def _vllm_parallel_ranks(group_type: _vLLMGroupType):
    return torch.distributed.get_process_group_ranks(_vllm_parallel_group(group_type))


class vLLMGroupCoordinator(GroupCoordinator):
    def __init__(
        self,
        group_type: _vLLMGroupType,
        group_name: Optional[str] = None,
    ):
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)

        # global rank
        self.rank = torch.distributed.get_rank()
        # global ranks in the group
        self.ranks = _vllm_parallel_ranks(group_type)
        # local rank used to assign devices
        self.local_rank = 0  # because we have set independent CUDA_VISIBLE_DEVICES

        # group for device communication
        self.device_group = _vllm_parallel_group(group_type)
        # group for cpu communication
        self.cpu_group = _vllm_cpu_parallel_group(group_type)

        # rank inside the group
        self.rank_in_group = _vllm_group_rank(group_type)

        # size of the group
        self.world_size = _vllm_group_size(group_type)

        if current_platform.is_cuda_alike():
            self.device = torch.device(f"cuda:{0}")
        else:
            self.device = torch.device("cpu")

        self.use_pynccl = False
        self.use_custom_allreduce = False
        self.use_tpu_communicator = False

        self.pynccl_comm = None
        self.ca_comm = None
        self.tpu_communicator = None
        self.mq_broadcaster = None


def init_vllm():
    # TODO: support speculative decoding, where the draft model can
    # have a different TP degree
    from vllm.distributed import parallel_state

    parallel_state._WORLD = vLLMGroupCoordinator(
        group_type=_vLLMGroupType.WORLD,
        group_name="world",
    )

    # Build the tensor model-parallel groups.
    assert (
        parallel_state._TP is None
    ), "vLLM tensor model parallel group is already initialized"
    parallel_state._TP = vLLMGroupCoordinator(
        group_type=_vLLMGroupType.TP,
        group_name="tp",
    )

    # Build the pipeline model-parallel groups.
    assert (
        parallel_state._PP is None
    ), "vLLM pipeline model parallel group is already initialized"
    parallel_state._PP = vLLMGroupCoordinator(
        group_type=_vLLMGroupType.PP,
        group_name="pp",
    )

    assert parallel_state.model_parallel_is_initialized()
