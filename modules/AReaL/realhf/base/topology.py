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

import os
from itertools import permutations
from itertools import product as cartesian_product
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch.distributed as dist

import realhf.base.logging as logging
from realhf.base.constants import NCCL_DEFAULT_TIMEOUT

logger = logging.getLogger("Topology")

GLOBAL_PROCESS_GROUP_REGISTRY: Dict[Tuple[Tuple[int], str], dist.ProcessGroup] = {}


def new_or_get_group(ranks: List[int], backend=None):
    if backend is None:
        backend = dist.get_backend()
    ranks = tuple(sorted(ranks))
    global GLOBAL_PROCESS_GROUP_REGISTRY
    key = (ranks, backend)
    if key not in GLOBAL_PROCESS_GROUP_REGISTRY:
        GLOBAL_PROCESS_GROUP_REGISTRY[key] = dist.new_group(
            ranks, backend=backend, timeout=NCCL_DEFAULT_TIMEOUT
        )
    return GLOBAL_PROCESS_GROUP_REGISTRY[key]


def destroy_all_comm_groups():
    if not dist.is_initialized():
        return
    global GLOBAL_PROCESS_GROUP_REGISTRY
    for group in GLOBAL_PROCESS_GROUP_REGISTRY.values():
        dist.destroy_process_group(group)
    GLOBAL_PROCESS_GROUP_REGISTRY = {}
    # Destroying the default group will raise an error
    # for the latest pytorch release. Omit it as a workaround.
    # dist.destroy_process_group()


def decompose_to_three_factors(n: int) -> List[Tuple[int, int, int]]:
    factors = []
    for i in range(1, int(n ** (1 / 2)) + 1):
        if n % i == 0:
            for j in range(i, int((n // i) ** (1 / 2)) + 1):
                if (n // i) % j == 0:
                    k = (n // i) // j
                    factors += list(set(permutations([i, j, k])))
    return factors


class PipeDataTensorProcessCoord(NamedTuple):
    pipe: int
    data: int
    tensor: int


class DataPipeTensorProcessCoord(NamedTuple):
    data: int
    pipe: int
    tensor: int


# Explicitly define these class to allow pickling.
PROCESS_COORD_REGISTRY = {
    "pipe#data#tensor": PipeDataTensorProcessCoord,
    "data#pipe#tensor": DataPipeTensorProcessCoord,
}


class ProcessTopology:
    """Manages the mapping of n-dimensional Cartesian coordinates to linear
    indices. This mapping is used to map the rank of processes to the grid for
    various forms of parallelism.

    Each axis of the tensor is accessed by its name. The provided
    ordering of the axes defines the layout of the topology.
    ProcessTopology uses a "row-major" layout of the tensor axes, and so
    axes=['x', 'y'] would map coordinates (x,y) and (x,y+1) to adjacent
    linear indices. If instead axes=['y', 'x'] was used, coordinates
    (x,y) and (x+1,y) would be adjacent.

    Some methods return ProcessCoord namedtuples.
    """

    def __init__(self, axes, dims):
        """Create a mapping of n-dimensional tensor coordinates to linear
        indices.

        Arguments:
            axes (list): the names of the tensor axes
            dims (list): the dimension (length) of each axis of the topology tensor
        """

        self.axes = axes  # names of each topology axis
        self.dims = dims  # length of each topology axis

        # This is actually a class that lets us hash {'row':3, 'col':2} mappings
        try:
            self.ProcessCoord = PROCESS_COORD_REGISTRY["#".join(axes)]
        except KeyError as e:
            raise KeyError(
                f"Corresponding coordinate namedtuple not implemented for axes {axes}. "
                "Check base/topology.py and implement explicitly."
            ) from e

        self.mapping = {}
        ranges = [range(d) for d in dims]
        # example: 1, (0,0,1)
        for global_rank, coord in enumerate(cartesian_product(*ranges)):
            key = {axis: coord[self.axes.index(axis)] for axis in self.axes}
            key = self.ProcessCoord(**key)
            # for example, {ProcessCoord(row=0, col=1) : 1}
            self.mapping[key] = global_rank

    def __eq__(self, other):
        if not isinstance(other, ProcessTopology):
            return False
        return self.mapping == other.mapping

    def __repr__(self):
        return f"ProcessTopology(axes={self.axes}, dims={self.dims})"

    def get_rank(self, **coord_kwargs):
        """Return the global rank of a process via its coordinates.

        Coordinates are specified as kwargs. For example:

            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> X.get_rank(x=0, y=1)
            1
        """
        if len(coord_kwargs) != len(self.axes):
            raise ValueError("get_rank() does not support slices. Use filter_match())")

        key = self.ProcessCoord(**coord_kwargs)
        assert (
            key in self.mapping
        ), f"key {coord_kwargs} invalid, mapping: {self.mapping}, key: {key}"
        return self.mapping[key]

    def get_axis_names(self):
        """Return a list of the axis names in the ordering of the topology."""
        return self.axes

    def get_rank_repr(
        self, rank, omit_axes=["data", "pipe"], inner_sep="_", outer_sep="-"
    ):
        """Return a string representation of a rank.

        This method is primarily used for checkpointing model data.

        For example:
            >>> topo = Topo(axes=['a', 'b'], dims=[2, 2])
            >>> topo.get_rank_repr(rank=3)
            'a_01-b_01'
            >>> topo.get_rank_repr(rank=3, omit_axes=['a'])
            'b_01'

        Args:
            rank (int): A rank in the topology.
            omit_axes (list, optional): Axes that should not be in the representation. Defaults to ['data', 'pipe'].
            inner_sep (str, optional): [description]. Defaults to '_'.
            outer_sep (str, optional): [description]. Defaults to '-'.

        Returns:
            str: A string representation of the coordinate owned by ``rank``.
        """
        omit_axes = frozenset(omit_axes)
        axes = [a for a in self.get_axis_names() if a not in omit_axes]
        names = []
        n = len(str(len(self.mapping)))
        for ax in axes:
            ax_rank = getattr(self.get_coord(rank=rank), ax)
            names.append(f"{ax}{inner_sep}{ax_rank:0{n}d}")
        return outer_sep.join(names)

    def get_dim(self, axis):
        """Return the number of processes along the given axis.

        For example:
            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> X.get_dim('y')
            3
        """
        if axis not in self.axes:
            return 0
        return self.dims[self.axes.index(axis)]

    def get_coord(self, rank):
        """Return the coordinate owned by a process rank.

        The axes of the returned namedtuple can be directly accessed as members. For
        example:
            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> coord = X.get_coord(rank=1)
            >>> coord.x
            0
            >>> coord.y
            1
        """
        for coord, idx in self.mapping.items():
            if idx == rank:
                return coord
        raise ValueError(f"rank {rank} not found in topology.")

    def get_axis_comm_lists(self, axis):
        """Construct lists suitable for a communicator group along axis
        ``axis``.

        Example:
            >>> topo = Topo(axes=['pipe', 'data', 'model'], dims=[2, 2, 2])
            >>> topo.get_axis_comm_lists('pipe')
            [
                [0, 4], # data=0, model=0
                [1, 5], # data=0, model=1
                [2, 6], # data=1, model=0
                [3, 7], # data=1, model=1
            ]

        Returns:
            A list of lists whose coordinates match in all axes *except* ``axis``.
        """

        # We don't want to RuntimeError because it allows us to write more generalized
        # code for hybrid parallelisms.
        if axis not in self.axes:
            return []

        # Grab all axes but `axis`
        other_axes = [a for a in self.axes if a != axis]

        lists = []

        # Construct all combinations of coords with other_axes
        ranges = [range(self.get_dim(a)) for a in other_axes]
        for coord in cartesian_product(*ranges):
            other_keys = {a: coord[other_axes.index(a)] for a in other_axes}
            # now go over all ranks in `axis`.
            sub_list = []
            for axis_key in range(self.get_dim(axis)):
                key = self.ProcessCoord(**other_keys, **{axis: axis_key})
                sub_list.append(self.mapping[key])
            lists.append(sub_list)

        return lists

    def filter_match(self, **filter_kwargs):
        """Return the list of ranks whose coordinates match the provided
        criteria.

        Example:
            >>> X = ProcessTopology(axes=['pipe', 'data', 'model'], dims=[2, 2, 2])
            >>> X.filter_match(pipe=0, data=1)
            [2, 3]
            >>> [X.get_coord(rank) for rank in X.filter_match(pipe=0, data=1)]
            [ProcessCoord(pipe=0, data=1, model=0), ProcessCoord(pipe=0, data=1, model=1)]

        Arguments:
            **filter_kwargs (dict): criteria used to select coordinates.

        Returns:
            The list of ranks whose coordinates match filter_kwargs.
        """

        def _filter_helper(x):
            for key, val in filter_kwargs.items():
                if getattr(x, key) != val:
                    return False
            return True

        coords = filter(_filter_helper, self.mapping.keys())
        return [self.mapping[coord] for coord in coords]

    def get_axis_list(self, axis, idx):
        """Returns the list of global ranks whose coordinate in an axis is idx.

        For example:
            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> X.get_axis_list(axis='x', idx=0)
            [0, 1, 2]
            >>> X.get_axis_list(axis='y', idx=0)
            [0, 3]
        """

        # This could be faster by generating the desired keys directly instead of
        # filtering.
        axis_num = self.axes.index(axis)
        ranks = [self.mapping[k] for k in self.mapping.keys() if k[axis_num] == idx]
        return ranks

    def world_size(self):
        return len(self.mapping)

    def __str__(self):
        return str(self.mapping)


def _prime_factors(N):
    """Returns the prime factorization of positive integer N."""
    if N <= 0:
        raise ValueError("Values must be strictly positive.")

    primes = []
    while N != 1:
        for candidate in range(2, N + 1):
            if N % candidate == 0:
                primes.append(candidate)
                N //= candidate
                break
    return primes


class PipeDataTensorParallelTopology(ProcessTopology):
    """A topology for hybrid pipeline, model, and data parallelism."""

    def __init__(
        self,
        num_pp: int,
        num_tp: int,
        num_dp: int,
        sequence_parallel: bool,
        gradient_checkpointing: bool,
        gradient_accumulation_fusion: bool,
        max_prompt_len: Optional[int] = None,
    ):
        super().__init__(axes=["pipe", "data", "tensor"], dims=[num_pp, num_dp, num_tp])

        self.sequence_parallel = sequence_parallel
        self.gradient_checkpointing = gradient_checkpointing
        self.max_prompt_len = max_prompt_len
        self.gradient_accumulation_fusion = gradient_accumulation_fusion


class DataPipeTensorParallelTopology(ProcessTopology):
    """A topology for hybrid data, pipeline, and tensor parallelism.

    Note that DP is the most outer dimension. Used for inference only.
    """

    def __init__(
        self,
        num_pp: int,
        num_tp: int,
        num_dp: int,
        sequence_parallel: bool,
        max_prompt_len: Optional[int] = None,
    ):
        super().__init__(axes=["data", "pipe", "tensor"], dims=[num_dp, num_pp, num_tp])
        self.sequence_parallel = sequence_parallel
        self.max_prompt_len = max_prompt_len


class ParallelGrid:
    """Implements a grid object that stores the data parallel ranks
    corresponding to each of the model parallel stages.

    The grid object organizes the processes in a distributed pytorch job
    into a 2D grid, of stage_id and data_parallel_id.

    self.stage_id and self.data_parallel_id stores the stage id and the
    data parallel id of current process.

    self.dp_group groups the processes by stage_id. self.dp_group[i], is
    a list containing all process ranks whose stage_id is i.

    self.p2p_groups stores a list of tuple, where each tuple stores
    process ranks of adjacent stages for a given data_parallel_id. For
    example if num_stage is 5 then a tuple [7,8] represents stages [3,
    4], with data_parallel id = 1. A stage wrap around will appear as
    non-adjacent ranks, for example tuple [4,0] with representing wrap-
    around stage 4 and 0, for data_parallel_id = 0, or similarly [9,5]
    represents wrapped around stages [4,0] for data_parallel_id = 1.
    """

    def __init__(
        self,
        topology: ProcessTopology,
        process_group: dist.ProcessGroup,
        rank_mapping: Optional[Dict[int, int]] = None,
    ):
        # NOTE: DeepSpeed will access *EVERY* attribute they defined. We need to remain
        # the original semantics, so the attribute name may not truthfully mean what the attribute is.
        # E.g., self.global_rank is not the global rank of the whole world including the master worker,
        # but the rank in the 3D parallelism group of this specific model.

        if rank_mapping is None:
            rank_mapping = {i: i for i in range(topology.world_size())}
        self.rank_mapping = rank_mapping

        self.global_rank = dist.get_rank(group=process_group)
        # NOTE: we don't use dist.get_world_size(process_group) because it will only return the true
        # world size when this process belongs to the process group, otherwise it will return -1.
        # self.world_size=-1 will trigger assertion errors.
        self.world_size = topology.world_size()

        self._topo = topology

        self.data_parallel_size = max(self._topo.get_dim("data"), 1)
        self.pipe_parallel_size = max(self._topo.get_dim("pipe"), 1)
        self.model_parallel_size = max(self._topo.get_dim("tensor"), 1)
        self.slice_parallel_size = self.model_parallel_size
        assert self._is_grid_valid(), (
            "Invalid Grid",
            topology,
            self.world_size,
        )

        self.stage_id = self.get_stage_id()
        self.data_parallel_id = self.get_data_parallel_id()

        # Create new ProcessGroups for TP + PP parallelism approaches.
        self.ds_model_proc_group = None
        self.ds_model_proc_group_gloo = None
        self.ds_model_rank = -1
        for dp in range(self.data_parallel_size):
            ranks = sorted(self._topo.get_axis_list(axis="data", idx=dp))
            proc_group = new_or_get_group(ranks=[rank_mapping[rank] for rank in ranks])
            proc_group_gloo = new_or_get_group(
                ranks=[rank_mapping[rank] for rank in ranks], backend="gloo"
            )
            if self.global_rank in ranks:
                self.ds_model_proc_group = proc_group
                self.ds_model_proc_group_gloo = proc_group_gloo
                self.ds_model_world_size = len(ranks)
                self.ds_model_rank = ranks.index(self.global_rank)
        if self.global_rank != -1:
            assert self.ds_model_rank > -1
            assert self.ds_model_proc_group is not None
        else:
            assert self.ds_model_rank == -1
            assert self.ds_model_proc_group is None

        # Create new ProcessGroup for gradient all-reduces - these are the data parallel groups
        self.dp_group = []
        self.dp_groups = self._topo.get_axis_comm_lists("data")
        self.dp_proc_group = self.dp_proc_group_gloo = None
        for g in self.dp_groups:
            proc_group = new_or_get_group(ranks=[rank_mapping[x] for x in g])
            # NOTE: We must create the GLOO group for vLLM's usage.
            # It will be used to broadcast CPU objects.
            proc_group_gloo = new_or_get_group(
                ranks=[rank_mapping[x] for x in g], backend="gloo"
            )
            if self.global_rank in g:
                self.dp_group = g
                self.dp_proc_group = proc_group
                self.dp_proc_group_gloo = proc_group_gloo

        self.is_first_stage = self.stage_id == 0
        self.is_last_stage = self.stage_id == (self.pipe_parallel_size - 1)

        # These ranks will be used only when NCCL P2P is not supported (torch version <= 1.8).
        # Otherwise we don't need to create groups for P2P communication.
        self.p2p_groups: List[List[int]] = self._build_p2p_groups()

        # Create new ProcessGroup for pipeline collectives - these are pipe parallel groups
        self.pp_group = []
        self.pp_proc_group = self.pp_proc_group_gloo = None
        self.embedding_proc_group = None
        self.position_embedding_proc_group = None
        self.pipe_groups = self._topo.get_axis_comm_lists("pipe")
        for ranks in self.pipe_groups:
            proc_group = new_or_get_group(ranks=[rank_mapping[rank] for rank in ranks])
            # NOTE: We must create the GLOO group for vLLM's usage.
            # It will be used to broadcast CPU objects.
            cpu_group = new_or_get_group(
                ranks=[rank_mapping[rank] for rank in ranks], backend="gloo"
            )
            if self.global_rank in ranks:
                self.pp_group = ranks
                self.pp_proc_group = proc_group
                # NOTE: we must create the GLOO group for vLLM's usage
                self.pp_proc_group_gloo = cpu_group

            # Create embedding groups for communicating gradients between LM head and the embedding.
            # Used to tie gradients of the embedding layer, e.g. for GPT.
            if len(ranks) > 1:
                embedding_ranks = [ranks[0], ranks[-1]]
                position_embedding_ranks = [ranks[0]]
            else:
                embedding_ranks = position_embedding_ranks = ranks
            embedding_group = new_or_get_group(
                ranks=[rank_mapping[rank] for rank in embedding_ranks]
            )
            if self.global_rank in embedding_ranks:
                self.embedding_proc_group = embedding_group

            # TODO: support pipline_model_parallel_split_rank
            position_embedding_group = new_or_get_group(
                ranks=[rank_mapping[rank] for rank in position_embedding_ranks]
            )
            if self.global_rank in position_embedding_ranks:
                self.position_embedding_proc_group = position_embedding_group

        if self.global_rank != -1:
            assert self.pp_proc_group is not None
        else:
            assert self.pp_proc_group is None

        # Create new ProcessGroup for model (tensor-slicing) collectives

        # Short circuit case without model parallelism.
        self.slice_group = None
        self.slice_proc_group = self.slice_proc_group_gloo = None
        self.mp_group = []
        self.model_groups = self._topo.get_axis_comm_lists("tensor")
        for g in self.model_groups:
            proc_group = new_or_get_group(ranks=[rank_mapping[x] for x in g])
            # NOTE: We must create the GLOO group for vLLM's usage.
            # It will be used to broadcast CPU objects.
            cpu_group = new_or_get_group(
                ranks=[rank_mapping[x] for x in g], backend="gloo"
            )
            if self.global_rank in g:
                self.slice_group = g
                self.slice_proc_group = proc_group
                self.slice_proc_group_gloo = cpu_group

        # Create tp+dp group to be consistent with Megatron.
        self.tp_dp_proc_group = None
        for pp in range(self.pipe_parallel_size):
            ranks = sorted(self._topo.get_axis_list(axis="pipe", idx=pp))
            proc_group = new_or_get_group(ranks=[rank_mapping[rank] for rank in ranks])
            if self.global_rank in ranks:
                self.tp_dp_proc_group = proc_group

    def get_stage_id(self):
        if self.global_rank == -1:
            return -1
        return self._topo.get_coord(rank=self.global_rank).pipe

    def get_data_parallel_id(self):
        if self.global_rank == -1:
            return -1
        return self._topo.get_coord(rank=self.global_rank).data

    def _build_p2p_groups(self):
        """Groups for sending and receiving activations and gradients across
        model parallel stages."""
        comm_lists = self._topo.get_axis_comm_lists("pipe")
        p2p_lists = []
        for rank in range(self.world_size):
            for l in comm_lists:
                assert len(l) == self.pipe_parallel_size
                if rank in l:
                    idx = l.index(rank)
                    buddy_rank = l[(idx + 1) % self.pipe_parallel_size]
                    p2p_lists.append(
                        [self.rank_mapping[rank], self.rank_mapping[buddy_rank]]
                    )
                    break  # next global rank
        assert len(p2p_lists) == self.world_size
        return p2p_lists

    def _is_grid_valid(self):
        ranks = 1
        for ax in self._topo.get_axis_names():
            ranks *= self._topo.get_dim(ax)
        return ranks == self.world_size

    # returns the global rank of the process with the provided stage id
    # which has the same data_parallel_id as caller process
    def stage_to_global(self, stage_id, **kwargs):
        if self.global_rank == -1:
            return -1
        me = self._topo.get_coord(self.global_rank)
        transform = me._replace(pipe=stage_id, **kwargs)._asdict()
        return self._topo.get_rank(**transform)

    def topology(self) -> ProcessTopology:
        return self._topo

    # MPU functions for DeepSpeed integration
    def get_global_rank(self):
        return self.global_rank

    def get_pipe_parallel_rank(self):
        """The stage of the pipeline this rank resides in."""
        return self.get_stage_id()

    def get_pipe_parallel_world_size(self):
        """The number of stages in the pipeline."""
        return self.pipe_parallel_size

    def get_pipe_parallel_group(self):
        """The group of ranks within the same pipeline."""
        return self.pp_proc_group

    def get_data_parallel_rank(self):
        """Which pipeline this rank resides in."""
        return self.data_parallel_id

    def get_data_parallel_world_size(self):
        """The number of pipelines."""
        return self.data_parallel_size

    def get_data_parallel_group(self):
        """The group of ranks within the same stage of all pipelines."""
        return self.dp_proc_group

    def get_data_parallel_group_gloo(self):
        return self.dp_proc_group_gloo

    # These are model parallel groups across all types of model parallelism.
    # Deepspeed uses them to detect overflow, etc.
    # group of ranks with same dp rank
    def get_model_parallel_rank(self):
        return self.ds_model_rank

    def get_model_parallel_world_size(self):
        return self.ds_model_world_size

    def get_model_parallel_group(self):
        return self.ds_model_proc_group

    # For Megatron-style tensor slicing
    def get_tensor_model_parallel_rank(self):
        if self.global_rank == -1:
            return -1
        if "tensor" in self._topo.get_axis_names():
            return self._topo.get_coord(rank=self.global_rank).tensor
        else:
            return 0

    def get_tensor_model_parallel_world_size(self):
        return self.slice_parallel_size

    def get_tensor_model_parallel_group(self):
        return self.slice_proc_group

    def get_tensor_model_parallel_cpu_group(self):
        return self.slice_proc_group_gloo

    @property
    def topo(self):
        return self._topo


class FakeGrid:
    """Used for testing dynamic scheduling in none-GPU environment."""

    def __init__(self, rank: int, topo: ProcessTopology):
        self.rank = rank
        self._topo = topo

        self.data_parallel_size = max(self._topo.get_dim("data"), 1)
        self.pipe_parallel_size = max(self._topo.get_dim("pipe"), 1)
        self.model_parallel_size = max(self._topo.get_dim("tensor"), 1)

        self.coord = self._topo.get_coord(self.rank)
        self.dp_id = self.coord.data
        self.pp_id = self.coord.pipe
        self.mp_id = self.coord.tensor

        self.world_size = (
            self.data_parallel_size * self.pipe_parallel_size * self.model_parallel_size
        )

    def get_pipe_parallel_group(self):
        raise RuntimeError("No groups in fake grid.")

    def get_pipe_parallel_world_size(self):
        return self.pipe_parallel_size

    def get_pipe_parallel_rank(self):
        return self.pp_id

    def get_data_parallel_group(self):
        raise RuntimeError("No groups in fake grid.")

    def get_data_parallel_world_size(self):
        return self.data_parallel_size

    def get_data_parallel_rank(self):
        return self.dp_id

    def get_tensor_model_parallel_group(self):
        raise RuntimeError("No groups in fake grid.")

    def get_tensor_model_parallel_world_size(self):
        return self.model_parallel_size

    def get_tensor_model_parallel_rank(self):
        return self.mp_id
