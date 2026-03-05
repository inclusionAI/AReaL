"""MegatronTopology adapter implementing TopologyManager.

Wraps megatron-core's parallel_state module to provide a unified
topology interface. Only importable when megatron-core is available.
"""

from __future__ import annotations

import torch.distributed as dist
from megatron.core import parallel_state as mpu
from torch.distributed.device_mesh import DeviceMesh

__all__ = ["MegatronTopology"]


class MegatronTopology:
    """TopologyManager implementation backed by megatron-core parallel_state.

    Delegates all size/group queries to megatron-core's global parallel state.
    Must be instantiated after mpu.initialize_model_parallel() has been called.
    """

    # =========================================================================
    # Size Properties
    # =========================================================================

    @property
    def tp_size(self) -> int:
        return mpu.get_tensor_model_parallel_world_size()

    @property
    def dp_size(self) -> int:
        return mpu.get_data_parallel_world_size()

    @property
    def pp_size(self) -> int:
        return mpu.get_pipeline_model_parallel_world_size()

    @property
    def cp_size(self) -> int:
        return mpu.get_context_parallel_world_size()

    @property
    def ep_size(self) -> int:
        return mpu.get_expert_model_parallel_world_size()

    @property
    def etp_size(self) -> int:
        return mpu.get_expert_tensor_parallel_world_size()

    @property
    def world_size(self) -> int:
        return dist.get_world_size()

    # =========================================================================
    # Enabled Flags
    # =========================================================================

    @property
    def tp_enabled(self) -> bool:
        return self.tp_size > 1

    @property
    def dp_enabled(self) -> bool:
        return self.dp_size > 1

    @property
    def pp_enabled(self) -> bool:
        return self.pp_size > 1

    @property
    def cp_enabled(self) -> bool:
        return self.cp_size > 1

    @property
    def ep_enabled(self) -> bool:
        return self.ep_size > 1

    # =========================================================================
    # Access Methods
    # =========================================================================

    _GROUP_ACCESSORS: dict[str, str] = {
        "tp": "get_tensor_model_parallel_group",
        "dp": "get_data_parallel_group",
        "pp": "get_pipeline_model_parallel_group",
        "cp": "get_context_parallel_group",
        "ep": "get_expert_model_parallel_group",
    }

    def get_group(self, name: str) -> dist.ProcessGroup | None:
        accessor_name = self._GROUP_ACCESSORS.get(name)
        if accessor_name is None:
            return None
        accessor = getattr(mpu, accessor_name, None)
        if accessor is None:
            return None
        return accessor()

    def get_mesh(self, name: str) -> DeviceMesh | None:
        # Megatron doesn't use DeviceMesh
        return None

    # =========================================================================
    # Computed Properties
    # =========================================================================

    @property
    def gradient_divide_factor(self) -> int:
        return self.dp_size * self.cp_size

    @property
    def seq_len_divisor(self) -> int:
        return self.tp_size * self.cp_size

    @property
    def context_and_model_parallel_size(self) -> int:
        return self.cp_size * self.tp_size * self.pp_size
