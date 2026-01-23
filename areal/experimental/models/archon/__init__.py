# Import model modules to trigger registration
from areal.experimental.models.archon import (
    qwen2,  # noqa: F401
    qwen3,  # noqa: F401
)
from areal.experimental.models.archon.base import BaseStateDictAdapter
from areal.experimental.models.archon.expert_parallel import (
    ExpertParallel,
    ExpertTensorParallel,
    TensorParallel,
    apply_expert_parallel,
)
from areal.experimental.models.archon.model_spec import (
    ModelSpec,
    get_model_spec,
    get_supported_model_types,
    is_supported_model,
)
from areal.experimental.models.archon.moe_weight_converter import (
    MoEConversionState,
    MoEWeightConverter,
)
from areal.experimental.models.archon.parallel_dims import (
    ArchonParallelDims,
)

__all__ = [
    "ArchonParallelDims",
    "BaseStateDictAdapter",
    "ExpertParallel",
    "ExpertTensorParallel",
    "MoEConversionState",
    "MoEWeightConverter",
    "ModelSpec",
    "TensorParallel",
    "apply_expert_parallel",
    "get_model_spec",
    "get_supported_model_types",
    "is_supported_model",
]
