import dataclasses

from areal.api.alloc_mode import AllocationMode as MainAllocationMode
from areal.api.io_struct import *


@dataclasses.dataclass
class ParallelStrategy:
    """Basic 5D parallel strategy (tensor, pipeline, expert, context and data parallelism)
    that can be parsed from allocation mode.

    For details, refer to https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#moe-parallel-folding

    Note:
        Sequence parallelism is only used in combination with tensor-model parallelism.
    """

    tensor_parallel_size: int = field(
        default=1, metadata={"help": "Size of tensor-model parallelism"}
    )
    pipeline_parallel_size: int = field(
        default=1, metadata={"help": "Number of pipeline parallel stages"}
    )
    data_parallel_size: int = field(
        default=1, metadata={"help": "Data parallelism size for ZeRO optimization"}
    )
    context_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Context parallelism size for megatron modules. "
            "Note that context parallelism is only effective for attention modules."
        },
    )
    expert_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Expert parallelism size for megatron modules. "
            "Note that expert parallelism is only effective for expert modules."
        },
    )
    expert_tensor_parallel_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Tensor parallelism size for expert modules. "
            "If not set, expert modules will use `tensor_parallel_size`."
        },
    )

    def __post_init__(self):
        if self.expert_parallel_size > 1:
            self.expert_tensor_parallel_size = (
                self.tensor_parallel_size
                if self.expert_tensor_parallel_size is None
                else self.expert_tensor_parallel_size
            )
            self.expert_model_parallel_size = (
                self.pipeline_parallel_size
                * self.expert_tensor_parallel_size
                * self.expert_parallel_size
            )
            assert self.world_size % self.expert_model_parallel_size == 0, (
                f"Expert model parallel size {self.expert_model_parallel_size} "
                f"can not divide world size {self.world_size}. "
            )

    @property
    def world_size(self):
        return (
            self.data_parallel_size
            * self.context_parallel_size
            * self.tensor_parallel_size
            * self.pipeline_parallel_size
        )

    @property
    def expert_data_parallel_size(self):
        return self.world_size // self.expert_model_parallel_size

    def __str__(self):
        s = (
            f"Parallel(tp={self.tensor_parallel_size},"
            f"pp={self.pipeline_parallel_size},"
            f"dp={self.data_parallel_size}"
        )
        if self.context_parallel_size > 1:
            s += f",cp={self.context_parallel_size}"
        if self.expert_parallel_size > 1:
            s += f",ep={self.expert_parallel_size},ep_tp={self.expert_tensor_parallel_size}"
        s += ")"
        return s

    @staticmethod
    def parallelism_eq(this, other):
        """Compare parallelism configurations.

        Note:
            Implemented as static method to avoid OmegaConf compatibility issues.
        """
        return (
            (this.tensor_parallel_size == other.tensor_parallel_size)
            and (this.pipeline_parallel_size == other.pipeline_parallel_size)
            and (this.data_parallel_size == other.data_parallel_size)
            and (this.context_parallel_size == other.context_parallel_size)
            and (this.expert_parallel_size == other.expert_parallel_size)
            and (this.expert_tensor_parallel_size == other.expert_tensor_parallel_size)
        )


@dataclasses.dataclass
class MegatronParallelStrategy(ParallelStrategy):
    """Megatron parallel strategy with additional sequence parallelism and virtual pipeline parallelism."""

    # TODO: Add FSDP parallel strategy when moving out of experimental.
    virtual_pipeline_parallel_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Virtual pipeline parallelism size for megatron modules "
            "for interleaved pipeline schedule."
        },
    )
    use_sequence_parallel: bool = field(
        default=False,
        metadata={
            "help": "Enable sequence parallelism. Only used with tensor-model parallelism in Megatron",
        },
    )

    @staticmethod
    def parallelism_eq(this, other):
        """Compare Megatron parallelism configurations (excluding sequence parallelism)."""
        return super().parallelism_eq(this, other) and (
            (
                this.virtual_pipeline_parallel_size
                == other.virtual_pipeline_parallel_size
            )
        )


class AllocationType(enum.Enum):
    COLOCATE = 0
    DECOUPLED_TRAIN = 1
    LLM_SERVER_ONLY = 2
    DECOUPLED_EVAL = 3


class InvalidAllocationModeError(Exception):
    pass


@dataclasses.dataclass
class AllocationMode:
    # For details about 5D parallelism used in this class,
    # refer to: https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#moe-parallel-folding
    type_: AllocationType
    gen: Optional[ParallelStrategy] = None
    train: Optional[ParallelStrategy] = None
    gen_backend: Optional[str] = None

    @property
    def gen_tp_size(self) -> int:
        return self.gen.tensor_parallel_size

    @property
    def gen_pp_size(self) -> int:
        return self.gen.pipeline_parallel_size

    @property
    def gen_dp_size(self) -> int:
        return self.gen.data_parallel_size

    @property
    def gen_ep_size(self) -> int:
        return self.gen.expert_parallel_size

    @property
    def gen_world_size(self) -> int:
        return self.gen.world_size

    @property
    def gen_instance_size(self):
        # TODO: Consider SGLang DP attention
        assert self.gen_world_size % self.gen_dp_size == 0
        return self.gen_world_size // self.gen_dp_size

    @property
    def train_tp_size(self) -> int:
        return self.train.tensor_parallel_size

    @property
    def train_pp_size(self) -> int:
        return self.train.pipeline_parallel_size

    @property
    def train_dp_size(self) -> int:
        return self.train.data_parallel_size

    @property
    def train_ep_size(self) -> int:
        return self.train.expert_parallel_size

    @property
    def train_cp_size(self) -> int:
        return self.train.context_parallel_size

    @property
    def train_etp_size(self) -> int:
        return self.train.expert_tensor_parallel_size or self.train_tp_size

    @property
    def train_edp_size(self) -> int:
        return self.train.expert_data_parallel_size

    @property
    def train_world_size(self) -> int:
        return self.train.world_size

    @classmethod
    def from_str(cls, allocation_mode: str):
        # Use the main AllocationMode parser and convert the result
        main_alloc = MainAllocationMode.from_str(allocation_mode)

        # Convert ParallelStrategy objects from main to experimental format
        def convert_strategy(strategy):
            if strategy is None:
                return None
            return ParallelStrategy(
                tensor_parallel_size=strategy.tensor_parallel_size,
                pipeline_parallel_size=strategy.pipeline_parallel_size,
                data_parallel_size=strategy.data_parallel_size,
                context_parallel_size=strategy.context_parallel_size,
                expert_parallel_size=strategy.expert_parallel_size,
                expert_tensor_parallel_size=strategy.expert_tensor_parallel_size,
            )

        return cls(
            type_=main_alloc.type_,
            gen=convert_strategy(main_alloc.gen),
            train=convert_strategy(main_alloc.train),
            gen_backend=main_alloc.gen_backend,
        )
