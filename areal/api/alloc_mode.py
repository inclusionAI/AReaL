"""
Allocation mode parsing and management for distributed training and inference.

This module provides comprehensive support for parsing allocation mode expressions
that specify resource allocation strategies for distributed ML workloads, including
both training and inference configurations with various parallelism strategies.
"""

import enum
import math
from dataclasses import dataclass, field
from typing import Optional

from lark import Lark, Transformer


class AllocationType(enum.Enum):
    """Type of resource allocation strategy."""

    COLOCATE = 0  # Shared resources between training and inference (including SFT/training-only)
    DECOUPLED_TRAIN = 1  # Separate resources for training and inference
    LLM_SERVER_ONLY = 2  # Inference-only allocation
    DECOUPLED_EVAL = 3  # Separate resources for inference and evaluation


class AllocationValidationError(Exception):
    """Raised when allocation mode validation fails."""


class InvalidAllocationModeError(Exception):
    """Legacy exception for backward compatibility with existing code."""


@dataclass
class ParallelStrategy:
    """5D parallel strategy supporting tensor, pipeline, data, context, and expert parallelism.

    This class represents a comprehensive parallelization strategy for distributed ML workloads,
    particularly designed for large language models and mixture-of-experts architectures.

    The five dimensions of parallelism are:
    - Tensor parallelism: Splits individual operations (like matrix multiplications) across devices
    - Pipeline parallelism: Splits model layers across devices in a pipeline fashion
    - Data parallelism: Replicates the model and splits data across devices
    - Context parallelism: Splits sequence length across devices (attention-specific)
    - Expert parallelism: Splits experts in MoE models across devices

    For implementation details, refer to:
    https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#moe-parallel-folding

    Args:
        tensor_parallel_size: Number of devices for tensor model parallelism (default: 1)
        pipeline_parallel_size: Number of pipeline parallel stages (default: 1)
        data_parallel_size: Number of data parallel replicas for ZeRO optimization (default: 1)
        context_parallel_size: Number of devices for context parallelism in attention modules (default: 1)
        expert_parallel_size: Number of devices for expert parallelism in MoE models (default: 1)
        expert_tensor_parallel_size: Tensor parallelism size specifically for expert modules (default: 1)

    Note:
        - Context parallelism is only effective for attention modules
        - Expert parallelism is only effective for MoE (Mixture of Experts) modules
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
            "help": "Context parallelism size for attention modules. "
            "Note that context parallelism is only effective for attention modules."
        },
    )
    expert_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Expert parallelism size for MoE models. "
            "Note that expert parallelism is only effective for expert modules."
        },
    )
    expert_tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Tensor parallelism size for expert modules. "
            "By default, it is 1 which disables expert tensor parallelism."
        },
    )

    def __post_init__(self):
        """Initialize computed properties and validate configuration."""
        if self.expert_parallel_size > 1:
            # Calculate expert model parallel size for validation
            self.expert_model_parallel_size = (
                self.pipeline_parallel_size
                * self.expert_tensor_parallel_size
                * self.expert_parallel_size
            )

            # Validate that world size is divisible by expert model parallel size
            assert self.world_size % self.expert_model_parallel_size == 0, (
                f"Expert model parallel size {self.expert_model_parallel_size} "
                f"cannot divide world size {self.world_size}."
            )

    @property
    def expert_data_parallel_size(self) -> int:
        """Data parallelism size for expert modules in MoE models."""
        if not hasattr(self, "expert_model_parallel_size"):
            return self.data_parallel_size
        return self.world_size // self.expert_model_parallel_size

    # Abbreviated properties for convenience
    @property
    def tp_size(self) -> int:
        """Tensor parallelism size (abbreviated)."""
        return self.tensor_parallel_size

    @property
    def pp_size(self) -> int:
        """Pipeline parallelism size (abbreviated)."""
        return self.pipeline_parallel_size

    @property
    def dp_size(self) -> int:
        """Data parallelism size (abbreviated)."""
        return self.data_parallel_size

    @property
    def cp_size(self) -> int:
        """Context parallelism size (abbreviated)."""
        return self.context_parallel_size

    @property
    def ep_size(self) -> int:
        """Expert parallelism size (abbreviated)."""
        return self.expert_parallel_size

    @property
    def etp_size(self) -> int:
        """Expert tensor parallelism size (abbreviated)."""
        return self.expert_tensor_parallel_size

    @property
    def edp_size(self) -> int:
        """Expert data parallelism size (abbreviated)."""
        return self.expert_data_parallel_size

    @property
    def world_size(self) -> int:
        """Total number of devices required for this parallelization strategy."""
        return (
            self.data_parallel_size
            * self.context_parallel_size
            * self.tensor_parallel_size
            * self.pipeline_parallel_size
        )

    def __str__(self):
        """String representation showing all non-default parallelism dimensions."""
        parts = [
            f"tp={self.tensor_parallel_size}",
            f"pp={self.pipeline_parallel_size}",
            f"dp={self.data_parallel_size}",
        ]

        if self.context_parallel_size > 1:
            parts.append(f"cp={self.context_parallel_size}")
        if self.expert_parallel_size > 1:
            parts.append(f"ep={self.expert_parallel_size}")
            if self.expert_tensor_parallel_size != 1:
                parts.append(f"ep_tp={self.expert_tensor_parallel_size}")

        return f"Parallel({','.join(parts)})"

    @staticmethod
    def parallelism_eq(this, other):
        """Compare two parallelism configurations for equality.

        Args:
            this: First ParallelStrategy to compare
            other: Second ParallelStrategy to compare

        Returns:
            bool: True if all parallelism dimensions match

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


@dataclass
class FSDPParallelStrategy(ParallelStrategy):
    """FSDP parallel strategy."""

    @staticmethod
    def parallelism_eq(this, other):
        """Compare FSDP parallelism configurations."""
        return ParallelStrategy.parallelism_eq(this, other)


@dataclass
class MegatronParallelStrategy(ParallelStrategy):
    """Megatron parallel strategy with additional sequence parallelism and virtual pipeline parallelism."""

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
        return ParallelStrategy.parallelism_eq(this, other) and (
            (
                this.virtual_pipeline_parallel_size
                == other.virtual_pipeline_parallel_size
            )
        )


@dataclass
class AllocationMode:
    """Resource allocation configuration for distributed ML workloads.

    This class encapsulates the complete resource allocation strategy for distributed
    training and inference, including parallelization strategies, backend selection,
    and resource partitioning between different workload types.

    The allocation mode supports various deployment patterns:
    - COLOCATE: Training and inference share the same resources (including SFT/training-only)
    - DECOUPLED_TRAIN: Separate dedicated resources for training and inference
    - LLM_SERVER_ONLY: Inference-only deployment
    - DECOUPLED_EVAL: Inference with separate evaluation processes

    Args:
        type_: Type of allocation strategy (AllocationType enum)
        gen: Parallelization strategy for inference/generation (optional)
        train: Parallelization strategy for training (optional)
        gen_backend: Backend used for inference ("sglang", "vllm", etc.) (optional)

    Example:
        # Simple colocated allocation for SFT
        mode = AllocationMode.from_str("d4t2p1")

        # Disaggregated allocation with different backends
        mode = AllocationMode.from_str("sglang:d4t2+fsdp:d8")

        # Inference-only allocation
        mode = AllocationMode.from_str("vllm:d2t4")
    """

    type_: AllocationType
    gen: ParallelStrategy = field(default_factory=ParallelStrategy)
    train: Optional[ParallelStrategy] = None
    gen_backend: Optional[str] = None
    train_backend: Optional[str] = None

    @property
    def gen_instance_size(self) -> int:
        """Number of devices per inference instance (excluding data parallelism)."""
        return self.gen.tp_size * self.gen.pp_size

    @classmethod
    def from_str(cls, allocation_mode: str):
        """Parse allocation mode string into AllocationMode object.

        This method uses grammar-based parsing with Lark for modern
        allocation mode syntax with comprehensive validation rules.

        Examples:
        - "sglang:d4t2" - Inference-only with SGLang backend
        - "sglang:d4t2+fsdp:d8" - Disaggregated with different backends
        - "vllm:d2t4|megatron:d2t4p2" - Colocated allocation
        - "sglang:d4t2+eval" - Inference with evaluation processes
        - "d2p2t1" - Simple parallelism for offline training (e.g., SFT)

        Args:
            allocation_mode: String representation of allocation mode

        Returns:
            AllocationMode: Parsed allocation configuration

        Raises:
            AllocationValidationError: When validation rules are violated
            ValueError: When parsing fails
        """
        parser = _LLMParallelParser()
        result = parser.parse(allocation_mode)
        return parser._convert_to_allocation_mode(result)


# Grammar-based parser implementation using Lark
ALLOCATION_GRAMMAR = """
    start: expression

    expression: disaggregate_expr | colocate_expr | inf_para | eval_expr
        | train_para

    disaggregate_expr: inf_para "+" train_para
    colocate_expr: inf_para "|" train_para
    eval_expr: inf_para "+" EVAL

    inf_para: modern_inf_para | legacy_inf_para
    modern_inf_para: INFER_BACKEND ":" inf_dim+
    legacy_inf_para: INFER_BACKEND "." inf_dim+
    train_para: (TRAIN_BACKEND ":")? common_dim+
        | (TRAIN_BACKEND ":")? hybrid_moe_syntax

    hybrid_moe_syntax: "("? attn_section "|" ffn_section ")"?
    attn_section: "attn" ":" attn_dim+
    ffn_section: "ffn" ":" ffn_dim+

    // Training parallelism strategy
    common_dim: DIM_TYPE NUMBER
    attn_dim: ATTN_DIM_TYPE NUMBER
    ffn_dim: FFN_DIM_TYPE NUMBER

    // Inference parallelism strategy
    inf_dim: INF_DIM_TYPE NUMBER

    DIM_TYPE: "p" | "d" | "t" | "c" | "e"
    ATTN_DIM_TYPE: "c" | "d" | "t" | "p"
    FFN_DIM_TYPE: "d" | "e" | "t" | "p"
    INF_DIM_TYPE: "d" | "t" | "p"

    EVAL: "cpu" | "eval"
    INFER_BACKEND: "sglang" | "vllm"
    TRAIN_BACKEND: "fsdp" | "megatron"

    NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
    NUMBER: /[1-9][0-9]*/

    %import common.WS
    %ignore WS
"""


@dataclass
class ParallelDimension:
    """Single parallelism dimension with type and size.

    Used internally by the grammar parser to represent individual
    parallelism specifications before combining them into strategies.
    """

    type_: str  # Dimension type ("d", "t", "p", "c", "e")
    size: int  # Parallelism degree

    def __str__(self):
        return f"{self.type_}{self.size}"


@dataclass
class InferenceParallelism:
    """Inference parallelism configuration with backend and validation.

    Represents the parallelization strategy for inference workloads,
    including the specific backend (SGLang, vLLM) and associated
    validation rules.
    """

    backend: str
    strategy: ParallelStrategy

    def __str__(self):
        dims = []
        if self.strategy.data_parallel_size != 1:
            dims.append(f"d{self.strategy.data_parallel_size}")
        if self.strategy.tensor_parallel_size != 1:
            dims.append(f"t{self.strategy.tensor_parallel_size}")
        if self.strategy.pipeline_parallel_size != 1:
            dims.append(f"p{self.strategy.pipeline_parallel_size}")
        if not dims:  # Show at least data parallel if all dimensions are 1
            dims.append(f"d{self.strategy.data_parallel_size}")
        return f"{self.backend}:{''.join(dims)}"


@dataclass
class TrainingParallelism:
    """Training parallelism configuration with automatic backend selection.

    Represents the parallelization strategy for training workloads,
    with automatic backend selection based on parallelism requirements
    and comprehensive validation rules.
    """

    backend: Optional[str] = None
    strategy: ParallelStrategy = field(default_factory=lambda: ParallelStrategy())

    def __post_init__(self):
        # Auto-select backend only when needed for specific parallelism
        if self.backend is None:
            # Only auto-select megatron for complex parallelism that requires it
            if (
                self.strategy.pipeline_parallel_size > 1
                or self.strategy.expert_parallel_size > 1
            ):
                self.backend = "megatron"
            else:
                self.backend = "fsdp"

        # TODO: enable the following backend selection logic
        # if self.backend == "fsdp":
        #     if (
        #         or self.strategy.pipeline_parallel_size > 1
        #         or self.strategy.expert_parallel_size > 1
        #     ):
        #         raise AllocationValidationError(
        #             f"Currently FSDP backend only supports data parallelism. "
        #             f"Got strategy: {self.strategy}"
        #         )

    def __str__(self):
        dims = []
        if self.strategy.data_parallel_size != 1:
            dims.append(f"d{self.strategy.data_parallel_size}")
        if self.strategy.pipeline_parallel_size != 1:
            dims.append(f"p{self.strategy.pipeline_parallel_size}")
        if self.strategy.tensor_parallel_size != 1:
            dims.append(f"t{self.strategy.tensor_parallel_size}")
        if self.strategy.context_parallel_size != 1:
            dims.append(f"c{self.strategy.context_parallel_size}")
        if self.strategy.expert_parallel_size != 1:
            dims.append(f"e{self.strategy.expert_parallel_size}")

        if not dims:  # Show at least data parallel if all dimensions are 1
            dims.append(f"d{self.strategy.data_parallel_size}")

        result = "".join(dims)
        if self.backend:
            result = f"{self.backend}:{result}"
        return result


@dataclass
class EvalType:
    """Evaluation expression type (cpu or eval)."""

    eval_type: str

    def __str__(self):
        return self.eval_type


# Allocation expression types for grammar-based parsing
@dataclass
class AllocationExpression:
    """Base class for allocation expressions."""


@dataclass
class InferenceOnlyExpression(AllocationExpression):
    """Inference-only allocation expression."""

    inference: InferenceParallelism

    def __str__(self):
        return str(self.inference)


@dataclass
class TrainingOnlyExpression(AllocationExpression):
    """Training-only allocation expression."""

    training: TrainingParallelism

    def __str__(self):
        return str(self.training)


@dataclass
class DisaggregatedExpression(AllocationExpression):
    """Disaggregated allocation expression (inference + training)."""

    inference: InferenceParallelism
    training: TrainingParallelism

    def __str__(self):
        return f"{self.inference}+{self.training}"


@dataclass
class ColocatedExpression(AllocationExpression):
    """Colocated allocation expression (inference | training)."""

    inference: InferenceParallelism
    training: TrainingParallelism

    def __post_init__(self):
        """Validate colocated expression."""
        # World sizes must match for colocated expressions
        w1 = self.inference.strategy.world_size
        w2 = self.training.strategy.world_size

        if w1 != w2:
            raise AllocationValidationError(
                f"World sizes must match for colocated expressions. "
                f"Inference world size: {w1}, "
                f"Training world size: {w1}"
            )

    def __str__(self):
        return f"{self.inference}|{self.training}"


@dataclass
class EvalAllocationExpression(AllocationExpression):
    """Evaluation allocation expression (inference + eval)."""

    inference: InferenceParallelism
    eval_type: EvalType

    def __str__(self):
        return f"{self.inference}+{self.eval_type}"


class _ParallelStrategyTransformer(Transformer):
    """Lark transformer to convert parse tree to Python objects.

    This transformer walks the parse tree generated by Lark and converts
    it into the appropriate Python data structures, applying validation
    rules during the transformation process.
    """

    def start(self, items):
        return items[0]

    def expression(self, items):
        return items[0]

    def disaggregate_expr(self, items):
        inf_para = items[0]
        train_para = items[1]
        return DisaggregatedExpression(inference=inf_para, training=train_para)

    def colocate_expr(self, items):
        inf_para = items[0]
        train_para = items[1]
        return ColocatedExpression(inference=inf_para, training=train_para)

    def eval_expr(self, items):
        inf_para = items[0]
        eval_type = items[1]
        return EvalAllocationExpression(inference=inf_para, eval_type=eval_type)

    def inf_para(self, items):
        # This method receives either a modern_inf_para or legacy_inf_para result
        return items[0]

    def modern_inf_para(self, items):
        backend = str(items[0])
        dimensions = items[1:]

        # Build ParallelStrategy from dimensions
        strategy_kwargs = {}
        for dim in dimensions:
            if dim.type_ == "d":
                strategy_kwargs["data_parallel_size"] = dim.size
            elif dim.type_ == "t":
                strategy_kwargs["tensor_parallel_size"] = dim.size
            elif dim.type_ == "p":
                strategy_kwargs["pipeline_parallel_size"] = dim.size

        strategy = ParallelStrategy(**strategy_kwargs)

        # Modern syntax - strict validation (no pipeline parallelism for inference)
        inference_parallelism = InferenceParallelism(backend=backend, strategy=strategy)
        return inference_parallelism

    def legacy_inf_para(self, items):
        backend = str(items[0])
        dimensions = items[1:]

        # Build ParallelStrategy from dimensions
        strategy_kwargs = {}
        for dim in dimensions:
            if dim.type_ == "d":
                strategy_kwargs["data_parallel_size"] = dim.size
            elif dim.type_ == "t":
                strategy_kwargs["tensor_parallel_size"] = dim.size
            elif dim.type_ == "p":
                strategy_kwargs["pipeline_parallel_size"] = dim.size

        strategy = ParallelStrategy(**strategy_kwargs)

        # Legacy syntax - permissive validation (allow pipeline parallelism for backward compatibility)
        inference_parallelism = InferenceParallelism(backend=backend, strategy=strategy)
        return inference_parallelism

    def train_para(self, items):
        backend = None
        dimensions = []

        i = 0
        # Check if first item is a backend specification
        if (
            len(items) > 0
            and isinstance(items[0], str)
            and items[0] in ["fsdp", "megatron"]
        ):
            backend = str(items[0])
            i = 1

        # Get remaining items
        remaining_items = items[i:]

        # Check if this is a pre-processed result from hybrid_moe_syntax
        if len(remaining_items) == 1 and isinstance(
            remaining_items[0], TrainingParallelism
        ):
            # This came from hybrid_moe_syntax, but we need to set the backend
            result = remaining_items[0]
            if backend is not None:
                # Create a new TrainingParallelism with the correct backend
                result = TrainingParallelism(backend=backend, strategy=result.strategy)
            return result

        # Regular dimension handling
        dimensions = remaining_items

        # Build ParallelStrategy from dimensions
        strategy_kwargs = {}
        for dim in dimensions:
            if dim.type_ == "d":
                strategy_kwargs["data_parallel_size"] = dim.size
            elif dim.type_ == "t":
                strategy_kwargs["tensor_parallel_size"] = dim.size
            elif dim.type_ == "p":
                strategy_kwargs["pipeline_parallel_size"] = dim.size
            elif dim.type_ == "c":
                strategy_kwargs["context_parallel_size"] = dim.size
            elif dim.type_ == "e":
                strategy_kwargs["expert_parallel_size"] = dim.size

        strategy = ParallelStrategy(**strategy_kwargs)
        return TrainingParallelism(backend=backend, strategy=strategy)

    def common_dim(self, items):
        dim_type = str(items[0])
        size = int(items[1])
        return ParallelDimension(type_=dim_type, size=size)

    def attn_dim(self, items):
        dim_type = str(items[0])
        size = int(items[1])
        return ParallelDimension(type_=dim_type, size=size)

    def ffn_dim(self, items):
        dim_type = str(items[0])
        size = int(items[1])
        return ParallelDimension(type_=dim_type, size=size)

    def inf_dim(self, items):
        dim_type = str(items[0])
        size = int(items[1])
        return ParallelDimension(type_=dim_type, size=size)

    def expert_dim(self, items):
        dim_type = str(items[0])
        size = int(items[1])
        return ParallelDimension(type_=dim_type, size=size)

    def attn_para(self, items):
        return items  # Return list of dimensions

    def expert_para(self, items):
        return items  # Return list of dimensions

    def hybrid_train_para(self, items):
        attn_dims = items[0]  # List of dimensions for attention modules
        expert_dims = items[1]  # List of dimensions for expert modules

        # Build attention strategy
        attn_kwargs = {
            "data_parallel_size": 1,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "context_parallel_size": 1,
        }
        for dim in attn_dims:
            if dim.type_ == "d":
                attn_kwargs["data_parallel_size"] = dim.size
            elif dim.type_ == "t":
                attn_kwargs["tensor_parallel_size"] = dim.size
            elif dim.type_ == "p":
                attn_kwargs["pipeline_parallel_size"] = dim.size
            elif dim.type_ == "c":
                attn_kwargs["context_parallel_size"] = dim.size

        # Build expert strategy parameters
        expert_data_parallel_size = None
        expert_pipeline_parallel_size = None
        expert_tensor_parallel_size = 1
        expert_parallel_size = 1

        for dim in expert_dims:
            if dim.type_ == "d":
                expert_data_parallel_size = dim.size
            elif dim.type_ == "p":
                expert_pipeline_parallel_size = dim.size
            elif dim.type_ == "t":
                expert_tensor_parallel_size = dim.size
            elif dim.type_ == "e":
                expert_parallel_size = dim.size

        # Validate that pipeline parallel sizes match between attention and FFN sections
        if (
            expert_pipeline_parallel_size is not None
            and expert_pipeline_parallel_size != attn_kwargs["pipeline_parallel_size"]
        ):
            raise AllocationValidationError(
                f"Pipeline parallel size for attention and FFN modules must be identical. "
                f"Got attention: {attn_kwargs['pipeline_parallel_size']}, FFN: {expert_pipeline_parallel_size}."
            )

        # Validate that world sizes match
        attn_world_size = math.prod(
            [
                attn_kwargs["data_parallel_size"],
                attn_kwargs["tensor_parallel_size"],
                attn_kwargs["pipeline_parallel_size"],
                attn_kwargs["context_parallel_size"],
            ]
        )
        expert_world_size = math.prod(
            [
                expert_data_parallel_size or attn_kwargs["data_parallel_size"],
                expert_pipeline_parallel_size or attn_kwargs["pipeline_parallel_size"],
            ]
            + [
                expert_tensor_parallel_size,
                expert_parallel_size,
            ]
        )

        if attn_world_size != expert_world_size:
            raise InvalidAllocationModeError(
                f"World size for expert modules and attention modules must be identical. "
                f"Got attention: {attn_world_size}, expert: {expert_world_size}."
            )

        # Create final strategy combining both
        final_strategy_kwargs = attn_kwargs.copy()
        final_strategy_kwargs["expert_parallel_size"] = expert_parallel_size
        final_strategy_kwargs["expert_tensor_parallel_size"] = (
            expert_tensor_parallel_size
        )

        strategy = ParallelStrategy(**final_strategy_kwargs)
        return TrainingParallelism(strategy=strategy, backend="megatron")

    def hybrid_moe_syntax(self, items):
        # items should be [attn_section_result, ffn_section_result]
        attn_dims = items[0]
        ffn_dims = items[1]
        return self.hybrid_train_para([attn_dims, ffn_dims])

    def attn_section(self, items):
        # items will be the attn_dim+ results (ignoring "attn" and ":" literals)
        return items

    def ffn_section(self, items):
        # items will be the ffn_dim+ results (ignoring "ffn" and ":" literals)
        return items

    def DIM_TYPE(self, token):
        return str(token)

    def ATTN_DIM_TYPE(self, token):
        return str(token)

    def FFN_DIM_TYPE(self, token):
        return str(token)

    def EXPERT_DIM_TYPE(self, token):
        return str(token)

    def INF_DIM_TYPE(self, token):
        return str(token)

    def EVAL(self, token):
        return EvalType(eval_type=str(token))

    def INFER_BACKEND(self, token):
        return str(token)

    def TRAIN_BACKEND(self, token):
        return str(token)

    def NUMBER(self, token):
        return int(token)


class _LLMParallelParser:
    """Internal LLM parallel strategy parser using Lark grammar.

    This parser handles the modern allocation mode syntax with explicit
    backend specifications, comprehensive validation, and support for
    complex allocation patterns including disaggregated and colocated
    configurations.
    """

    def __init__(self):
        self.parser = Lark(ALLOCATION_GRAMMAR, parser="earley", ambiguity="explicit")

    def parse(self, expression: str):
        """Parse allocation mode expression using grammar-based parsing.

        Args:
            expression: Allocation mode string to parse

        Returns:
            AllocationExpression: Parsed expression object

        Raises:
            AllocationValidationError: When validation rules are violated
            ValueError: When parsing fails
        """
        try:
            tree = self.parser.parse(expression)
            transformer = _ParallelStrategyTransformer()
            result = transformer.transform(tree)
            return result
        except (AllocationValidationError, InvalidAllocationModeError):
            # Re-raise validation errors without modification
            raise
        except Exception as e:
            # Check for wrapped validation errors in lark VisitError
            import traceback

            tb = traceback.format_exception(type(e), e, e.__traceback__)
            tb_str = "".join(tb)

            if "AllocationValidationError" in tb_str:
                # Extract the validation error message
                lines = tb_str.split("\n")
                for line in lines:
                    if "AllocationValidationError:" in line:
                        msg = line.split("AllocationValidationError:")[-1].strip()
                        raise AllocationValidationError(msg)
                raise AllocationValidationError(str(e))
            elif "InvalidAllocationModeError" in tb_str:
                # Extract the invalid allocation error message
                lines = tb_str.split("\n")
                for line in lines:
                    if "InvalidAllocationModeError:" in line:
                        msg = line.split("InvalidAllocationModeError:")[-1].strip()
                        raise InvalidAllocationModeError(msg)
                raise InvalidAllocationModeError(str(e))

            raise ValueError(f"Parsing error: {e}")

    def _convert_to_allocation_mode(self, result):
        """Convert parsed expression to AllocationMode object.

        Args:
            result: Parsed AllocationExpression object

        Returns:
            AllocationMode: Converted allocation mode configuration

        Raises:
            ValueError: When expression type is not recognized
        """
        if isinstance(result, InferenceOnlyExpression):
            return AllocationMode(
                type_=AllocationType.LLM_SERVER_ONLY,
                gen=result.inference.strategy,
                train=None,
                gen_backend=result.inference.backend,
                train_backend=None,
            )
        elif isinstance(result, TrainingOnlyExpression):
            return AllocationMode(
                type_=AllocationType.COLOCATE,
                gen=result.training.strategy,
                train=result.training.strategy,
                gen_backend=None,
                train_backend=result.training.backend,
            )
        elif isinstance(result, DisaggregatedExpression):
            return AllocationMode(
                type_=AllocationType.DECOUPLED_TRAIN,
                gen=result.inference.strategy,
                train=result.training.strategy,
                gen_backend=result.inference.backend,
                train_backend=result.training.backend,
            )
        elif isinstance(result, ColocatedExpression):
            return AllocationMode(
                type_=AllocationType.COLOCATE,
                gen=result.inference.strategy,
                train=result.training.strategy,
                gen_backend=result.inference.backend,
                train_backend=result.training.backend,
            )
        elif isinstance(result, EvalAllocationExpression):
            return AllocationMode(
                type_=AllocationType.DECOUPLED_EVAL,
                gen=result.inference.strategy,
                train=None,
                gen_backend=result.inference.backend,
                train_backend=None,
            )
        elif isinstance(result, InferenceParallelism):
            return AllocationMode(
                type_=AllocationType.LLM_SERVER_ONLY,
                gen=result.strategy,
                train=None,
                gen_backend=result.backend,
                train_backend=None,
            )
        elif isinstance(result, TrainingParallelism):
            return AllocationMode(
                type_=AllocationType.COLOCATE,
                gen=result.strategy,
                train=result.strategy,
                gen_backend=None,
                train_backend=result.backend,
            )
        else:
            raise ValueError(f"Unknown allocation expression type: {type(result)}")
