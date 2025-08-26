from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from lark import Lark, Transformer, Tree

GRAMMAR = """
    start: expression

    expression: train_para | disaggregate_expr | colocate_expr
        | inf_para | eval_expr

    disaggregate_expr: inf_para "+" train_para
    colocate_expr: inf_para "|" train_para
    eval_expr: inf_para "+" EVAL

    inf_para: INFER_BACKEND ":" inf_dim+
    train_para: (TRAIN_BACKEND ":")? common_dim+
        | ( "(" "attn" ":" attn_dim+ "|" "ffn" ":" ffn_dim+ ")" )

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
    INFER_BACKEND: "sglang"
    TRAIN_BACKEND: "fsdp" | "megatron"

    NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
    NUMBER: /[1-9][0-9]*/

    %import common.WS
    %ignore WS
"""


@dataclass
class ParallelDimension:
    type: str
    size: int

    def __str__(self):
        return f"{self.type}{self.size}"


@dataclass
class InferenceParallelism:
    """Inference parallelism configuration"""

    backend: str
    strategy: "ParallelStrategy"

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Validate inference parallelism configuration"""
        # Rule 5: No pipeline parallelism for inference
        if self.strategy.pipeline_parallel_size > 1:
            raise AllocationValidationError(
                f"Pipeline parallelism not supported for inference backend '{self.backend}'. "
                f"Got pipeline_parallel_size={self.strategy.pipeline_parallel_size}"
            )

    @property
    def data_parallel_size(self) -> int:
        return self.strategy.data_parallel_size

    @property
    def tensor_parallel_size(self) -> int:
        return self.strategy.tensor_parallel_size

    @property
    def pipeline_parallel_size(self) -> int:
        return self.strategy.pipeline_parallel_size

    def __str__(self):
        dims = []
        if self.strategy.data_parallel_size != 1:
            dims.append(f"d{self.strategy.data_parallel_size}")
        if self.strategy.tensor_parallel_size != 1:
            dims.append(f"t{self.strategy.tensor_parallel_size}")
        if self.strategy.pipeline_parallel_size != 1:
            dims.append(f"p{self.strategy.pipeline_parallel_size}")
        if not dims:  # If all dimensions are 1, show at least data parallel
            dims.append(f"d{self.strategy.data_parallel_size}")
        return f"{self.backend}:{''.join(dims)}"


@dataclass
class TrainingParallelism:
    """Training parallelism configuration"""

    backend: Optional[str] = None
    strategy: "ParallelStrategy" = field(default_factory=lambda: ParallelStrategy())

    def __post_init__(self):
        # Rule 7: Auto-select backend if not specified
        if self.backend is None:
            if (
                self.strategy.pipeline_parallel_size > 1
                or self.strategy.expert_parallel_size > 1
            ):
                self.backend = "megatron"
            else:
                self.backend = "fsdp"

        self._validate()

    def _validate(self):
        """Validate training parallelism configuration"""
        # Rule 3: FSDP only supports data parallelism
        if self.backend == "fsdp":
            if (
                self.strategy.tensor_parallel_size > 1
                or self.strategy.pipeline_parallel_size > 1
                or self.strategy.context_parallel_size > 1
                or self.strategy.expert_parallel_size > 1
            ):
                raise AllocationValidationError(
                    f"FSDP backend only supports data parallelism. "
                    f"Got strategy: {self.strategy}"
                )

        # Rule 4: Currently don't support megatron backend
        if self.backend == "megatron":
            raise AllocationValidationError(
                f"Megatron backend is not currently supported"
            )

    @property
    def data_parallel_size(self) -> int:
        return self.strategy.data_parallel_size

    @property
    def tensor_parallel_size(self) -> int:
        return self.strategy.tensor_parallel_size

    @property
    def pipeline_parallel_size(self) -> int:
        return self.strategy.pipeline_parallel_size

    @property
    def context_parallel_size(self) -> int:
        return self.strategy.context_parallel_size

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

        if not dims:  # If all dimensions are 1, show at least data parallel
            dims.append(f"d{self.strategy.data_parallel_size}")

        result = "".join(dims)
        if self.backend:
            result = f"{self.backend}:{result}"
        return result


class AllocationValidationError(Exception):
    """Raised when allocation mode validation fails"""

    pass


@dataclass
class EvalType:
    """Evaluation expression (cpu or eval)"""

    eval_type: str

    def __str__(self):
        return self.eval_type


@dataclass
class AllocationExpression:
    """Base allocation expression"""

    pass


@dataclass
class InferenceOnlyExpression(AllocationExpression):
    """Inference-only allocation"""

    inference: InferenceParallelism

    def __str__(self):
        return str(self.inference)


@dataclass
class TrainingOnlyExpression(AllocationExpression):
    """Training-only allocation"""

    training: TrainingParallelism

    def __str__(self):
        return str(self.training)


@dataclass
class DisaggregatedExpression(AllocationExpression):
    """Disaggregated allocation (inference + training)"""

    inference: InferenceParallelism
    training: TrainingParallelism

    def __str__(self):
        return f"{self.inference}+{self.training}"


@dataclass
class ColocatedExpression(AllocationExpression):
    """Colocated allocation (inference | training)"""

    inference: InferenceParallelism
    training: TrainingParallelism

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Validate colocated expression"""
        # Rule 2: World sizes must match for colocated expressions
        inf_world_size = self.inference.strategy.world_size
        train_world_size = self.training.strategy.world_size

        if inf_world_size != train_world_size:
            raise AllocationValidationError(
                f"World sizes must match for colocated expressions. "
                f"Inference world size: {inf_world_size}, "
                f"Training world size: {train_world_size}"
            )

    def __str__(self):
        return f"{self.inference}|{self.training}"


@dataclass
class EvalAllocationExpression(AllocationExpression):
    """Evaluation allocation (inference + eval)"""

    inference: InferenceParallelism
    eval_type: EvalType

    def __str__(self):
        return f"{self.inference}+{self.eval_type}"


@dataclass
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


class ParallelStrategyTransformer(Transformer):
    """Lark transformer to convert parse tree to Python objects"""

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
        backend = str(items[0])
        dimensions = items[1:]

        # Build ParallelStrategy from dimensions
        strategy_kwargs = {}
        for dim in dimensions:
            if dim.type == "d":
                strategy_kwargs["data_parallel_size"] = dim.size
            elif dim.type == "t":
                strategy_kwargs["tensor_parallel_size"] = dim.size
            elif dim.type == "p":
                strategy_kwargs["pipeline_parallel_size"] = dim.size

        strategy = ParallelStrategy(**strategy_kwargs)
        return InferenceParallelism(backend=backend, strategy=strategy)

    def train_para(self, items):
        backend = None
        dimensions = []

        i = 0
        # Check if first item is a backend
        if (
            len(items) > 0
            and isinstance(items[0], str)
            and items[0] in ["fsdp", "megatron"]
        ):
            backend = str(items[0])
            i = 1

        # Get remaining dimensions
        dimensions = items[i:]

        # Build ParallelStrategy from dimensions
        strategy_kwargs = {}
        for dim in dimensions:
            if dim.type == "d":
                strategy_kwargs["data_parallel_size"] = dim.size
            elif dim.type == "t":
                strategy_kwargs["tensor_parallel_size"] = dim.size
            elif dim.type == "p":
                strategy_kwargs["pipeline_parallel_size"] = dim.size
            elif dim.type == "c":
                strategy_kwargs["context_parallel_size"] = dim.size
            elif dim.type == "e":
                strategy_kwargs["expert_parallel_size"] = dim.size

        strategy = ParallelStrategy(**strategy_kwargs)
        return TrainingParallelism(backend=backend, strategy=strategy)

    def common_dim(self, items):
        dim_type = str(items[0])
        size = int(items[1])
        return ParallelDimension(type=dim_type, size=size)

    def attn_dim(self, items):
        dim_type = str(items[0])
        size = int(items[1])
        return ParallelDimension(type=dim_type, size=size)

    def ffn_dim(self, items):
        dim_type = str(items[0])
        size = int(items[1])
        return ParallelDimension(type=dim_type, size=size)

    def inf_dim(self, items):
        dim_type = str(items[0])
        size = int(items[1])
        return ParallelDimension(type=dim_type, size=size)

    def DIM_TYPE(self, token):
        return str(token)

    def ATTN_DIM_TYPE(self, token):
        return str(token)

    def FFN_DIM_TYPE(self, token):
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


class LLMParallelParser:
    """LLM并行策略解析器

    支持的表达式类型:
    1. 推理配置: sglang:d4t2p1
    2. 分离式配置: sglang:d4t2+fsdp:d4p1t2
    3. 评估配置: sglang:d4t2+eval

    使用示例:
    >>> parser = LLMParallelParser()
    >>> result = parser.parse("sglang:d4t2+fsdp:d4p1")
    >>> print(result)  # sglang:d4t2+fsdp:d4p1
    >>> inf_config = parser.get_inference_config(result)
    >>> print(inf_config.data_parallel_size)  # 4
    """

    def __init__(self):
        self.parser = Lark(GRAMMAR, parser="lalr")

    def parse(self, expression: str):
        """解析并行策略表达式"""
        try:
            tree = self.parser.parse(expression)
            transformer = ParallelStrategyTransformer()
            result = transformer.transform(tree)
            return result
        except AllocationValidationError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            raise ValueError(f"解析错误: {e}")

    def get_inference_config(self, result):
        """从解析结果获取推理配置"""
        if isinstance(result, InferenceParallelism):
            return result
        elif isinstance(
            result,
            (DisaggregatedExpression, ColocatedExpression, EvalAllocationExpression),
        ):
            return result.inference
        return None

    def get_training_config(self, result):
        """从解析结果获取训练配置"""
        if isinstance(result, TrainingParallelism):
            return result
        elif isinstance(result, (DisaggregatedExpression, ColocatedExpression)):
            return result.training
        return None


# 测试和演示代码
def main():
    parser = LLMParallelParser()

    # 测试用例
    test_cases = [
        # 分离式配置 (disaggregate)
        "sglang:d4t2+fsdp:d4p1",  # 推理 + 训练后端
        "sglang:d4t2+d4p1",  # 推理 + 训练(无后端)
        # 单纯推理配置
        "sglang:d4t2",  # 推理唯一
        "sglang:d8t1p2",  # 推理三维
        # 评估配置
        "sglang:d4t2+eval",  # 推理 + 评估
        "sglang:d4t2+cpu",  # 推理 + CPU
        # 训练配置测试
        "sglang:d2t4+megatron:d2p2t4c2",  # 复杂训练配置
    ]

    for expr in test_cases:
        try:
            print(f"\n表达式: {expr}")
            print("=" * 50)

            result = parser.parse(expr)
            print(f"解析结果: {result}")
            print(f"类型: {type(result).__name__}")

            # Show detailed information based on result type
            if isinstance(result, InferenceParallelism):
                print(f"推理后端: {result.backend}")
                print(f"数据并行: {result.data_parallel_size}")
                print(f"张量并行: {result.tensor_parallel_size}")
                print(f"流水线并行: {result.pipeline_parallel_size}")
            elif isinstance(result, DisaggregatedExpression):
                print(f"推理配置: {result.inference}")
                print(f"训练配置: {result.training}")
            elif isinstance(result, EvalAllocationExpression):
                print(f"推理配置: {result.inference}")
                print(f"评估类型: {result.eval_type}")

            # Demonstrate utility methods
            inf_config = parser.get_inference_config(result)
            train_config = parser.get_training_config(result)
            if inf_config:
                print(
                    f"推理并行度 - 数据:{inf_config.data_parallel_size}, 张量:{inf_config.tensor_parallel_size}, 流水线:{inf_config.pipeline_parallel_size}"
                )
            if train_config:
                print(
                    f"训练并行度 - 数据:{train_config.data_parallel_size}, 张量:{train_config.tensor_parallel_size}, 流水线:{train_config.pipeline_parallel_size}"
                )

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"错误: {e}")


if __name__ == "__main__":
    main()
