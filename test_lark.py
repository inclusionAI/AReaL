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
    common_dim: ("p" | "d" | "t" | "c") NUMBER
    attn_dim: ("c" | "d" | "t" | "p") NUMBER
    ffn_dim: ("d" | "e" | "t" | "p") NUMBER

    // Inference parallelism strategy
    inf_dim: ("d" | "t" | "p") NUMBER

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
    """语法树转换器"""

    # TODO


class LLMParallelParser:
    """LLM并行策略解析器"""

    def __init__(self):
        self.parser = Lark(
            GRAMMAR, parser="lalr", transformer=ParallelStrategyTransformer()
        )

    def parse(self, expression: str):
        """解析并行策略表达式"""
        try:
            result = self.parser.parse(expression)
            print(type(result))
            return result
        except Exception as e:
            raise ValueError(f"解析错误: {e}")


# 测试和演示代码
def main():
    parser = LLMParallelParser()

    # 测试用例
    test_cases = [
        "sglang:d4t2+fsdp:d4p1",
        "sglang:d4t2+d4p1",
        "sglang:d4t2",
        "sglang:d4t2+eval",
    ]

    for expr in test_cases:
        try:
            print(f"\n表达式: {expr}")
            print("=" * 50)

            analysis = parser.parse(expr)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"错误: {e}")


if __name__ == "__main__":
    main()
