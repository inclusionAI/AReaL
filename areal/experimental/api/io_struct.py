import dataclasses
import math

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
        # 1. A string like "d2p2t1", representing the N-d parallelism strategy
        para = AllocationMode.extract_parallelism_strategy(allocation_mode)
        if para:
            return cls(
                type_=AllocationType.COLOCATE,
                gen=para,
                train=para,
                gen_backend=AllocationMode.get_gen_backend(allocation_mode),
            )
        gen_para, train_para = AllocationMode.extract_decoupled_alloc(allocation_mode)
        # 2. A string like "sglang.d4p1t1+d2p1t2", representing a decoupled
        # allocation with 4 GPUs dedicated for SGLang inference and
        # other 4 GPUs dedicated for training
        if train_para:
            return cls(
                type_=AllocationType.DECOUPLED_TRAIN,
                gen=gen_para,
                train=train_para,
                gen_backend=AllocationMode.get_gen_backend(allocation_mode),
            )
        # 3. A string like "sglang.d4p1t1+eval" or "sglang.d4p1t1+cpu",
        # representing a decoupled allocation with 4 GPUs for SGLang server
        # and several CPU client processes to send requests.
        # The number of CPU processes depend on the launcher type.
        # One process for local launcher and `n_gpus_per_node` processes for Ray/SLURM.
        if gen_para:
            return cls(
                type_=AllocationType.DECOUPLED_EVAL,
                gen=gen_para,
                train=None,
                gen_backend=AllocationMode.get_gen_backend(allocation_mode),
            )
        # 4. A string like "sglang.d4p1t1"", representing SGLang server-only allocation.
        para = AllocationMode.extract_gen_alloc(allocation_mode)
        if para:
            return cls(
                type_=AllocationType.LLM_SERVER_ONLY,
                gen=para,
                train=None,
                gen_backend=AllocationMode.get_gen_backend(allocation_mode),
            )
        raise NotImplementedError(f"Failed to parse allocation: {allocation_mode}")

    @staticmethod
    def extract_parallelism_strategy(
        allocation_mode: str,
    ) -> ParallelStrategy | None:
        """
        Extract pattern-based parallelism strategy from the allocation mode string:

        Pattern-based allocation mode designate number of GPU used and parallel strategies of train engines as
        well as generation servers. The 5D parallel strategy can be represented as a
        string with following rules:

            Parallel degrees are represented by single characters:
            - 't': Tensor parallel
            - 'p': Pipeline parallel
            - 'd': Data parallel
            - 'c': Context parallel
            - 'e': Expert parallel

            Pattern Rules:
            1. Format: Each parallel degree is immediately followed by its integer size (e.g., 'd4' = data parallel size 4)
            2. Minimum requirements: Must include at least 3 core degrees: tensor ('t'), pipeline ('p'), and data ('d')
            - 3D parallelism: "d2p4t1" (data=2, pipeline=4, tensor=1)
            3. Adding context/expert parallel degrees:
            - + context parallel: "d2p2t2c2"
            - + expert parallel: "d2p2t2e2"
            - 5D parallelism: "d2p2t2c2e2"
            4. Hybrid configuration:
            - Use '/' to separate two strategies for attention modules (context parallel) and expert modules for MoE (expert parallel)
            with different tensor/data parallel degrees.
            - This mode only supports training/inference of MoE models implemented by megatron-core.
            - Constraints:
                a) Total GPUs (degree product) must match between strategies
                b) Pipeline degree ('p') must be identical in both strategies
            - Example: "d4p2t2c2/d2p2t2e4"
                a) Attention: data=4, pipeline=2, tensor=2, context=2 -> 4*2*2*2=32 GPUs
                b) Expert: data=2, pipeline=2, tensor=2, expert=4 -> 2*2*2*4=32 GPUs
                c) Pipeline size 'p2' matches in both segments
            - For details, refer to: https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#moe-parallel-folding
        """

        def extract_from_pattern(_allocation_mode: str, pattern_list: List[str]):
            for names in itertools.permutations(pattern_list):
                pattern = rf"^"
                for name in names:
                    pattern += rf"{name}(\d+)"
                pattern += rf"$"
                m = re.match(pattern, _allocation_mode)
                if not m:
                    continue
                groups = map(int, m.groups())
                # to be consistent with the key-value pattern
                return {name: val for name, val in zip(names, groups)}
            return {}

        pattern_lists = [
            ["p", "d", "t"],  # 3d parallel
            ["p", "d", "c", "t"],  # context parallel
            ["p", "d", "e", "t"],  # expert parallel
            [
                "p",
                "d",
                "e",
                "c",
                "t",
            ],  # full 5d parallel, expert_tensor_size==tensor_size
        ]
        for pattern_list in pattern_lists:
            res = extract_from_pattern(allocation_mode, pattern_list)
            if res:
                return ParallelStrategy(
                    tensor_parallel_size=res.get("t", 1),
                    pipeline_parallel_size=res.get("p", 1),
                    data_parallel_size=res.get("d", 1),
                    context_parallel_size=res.get("c", 1),
                    expert_parallel_size=res.get("e", 1),
                )

        # context parallel + expert parallel, and expert_tensor_size != tensor_size
        if "/" in allocation_mode:
            cp_allocation_mode, ep_allocation_mode = allocation_mode.split("/")
            cp_pattern_lists = [["p", "d", "t"], ["p", "d", "c", "t"]]
            cp_pattern = None
            for cp_pattern_list in cp_pattern_lists:
                cp_pattern = extract_from_pattern(cp_allocation_mode, cp_pattern_list)
            ep_pattern = extract_from_pattern(ep_allocation_mode, ["p", "d", "e", "t"])

            if ep_pattern and cp_pattern:
                if cp_pattern["p"] != ep_pattern["p"]:
                    raise InvalidAllocationModeError(
                        "Pipeline parallel size for expert modules and other modules should be the same, "
                        f"current allocation mode: {allocation_mode}"
                    )
                cp_pattern_world_size = math.prod(cp_pattern.values())
                ep_pattern_world_size = math.prod(ep_pattern.values())
                if cp_pattern_world_size != ep_pattern_world_size:
                    raise InvalidAllocationModeError(
                        "World size for expert modules and other modules should be the same,"
                        f"current allocation mode: {allocation_mode}"
                    )
                return ParallelStrategy(
                    tensor_parallel_size=cp_pattern.get("t", 1),
                    pipeline_parallel_size=cp_pattern.get("p", 1),
                    data_parallel_size=cp_pattern.get("d", 1),
                    context_parallel_size=cp_pattern.get("c", 1),
                    expert_parallel_size=ep_pattern.get("e", 1),
                    expert_tensor_parallel_size=ep_pattern.get("t", 1),
                )
        return None

    @staticmethod
    def extract_gen_alloc(allocation_mode: str) -> ParallelStrategy | None:
        pattern = re.compile(r"^(?:vllm|sglang)\.(.+)$")
        m = pattern.match(allocation_mode)
        if not m:
            return None
        return AllocationMode.extract_parallelism_strategy(m.group(1))

    @staticmethod
    def extract_decoupled_alloc(
        allocation_mode: str,
    ) -> Tuple[ParallelStrategy | None, ParallelStrategy | None]:
        pattern = re.compile(
            r"^(?:(?:vllm|sglang)\.(.+?)\+(.+))|(?:(.+?)\+(?:vllm|sglang)\.(.+))$"
        )
        m = pattern.match(allocation_mode)
        if not m:
            return None, None
        if m.group(1):
            gen_alloc = m.group(1)
            other_alloc = m.group(2)
        else:
            gen_alloc = m.group(4)
            other_alloc = m.group(3)
        gen_alloc = AllocationMode.extract_parallelism_strategy(gen_alloc)
        other_alloc = AllocationMode.extract_parallelism_strategy(other_alloc)
        return gen_alloc, other_alloc

    @staticmethod
    def get_gen_backend(allocation_mode: str) -> Optional[str]:
        if "vllm" in allocation_mode:
            return "vllm"
        if "sglang" in allocation_mode:
            return "sglang"
        return None
