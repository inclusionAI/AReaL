# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
import enum
from typing import Any, Dict, List, Optional

import realhf.base.topology as topology


@dataclasses.dataclass
class DatasetAbstraction:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class EnvServiceAbstraction:
    type_: str = "null"
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class AgentAbstraction:
    type_: str = "null"
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ModelWrapperAbstraction:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ModelAbstraction:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)
    wrappers: List[ModelWrapperAbstraction] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ModelBackendAbstraction:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ModelInterfaceAbstraction:
    type_: str  # This type is the
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


class ModelInterfaceType(enum.Enum):
    GENERATE = "generate"
    TRAIN_STEP = "train_step"
    EVALUATE = "evaluate"
    INFERENCE = "inference"


@dataclasses.dataclass(unsafe_hash=True, order=True, frozen=True)
class ModelName:
    """A unique identifier for a model.

    :param role: The role of the model, e.g., "actor" or "critic".
    :type role: str
    :param replica_id: The replica ID of the model. Different replicas
        of the same role have the same set of parameters but different
        memory locations. For example, if actor generation and training
        in PPO use different parallel strategies, they will have the
        same role but different replica IDs.
    :type replica_id: int
    """

    role: str
    replica_id: int

    @property
    def name(self):
        return str(self)


@dataclasses.dataclass
class ModelShardID:
    """The ID of a model shard in a specific model worker.

    This ID is essentially a combination of the model name and the 3D
    parallelism rank, and can be used as a dictionary key. It represents
    the identity of a "model handler". The master worker maintains a
    lookup table mapping the ModelShardID to the model worker index,
    which can be a many-to-one mapping. Requests are created with the
    ModelShardID; for example, actors with ranks (dp=*, mp=0, pp=0)
    should transfer data to the critics. The ModelShardID is then mapped
    to the model worker index, and the requests are sent to the
    corresponding model workers.

    :param model_name: The name of the model.
    :type model_name: ModelName
    :param dp_rank: The data parallel rank.
    :type dp_rank: int
    :param tp_rank: The tensor-model parallel rank.
    :type tp_rank: int
    :param pp_rank: The pipeline-model parallel rank.
    :type pp_rank: int
    :param topo: The 3D parallelism topology of this model.
    :type topo: ProcessTopology
    """

    model_name: ModelName
    dp_rank: int
    tp_rank: int
    pp_rank: int
    topo: topology.ProcessTopology

    def __post_init__(self):
        assert self.dp_rank >= 0 and self.tp_rank >= 0 and self.pp_rank >= 0
        if "@" in self.model_name.role:
            raise ValueError("model_name cannot contain @")
        assert self.dp_rank < self.topo.get_dim("data")
        assert self.tp_rank < self.topo.get_dim("tensor")
        assert self.pp_rank < self.topo.get_dim("pipe")

    @property
    def parallelism_rank(self):
        return self.topo.get_rank(
            data=self.dp_rank, tensor=self.tp_rank, pipe=self.pp_rank
        )

    @classmethod
    def from_parallelism_rank(cls, model_name, topo, parallelism_rank):
        c = topo.get_coord(parallelism_rank)
        return cls(
            model_name=model_name,
            dp_rank=c.data,
            tp_rank=c.tensor,
            pp_rank=c.pipe,
            topo=topo,
        )

    def __repr__(self):
        n = len(str(self.topo.world_size()))
        return f"{self.model_name}@pp{self.pp_rank:0{n}d}@tp{self.tp_rank:0{n}d}@dp{self.dp_rank:0{n}d}"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        # Compare the key attribute for equality
        if isinstance(other, ModelShardID):
            return (
                self.model_name == other.model_name
                and self.dp_rank == other.dp_rank
                and self.tp_rank == other.tp_rank
                and self.pp_rank == other.pp_rank
            )
        return False


@dataclasses.dataclass
class StandaloneModelShardAbstraction:
    id: ModelShardID
    model: ModelAbstraction
    backend: ModelBackendAbstraction
    # evaluation
    eval_dataset: Optional[DatasetAbstraction] = None
    eval_bs: int = 128
    should_instantiate: bool = True
