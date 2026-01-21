# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import collections
import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx

import realhf.base.logging as logging
from realhf.api.core.config import (
    ModelInterfaceAbstraction,
    ModelInterfaceType,
    ModelName,
)
from realhf.api.core.data_api import MicroBatchSpec

logger = logging.getLogger("DataFlowGraph", "benchmark")


@dataclasses.dataclass
class OffloadHook:
    pass


@dataclasses.dataclass
class ParamReallocHook:
    """Hook for reallocating weights between source and target.

    Weights are transferred from the source model to the target model.
    Only one of `source` or `target` should be provided; the other should be
    the model name of the hooked MFC.

    The weights are updated using the formula: `target = eta * source + (1 - eta) * target`.

    :param source: The model name of the source from which weights are transferred.
    :type source: Optional[ModelName]
    :param target: The model name of the target to which weights are transferred.
    :type target: Optional[ModelName]
    :param eta: The weight for the source in the update formula. The default is 1.0,
        meaning that the target will be completely overwritten by the source.
    :type eta: float
    """

    source: Optional[ModelName] = None
    target: Optional[ModelName] = None
    eta: float = 1.0


RPCHook = Union[OffloadHook, ParamReallocHook]


@dataclasses.dataclass
class MFCDef:
    """A model function call (MFC) object used by the workers.

    MFC stands for Model Function Call. This object serves as the interface for
    developing new algorithms and will be inserted into an `nx.DiGraph` as nodes.
    Edges will be automatically resolved based on input/output keys.

    Fields starting with an underscore are filled automatically.

    **Note:** In the ReaL implementation, the term RPC also refers to MFC.

    :param name: The unique identifier for this model function call.
    :type name: str
    :param n_seqs: The number of sequences to be processed in a batch.
    :type n_seqs: int
    :param interface_type: The type of interface used by the node (e.g., generate, train_step).
    :type interface_type: ModelInterfaceType
    :param interface_impl: The actual implementation of the interface when running this node.
    :type interface_impl: ModelInterface
    :param model_name: The model identifier used by the node, corresponding to a unique LLM.
        The user-provided model name can be a string; the replica ID will be resolved in ReaL.
    :type model_name: str or ModelName
    :param input_keys: Input data keys used to resolve dependencies.
    :type input_keys: Tuple
    :param output_keys: Output data keys used to resolve dependencies.
    :type output_keys: Tuple
    :param input_key_remap: Remap input keys to identifiers recognized by the interface implementation.
        Keys are from `input_keys` and values are identifiers known to the interface.
    :type input_key_remap: Dict[str, str]
    :param output_key_remap: Remap output keys to identifiers recognized by MFC.
        Keys are identifiers known to the interface, and values are from `output_keys`.
    :type output_key_remap: Dict[str, str]
    :param mb_spec: The approach to dividing micro-batches. Check MicroBatchSpec for details.
    :type mb_spec: MicroBatchSpec.
    :param min_n_seqs_per_pass: The minimum number of sequences for each model interface pass.
        If the interface does not further split the batch, this value should be 1. Otherwise,
        it should be the minimum number of required mini-batches, e.g., PPO minibatch.
    :type min_n_seqs_per_pass: int
    :param log_return_value: Whether to log the return value of the interface implementation.
    :type log_return_value: bool
    """

    # The unique identifier of this model function call.
    name: str

    # batch size
    n_seqs: int

    # The interface type to be used by the node (e.g., generate, train_step).
    interface_type: ModelInterfaceType
    interface_impl: ModelInterfaceAbstraction

    # The model identifier to be used by the node.
    model_name: str | ModelName

    # Input and output keys, used to resolve dependencies.
    input_keys: Tuple = dataclasses.field(default_factory=tuple)
    input_key_remap: Dict[str, str] = dataclasses.field(default_factory=lambda: {})
    output_keys: Tuple = dataclasses.field(default_factory=tuple)
    output_key_remap: Dict[str, str] = dataclasses.field(default_factory=lambda: {})

    mb_spec: MicroBatchSpec = dataclasses.field(default_factory=MicroBatchSpec)
    min_n_seqs_per_pass: int | float = 1
    log_return_value: bool = False

    # Reserved dataclasses.fields. Should not be set by the user.
    _G: nx.DiGraph = None
    _pre_hooks: List[RPCHook] = dataclasses.field(default_factory=lambda: [])
    _post_hooks: List[RPCHook] = dataclasses.field(default_factory=lambda: [])

    def __post_init__(self):
        if isinstance(self.model_name, str):
            self.model_name = ModelName(role=self.model_name, replica_id=0)

    def __repr__(self):
        return f"MFCDef[{self.name}]"

    def __hash__(self):
        return hash(self.name)

    @property
    def role(self):
        return self.model_name.role

    def is_train(self):
        return self.interface_type in [ModelInterfaceType.TRAIN_STEP]

    def is_inference(self):
        return self.interface_type in [ModelInterfaceType.INFERENCE]

    def is_generate(self):
        return self.interface_type in [ModelInterfaceType.GENERATE]

    def add_pre_hook(self, h: RPCHook):
        assert isinstance(h, RPCHook), type(h)
        if isinstance(h, ParamReallocHook):
            assert h.target is None or h.source is None
        if isinstance(h, OffloadHook):
            raise ValueError("Offload can only be post hooks!")
        self._pre_hooks.append(h)

    def add_post_hook(self, h: RPCHook):
        if isinstance(h, ParamReallocHook):
            assert h.target is None or h.source is None
        self._post_hooks.append(h)

    @property
    def is_src(self):
        return len(list(self._G.predecessors(self.name))) == 0

    @property
    def is_dst(self):
        return len(list(self._G.successors(self.name))) == 0

    @property
    def data_producers(self) -> Dict[str, ModelName]:
        return self._G.graph["data_producers"]

    @property
    def data_consumers(self) -> Dict[str, List[str]]:
        return self._G.graph["data_consumers"]

    @property
    def parents(self) -> List["MFCDef"]:
        return [self._G.nodes[x]["object"] for x in self._G.predecessors(self.name)]

    @property
    def children(self) -> List["MFCDef"]:
        return [self._G.nodes[x]["object"] for x in self._G.successors(self.name)]

    def all_successors(self) -> List["MFCDef"]:
        names = list(nx.dfs_preorder_nodes(self._G, self.name))
        names.remove(self.name)
        return [self._G.nodes[x]["object"] for x in names]

    @property
    def is_dst_of_model_role(self):

        def _has_children_of_model_name(rpc: "MFCDef", model_name: ModelName):
            if rpc.is_dst:
                return False
            return any(
                [
                    r.role == model_name.role
                    or _has_children_of_model_name(r, model_name)
                    for r in rpc.children
                ]
            )

        return not _has_children_of_model_name(self, self.model_name)


def _draw_topo_sorted_digraph(G: nx.DiGraph, graph_path: str):
    topological_order = list(nx.topological_sort(G))
    # Initialize a dictionary to store the depth of each node
    node_depth = {node: 0 for node in G.nodes()}

    # Calculate the depth of each node
    for node in topological_order:
        for neighbor in G.successors(node):
            node_depth[neighbor] = max(node_depth[neighbor], node_depth[node] + 1)

    layers = {
        i: [node for node, depth in node_depth.items() if depth == i]
        for i in range(max(node_depth.values()) + 1)
    }
    pos = nx.multipartite_layout(G, subset_key=layers)
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=4000,
        node_color="lightblue",
        arrows=True,
        arrowsize=20,
        width=1.5,
    )
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_color="black")
    plt.savefig(graph_path, dpi=300)


def build_graph(
    nodes: List[MFCDef],
    verbose: bool = False,
    graph_path: Optional[str] = None,
) -> nx.DiGraph:
    if len(set(node.name for node in nodes)) != len(nodes):
        raise ValueError(
            "Each model function call should have an unique name. "
            f"Got {[node.name for node in nodes]}."
        )

    _G = nx.DiGraph()
    _G.add_nodes_from([(node.name, dict(object=node)) for node in nodes])

    data_producers: Dict[str, MFCDef] = {}
    data_consumers: Dict[str, List[MFCDef]] = collections.defaultdict(list)
    for node in nodes:
        for k in node.output_keys:
            data_producers[k] = node
        for k in node.input_keys:
            data_consumers[k].append(node)

    for node in nodes:
        for k in node.input_keys:
            if k not in data_producers:
                # This is a key from the dataset.
                continue
            src, dst = data_producers[k].name, node.name
            if _G.has_edge(src, dst):
                _G[src][dst]["keys"].append(k)
            else:
                _G.add_edge(src, dst, keys=[k])
    if verbose:
        for u, v, data in _G.edges(data=True):
            logger.info(f"Edge: {u} -> {v} with keys {data['keys']}")
        if graph_path is not None:
            _draw_topo_sorted_digraph(_G, graph_path)
            logger.info(
                f"> Visualization of the dataflow graph in "
                f"this experiment is saved to: {graph_path}."
            )

    if len(nodes) != len(_G.nodes):
        raise ValueError("There are replicated nodes in the graph!")

    # Store useful metadata
    _G.graph["data_producers"] = {k: v.model_name for k, v in data_producers.items()}
    _G.graph["data_consumers"] = {
        k: [v.model_name for v in vs] for k, vs in data_consumers.items()
    }

    return _G
