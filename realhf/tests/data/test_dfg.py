# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import collections
import dataclasses
import pathlib
import pickle
from typing import *

import matplotlib.pyplot as plt
import networkx as nx
import pytest

from realhf.api.core.config import ModelInterfaceAbstraction, ModelName
from realhf.api.core.dfg import MFCDef, ModelInterfaceType, build_graph
from realhf.base import logging


def _get_ppo_rpcs() -> List[MFCDef]:
    actor_gen = MFCDef(
        n_seqs=1,
        name="actorGen",
        interface_type=ModelInterfaceType.GENERATE,
        interface_impl=ModelInterfaceAbstraction(""),
        model_name="actor",
        input_keys=["prompt"],
        output_keys=["input_ids", "oldlogp"],
    )
    rew_inf = MFCDef(
        n_seqs=1,
        name="rewInf",
        interface_type=ModelInterfaceType.INFERENCE,
        interface_impl=ModelInterfaceAbstraction(""),
        model_name="rew",
        input_keys=["input_ids"],
        output_keys=["rew"],
    )
    ref_inf = MFCDef(
        n_seqs=1,
        name="refInf",
        interface_type=ModelInterfaceType.INFERENCE,
        interface_impl=ModelInterfaceAbstraction(""),
        model_name="ref",
        input_keys=["input_ids"],
        output_keys=["reflogp"],
    )
    critic_inf = MFCDef(
        n_seqs=1,
        name="criticInf",
        interface_type=ModelInterfaceType.INFERENCE,
        interface_impl=ModelInterfaceAbstraction(""),
        model_name=ModelName("critic", 0),
        input_keys=["input_ids"],
        output_keys=["oldvalue"],
    )
    actor_train = MFCDef(
        n_seqs=1,
        name="actorTrain",
        interface_type=ModelInterfaceType.TRAIN_STEP,
        interface_impl=ModelInterfaceAbstraction(""),
        model_name=ModelName("actor", 1),
        input_keys=["input_ids", "oldlogp", "reflogp", "rew", "oldvalue"],
    )
    critic_train = MFCDef(
        n_seqs=1,
        name="criticTrain",
        interface_type=ModelInterfaceType.TRAIN_STEP,
        interface_impl=ModelInterfaceAbstraction(""),
        model_name=ModelName("critic", 1),
        input_keys=["input_ids", "oldlogp", "reflogp", "rew", "oldvalue"],
    )
    return [actor_gen, rew_inf, ref_inf, critic_inf, actor_train, critic_train]


def _get_reinforce_rpcs():
    actor_gen = MFCDef(
        n_seqs=1,
        name="greedyGen",
        interface_type=ModelInterfaceType.GENERATE,
        interface_impl=ModelInterfaceAbstraction(""),
        model_name="actor",
        input_keys=["prompt"],
        output_keys=["greedy_input_ids"],
    )
    actor_sample = MFCDef(
        n_seqs=1,
        name="sampleGen",
        interface_type=ModelInterfaceType.GENERATE,
        interface_impl=ModelInterfaceAbstraction(""),
        model_name="actor",
        input_keys=["prompt"],
        output_keys=["sampled_input_ids"],
    )
    greedy_inf = MFCDef(
        n_seqs=1,
        name="greedy_inf",
        interface_type=ModelInterfaceType.INFERENCE,
        interface_impl=ModelInterfaceAbstraction(""),
        model_name="rew",
        input_keys=["greedy_input_ids"],
        output_keys=["greedy_rew"],
    )
    sample_inf = MFCDef(
        n_seqs=1,
        name="sample_inf",
        interface_type=ModelInterfaceType.INFERENCE,
        interface_impl=ModelInterfaceAbstraction(""),
        model_name="rew",
        input_keys=["sampled_input_ids"],
        output_keys=["sample_rew"],
    )
    actor_train = MFCDef(
        n_seqs=1,
        name="actorTrain",
        interface_type=ModelInterfaceType.TRAIN_STEP,
        interface_impl=ModelInterfaceAbstraction(""),
        model_name=ModelName("actor", 1),
        input_keys=["sampled_input_ids", "greedy_rew", "sample_rew"],
    )
    return [actor_gen, actor_train, greedy_inf, actor_sample, sample_inf]


@pytest.mark.parametrize("rpcs", [_get_ppo_rpcs(), _get_reinforce_rpcs()])
def test_build_graph(tmp_path: pathlib.Path, rpcs: List[MFCDef]):
    G = build_graph(rpcs, verbose=True, graph_path=str(tmp_path / "dfg.png"))
    assert nx.is_directed_acyclic_graph(G)
    for node in rpcs:
        node._G = G

    for node in rpcs:
        # Ensure that all attributes are accessible.
        res = dict(
            name=node.name,
            role=node.role,
            is_src=node.is_src,
            is_dst=node.is_dst,
            data_producers=node.data_producers,
            data_consumers=node.data_consumers,
            parents=node.parents,
            children=node.children,
            is_dst_of_model_role=node.is_dst_of_model_role,
        )

        # Ensure node is picklable.
        node_ = pickle.loads(pickle.dumps(node))
        for k, v in dataclasses.asdict(node_).items():
            if k.startswith("_"):
                continue
            assert v == dataclasses.asdict(node)[k]
